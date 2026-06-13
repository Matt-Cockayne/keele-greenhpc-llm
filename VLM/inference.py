"""
VLM/inference.py -- Basic single-prompt vision-language model inference example.

Loads a locally downloaded VLM and generates a response to a single prompt,
optionally conditioned on an image.  The prompt (and optional image path) can
be supplied via --prompt / --image-path flags, read from a JSON file via
--prompt-file, or defaults to a built-in demo text-only prompt.

JSON prompt file format (--prompt-file):
    {
        "prompt"    : "Describe the findings in this image.",
        "image_path": "/path/to/image.jpg",   // optional -- omit for text-only
        "system"    : "You are an expert radiologist."  // optional
    }

Supported model families (auto-detected from config.json model_type):
    MedGemma  -- /home/xrai/models/medgemma-4b-it
    LLaVA-Med -- /home/xrai/models/llava-med-v1.5-mistral-7b

For batch multi-prompt inference see: multi-prompt.py

HPC notes:
    - Weights must be pre-downloaded on the head node; compute nodes on Keele
      GreenHPC have no outbound internet access.
    - TRANSFORMERS_OFFLINE=1 is set in the SLURM script to enforce this.
    - bfloat16 halves VRAM vs float32 with negligible quality loss on A100/H100.
    - device_map="auto" spreads the model across available GPU slices.
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    LlavaForConditionalGeneration,
)

# ---------------------------------------------------------------------------
# Defaults -- override via CLI arguments or environment variables.
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = "/home/xrai/models/medgemma-4b-it"
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "vlm_results")
DEFAULT_PROMPT = "Describe what you observe in this image in detail."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Basic VLM inference")
    p.add_argument(
        "--model-path",
        default=os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH),
        help="Local path to model weights directory.",
    )
    p.add_argument(
        "--model-type",
        choices=["auto", "medgemma", "llava"],
        default="auto",
        help="Model family.  'auto' detects from config.json (default).",
    )
    p.add_argument(
        "--prompt",
        default=None,
        help="Prompt string (takes precedence over --prompt-file).",
    )
    p.add_argument(
        "--image-path",
        default=None,
        help="Path to image file.  Omit for text-only inference.",
    )
    p.add_argument(
        "--prompt-file",
        default=None,
        help='Path to a JSON file: {"prompt": "...", "image_path": "...", "system": "..."}. '
             "Falls back to built-in demo prompt when omitted.",
    )
    p.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the JSON result file will be written.",
    )
    p.add_argument("--max-new-tokens", type=int, default=512)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model-family detection
# ---------------------------------------------------------------------------

def detect_model_family(model_path: str) -> str:
    """Return 'medgemma' or 'llava', inferred from config.json model_type.

    Falls back to a directory-name heuristic if config.json is absent.
    """
    cfg_path = Path(model_path) / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        mt = cfg.get("model_type", "").lower()
        if mt in ("paligemma", "gemma", "gemma2", "gemma3"):
            return "medgemma"
        if mt in ("llava", "llava_next", "llava_mistral"):
            return "llava"
    # Heuristic fallback: check directory name.
    name = Path(model_path).name.lower()
    if "gemma" in name:
        return "medgemma"
    if "llava" in name:
        return "llava"
    return "medgemma"  # safe default


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str, model_family: str):
    """Load processor and model for the given model family."""
    print(f"Loading {model_family} processor from: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)

    print(f"Loading {model_family} model from: {model_path}")
    if model_family == "medgemma":
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        )
    else:  # llava
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        )
    model.eval()
    return processor, model


# ---------------------------------------------------------------------------
# Inference helpers (one per model family)
# ---------------------------------------------------------------------------

def _run_medgemma(
    processor,
    model,
    prompt: str,
    image: Image.Image | None,
    system: str | None,
    max_new_tokens: int,
) -> tuple[str, float, float]:
    system_text = system or "You are a helpful medical AI assistant."
    user_content: list = []
    if image is not None:
        user_content.append({"type": "image", "image": image})
    user_content.append({"type": "text", "text": prompt})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": user_content},
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        t0 = time.perf_counter()
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        t1 = time.perf_counter()

    latency_s = t1 - t0
    generated_ids = output_ids[0][input_len:]
    n_tokens = len(generated_ids)
    tokens_per_sec = n_tokens / latency_s if latency_s > 0 else 0.0
    text = processor.decode(generated_ids, skip_special_tokens=True)
    return text, latency_s, tokens_per_sec


def _run_llava(
    processor,
    model,
    prompt: str,
    image: Image.Image | None,
    system: str | None,
    max_new_tokens: int,
) -> tuple[str, float, float]:
    # LLaVA 1.5 uses the vicuna/sharegpt prompt format.
    # The <image> token must appear in the text for image-conditioned inference.
    if image is not None:
        prompt_text = f"USER: <image>\n{prompt}\nASSISTANT:"
        inputs = processor(
            text=prompt_text, images=image, return_tensors="pt"
        ).to(model.device)
    else:
        prompt_text = f"USER: {prompt}\nASSISTANT:"
        inputs = processor(text=prompt_text, return_tensors="pt").to(model.device)

    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        t0 = time.perf_counter()
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        t1 = time.perf_counter()

    latency_s = t1 - t0
    generated_ids = output_ids[0][input_len:]
    n_tokens = len(generated_ids)
    tokens_per_sec = n_tokens / latency_s if latency_s > 0 else 0.0
    text = processor.decode(generated_ids, skip_special_tokens=True)
    return text, latency_s, tokens_per_sec


def run_inference(
    processor,
    model,
    model_family: str,
    prompt: str,
    image: Image.Image | None,
    system: str | None,
    max_new_tokens: int,
) -> tuple[str, float, float]:
    """Dispatch to the correct inference function for the model family."""
    if model_family == "medgemma":
        return _run_medgemma(processor, model, prompt, image, system, max_new_tokens)
    return _run_llava(processor, model, prompt, image, system, max_new_tokens)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Resolve prompt, image path, and optional system message.
    # ------------------------------------------------------------------
    image_path_str: str | None = None
    system: str | None = None

    if args.prompt:
        prompt = args.prompt
        image_path_str = args.image_path
    elif args.prompt_file:
        with open(args.prompt_file) as f:
            data = json.load(f)
        prompt = data["prompt"]
        image_path_str = data.get("image_path") or args.image_path
        system = data.get("system")
        print(f"Loaded prompt from: {args.prompt_file}")
    else:
        prompt = DEFAULT_PROMPT
        print("No prompt provided; using built-in demo prompt.")

    image: Image.Image | None = None
    if image_path_str:
        image = Image.open(image_path_str).convert("RGB")
        print(f"Image loaded: {image_path_str}  ({image.width}x{image.height})")
    else:
        print("No image provided; running in text-only mode.")

    # ------------------------------------------------------------------
    # Detect model family and load model.
    # ------------------------------------------------------------------
    model_family = args.model_type
    if model_family == "auto":
        model_family = detect_model_family(args.model_path)
    print(f"Model family: {model_family}")

    t_load = time.perf_counter()
    processor, model = load_model(args.model_path, model_family)
    load_time_s = time.perf_counter() - t_load
    if hasattr(model, "hf_device_map"):
        print(f"Device map: {model.hf_device_map}")
    print(f"Model load time: {load_time_s:.1f}s")

    # ------------------------------------------------------------------
    # Run inference.
    # ------------------------------------------------------------------
    sep = "=" * 60
    print(f"\n{sep}\nPROMPT\n{sep}")
    print(prompt)

    text, latency_s, tokens_per_sec = run_inference(
        processor, model, model_family, prompt, image, system, args.max_new_tokens,
    )
    n_tokens = int(latency_s * tokens_per_sec)

    print(f"\n{sep}\nRESPONSE\n{sep}")
    print(text)
    print(f"\nLatency: {latency_s:.2f}s  |  {n_tokens} tokens  |  {tokens_per_sec:.1f} tok/s")

    # ------------------------------------------------------------------
    # Save result.
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "timestamp": timestamp,
        "model_path": args.model_path,
        "model_family": model_family,
        "generation_config": {
            "max_new_tokens": args.max_new_tokens,
        },
        "prompt": prompt,
        "system": system,
        "image_path": image_path_str,
        "response": text,
        "metrics": {
            "model_load_time_s": round(load_time_s, 3),
            "latency_s": round(latency_s, 3),
            "tokens_generated": n_tokens,
            "tokens_per_second": round(tokens_per_sec, 1),
        },
    }
    out_path = output_dir / f"vlm_inference_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved to: {out_path}")


if __name__ == "__main__":
    main()
