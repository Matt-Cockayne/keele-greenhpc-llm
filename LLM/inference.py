"""
LLM/inference.py -- Basic single-prompt causal LM inference example.

Loads a locally downloaded causal LM and generates a response to a single
prompt. The prompt can be supplied inline via --prompt, read from a JSON file
via --prompt-file, or defaults to a built-in demo prompt.

JSON prompt file format (--prompt-file):
    {
        "prompt": "What are the main causes of heart failure?",
        "system": "You are a helpful medical assistant."   // optional
    }

For batch multi-prompt inference see: multi-prompt.py

Models tested:
    /home/xrai/models/Meta-Llama-3.1-8B-Intstruct

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
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Defaults -- override via CLI arguments or environment variables.
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = "/home/xrai/models/Meta-Llama-3.1-8B-Intstruct"
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "llm_results")
DEFAULT_PROMPT = "Explain the difference between transformer encoder and decoder architectures."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Basic causal LLM inference")
    p.add_argument(
        "--model-path",
        default=os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH),
        help="Local path to model weights directory.",
    )
    p.add_argument(
        "--prompt",
        default=None,
        help="Prompt string (takes precedence over --prompt-file).",
    )
    p.add_argument(
        "--prompt-file",
        default=None,
        help='Path to a JSON file: {"prompt": "...", "system": "..."}. '
             "Falls back to built-in demo prompt when omitted.",
    )
    p.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the JSON result file will be written.",
    )
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0.0 = greedy (deterministic); >0 enables sampling.",
    )
    p.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling mass; only used when temperature > 0.",
    )
    return p.parse_args()


def load_model(model_path: str):
    """Load tokeniser and causal LM from a local weights directory."""
    print(f"Loading tokeniser from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    # Many instruction-tuned models omit pad_token; reuse eos_token to avoid
    # a ValueError during generation.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()
    return tokenizer, model


def run_inference(
    tokenizer,
    model,
    prompt: str,
    system: str | None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[str, float, float]:
    """Tokenise a prompt and run generation.

    Returns (generated_text, latency_s, tokens_per_sec).
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    # apply_chat_template formats the conversation into the token sequence the
    # model was instruction-tuned on.  Falls back to a plain string if the
    # tokeniser has no chat template.
    if tokenizer.chat_template is not None:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted = prompt

    inputs = tokenizer(formatted, return_tensors="pt", padding=True).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    do_sample = temperature > 0.0
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.inference_mode():
        t0 = time.perf_counter()
        output_ids = model.generate(**gen_kwargs)
        t1 = time.perf_counter()

    latency_s = t1 - t0
    generated_ids = output_ids[0][input_len:]
    n_tokens = len(generated_ids)
    tokens_per_sec = n_tokens / latency_s if latency_s > 0 else 0.0

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text, latency_s, tokens_per_sec


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Resolve prompt and optional system message.
    # ------------------------------------------------------------------
    system = None
    if args.prompt:
        prompt = args.prompt
    elif args.prompt_file:
        with open(args.prompt_file) as f:
            data = json.load(f)
        prompt = data["prompt"]
        system = data.get("system")
        print(f"Loaded prompt from: {args.prompt_file}")
    else:
        prompt = DEFAULT_PROMPT
        print("No prompt provided; using built-in demo prompt.")

    # ------------------------------------------------------------------
    # Load model.
    # ------------------------------------------------------------------
    t_load = time.perf_counter()
    tokenizer, model = load_model(args.model_path)
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
        tokenizer, model, prompt, system,
        args.max_new_tokens, args.temperature, args.top_p,
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
        "generation_config": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "prompt": prompt,
        "system": system,
        "response": text,
        "metrics": {
            "model_load_time_s": round(load_time_s, 3),
            "latency_s": round(latency_s, 3),
            "tokens_generated": n_tokens,
            "tokens_per_second": round(tokens_per_sec, 1),
        },
    }
    out_path = output_dir / f"llm_inference_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved to: {out_path}")


if __name__ == "__main__":
    main()
