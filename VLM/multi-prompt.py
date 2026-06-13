"""
VLM/multi-prompt.py -- Batch multi-prompt VLM inference from a JSON file.

Loads the model once, then iterates sequentially through all entries in the
prompts file, recording each response before moving to the next.  Each entry
can optionally reference a different image, so the same job can answer
questions about multiple images without reloading the model.

Prompts JSON file format (--prompts-file):
    A JSON array where each element is an object with keys:
        "id"         (optional str) : label carried through to the output
        "prompt"     (required str) : the user prompt
        "image_path" (optional str) : path to an image file; omit for text-only
        "system"     (optional str) : system message override for this entry

    Example (VLM/prompts/multi.json):
        [
            {
                "id": "cxr_001",
                "prompt": "Describe the key findings in this chest X-ray.",
                "image_path": "/data/images/cxr_001.png",
                "system": "You are an expert radiologist."
            },
            {
                "id": "text_only",
                "prompt": "What are the typical radiological signs of pneumonia?"
            }
        ]

Output JSON (data/vlm_results/vlm_multi_<timestamp>.json):
    {
        "timestamp": "...",
        "model_path": "...",
        "model_family": "medgemma" | "llava",
        "prompts_file": "...",
        "generation_config": { ... },
        "aggregate_metrics": { ... },
        "results": [
            {
                "id": "cxr_001",
                "prompt": "...",
                "system": null,
                "image_path": "...",
                "response": "...",
                "metrics": { "latency_s": ..., "tokens_generated": ..., "tokens_per_second": ... }
            },
            ...
        ]
    }

Supported models:
    /home/xrai/models/medgemma-4b-it
    /home/xrai/models/llava-med-v1.5-mistral-7b

HPC notes:
    - Submit via VLM/multi-prompt-job.slurm; set MODEL_PATH and PROMPTS_FILE.
    - Loading takes ~30-90 s on a cold A100; subsequent prompts are fast.
    - Each entry can use a different image; images are loaded on demand and not
      cached -- keep batch sizes moderate if images are large.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from PIL import Image

# Share detection and model code from the sibling inference.py.
sys.path.insert(0, os.path.dirname(__file__))
from inference import detect_model_family, load_model, run_inference

DEFAULT_MODEL_PATH = "/home/xrai/models/medgemma-4b-it"
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "vlm_results")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch multi-prompt VLM inference")
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
        "--prompts-file",
        required=True,
        help="Path to a JSON array of prompt entries.  See module docstring for format.",
    )
    p.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the JSON results file will be written.",
    )
    p.add_argument("--max-new-tokens", type=int, default=512)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load prompts from JSON file.
    # ------------------------------------------------------------------
    prompts_path = Path(args.prompts_file).resolve()
    with open(prompts_path) as f:
        entries = json.load(f)
    if not isinstance(entries, list):
        raise ValueError("--prompts-file must contain a JSON array at the top level.")
    print(f"Loaded {len(entries)} entries from: {prompts_path}")

    # ------------------------------------------------------------------
    # Detect model family and load model once for the whole batch.
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
    print(f"Model load time: {load_time_s:.1f}s\n")

    # ------------------------------------------------------------------
    # Iterate over all entries and record responses.
    # ------------------------------------------------------------------
    sep = "=" * 60
    all_results = []
    total_latency = 0.0
    total_tokens = 0

    for i, entry in enumerate(entries, start=1):
        prompt_id = entry.get("id", str(i))
        prompt = entry["prompt"]
        system = entry.get("system")
        image_path_str: str | None = entry.get("image_path")

        image: Image.Image | None = None
        if image_path_str:
            image = Image.open(image_path_str).convert("RGB")

        print(f"{sep}")
        print(f"ENTRY {i}/{len(entries)}  (id={prompt_id})")
        print(f"{sep}")
        if image_path_str:
            print(f"Image : {image_path_str}")
        print(f"Prompt: {prompt}")

        text, latency_s, tokens_per_sec = run_inference(
            processor, model, model_family, prompt, image, system, args.max_new_tokens,
        )
        n_tokens = int(latency_s * tokens_per_sec)
        total_latency += latency_s
        total_tokens += n_tokens

        print(f"\n{sep}")
        print("RESPONSE")
        print(f"{sep}")
        print(text)
        print(f"\n  Latency : {latency_s:.2f}s")
        print(f"  Tokens  : {n_tokens}  ({tokens_per_sec:.1f} tok/s)")

        all_results.append({
            "id": prompt_id,
            "prompt": prompt,
            "system": system,
            "image_path": image_path_str,
            "response": text,
            "metrics": {
                "latency_s": round(latency_s, 3),
                "tokens_generated": n_tokens,
                "tokens_per_second": round(tokens_per_sec, 1),
            },
        })

    # ------------------------------------------------------------------
    # Print aggregate metrics.
    # ------------------------------------------------------------------
    mean_latency = total_latency / len(entries)
    mean_throughput = total_tokens / total_latency if total_latency > 0 else 0.0

    print(f"\n{sep}")
    print("BATCH COMPLETE")
    print(f"{sep}")
    print(f"  Entries processed   : {len(entries)}")
    print(f"  Model load time     : {load_time_s:.2f}s")
    print(f"  Total tokens gen.   : {total_tokens}")
    print(f"  Mean latency/entry  : {mean_latency:.2f}s")
    print(f"  Mean throughput     : {mean_throughput:.1f} tok/s")

    # ------------------------------------------------------------------
    # Save all results to a single JSON file.
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_doc = {
        "timestamp": timestamp,
        "model_path": args.model_path,
        "model_family": model_family,
        "prompts_file": str(prompts_path),
        "generation_config": {
            "max_new_tokens": args.max_new_tokens,
        },
        "aggregate_metrics": {
            "model_load_time_s": round(load_time_s, 3),
            "total_entries": len(entries),
            "total_tokens_generated": total_tokens,
            "mean_latency_s": round(mean_latency, 3),
            "mean_tokens_per_second": round(mean_throughput, 1),
        },
        "results": all_results,
    }
    out_path = output_dir / f"vlm_multi_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(results_doc, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
