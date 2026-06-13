"""
LLM/multi-prompt.py -- Batch multi-prompt inference from a JSON file.

Loads the model once, then iterates sequentially through all prompts,
recording each response before moving to the next.  The model is not reloaded
between prompts, so total wall time is model-load + sum(per-prompt latency).

Prompts JSON file format (--prompts-file):
    A JSON array where each element is either:
      • A plain string  -- used directly as the user prompt.
      • An object with keys:
            "id"     (optional str) : label carried through to the output
            "prompt" (required str) : the user prompt
            "system" (optional str) : system message override for this entry

    Example (LLM/prompts/multi.json):
        [
            "Explain transformer encoder vs decoder architectures.",
            {
                "id": "cardiology_q",
                "prompt": "What are the main causes of heart failure?",
                "system": "You are a helpful cardiologist."
            }
        ]

Output JSON (data/llm_results/llm_multi_<timestamp>.json):
    {
        "timestamp": "...",
        "model_path": "...",
        "prompts_file": "...",
        "generation_config": { ... },
        "aggregate_metrics": { ... },
        "results": [
            {
                "id": "1",
                "prompt": "...",
                "system": null,
                "response": "...",
                "metrics": { "latency_s": ..., "tokens_generated": ..., "tokens_per_second": ... }
            },
            ...
        ]
    }

Models tested:
    /home/xrai/models/Meta-Llama-3.1-8B-Intstruct

HPC notes:
    - Submit via LLM/multi-prompt-job.slurm; set PROMPTS_FILE before sbatch.
    - Loading takes ~30-60 s on a cold A100; subsequent prompts are fast.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Share model loading and inference logic from the sibling inference.py.
sys.path.insert(0, os.path.dirname(__file__))
from inference import load_model, run_inference

DEFAULT_MODEL_PATH = "/home/xrai/models/Meta-Llama-3.1-8B-Intstruct"
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "llm_results")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch multi-prompt causal LLM inference")
    p.add_argument(
        "--model-path",
        default=os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH),
        help="Local path to model weights directory.",
    )
    p.add_argument(
        "--prompts-file",
        required=True,
        help="Path to a JSON array of prompts.  See module docstring for format.",
    )
    p.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the JSON results file will be written.",
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


def normalise_entry(entry) -> tuple[str | None, str, str | None]:
    """Return (id, prompt, system) from a prompt list entry (str or dict)."""
    if isinstance(entry, str):
        return None, entry, None
    return entry.get("id"), entry["prompt"], entry.get("system")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load prompts from JSON file.
    # ------------------------------------------------------------------
    prompts_path = Path(args.prompts_file).resolve()
    with open(prompts_path) as f:
        raw_prompts = json.load(f)
    if not isinstance(raw_prompts, list):
        raise ValueError("--prompts-file must contain a JSON array at the top level.")
    print(f"Loaded {len(raw_prompts)} prompts from: {prompts_path}")

    # ------------------------------------------------------------------
    # Load model once for the whole batch.
    # ------------------------------------------------------------------
    t_load = time.perf_counter()
    tokenizer, model = load_model(args.model_path)
    load_time_s = time.perf_counter() - t_load
    if hasattr(model, "hf_device_map"):
        print(f"Device map: {model.hf_device_map}")
    print(f"Model load time: {load_time_s:.1f}s\n")

    # ------------------------------------------------------------------
    # Iterate over all prompts and record responses.
    # ------------------------------------------------------------------
    sep = "=" * 60
    all_results = []
    total_latency = 0.0
    total_tokens = 0

    for i, entry in enumerate(raw_prompts, start=1):
        prompt_id, prompt, system = normalise_entry(entry)
        label = prompt_id or str(i)

        print(f"{sep}")
        print(f"PROMPT {i}/{len(raw_prompts)}  (id={label})")
        print(f"{sep}")
        print(prompt)

        text, latency_s, tokens_per_sec = run_inference(
            tokenizer, model, prompt, system,
            args.max_new_tokens, args.temperature, args.top_p,
        )
        n_tokens = int(latency_s * tokens_per_sec)
        total_latency += latency_s
        total_tokens += n_tokens

        print(f"\n{sep}")
        print("RESPONSE")
        print(f"{sep}")
        print(text)
        print(f"\n  Latency   : {latency_s:.2f}s")
        print(f"  Tokens    : {n_tokens}  ({tokens_per_sec:.1f} tok/s)")

        all_results.append({
            "id": label,
            "prompt": prompt,
            "system": system,
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
    mean_latency = total_latency / len(raw_prompts)
    mean_throughput = total_tokens / total_latency if total_latency > 0 else 0.0

    print(f"\n{sep}")
    print("BATCH COMPLETE")
    print(f"{sep}")
    print(f"  Prompts processed   : {len(raw_prompts)}")
    print(f"  Model load time     : {load_time_s:.2f}s")
    print(f"  Total tokens gen.   : {total_tokens}")
    print(f"  Mean latency/prompt : {mean_latency:.2f}s")
    print(f"  Mean throughput     : {mean_throughput:.1f} tok/s")

    # ------------------------------------------------------------------
    # Save all results to a single JSON file.
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_doc = {
        "timestamp": timestamp,
        "model_path": args.model_path,
        "prompts_file": str(prompts_path),
        "generation_config": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "aggregate_metrics": {
            "model_load_time_s": round(load_time_s, 3),
            "total_prompts": len(raw_prompts),
            "total_tokens_generated": total_tokens,
            "mean_latency_s": round(mean_latency, 3),
            "mean_tokens_per_second": round(mean_throughput, 1),
        },
        "results": all_results,
    }
    out_path = output_dir / f"llm_multi_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(results_doc, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
