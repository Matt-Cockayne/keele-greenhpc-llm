# Keele GreenHPC -- LLM and VLM Inference

A repo for getting started with large language model (LLMs) and vision-language model (VLM) inference experiments on the Keele GreenHPC.

---

## How a HPC Cluster Works (Quick Primer)

A cluster has two types of nodes (servers):

- **Head node (login node)**: where you land when you SSH in. Has internet access. Use it for setup, downloads, and submitting jobs. Do **not** run heavy computation here.
- **Compute nodes**: the powerful GPU servers that actually run your experiments. You never SSH into them directly, instead you submit a *job* and Slurm schedules it for you.

**Slurm** is the job scheduler. You write a shell script (`.slurm`) that describes what resources you need and what to run, then submit it with `sbatch`. Slurm queues it, allocates a GPU node, runs it, and writes the output to a log file.

Compute nodes have **no internet access**, so all model weights must be downloaded on the head node before submitting a job.

---

## Repository Structure

```
keele-greenhpc-llm/
├── LLM/
│   ├── inference.py              # Single-prompt text inference
│   ├── multi-prompt.py           # Batch inference from a JSON prompts file
│   ├── inference-job.slurm       # Slurm job: single prompt
│   ├── multi-prompt-job.slurm    # Slurm job: batch prompts
│   └── prompts/
│       ├── single.json           # Example single-prompt input
│       └── multi.json            # Example multi-prompt batch input
├── VLM/
│   ├── inference.py              # Single-prompt vision-language inference
│   ├── multi-prompt.py           # Batch VLM inference from a JSON prompts file
│   ├── inference-job.slurm       # Slurm job: single prompt
│   ├── multi-prompt-job.slurm    # Slurm job: batch prompts
│   └── prompts/
│       ├── single.json           # Example single-prompt input
│       └── multi.json            # Example multi-prompt batch input
├── data/
│   ├── llm_results/              # LLM output JSON files (auto-created)
│   └── vlm_results/              # VLM output JSON files (auto-created)
└── README.md
```

---

## Supported Models

The following models are already downloaded to the shared project space:

| Model | Path | Type |
|---|---|---|
| Meta Llama 3.1 8B Instruct | `/home/xrai/models/Meta-Llama-3.1-8B-Intstruct` | LLM (text only) |
| MedGemma 4B-IT | `/home/xrai/models/medgemma-4b-it` | VLM (text + image) |
| LLaVA-Med v1.5 Mistral 7B | `/home/xrai/models/llava-med-v1.5-mistral-7b` | VLM (text + image) |

The VLM scripts auto-detect which model family is being used from its `config.json` — no code changes are needed to switch between MedGemma and LLaVA-Med.

---

## Step 1 — SSH into the Head Node

```bash
ssh <username>@keele-hpc-address
```
I recommend using the built-in VSCode remote explorer to ssh into the hpc.
https://code.visualstudio.com/docs/remote/ssh  

All remaining steps are run on the head node unless stated otherwise.

---

## Step 2 — Clone This Repository

```bash
cd ~
git clone <repo-url> projects/keele-greenhpc-llm
cd projects/keele-greenhpc-llm
```

---

## Step 3 — Set Up a Conda Environment

**Conda** is a package and environment manager. An *environment* is an isolated Python installation with its own packages. This prevents conflicts between projects.

```bash
# Create a new environment called keele-llm with Python 3.11.
conda create -n keele-llm python=3.11 -y

# Activate it (you need to do this at the start of each terminal session).
conda activate keele-llm

# Install PyTorch with CUDA 12.1 using the official pip wheels.
# Do NOT use 'conda install pytorch' — the conda-forge MKL version causes
# an 'undefined symbol: iJIT_NotifyEvent' crash at import time.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install the HuggingFace libraries and other dependencies.
pip install "transformers>=4.47.0" accelerate Pillow numpy tqdm sentencepiece tiktoken


# Install matplotlib for example plot generation 
conda install matplotlib
```

The environment lives on the shared filesystem, so compute nodes can use it automatically (you do not need to reinstall anything on the compute nodes.)

> **First time with conda?** Run `conda init bash` once after installing conda, then open a new terminal. You should see `(base)` in your prompt. After `conda activate keele-llm` it changes to `(keele-llm)`.

---

## Step 4 — File Storage

GreenHPC home directories (`/home/<username>/`) have a ~50 GB quota. Large model weights will fill this quickly. Use the shared project space instead:

| What | Where |
|---|---|
| Model weights | `/home/xrai/models/` |
| HuggingFace cache | `/home/xrai/.cache/huggingface/` |
| Your code and results | `/home/<username>/projects/` |

The Slurm scripts set `HF_HOME=/home/xrai/.cache/huggingface` automatically so HuggingFace artefacts go to the right place.

---

## Step 5 — (Optional) Download Additional Models

The three models listed above are already available. If you want to add a new model, download it on the head node:

```bash
# Log in to HuggingFace (one-time setup; needed for gated models).
hf auth login

# Example: download a model to the shared weights directory.
hf download <org/model-name> --local-dir /home/xrai/models/<model-name>
```

---

## Step 6 — Prompt File Formats

The scripts accept prompts from JSON files, which makes it easy to run the same experiment repeatedly or submit batches without editing code.

### Single prompt (`--prompt-file`)

```json
{
  "prompt": "What are the main causes of heart failure?",
  "system": "You are a helpful medical assistant."
}
```

`system` sets the model's persona and is optional. For VLM, add `image_path` to include an image:

```json
{
  "prompt": "Describe the key findings in this chest X-ray.",
  "image_path": "/absolute/path/to/image.png",
  "system": "You are an expert radiologist."
}
```

Omit `image_path` for text-only VLM inference.

### Multi-prompt batch (`--prompts-file`)

A JSON array — the model loads once and answers every prompt in order:

```json
[
  {
    "id": "q1",
    "prompt": "Explain transformer encoder vs decoder architectures."
  },
  {
    "id": "cxr_report",
    "prompt": "Describe the findings in this chest X-ray.",
    "image_path": "/data/images/cxr_001.png",
    "system": "You are an expert radiologist."
  }
]
```

`id` is an optional label that appears in the output JSON. See `LLM/prompts/multi.json` and `VLM/prompts/multi.json` for ready-to-run examples.

---

## Step 7 — Running Inference on hpc

You can run the scripts directly for quick tests (don't do this), or submit them to the cluster via Slurm (do this).

### Run directly (quick test)

```bash
cd ~/projects/keele-greenhpc-llm
conda activate keele-llm

# LLM — single prompt (uses built-in demo if no file given):
python LLM/inference.py --prompt-file LLM/prompts/single.json

# LLM — batch (loads model once, answers all prompts):
python LLM/multi-prompt.py --prompts-file LLM/prompts/multi.json

# VLM — single prompt (MedGemma by default):
python VLM/inference.py --prompt-file VLM/prompts/single.json

# VLM — batch:
python VLM/multi-prompt.py --prompts-file VLM/prompts/multi.json

# VLM — switch to LLaVA-Med:
python VLM/inference.py \
    --model-path /home/xrai/models/llava-med-v1.5-mistral-7b \
    --prompt-file VLM/prompts/single.json
```

### Submit to the cluster via Slurm

Slurm jobs run on a GPU compute node. Submit from the repository root:

```bash
cd ~/projects/keele-greenhpc-llm

# LLM single prompt:
PROMPT_FILE=LLM/prompts/single.json sbatch LLM/inference-job.slurm

# LLM batch:
PROMPTS_FILE=LLM/prompts/multi.json sbatch LLM/multi-prompt-job.slurm

# VLM single prompt (MedGemma):
PROMPT_FILE=VLM/prompts/single.json sbatch VLM/inference-job.slurm

# VLM batch (MedGemma):
PROMPTS_FILE=VLM/prompts/multi.json sbatch VLM/multi-prompt-job.slurm

# VLM batch (LLaVA-Med):
MODEL_PATH=/home/xrai/models/llava-med-v1.5-mistral-7b \
PROMPTS_FILE=VLM/prompts/multi.json \
sbatch VLM/multi-prompt-job.slurm

```

> **How `VAR=value sbatch ...` works:** setting a variable immediately before `sbatch` passes it as an environment variable to that command only. The Slurm scripts read these variables (e.g. `MODEL_PATH`, `PROMPT_FILE`) to configure the job without editing the script itself.

---

## Step 8 — Monitoring Jobs

After submitting, Slurm prints a job ID, e.g. `Submitted batch job 12345`.

```bash
# Check the status of your jobs (PD = pending, R = running, CG = completing).
squeue -u $USER

# See detailed info about a job (useful if it fails immediately).
scontrol show job 12345

# Cancel a job.
scancel 12345
```

While the job runs, Slurm writes output to `logs/` in the repository root:

```bash
# Stream live output from a running job:
tail -f logs/llm_inference_12345.out
tail -f logs/vlm_multi_12345.out
```

---

## Step 9 — Viewing Results

Each run writes a timestamped JSON file to `data/`:

| Script | Output |
|---|---|
| `LLM/inference.py` | `data/llm_results/llm_inference_YYYYMMDD_HHMMSS.json` |
| `LLM/multi-prompt.py` | `data/llm_results/llm_multi_YYYYMMDD_HHMMSS.json` |
| `VLM/inference.py` | `data/vlm_results/vlm_inference_YYYYMMDD_HHMMSS.json` |
| `VLM/multi-prompt.py` | `data/vlm_results/vlm_multi_YYYYMMDD_HHMMSS.json` |

```bash
# Pretty-print the most recent LLM result:
cat $(ls -t data/llm_results/llm_inference_*.json | head -1) | python -m json.tool

# Most recent VLM batch result:
cat $(ls -t data/vlm_results/vlm_multi_*.json | head -1) | python -m json.tool
```

The batch output contains an `aggregate_metrics` block and a `results` array with one entry per prompt (including the `id` you set, the full response, and per-prompt latency).

---

## Performance Reference

Typical figures on a single A100 40 GB (bfloat16, greedy decoding):

| Model | Load time | Throughput |
|---|---|---|
| MedGemma 4B | ~30 s | ~60 tokens/s |
| Llama 3.1 8B | ~45 s | ~40 tokens/s |
| LLaVA-Med 7B | ~60 s | ~30 tokens/s |

For batch jobs, the model loads once and all subsequent prompts run at full throughput.

---

## Troubleshooting

**`conda activate` says "conda: command not found"**
Run `conda init bash` and open a new terminal.

**"OSError: We couldn't connect to huggingface.co"**
The model path is wrong or the download is incomplete. Confirm `ls /home/xrai/models/<model>/config.json` exists.

**"CUDA out of memory"**
Lower `--max-new-tokens`, or request two GPUs in the SLURM script (`--gres=gpu:2`).

**Job stays in state `PD` (pending) for a long time**
The GPU partition is busy. Check availability with `sinfo` and your priority with `sshare -u $USER`.

**VLM model not detected correctly**
Pass `--model-type medgemma` or `--model-type llava` explicitly to bypass auto-detection.

**HuggingFace warnings about `padding_side`**
Cosmetic warnings; they do not affect output.

**"Permission denied" on `/var/spool/slurmd/...` / "No such file or directory" for the Python script**
You ran `sbatch` from the wrong directory. Always submit from the repository root:
```bash
cd ~/projects/keele-greenhpc-llm
PROMPTS_FILE=VLM/prompts/multi.json sbatch VLM/multi-prompt-job.slurm
```
Slurm copies the script to a spool directory before running it, so paths inside the script must be resolved via `$SLURM_SUBMIT_DIR` (the directory you called `sbatch` from) rather than relative to the script file.
