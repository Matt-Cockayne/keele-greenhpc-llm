#!/bin/bash
#SBATCH --job-name=llm_inference
#SBATCH --output=logs/llm_inference_%j.out
#SBATCH --error=logs/llm_inference_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4          # 4 CPU workers for tokenisation and I/O
#SBATCH --mem=32G                  # system RAM; model weights live in GPU VRAM
#SBATCH --gres=gpu:1               # 1 GPU; an 8B model fits on a single A100 40 GB
#SBATCH --partition=gpu

# =============================================================================
# LLM Inference Job -- Text-only causal language model
#
# Works with any HuggingFace AutoModelForCausalLM-compatible model
# (Llama 3, Mistral, Phi-3, etc.) that has been downloaded locally.
#
# Prerequisites (run once on the head node, not here):
#   1. conda create -n keele-llm python=3.11 -y
#   2. See README.md section "Environment Setup" for full package list.
#   3. Download the model weights (see README.md "Model Downloads").
#
# Submit:
#   cd projects/keele-greenhpc-llm
#   sbatch LLM/inference-job.slurm
#
# Optionally pass a custom prompts file:
#   PROMPTS_FILE=/path/to/prompts.json sbatch LLM/inference-job.slurm
# =============================================================================

echo "=========================================="
echo "LLM INFERENCE"
echo "=========================================="
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURM_NODELIST"
echo "GPUs     : $SLURM_GPUS_ON_NODE"
echo "Start    : $(date)"
echo "=========================================="

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
source ~/.bashrc
conda activate keele-llm

# HF_HOME controls where HuggingFace caches tokeniser configs, vocab files, etc.
# Point it to the shared project space to avoid consuming your home quota (~50 GB).
export HF_HOME="/home/xrai/.cache/huggingface"

# Block any outbound HuggingFace network calls at runtime.
# All weights must already be present at MODEL_PATH before job submission.
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Set MODEL_PATH here or export it before calling sbatch:
#   MODEL_PATH=/home/xrai/models/my-model sbatch LLM/inference-job.slurm
export MODEL_PATH="${MODEL_PATH:-/home/xrai/models/Meta-Llama-3.1-8B-Intstruct}"

# Optional: path to a single-prompt JSON file.
# Format: {"prompt": "...", "system": "..."}  (system is optional)
# Leave unset to use the built-in demo prompt in inference.py.
# For multi-prompt batches use multi-prompt-job.slurm instead.
PROMPT_FILE="${PROMPT_FILE:-}"

echo ""
echo "Model       : $MODEL_PATH"
echo "Prompt file : ${PROMPT_FILE:-<built-in demo prompt>}"
echo ""

# Log GPU visibility for diagnostics.
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# -----------------------------------------------------------------------------
# Pre-flight checks
# -----------------------------------------------------------------------------
# SLURM_SUBMIT_DIR is set by Slurm to the directory sbatch was called from.
REPO_DIR="$SLURM_SUBMIT_DIR"
SCRIPT_DIR="${REPO_DIR}/LLM"

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "       Download weights on the head node first. See README.md."
    exit 1
fi
echo "Model directory found: $MODEL_PATH"

if [[ -n "$PROMPT_FILE" && ! -f "$PROMPT_FILE" ]]; then
    echo "ERROR: Prompt file not found: $PROMPT_FILE"
    exit 1
fi

# Create required directories.
mkdir -p "${REPO_DIR}/logs"
mkdir -p "${REPO_DIR}/data/llm_results"

echo "Pre-flight checks passed."
echo ""

# -----------------------------------------------------------------------------
# Run inference
# -----------------------------------------------------------------------------
echo "=========================================="
echo "Running inference"
echo "=========================================="

EXTRA_ARGS=""
if [[ -n "$PROMPT_FILE" ]]; then
    EXTRA_ARGS="--prompt-file $PROMPT_FILE"
fi

python "${SCRIPT_DIR}/inference.py" \
    --model-path "$MODEL_PATH" \
    --output-dir "${REPO_DIR}/data/llm_results" \
    --max-new-tokens 512 \
    --temperature 0.0 \
    $EXTRA_ARGS

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job complete  |  Exit code: $EXIT_CODE"
echo "End : $(date)"
echo "=========================================="

if [[ $EXIT_CODE -ne 0 ]]; then
    echo "ERROR: inference.py exited with code $EXIT_CODE"
    exit $EXIT_CODE
fi

LATEST=$(ls -t "${REPO_DIR}/data/llm_results"/llm_inference_*.json 2>/dev/null | head -1)
if [[ -n "$LATEST" ]]; then
    echo "Latest result: $LATEST"
    echo "View with   : cat $LATEST"
fi
