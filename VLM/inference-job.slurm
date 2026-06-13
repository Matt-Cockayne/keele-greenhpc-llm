#!/bin/bash
#SBATCH --job-name=vlm_inference
#SBATCH --output=logs/vlm_inference_%j.out
#SBATCH --error=logs/vlm_inference_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4          # 4 CPU workers for DataLoader / image preprocessing
#SBATCH --mem=32G                  # system RAM; the model itself lives in GPU VRAM
#SBATCH --gres=gpu:1               # request 1 GPU; MedGemma 4B / LLaVA-Med 7B fit on a single A100 40 GB
#SBATCH --partition=gpu

# =============================================================================
# VLM Inference Job -- Single-prompt vision-language model inference
#
# Supports:
#   MedGemma 4B-IT  --  /home/xrai/models/medgemma-4b-it
#   LLaVA-Med 1.5   --  /home/xrai/models/llava-med-v1.5-mistral-7b
#
# Prerequisites (run once on the head node, not here):
#   1. conda create -n keele-llm python=3.11 -y
#   2. See README.md section "Environment Setup" for full package list.
#
# Submit:
#   cd projects/keele-greenhpc-llm
#   sbatch VLM/inference-job.slurm
#
# Override model or prompt file before sbatch:
#   MODEL_PATH=/home/xrai/models/llava-med-v1.5-mistral-7b \
#   PROMPT_FILE=VLM/prompts/single.json \
#   sbatch VLM/inference-job.slurm
#
# For multi-prompt batches use multi-prompt-job.slurm instead.
# =============================================================================

echo "=========================================="
echo "VLM INFERENCE"
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
# Point it away from $HOME to avoid hitting the home-directory quota (~50 GB).
# /home/xrai/ is a shared project space with a larger quota on Keele GreenHPC.
export HF_HOME="/home/xrai/.cache/huggingface"

# Prevent the transformers library from attempting any network request.
# All weights must already be present at MODEL_PATH.
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Path to the locally downloaded model weights directory.
# Defaults to MedGemma; override with the MODEL_PATH env var.
export MODEL_PATH="${MODEL_PATH:-/home/xrai/models/medgemma-4b-it}"

# Optional: path to a single-prompt JSON file.
# Format: {"prompt": "...", "image_path": "...", "system": "..."}
# image_path and system are optional.  Leave PROMPT_FILE unset to use
# the built-in demo prompt in inference.py.
# For multi-prompt batches use multi-prompt-job.slurm instead.
PROMPT_FILE="${PROMPT_FILE:-}"

# Log GPU visibility so the job output is self-documenting.
echo ""
echo "Model       : $MODEL_PATH"
echo "Prompt file : ${PROMPT_FILE:-<built-in demo prompt>}"
echo ""
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# -----------------------------------------------------------------------------
# Pre-flight checks
# -----------------------------------------------------------------------------
# SLURM_SUBMIT_DIR is set by Slurm to the directory sbatch was called from.
REPO_DIR="$SLURM_SUBMIT_DIR"
SCRIPT_DIR="${REPO_DIR}/VLM"

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "       Download weights on the head node first. See README.md."
    exit 1
fi

if [[ -n "$PROMPT_FILE" && ! -f "$PROMPT_FILE" ]]; then
    echo "ERROR: Prompt file not found: $PROMPT_FILE"
    exit 1
fi

echo "Pre-flight checks passed."
echo ""

# Create required directories.
mkdir -p "${REPO_DIR}/logs"
mkdir -p "${REPO_DIR}/data/vlm_results"

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
    --output-dir "${REPO_DIR}/data/vlm_results" \
    --max-new-tokens 512 \
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

LATEST=$(ls -t "${REPO_DIR}/data/vlm_results"/vlm_inference_*.json 2>/dev/null | head -1)
if [[ -n "$LATEST" ]]; then
    echo "Latest result: $LATEST"
    echo "View with   : cat $LATEST"
fi