#!/bin/bash
#SBATCH --job-name=llm_multi
#SBATCH --output=logs/llm_multi_%j.out
#SBATCH --error=logs/llm_multi_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# =============================================================================
# LLM Multi-Prompt Batch Inference Job
#
# Loads the model once and processes all prompts from a JSON file
# sequentially, writing all responses to a single output JSON.
#
# Prompt file format (--prompts-file / PROMPTS_FILE):
#   A JSON array of strings or objects.  Each object may have:
#       "id"     (optional) : label in output
#       "prompt" (required) : user message
#       "system" (optional) : system message override
#   See LLM/prompts/multi.json for an example.
#
# Submit:
#   cd projects/keele-greenhpc-llm
#   PROMPTS_FILE=LLM/prompts/multi.json sbatch LLM/multi-prompt-job.slurm
#
# Override model:
#   MODEL_PATH=/home/xrai/models/other-model \
#   PROMPTS_FILE=LLM/prompts/multi.json \
#   sbatch LLM/multi-prompt-job.slurm
# =============================================================================

echo "=========================================="
echo "LLM MULTI-PROMPT INFERENCE"
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

export HF_HOME="/home/xrai/.cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export MODEL_PATH="${MODEL_PATH:-/home/xrai/models/Meta-Llama-3.1-8B-Intstruct}"
PROMPTS_FILE="${PROMPTS_FILE:-}"

echo ""
echo "Model        : $MODEL_PATH"
echo "Prompts file : ${PROMPTS_FILE}"
echo ""

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
    exit 1
fi

if [[ -z "$PROMPTS_FILE" ]]; then
    echo "ERROR: PROMPTS_FILE is not set."
    echo "       Usage: PROMPTS_FILE=LLM/prompts/multi.json sbatch LLM/multi-prompt-job.slurm"
    exit 1
fi

if [[ ! -f "$PROMPTS_FILE" ]]; then
    echo "ERROR: Prompts file not found: $PROMPTS_FILE"
    exit 1
fi

mkdir -p "${REPO_DIR}/logs"
mkdir -p "${REPO_DIR}/data/llm_results"

echo "Pre-flight checks passed."
echo ""

# -----------------------------------------------------------------------------
# Run multi-prompt inference
# -----------------------------------------------------------------------------
echo "=========================================="
echo "Running multi-prompt inference"
echo "=========================================="

python "${SCRIPT_DIR}/multi-prompt.py" \
    --model-path "$MODEL_PATH" \
    --prompts-file "$PROMPTS_FILE" \
    --output-dir "${REPO_DIR}/data/llm_results" \
    --max-new-tokens 512 \
    --temperature 0.0

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job complete  |  Exit code: $EXIT_CODE"
echo "End : $(date)"
echo "=========================================="

if [[ $EXIT_CODE -ne 0 ]]; then
    echo "ERROR: multi-prompt.py exited with code $EXIT_CODE"
    exit $EXIT_CODE
fi

LATEST=$(ls -t "${REPO_DIR}/data/llm_results"/llm_multi_*.json 2>/dev/null | head -1)
if [[ -n "$LATEST" ]]; then
    echo "Latest result: $LATEST"
    echo "View with   : cat $LATEST"
fi
