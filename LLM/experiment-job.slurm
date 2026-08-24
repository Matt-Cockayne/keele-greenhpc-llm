#!/bin/bash
#SBATCH --job-name=llm_experiment
#SBATCH --output=logs/llm_experiment_%j.out
#SBATCH --error=logs/llm_experiment_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# =============================================================================
# LLM Prompt Experiment Job
#
# Runs a set of prompts through the model and saves results for comparison.
# Each entry in the prompts file is one experimental condition.
#
# To design a new experiment:
#   1. Copy LLM/prompts/system_prompt_experiment.json and edit the prompts.
#      - Keep the same "prompt" across conditions to isolate one variable.
#      - Change "system", phrasing, or add context to define each condition.
#      - Give each entry a descriptive "id" so results are easy to compare.
#   2. Set EXPERIMENT_NAME — results are saved under data/llm_results/<name>/.
#   3. Set PROMPTS_FILE to point at your new prompts file.
#   4. Submit: EXPERIMENT_NAME=my_experiment PROMPTS_FILE=... sbatch LLM/experiment-job.slurm
#
# To run multiple experiments back-to-back, submit this script once per
# prompts file with a different EXPERIMENT_NAME each time.
# =============================================================================

echo "=========================================="
echo "LLM EXPERIMENT"
echo "=========================================="
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURM_NODELIST"
echo "Start    : $(date)"
echo "=========================================="

source ~/.bashrc
conda activate keele-llm

export HF_HOME="/home/xrai/.cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# --- Change these to configure your experiment ----------------------------
export MODEL_PATH="${MODEL_PATH:-/home/xrai/models/Meta-Llama-3.1-8B-Intstruct}"
PROMPTS_FILE="${PROMPTS_FILE:-LLM/prompts/system_prompt_experiment.json}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-system_prompt_experiment}"
# --------------------------------------------------------------------------

REPO_DIR="$SLURM_SUBMIT_DIR"
OUTPUT_DIR="${REPO_DIR}/data/llm_results/${EXPERIMENT_NAME}"

echo "Model      : $MODEL_PATH"
echo "Prompts    : $PROMPTS_FILE"
echo "Experiment : $EXPERIMENT_NAME"
echo "Output     : $OUTPUT_DIR"
echo ""

# Checks
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: Model not found at $MODEL_PATH"; exit 1
fi
if [[ ! -f "$PROMPTS_FILE" ]]; then
    echo "ERROR: Prompts file not found: $PROMPTS_FILE"; exit 1
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p "${REPO_DIR}/logs"

echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

python "${REPO_DIR}/LLM/multi-prompt.py" \
    --model-path "$MODEL_PATH" \
    --prompts-file "$PROMPTS_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --max-new-tokens 512 \
    --temperature 0.0        # set > 0 to introduce randomness across runs

EXIT_CODE=$?

echo ""
echo "Exit code : $EXIT_CODE"
echo "End       : $(date)"

if [[ $EXIT_CODE -eq 0 ]]; then
    LATEST=$(ls -t "${OUTPUT_DIR}"/llm_multi_*.json 2>/dev/null | head -1)
    echo "Results   : $LATEST"
fi

exit $EXIT_CODE
