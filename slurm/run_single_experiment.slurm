#!/bin/bash
#SBATCH --job-name=single_experiment
#SBATCH --output=logs/single_experiment_%j.out
#SBATCH --error=logs/single_experiment_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1

# Single Experiment Runner with Configurable Backbone
# 
# Usage:
#   sbatch slurm/run_single_experiment.slurm
#   sbatch --export=BACKBONE=resnet50,SEED=123 slurm/run_single_experiment.slurm

echo "=================================================="
echo "CBM SINGLE EXPERIMENT"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Configuration (can be overridden via --export)
BACKBONE="${BACKBONE:-vit_base_patch16_224}"
SEED="${SEED:-42}"
N_SAMPLES="${N_SAMPLES:-500}"

# Results directory organized by backbone
RESULTS_BASE="/home/csc29/projects/keele-greenhpc-llm/slurm/outputs"
RESULTS_DIR="$RESULTS_BASE/$BACKBONE/seed_$SEED"

echo "Configuration:"
echo "  Backbone: $BACKBONE"
echo "  Random Seed: $SEED"
echo "  Results: $RESULTS_DIR"
echo ""

# Init conda
source ~/.bashrc
conda activate keele-llm

echo "Python: $(python --version)"
echo ""

mkdir -p "$RESULTS_DIR"
mkdir -p logs

# Run experiment
echo "Starting experiment with $BACKBONE..."
python "${SLURM_SUBMIT_DIR}/slurm/run_experiment.py" \
    --backbone "$BACKBONE" \
    --seed "$SEED" \
    --n_samples "$N_SAMPLES" \
    --results_dir "$RESULTS_DIR"

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "EXPERIMENT COMPLETED"
echo "=================================================="
echo "Backbone: $BACKBONE"
echo "Seed: $SEED"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS"
    echo "Results : $RESULTS_DIR/results.json"
    echo "Plot    : $RESULTS_DIR/distributions.png"
else
    echo "FAILED"
    echo "Check error logs for details"
fi
echo "=================================================="
