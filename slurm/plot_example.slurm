#!/bin/bash
#SBATCH --job-name=plot_example
#SBATCH --output=logs/plot_example_%j.out
#SBATCH --error=logs/plot_example_%j.err
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=gpu

echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURM_NODELIST"
echo "Start  : $(date)"

source ~/.bashrc
conda activate keele-llm

REPO_DIR="$SLURM_SUBMIT_DIR"
OUTPUT_DIR="${REPO_DIR}/slurm/outputs"

mkdir -p "${REPO_DIR}/logs"
mkdir -p "$OUTPUT_DIR"

python "${REPO_DIR}/slurm/plot_example.py" --output-dir "$OUTPUT_DIR"

EXIT_CODE=$?
echo "End    : $(date)"
echo "Exit   : $EXIT_CODE"
exit $EXIT_CODE
