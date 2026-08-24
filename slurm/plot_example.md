# plot_example

Submits `plot_example.py` to the GPU partition. The script generates sin/cos arrays and saves a plot to `slurm/outputs/plot.png`.

## Submit

```bash
cd ~/projects/keele-greenhpc-llm
sbatch slurm/plot_example.slurm
```

## What the script does

| Line | Purpose |
|---|---|
| `#SBATCH --partition=gpu` | Only one partition available; the job runs on a GPU node even though no GPU is used |
| `#SBATCH --mem=4G` | 4 GB RAM is enough for NumPy/Matplotlib |
| `#SBATCH --time=00:05:00` | 5-minute limit; the job finishes in seconds |
| `source ~/.bashrc` + `conda activate` | Makes the conda environment available on the compute node |
| `REPO_DIR="$SLURM_SUBMIT_DIR"` | Captures the directory you ran `sbatch` from so relative paths work correctly |
| `mkdir -p logs` | Creates the log directory if it doesn't exist yet |
| `EXIT_CODE=$?` | Captures whether Python succeeded or failed; passed to `exit` so Slurm marks the job accordingly |

## Monitor

```bash
squeue -u $USER
tail -f logs/plot_example_<job_id>.out
```
