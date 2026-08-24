# run_single_experiment

Submits `run_experiment.py` as a configurable GPU job. The script generates two random distributions, saves a summary to `results.json`, and plots a histogram to `distributions.png`.

## Submit

```bash
cd ~/projects/keele-greenhpc-llm
sbatch slurm/run_single_experiment.slurm
```

## Override parameters

Variables have defaults but can be changed at submission time without editing the script:

```bash
# Change backbone and seed
BACKBONE=resnet50 SEED=123 sbatch slurm/run_single_experiment.slurm

# Or use --export
sbatch --export=BACKBONE=resnet50,SEED=123,N_SAMPLES=1000 slurm/run_single_experiment.slurm
```

## Parameters

| Variable | Default | Description |
|---|---|---|
| `BACKBONE` | `vit_base_patch16_224` | Label used to organise output folders |
| `SEED` | `42` | Random seed for reproducibility |
| `N_SAMPLES` | `500` | Number of samples to generate |

Results are written to `slurm/outputs/<BACKBONE>/seed_<SEED>/`.

## What the script does

| Section | Purpose |
|---|---|
| `#SBATCH --gres=gpu:1` | Requests one GPU (swap to `--partition=cpu` and remove this line if no GPU is needed) |
| `${BACKBONE:-vit_base_patch16_224}` | Default value pattern — uses the env var if set, otherwise falls back to the default |
| `RESULTS_DIR="$RESULTS_BASE/$BACKBONE/seed_$SEED"` | Organises outputs so runs with different settings don't overwrite each other |
| `conda activate keele-llm` | Activates the shared conda environment on the compute node |
| `mkdir -p "$RESULTS_DIR"` | Creates the output directory before Python runs |
| `EXIT_CODE=$?` | Captures the Python exit code so Slurm marks the job as failed if the script crashes |

## Monitor

```bash
squeue -u $USER
tail -f logs/single_experiment_<job_id>.out
```
