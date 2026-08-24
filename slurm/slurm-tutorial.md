# Running Python Experiments on the HPC with Slurm

A beginner's guide to submitting and managing jobs on a High-Performance Computing (HPC) cluster using Slurm.

---

## 1. What is an HPC and Why Do We Use It?

A High-Performance Computing (HPC) cluster is a collection of powerful servers linked together. When you need to train a machine learning model or run inference with a large neural network, your laptop or desktop simply doesn't have enough memory or compute power. An HPC gives you access to high-end GPUs (e.g. NVIDIA A100s) and large amounts of RAM without having to own them.

**The key idea:** you do not run your code interactively. Instead, you write a job script that describes what you want to run and what resources you need, then hand it to the cluster's job scheduler. The scheduler queues it and runs it when resources become available.

---

## 2. The Two Types of Nodes

When you connect to the HPC, you land on the **head node** (also called the login node). There are two distinct environments to be aware of:

| | Head Node | Compute Nodes |
|---|---|---|
| What it's for | Setup, file editing, submitting jobs | Running your actual experiments |
| Internet access | Yes | No |
| GPU | No | Yes (e.g. A100 40 GB) |
| How you use it | Interactively via SSH | Automatically, via Slurm |

> **Important:** Never run heavy computation on the head node. It is shared by everyone on the cluster. Running a GPU job directly there will slow it down for all users and your job may be killed.

---

## 3. What is Slurm?

**Slurm** (Simple Linux Utility for Resource Management) is the job scheduler used on most HPC clusters. You interact with it using a handful of commands:

| Command | What it does |
|---|---|
| `sbatch script.slurm` | Submit a job script to the queue |
| `squeue -u $USER` | Check the status of your submitted jobs |
| `scancel <job_id>` | Cancel a queued or running job |
| `scontrol show job <job_id>` | Show detailed info about a job |
| `sinfo` | Show the state of all partitions and nodes |

When you run `sbatch`, Slurm prints a job ID:

```
Submitted batch job 12345
```

This ID is how you refer to your job when monitoring or cancelling it.

**Job states** you will see in `squeue`:

| State | Meaning |
|---|---|
| `PD` | Pending — waiting for resources to become free |
| `R` | Running |
| `CG` | Completing — wrapping up |
| `F` | Failed |

---

## 4. Anatomy of a Slurm Script

A `.slurm` file is a regular Bash shell script with a special header block. Every line in the header starts with `#SBATCH` and tells Slurm about the resources and settings for the job.

Here is a minimal example followed by an explanation of each directive:

```bash
#!/bin/bash
#SBATCH --job-name=my_experiment        # Name shown in squeue
#SBATCH --output=logs/job_%j.out        # Stdout log file (%j = job ID)
#SBATCH --error=logs/job_%j.err         # Stderr log file
#SBATCH --time=01:00:00                 # Max wall time (HH:MM:SS)
#SBATCH --nodes=1                       # Number of compute nodes
#SBATCH --ntasks=1                      # Number of parallel tasks (usually 1)
#SBATCH --cpus-per-task=4               # CPU cores (for data loading etc.)
#SBATCH --mem=32G                       # System RAM
#SBATCH --gres=gpu:1                    # Number of GPUs
#SBATCH --partition=gpu                 # Which queue/partition to use

# Everything below here is ordinary Bash.
source ~/.bashrc
conda activate my-env

python my_script.py --some-arg value
```

**What each `#SBATCH` line does:**

- `--job-name` — label shown in `squeue`; pick something descriptive
- `--output` / `--error` — where to write `stdout` and `stderr`; `%j` is replaced with the job ID automatically
- `--time` — the job is killed after this wall-clock time; set it longer than your expected runtime but not excessively long (it affects your queue priority)
- `--mem` — total system RAM for the job; the GPU has its own VRAM on top of this
- `--gres=gpu:1` — request one GPU; most single-model jobs fit on one A100
- `--partition` — the name of the resource pool; `gpu` is the partition that has GPU nodes

---

## 5. A Real Example: `inference-job.slurm`

Let's walk through the VLM inference job script (`VLM/inference-job.slurm`) piece by piece.

### 5a. The resource header

```bash
#SBATCH --job-name=vlm_inference
#SBATCH --output=logs/vlm_inference_%j.out
#SBATCH --error=logs/vlm_inference_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
```

This requests: 1 GPU node, 4 CPU cores (used by the PyTorch DataLoader for image pre-processing), 32 GB of system RAM, and up to 1 hour of runtime.

### 5b. Activating the conda environment

```bash
source ~/.bashrc
conda activate keele-llm
```

The compute node has access to the same shared filesystem as the head node, so the conda environment you created on the head node is immediately available here. You don't need to reinstall anything.

### 5c. Environment variables

```bash
export HF_HOME="/home/xrai/.cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export MODEL_PATH="${MODEL_PATH:-/home/xrai/models/medgemma-4b-it}"
```

- `HF_HOME` redirects the HuggingFace cache to a shared directory with a larger quota.
- `TRANSFORMERS_OFFLINE=1` prevents the transformers library from trying to reach `huggingface.co` — this is necessary because compute nodes have no internet.
- `MODEL_PATH` uses a default value but can be overridden (see Section 7).

### 5d. Pre-flight checks

```bash
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi
```

The script validates that the model weights directory exists before doing any work. If you forget to download the model first, the job exits immediately with a clear error rather than failing deep into a multi-hour run.

### 5e. Running the Python script

```bash
python "${SCRIPT_DIR}/inference.py" \
    --model-path "$MODEL_PATH" \
    --output-dir "${REPO_DIR}/data/vlm_results" \
    --max-new-tokens 512 \
    $EXTRA_ARGS
```

This is the actual work. Everything above it was setup. The Python script receives its configuration through command-line arguments, which the Slurm script assembles.

---

## 6. A Real Example: `run_single_experiment.slurm`

The `slurm/run_single_experiment.slurm` script follows the same pattern but is more configurable — it exposes multiple parameters that can be overridden at submission time.

### 6a. Configuration block

```bash
BACKBONE="${BACKBONE:-vit_base_patch16_224}"
SEED="${SEED:-42}"
DATA_DIR="${DATA_DIR:-/home/csc29/projects/SkinCAP/skincap}"
```

Each variable uses the `${VAR:-default}` pattern: if `VAR` is already set in the environment, use that value; otherwise fall back to the default. This is the standard way to make Slurm scripts configurable without editing them.

### 6b. Loading modules

```bash
module use /opt/nvidia/hpc_sdk/modulefiles
module load nvhpc-hpcx-cuda12/24.5
```

Some clusters manage system-level libraries (CUDA, MPI, compilers) with the `module` system. `module load` makes the required libraries available to your job.

### 6c. Building and running the command

```bash
CMD="python scripts/test_minimal_curriculum.py \
    --data_dir \"$DATA_DIR\" \
    --csv_path \"$CSV_PATH\" \
    --backbone \"$BACKBONE\" \
    --random_seed \"$SEED\""

eval $CMD
```

Building the command as a string first and then running it with `eval` makes it easy to conditionally add extra arguments (e.g. `--experiments`) based on environment variables.

---

## 7. Submitting a Job

### Basic submission

Always submit from the repository root directory so that relative paths in the script resolve correctly:

```bash
cd ~/projects/keele-greenhpc-llm
sbatch VLM/inference-job.slurm
```

### Passing configuration without editing the script

You can set environment variables immediately before `sbatch`. They are passed into the job automatically:

```bash
# Use a different model
MODEL_PATH=/home/xrai/models/llava-med-v1.5-mistral-7b sbatch VLM/inference-job.slurm

# Run a specific prompt file
PROMPT_FILE=VLM/prompts/single.json sbatch VLM/inference-job.slurm

# Combine multiple overrides
MODEL_PATH=/home/xrai/models/llava-med-v1.5-mistral-7b \
PROMPT_FILE=VLM/prompts/single.json \
sbatch VLM/inference-job.slurm
```

For `run_single_experiment.slurm`, the same approach applies:

```bash
# Run with a different backbone and seed
BACKBONE=resnet50 SEED=123 sbatch slurm/run_single_experiment.slurm

# Alternatively, use --export
sbatch --export=BACKBONE=resnet50,SEED=123 slurm/run_single_experiment.slurm
```

> **Why `VAR=value sbatch ...` works:** setting a variable immediately before a command passes it as an environment variable to that single command only. It does not change your current shell session.

---

## 8. Monitoring Your Job

Once submitted, you can monitor your job in real time:

```bash
# List all your active jobs
squeue -u $USER

# Stream live output as the job runs
tail -f logs/vlm_inference_12345.out

# Stream the error log (useful for debugging crashes)
tail -f logs/vlm_inference_12345.err

# Check detailed job info (start time, allocated node, reason for pending)
scontrol show job 12345

# Cancel a job
scancel 12345
```

**If your job is stuck in `PD` (pending):** the GPU partition is busy. Check availability with `sinfo`. You can also check your fairshare priority with `sshare -u $USER` — jobs from users who have used less recently get higher priority.

---

## 9. Reading the Output

Slurm writes everything your script prints to `stdout` into the `.out` log file and everything sent to `stderr` into the `.err` file. These are defined by `--output` and `--error` in the script header.

For the inference scripts, results are also saved as timestamped JSON files:

```bash
# View the most recent VLM result
cat $(ls -t data/vlm_results/vlm_inference_*.json | head -1) | python -m json.tool
```

---

## 10. Common Mistakes and How to Fix Them

| Problem | Cause | Fix |
|---|---|---|
| `conda: command not found` in job log | `~/.bashrc` not sourced | Add `source ~/.bashrc` before `conda activate` in your script |
| `OSError: couldn't connect to huggingface.co` | Model path wrong or weights not downloaded | Verify `ls /path/to/model/config.json` exists on the head node |
| `CUDA out of memory` | Model too large for GPU VRAM | Reduce `--max-new-tokens`, use quantisation, or request 2 GPUs with `--gres=gpu:2` |
| Job fails instantly (exit code 1) | Pre-flight check failed | Read the `.err` log; a missing file or directory is the usual cause |
| Job stays `PD` for a long time | GPU nodes are busy | Wait, or check `sinfo` for a less busy partition |
| Results not saving | `logs/` or `data/` directory doesn't exist | Add `mkdir -p logs data/results` near the top of your script |

---

## 11. Quick-Reference Checklist

Before submitting any job, run through this checklist on the head node:

- [ ] Conda environment is set up and all packages are installed
- [ ] Model weights / data files are downloaded and paths are correct
- [ ] The `logs/` directory exists (or your script creates it with `mkdir -p`)
- [ ] You are in the repository root directory when running `sbatch`
- [ ] `--time` is set generously (at least 2× your expected runtime)
- [ ] You know where the output will be written so you can find it afterwards

---

## 12. Further Reading

- [Slurm documentation](https://slurm.schedmd.com/documentation.html) — official reference
- [Conda cheatsheet](https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html)
- [HuggingFace offline mode](https://huggingface.co/docs/transformers/installation#offline-mode)
- `README.md` in this repository — Keele GreenHPC-specific setup steps
