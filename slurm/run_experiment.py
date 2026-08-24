import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Slurm demo — random number experiment")
parser.add_argument("--backbone",    default="vit_base_patch16_224")
parser.add_argument("--seed",        type=int, default=42)
parser.add_argument("--results_dir", required=True)
parser.add_argument("--n_samples",   type=int, default=500)
args = parser.parse_args()

rng = np.random.default_rng(args.seed)
os.makedirs(args.results_dir, exist_ok=True)

print(f"Backbone  : {args.backbone}")
print(f"Seed      : {args.seed}")
print(f"N samples : {args.n_samples}")
print(f"Results   : {args.results_dir}")

# Generate two labelled distributions to compare
group_a = rng.normal(loc=0.0, scale=1.0, size=args.n_samples)
group_b = rng.normal(loc=0.8, scale=1.2, size=args.n_samples)

summary = {
    "backbone": args.backbone,
    "seed": args.seed,
    "group_a": {"mean": float(group_a.mean()), "std": float(group_a.std())},
    "group_b": {"mean": float(group_b.mean()), "std": float(group_b.std())},
}
print(f"\nGroup A  mean={summary['group_a']['mean']:.4f}  std={summary['group_a']['std']:.4f}")
print(f"Group B  mean={summary['group_b']['mean']:.4f}  std={summary['group_b']['std']:.4f}")

results_path = os.path.join(args.results_dir, "results.json")
with open(results_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nResults saved to {results_path}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(group_a, bins=30, alpha=0.6, label="Group A")
ax.hist(group_b, bins=30, alpha=0.6, label="Group B")
ax.set_xlabel("Value")
ax.set_ylabel("Count")
ax.set_title(f"Distribution comparison — {args.backbone}  (seed={args.seed})")
ax.legend()
fig.tight_layout()

plot_path = os.path.join(args.results_dir, "distributions.png")
fig.savefig(plot_path, dpi=150)
print(f"Plot saved to {plot_path}")
