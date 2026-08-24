import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", default="./slurm/outputs")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

x = np.linspace(0, 2 * np.pi, 200)
y_sin = np.sin(x)
y_cos = np.cos(x)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, y_sin, label="sin(x)")
ax.plot(x, y_cos, label="cos(x)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Sin and Cos")
ax.legend()
fig.tight_layout()

output_path = os.path.join(args.output_dir, "plot.png")
fig.savefig(output_path, dpi=150)
print(f"Saved plot to {output_path}")
