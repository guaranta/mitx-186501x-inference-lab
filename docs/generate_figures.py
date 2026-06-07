"""Generate README figures for model drift detection."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "model-drift"))
from drift_detection import drift_report  # noqa: E402

rng = np.random.default_rng(42)
ref = rng.normal(0.72, 0.08, 2000)
cur = rng.normal(0.65, 0.10, 2000)
report = drift_report(ref, cur)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(ref, bins=50, alpha=0.6, label="Reference", color="#3b82f6", density=True)
axes[0].hist(cur, bins=50, alpha=0.6, label="Production", color="#f97316", density=True)
axes[0].set_title(f"Score distributions (KS p={report['p_value']:.2e})")
axes[0].legend()

# PSI bins visualization
breakpoints = np.percentile(ref, np.linspace(0, 100, 11))
breakpoints[0], breakpoints[-1] = -np.inf, np.inf
exp_pcts, act_pcts = [], []
for i in range(10):
    exp_pcts.append(np.mean((ref >= breakpoints[i]) & (ref < breakpoints[i + 1])))
    act_pcts.append(np.mean((cur >= breakpoints[i]) & (cur < breakpoints[i + 1])))
x = np.arange(10)
w = 0.35
axes[1].bar(x - w / 2, exp_pcts, w, label="Reference", color="#3b82f6")
axes[1].bar(x + w / 2, act_pcts, w, label="Production", color="#f97316")
axes[1].set_title(f"PSI = {report['psi']:.3f} (alert > 0.2)")
axes[1].set_xlabel("Decile bin")
axes[1].legend()
fig.suptitle("Model score drift — production monitoring", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "drift_detection.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"Saved figure to {OUT}")
