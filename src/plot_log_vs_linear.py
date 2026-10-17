'''
File:
plot_log_vs_linear.py

Description:
Part 2 (sensor) -- why the features go through log1p before scaling.
Plots the scaled IR of every TRAINING reading, grouped by its true
lighting condition, once with plain min-max scaling and once with log1p
first, with the GRID_N grid rows drawn behind. Linear scaling squeezes
shade, LED and fluorescent into grid row 0, where no byte table can tell
them apart; log1p spreads them over separate rows.

The true labels are used for grouping only; the pipeline never sees them.

Note:
- Run from src/ (paths are relative to it). Sensor dataset only.
- Reads datasets/sensor/data_complete.csv (readings plus label column).
- Writes outputs/sensor/figures/log_vs_linear.pdf.

___________________________________________________
Author      Date                Comment
Gib M.      Oct. 16th, 2026     rev 1
'''
import csv

import matplotlib.pyplot as plt
import numpy as np
from pipeline_common import DATA_DIR, FIG_DIR, GRID_N, split_data

LABELLED_CSV = DATA_DIR + "data_complete.csv"
ORDER = ["shade", "LED", "fluorescent", "sun"]          # dimmest to brightest
MARKERS = {"shade": "o", "LED": "s", "fluorescent": "^", "sun": "D"}

# --- Load the labelled readings and keep the training split ----------------
# split_data shuffles indices with the pipeline's seed, so these are the
# same training rows DBSCAN saw.
with open(LABELLED_CSV) as f:
    rows = list(csv.DictReader(f))
ir = np.array([float(r["IR_sensor"]) for r in rows])
labels = np.array([r["label"] for r in rows])
train_idx, _, _ = split_data(np.arange(len(rows)))
ir, labels = ir[train_idx], labels[train_idx]


def minmax(v):
    """Scale onto [0, 1] by the training range, as scale_features does."""
    return (v - v.min()) / (v.max() - v.min())


# --- Figure: linear vs log1p, side by side ----------------------------------
plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.3), sharey=True)
rng = np.random.default_rng(0)  # horizontal jitter only, for visibility

for ax, (title, values) in zip(axes, (("linear min-max", minmax(ir)),
                                      ("log1p, then min-max", minmax(np.log1p(ir))))):
    # The grid rows the byte tables are cut into.
    for k in range(GRID_N + 1):
        ax.axhline(k / GRID_N, color="0.85", lw=0.4, zorder=0)
    for i, c in enumerate(ORDER):
        m = labels == c
        ax.scatter(i + rng.uniform(-0.18, 0.18, m.sum()), values[m], s=4,
                   marker=MARKERS[c], facecolor="0.55", edgecolor="black",
                   linewidth=0.2)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(ORDER, rotation=35, ha="right")
    ax.set_title(title, fontsize=8)
    ax.set_ylim(-0.03, 1.03)

axes[0].annotate("all three indoor\nclasses in row 0", xy=(1, 0.01),
                 xytext=(1, 0.42), fontsize=7, ha="center", color="0.2",
                 arrowprops=dict(arrowstyle="->", color="0.3", lw=0.6))
axes[0].set_ylabel("scaled IR (lines: grid rows)")

fig.savefig(FIG_DIR + "log_vs_linear.pdf", bbox_inches="tight",
            transparent=True)
