'''
File:
cluster_dbscan.py

Description:
Part 1 -- scale the features, then discover the classes with DBSCAN.
There are no labels: DBSCAN groups points by density, so it can follow
any shape, and anything in a sparse region is marked as noise (-1).

The k-means panel in the figure is for contrast only. It is given the
correct number of clusters and still gets the shapes wrong, because
assigning each point to its nearest centre can only carve the plane into
straight-edged pieces.

Note:
- Run from src/ (paths are relative to it).
- Writes outputs/<DATASET>/labels.csv (one cluster label per TRAINING
  point, in training-split order) and figures/clustered_space.pdf.

___________________________________________________
Author      Date                Comment
Gib M.      Sept. 11th, 2026    rev 1
Gib M.      Oct. 14th, 2026     rev 2 -- paths and axis labels from
                                pipeline_common; labels.csv gets a
                                describing header; rewrote header/comments.
'''
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from pipeline_common import (load_split_scaled, VIEW_LIM, AXIS_LABELS,
                             LABELS_CSV, FIG_DIR)

EPS = 0.06         # neighbourhood radius -- only meaningful on scaled data
MIN_SAMPLES = 8    # points needed within EPS to form a dense core

# Clustering sees TRAINING data only; val/test stay unseen.
X = load_split_scaled()["train"]

# --- Discover the classes ---------------------------------------------------
labels = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit_predict(X)  # -1 = noise

classes = sorted(set(labels) - {-1})
print(f"{len(classes)} clusters; {(labels == -1).sum()} noise")
for c in classes:
    print(f"  cluster {c}: {(labels == c).sum()} points")

# Every later stage reads these labels instead of re-clustering.
np.savetxt(LABELS_CSV, labels, fmt="%d",
           header=f"DBSCAN cluster per training point (eps={EPS}, "
                  f"min_samples={MIN_SAMPLES}); -1 = noise")

# --- k-means, for contrast only ---------------------------------------------
kmeans_labels = KMeans(n_clusters=len(classes), n_init=10,
                       random_state=0).fit_predict(X)

# --- Figure: DBSCAN vs k-means side by side ---------------------------------
# Grayscale markers so the figure reads in black and white.
MARKERS = ["o", "s", "^", "D", "v", "P"]
GRAYS = ["0.75", "0.45", "0.15", "0.6", "0.3", "0.85"]


def draw(ax, lab, title, show_noise):
    """Scatter one labelling: a marker/shade per cluster, noise as x."""
    for i, c in enumerate(sorted(set(lab) - {-1})):
        pts = X[lab == c]
        ax.scatter(pts[:, 0], pts[:, 1], s=5,
                   marker=MARKERS[i % len(MARKERS)],
                   facecolor=GRAYS[i % len(GRAYS)],
                   edgecolor="black", linewidth=0.25,
                   label=f"class {c} (n={len(pts)})")
    if show_noise:
        noise = X[lab == -1]
        ax.scatter(noise[:, 0], noise[:, 1], s=9, c="0.45", marker="x",
                   linewidth=0.5, label=f"noise, $-1$ (n={len(noise)})")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(AXIS_LABELS[0])
    ax.set_aspect("equal")
    ax.set_xlim(*VIEW_LIM)
    ax.set_ylim(*VIEW_LIM)


fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharex=True, sharey=True)
draw(axes[0], labels, "DBSCAN: shapes stay intact", show_noise=True)
draw(axes[1], kmeans_labels, "k-means: shapes get sliced", show_noise=False)
axes[0].set_ylabel(AXIS_LABELS[1])
fig.savefig(FIG_DIR + "clustered_space.pdf", bbox_inches="tight",
            transparent=True)
