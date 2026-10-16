'''
File:
cluster_dbscan.py

Description:

Note:
- At the very end of this file, be aware that the save file path
  is relative to the current working directory.

___________________________________________________
Author      Date                Comment
Gib M.      Sept. 11th, 2026     rev 1
'''

"""Part 1 -- scale the features, then discover
classes with DBSCAN.

The k-means panel is for contrast only. It is
given the correct k and
still gets the shapes wrong: assigning each point
to its nearest centre
can only carve the plane into straight-edged
pieces.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from pipeline_common import load_split_scaled, VIEW_LIM, load_dataset

X = load_split_scaled()["train"]  # clustering sees TRAINING data only

db = DBSCAN(eps=0.06, min_samples=8)  # eps is meaningless unscaled
labels = db.fit_predict(X)  # -1 = noise

classes = sorted(set(labels) - {-1})
print(f"{len(classes)} clusters; {(labels == -1).sum()} noise")
for c in classes:
    print(f"  cluster {c}: {(labels == c).sum()} points")
np.savetxt("../datasets/labels.csv", labels, fmt="%d")

kmeans_labels = KMeans(
    n_clusters=len(classes),
    n_init=10,
    random_state=0
).fit_predict(X)

MARKERS = ["o", "s", "^", "D", "v", "P"]
GRAYS = ["0.75", "0.45", "0.15", "0.6", "0.3", "0.85"]


def draw(ax, lab, title, show_noise):
    for i, c in enumerate(sorted(set(lab) - {-1})):
        pts = X[lab == c]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=5,
            marker=MARKERS[i % len(MARKERS)],
            facecolor=GRAYS[i % len(GRAYS)],
            edgecolor="black",
            linewidth=0.25,
            label=f"class {c} (n={len(pts)})"
        )
    if show_noise:
        noise = X[lab == -1]
        ax.scatter(
            noise[:, 0],
            noise[:, 1],
            s=9,
            c="0.45",
            marker="x",
            linewidth=0.5,
            label=f"noise, $-1$ (n={len(noise)})"
        )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("scaled feature $x_1$")
    ax.set_aspect("equal")
    ax.set_xlim(*VIEW_LIM)
    ax.set_ylim(*VIEW_LIM)


fig, axes = plt.subplots(
    1, 2,
    figsize=(7.4, 3.4),
    sharex=True,
    sharey=True
)
draw(axes[0], labels, "DBSCAN: shapes stay intact", show_noise=True)
draw(axes[1], kmeans_labels, "k-means: shapes get sliced", show_noise=False)
axes[0].set_ylabel("scaled feature $x_2$")
fig.savefig(
    "../figures/generated/clustered_space.pdf",
    bbox_inches="tight",
    transparent=True
)