'''
File:
kde_class_contours.py

Description:
This program creates a KDE map per class on a lookup
grid. The peaks of each class is then normalized.That
data is then saved on a surfaces.npz file which
is used in later stages.

All these 3 maps are saved on the class_surfaces_continuous.pdf.

Note:
- At the very end of this file, be aware that the save file path
  is relative to the current working directory.

___________________________________________________
Author      Date                Comment
Gib M.      Sept. 23th, 2026     rev 1
'''

""" on the lookup
grid, peak-normalised.

Saves surfaces.npz, which every later stage reads.
"""
import matplotlib.pyplot as plt
import numpy as np
from pipeline_common import (load_split_scaled,
                             make_grid, fit_density,
                             evaluate_on_grid,
                             normalise_peak, VIEW_LIM)

BANDWIDTH = 0.05
DISPLAY_N = 200  # fine grid used only for the
                 # figure, not for storage

X = load_split_scaled()["train"]  # KDE is
                                  # fit on TRAINING data only
labels = np.loadtxt("labels.csv", dtype=int)
classes = sorted(set(labels) - {-1})
xx, yy, grid_points = make_grid()

# The stored maps live on the 16x16 lookup grid --
# this is what ships.
kdes = {c: fit_density(X[labels == c], BANDWIDTH)
        for c in classes}
surfaces = {c: normalise_peak(evaluate_on_grid(
    kdes[c], grid_points, xx))
            for c in classes}
np.savez("surfaces.npz", **{f"c{c}": s for c, s in
                           surfaces.items()})

# The figure shows the true KDE on a fine grid --
# the continuous density
# before the lookup grid coarsens it. Compare
# against the stored bytes in
# Part 3, where the same maps appear on the 16x16
# grid.
g = np.linspace(0, 1, DISPLAY_N)
fxx, fyy = np.meshgrid(g, g)
fine = np.column_stack([fxx.ravel(), fyy.ravel()])

fig, axes = plt.subplots(1, len(classes), figsize
                        =(3.0 * len(classes), 3.0),
                        sharex=True, sharey=True)
for ax, c in zip(np.atleast_1d(axes), classes):
    smooth = normalise_peak(np.exp(kdes[c].
                                   score_samples(fine)).reshape(fxx.shape))
    im = ax.imshow(smooth, extent=[0, 1, 0, 1],
                   origin="lower",
                   cmap="Greys", interpolation="bilinear",
                   aspect="equal",
                   vmin=0, vmax=1)
    ax.scatter(X[labels == c][:, 0], X[labels == c][:, 1],
               s=2, c="white", edgecolors="black",
               linewidths=0.2, alpha=0.7)
    ax.set_xlim(*VIEW_LIM)
    ax.set_ylim(*VIEW_LIM)
    ax.set_title(f"class {c}, continuous",
                 fontsize=9)
    ax.set_xlabel("scaled feature $x_1$")
    np.atleast_1d(axes)[0].set_ylabel("scaled feature $x_2$")
fig.colorbar(im, ax=axes, shrink=0.85, label="normalised density")
fig.savefig("../figures/generated/class_surfaces_continuous.pdf",
            bbox_inches="tight", transparent=True)