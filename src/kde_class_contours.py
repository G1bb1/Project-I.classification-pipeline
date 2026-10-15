'''
File:
kde_class_contours.py

Description:
Part 2 -- build one density map per class.

A Gaussian KDE is fit to each DBSCAN class, then evaluated at the centre
of every cell of the GRID_N x GRID_N lookup grid and normalised so each
class's own peak is 1.0. These per-class maps are what the classifier is
built from: they are saved to surfaces.npz and read by every later stage.

The figure shows the same KDEs on a fine 200x200 grid -- the continuous
density before the lookup grid coarsens it -- with one panel per class.
Compare it against Part 3's figure, where the same maps appear as stored
bytes on the coarse grid.

Note:
- Run from src/ (paths are relative to it), after cluster_dbscan.py.
- Reads outputs/<DATASET>/labels.csv.
- Writes outputs/<DATASET>/surfaces.npz (one array per class, keys
  "c0", "c1", ...) and figures/class_surfaces_continuous.pdf.

___________________________________________________
Author      Date                Comment
Gib M.      Sept. 23rd, 2026    rev 1
Gib M.      Oct. 14th, 2026     rev 2 -- BANDWIDTH, paths and axis labels
                                from pipeline_common; merged the Part 2
                                description into this header; rewrote
                                comments.
'''
import matplotlib.pyplot as plt
import numpy as np
from pipeline_common import (load_split_scaled, make_grid, fit_density,
                             evaluate_on_grid, normalise_peak, VIEW_LIM,
                             AXIS_LABELS, BANDWIDTH, LABELS_CSV,
                             SURFACES_NPZ, FIG_DIR)

DISPLAY_N = 200  # fine grid used only for the figure, not for storage

X = load_split_scaled()["train"]  # KDE is fit on TRAINING data only
labels = np.loadtxt(LABELS_CSV, dtype=int)
classes = sorted(set(labels) - {-1})
xx, yy, grid_points = make_grid()

# --- The stored maps: one KDE per class, sampled on the lookup grid ---------
kdes = {c: fit_density(X[labels == c], BANDWIDTH) for c in classes}
surfaces = {c: normalise_peak(evaluate_on_grid(kdes[c], grid_points, xx))
            for c in classes}
np.savez(SURFACES_NPZ, **{f"c{c}": s for c, s in surfaces.items()})

# --- Figure: the continuous KDE per class, on a fine grid -------------------
g = np.linspace(0, 1, DISPLAY_N)
fxx, fyy = np.meshgrid(g, g)
fine = np.column_stack([fxx.ravel(), fyy.ravel()])

fig, axes = plt.subplots(1, len(classes), figsize=(3.0 * len(classes), 3.0),
                         sharex=True, sharey=True)
for ax, c in zip(np.atleast_1d(axes), classes):
    # Darker = denser; each panel normalised to its own peak.
    smooth = normalise_peak(
        np.exp(kdes[c].score_samples(fine)).reshape(fxx.shape))
    im = ax.imshow(smooth, extent=[0, 1, 0, 1], origin="lower",
                   cmap="Greys", interpolation="bilinear", aspect="equal",
                   vmin=0, vmax=1)
    # The class's training points on top, so the fit can be judged.
    ax.scatter(X[labels == c][:, 0], X[labels == c][:, 1], s=2, c="white",
               edgecolors="black", linewidths=0.2, alpha=0.7)
    ax.set_xlim(*VIEW_LIM)
    ax.set_ylim(*VIEW_LIM)
    ax.set_title(f"class {c}, continuous", fontsize=9)
    ax.set_xlabel(AXIS_LABELS[0])
np.atleast_1d(axes)[0].set_ylabel(AXIS_LABELS[1])

fig.colorbar(im, ax=axes, shrink=0.85, label="normalised density")
fig.savefig(FIG_DIR + "class_surfaces_continuous.pdf", bbox_inches="tight",
            transparent=True)
