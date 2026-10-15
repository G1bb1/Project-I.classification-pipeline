'''
File:
discretize_surfaces.py

Description:
Part 3 -- store the class maps as bytes, and count the flash cost.

Every candidate storage format holds the same GRID_N x GRID_N values per
class; they differ only in bytes per cell. The table this prints is the
memory-footprint argument the report has to make, and the worst-case
rounding error is the accuracy side of the same argument. The figure
shows exactly what would be stored on the microcontroller.

Note:
- Run from src/ (paths are relative to it), after kde_class_contours.py.
- Reads outputs/<DATASET>/surfaces.npz.
- Writes outputs/<DATASET>/figures/class_surfaces_discretized.pdf.

___________________________________________________
Author      Date                Comment
Gib M.      Sept. 23rd, 2026    rev 1
Gib M.      Oct. 14th, 2026     rev 2 -- paths and axis labels from
                                pipeline_common; added header and comments.
'''
import matplotlib.pyplot as plt
import numpy as np
from pipeline_common import (quantize, dequantize, GRID_N, VIEW_LIM,
                             AXIS_LABELS, SURFACES_NPZ, FIG_DIR)

surfaces = np.load(SURFACES_NPZ)
classes = sorted(int(k[1:]) for k in surfaces.files)  # keys are "c0", "c1"...
K = len(classes)
cells = GRID_N * GRID_N

# --- Flash cost per storage format ------------------------------------------
FORMATS = [("uint8", 1), ("uint16", 2), ("float32", 4), ("float64", 8)]

print(f"{'storage':>9} {'B/cell':>7} {'per class':>10} "
      f"{'all ' + str(K):>8} {'vs uint8':>9}")
for name, width in FORMATS:
    print(f"{name:>9} {width:>7} {width * cells:>10} "
          f"{width * cells * K:>8} {width:>8}x")
# The scaling bounds also ship: a min and max for each of the 2 features,
# as 4-byte floats.
print(f"+ {2 * 2 * 4} bytes of scaling bounds (2 floats per feature)")

# --- Accuracy cost of uint8 --------------------------------------------------
# Rounding to one of 256 levels is off by at most half a level,
# i.e. 1/(2*255) = 0.196% of peak.
worst = max(np.abs(dequantize(quantize(surfaces[f"c{c}"]))
                   - surfaces[f"c{c}"]).max()
            for c in classes)
print(f"uint8 worst-case error: {100 * worst:.3f}% of peak")

# --- Figure: the stored bytes, one panel per class ---------------------------
fig, axes = plt.subplots(1, K, figsize=(3.0 * K, 3.0),
                         sharex=True, sharey=True)
for ax, c in zip(np.atleast_1d(axes), classes):
    # "nearest" interpolation: show the cells as the blocks they really are.
    im = ax.imshow(quantize(surfaces[f"c{c}"]), extent=[0, 1, 0, 1],
                   origin="lower", cmap="Greys", interpolation="nearest",
                   aspect="equal", vmin=0, vmax=255)
    ax.set_xlim(*VIEW_LIM)
    ax.set_ylim(*VIEW_LIM)
    ax.set_title(f"class {c}, stored uint8", fontsize=9)
    ax.set_xlabel(AXIS_LABELS[0])
np.atleast_1d(axes)[0].set_ylabel(AXIS_LABELS[1])

fig.colorbar(im, ax=axes, shrink=0.85, label="stored byte (0--255)")
fig.savefig(FIG_DIR + "class_surfaces_discretized.pdf", bbox_inches="tight",
            transparent=True)
