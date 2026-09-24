"""Part 3 -- store the maps as bytes, and count
the flash cost.

Every candidate format holds the same GRID_N x
GRID_N values per class;
they differ only in bytes per cell. The table this
prints is the
footprint argument your report has to make.
"""
import matplotlib.pyplot as plt
import numpy as np
from pipeline_common import quantize, dequantize, GRID_N, VIEW_LIM

surfaces = np.load("surfaces.npz")
classes = sorted(int(k[1:]) for k in surfaces.files)
K = len(classes)
cells = GRID_N * GRID_N

FORMATS = [("uint8", 1), ("uint16", 2), ("float32", 4), ("float64", 8)]

print(f"{'storage':>9} {'B/cell':>7} {'per class':>10} {'all ' + str(K):>8} {'vs uint8':>9}")
for name, width in FORMATS:
    print(f"{name:>9} {width:>7} {width * cells:>10} {width * cells * K:>8}"
          f" {width:>8}x")
print(f"+ {2 * 2 * 4} bytes of scaling bounds (2 floats per feature)")

# The accuracy side of the argument: rounding to
# one of 256 levels can be
# off by at most half a level, i.e. 1/(2*255) = 0.196% of peak.
worst = max(np.abs(dequantize(quantize(surfaces[f"c{c}"]))
                    - surfaces[f"c{c}"]).max()
            for c in classes)
print(f"uint8 worst-case error: {100 * worst:.3f}% of peak")

fig, axes = plt.subplots(1, K, figsize=(3.0 * K, 3.0),
                         sharex=True, sharey=True)
for ax, c in zip(np.atleast_1d(axes), classes):
    im = ax.imshow(quantize(surfaces[f"c{c}"]),
                   extent=[0, 1, 0, 1],
                   origin="lower", cmap="Greys",
                   interpolation="nearest",
                   aspect="equal", vmin=0, vmax=255)
    ax.set_xlim(*VIEW_LIM)
    ax.set_ylim(*VIEW_LIM)
    ax.set_title(f"class {c}, stored uint8",
                 fontsize=9)
    ax.set_xlabel("scaled feature $x_1$")
    np.atleast_1d(axes)[0].set_ylabel("scaled feature $x_2$")
fig.colorbar(im, ax=axes, shrink=0.85, label="stored byte (0--255)")
fig.savefig("../figures/generated/class_surfaces_discretized.pdf",
            bbox_inches="tight", transparent=True)