'''
File:
evaluate_test.py

Description:
Part 4 -- spend the TEST set once, on the frozen design.

There are no ground-truth labels (DBSCAN invented the classes), so the
honest question the test set can answer is: does the cheap shipped model
(the uint8 grid lookup) reproduce the decision of the expensive research
artifact (the continuous KDE) on data it never saw? Their agreement is the
accuracy figure, and the confusion table shows where the two disagree.

Frozen inputs: the bandwidth h and the grid were chosen on validation,
and the byte tables come from surfaces.npz (built from training data
only). Reusing test data to tune any of these would invalidate the
number this script prints.

Note:
- Run from src/ (paths are relative to it), after kde_class_contours.py.
- Reads outputs/<DATASET>/labels.csv and surfaces.npz. Prints only.

___________________________________________________
Author      Date                Comment
Gib M.      Sept. 23rd, 2026    rev 1
Gib M.      Oct. 14th, 2026     rev 2 -- BANDWIDTH and paths from
                                pipeline_common; added header and comments.
'''
import numpy as np
from pipeline_common import (load_split_scaled, make_grid, cell_of,
                             fit_density, quantize, dequantize, REJECT_TAU,
                             BANDWIDTH, LABELS_CSV, SURFACES_NPZ)

data = load_split_scaled()
Xtr, Xte = data["train"], data["test"]
labels = np.loadtxt(LABELS_CSV, dtype=int)
classes = sorted(set(labels) - {-1})
xx, yy, grid_points = make_grid()

# --- The two models being compared ------------------------------------------
# Continuous artifact: the per-class KDE, plus each class's peak on the grid
# (the same normaliser used to build the stored maps), so continuous and
# stored values are on the same 0-1 scale.
kdes, peaks = {}, {}
for c in classes:
    kde = fit_density(Xtr[labels == c], BANDWIDTH)
    kdes[c] = kde
    peaks[c] = np.exp(kde.score_samples(grid_points)).max()

# Stored artifact: one uint8 table per class, read from surfaces.npz.
surfaces = np.load(SURFACES_NPZ)
tables = {c: quantize(surfaces[f"c{c}"]) for c in classes}

LABELS = classes + [-1]  # -1 = "unknown" (rejected)


def continuous_decision(point):
    """Decision from the exact KDE values at the point."""
    vals = np.array([
        np.exp(kdes[c].score_samples([point])[0]) / peaks[c]
        for c in classes
    ])
    best = int(np.argmax(vals))
    return -1 if vals[best] < REJECT_TAU else classes[best]


def stored_decision(point):
    """Decision from the stored bytes of the point's grid cell."""
    r, c = cell_of(point)
    vals = np.array([
        dequantize(tables[k][r, c])
        for k in classes
    ])
    best = int(np.argmax(vals))
    return -1 if vals[best] < REJECT_TAU else classes[best]


# --- Score the test set ------------------------------------------------------
cont = np.array([continuous_decision(p) for p in Xte])
stor = np.array([stored_decision(p) for p in Xte])
agree = (cont == stor).mean()

print(f"test points: {len(Xte)}")
print(f"stored-vs-continuous agreement on held-out test: {100 * agree:.1f}%")

# --- Confusion table: where do the two models disagree? ---------------------
# Diagonal = agreement; anything off the diagonal is a point where the
# byte lookup and the KDE gave different answers.
print("\nconfusion (rows = continuous KDE, cols = stored uint8):")
head = "".join(f"{('unk' if l == -1 else 'c' + str(l)):>6}" for l in LABELS)
print(f"{'':>10}{head}")

for lt in LABELS:
    row = "".join(f"{int(((cont == lt) & (stor == lc)).sum()):>6}"
                  for lc in LABELS)
    name = "unknown" if lt == -1 else f"class {lt}"
    print(f"{name:>10}{row}")
