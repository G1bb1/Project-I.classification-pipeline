"""Part 4 -- spend the TEST set once, on the
frozen design.

There are no ground-truth labels (DBSCAN invented
the classes), so the
honest question the test set can answer is: does
the cheap shipped model
(the uint8 grid lookup) reproduce the decision of
the expensive research
artifact (the continuous KDE) on data it never saw?
Agreement is the
accuracy figure; the confusion view shows where
the two disagree.

Frozen inputs: bandwidth h and grid come from
validation; the byte tables
come from surfaces.npz (training data only).
Reusing test data to tune any
of these would invalidate the number below.
"""
import numpy as np
from pipeline_common import (load_split_scaled, make_grid, cell_of,
                             fit_density, normalise_peak, quantize,
                             dequantize, REJECT_TAU)

BANDWIDTH = 0.05  # frozen: validation-selected

data = load_split_scaled()
Xtr, Xte = data["train"], data["test"]
labels = np.loadtxt("labels.csv", dtype=int)
classes = sorted(set(labels) - {-1})
xx, yy, grid_points = make_grid()

# Continuous artifact: per-class KDE and its own
# peak (the normaliser used
# to build the stored maps), so continuous and
# stored values are comparable.
kdes, peaks = {}, {}
for c in classes:
    kde = fit_density(Xtr[labels == c], BANDWIDTH)
    kdes[c] = kde
    peaks[c] = np.exp(kde.score_samples(grid_points)).max()

# Stored artifact: one uint8 table per class, read
# from surfaces.npz.
surfaces = np.load("surfaces.npz")
tables = {c: quantize(surfaces[f"c{c}"]) for c in classes}

LABELS = classes + [-1]  # -1 = "unknown" (rejected)


def continuous_decision(point):
    vals = np.array([
        np.exp(kdes[c].score_samples([point])[0]) / peaks[c]
        for c in classes
    ])
    best = int(np.argmax(vals))
    return -1 if vals[best] < REJECT_TAU else classes[best]


def stored_decision(point):
    r, c = cell_of(point)
    vals = np.array([
        dequantize(tables[k][r, c])
        for k in classes
    ])
    best = int(np.argmax(vals))
    return -1 if vals[best] < REJECT_TAU else classes[best]


cont = np.array([continuous_decision(p) for p in Xte])
stor = np.array([stored_decision(p) for p in Xte])
agree = (cont == stor).mean()

print(f"test points: {len(Xte)}")
print(f"stored-vs-continuous agreement on held-out test: {100 * agree:.1f}%")

print("\nconfusion (rows = continuous KDE, cols = stored uint8):")
head = "".join(
    f"{('unk' if l == -1 else 'c' + str(l)):>6}"
    for l in LABELS
)
print(f"{'':>10}{head}")

for lt in LABELS:
    row = "".join(
        f"{int(((cont == lt) & (stor == lc)).sum()):>6}"
        for lc in LABELS
    )
    name = "unknown" if lt == -1 else f"class {lt}"
    print(f"{name:>10}{row}")