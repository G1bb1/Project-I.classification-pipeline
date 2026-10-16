"""Part 4 -- the online path: scale the query,
index the grid, read one
byte per class, decide or reject.

No KDE is refitted and no training point is
consulted, so the cost does
not depend on how much data built the tables.
"""
import matplotlib.pyplot as plt
import numpy as np
from pipeline_common import (load_split_scaled, scale_features, cell_of,
                             quantize, dequantize,
                             GRID_N, REJECT_TAU, VIEW_LIM)

data = load_split_scaled()
Xtest, bounds = data["test"], data["bounds"]
surfaces = np.load("../figures/surfaces.npz")
classes = sorted(int(k[1:]) for k in surfaces.files)

# The shipped model: one uint8 table per class, plus `bounds`.
tables = {c: quantize(surfaces[f"c{c}"]) for c in classes}

# The query is a genuine HELD-OUT test point, not a hand-picked one: among
# the test points confidently classified and away from the plot edges (so
# the marker is visible), take the one the stored model claims most
# strongly. The figure then shows a real unseen point being classified.
def _win_value(p):
    return max(dequantize(tables[k][cell_of(p)]) for k in classes)


_interior = [i for i, p in enumerate(Xtest)
             if _win_value(p) >= REJECT_TAU
             and all(2 <= v <= GRID_N - 3 for v in cell_of(p))]
QUERY = Xtest[max(_interior, key=lambda i: _win_value(Xtest[i]))]
QUERY_RAW = QUERY * (bounds[1] - bounds[0]) + bounds[0]  # back to raw units


def lookup_nearest(point):
    """One byte per class, from the cell containing the point."""
    r, c = cell_of(point)
    return np.array([dequantize(tables[k][r, c]) for k in classes])


def lookup_bilinear(point):
    """Blend the four surrounding cells: 4 bytes per class, smoother."""
    fc, fr = np.clip(np.asarray(point) * GRID_N - 0.5, 0, GRID_N - 1)
    c0, r0 = int(fc), int(fr)
    c1, r1 = min(c0 + 1, GRID_N - 1), min(r0 + 1, GRID_N - 1)
    a, b = fc - c0, fr - r0
    out = []
    for k in classes:
        T = dequantize(tables[k])
        out.append(T[r0, c0] * (1 - a) * (1 - b) +
                   T[r0, c1] * a * (1 - b) +
                   T[r1, c0] * (1 - a) * b +
                   T[r1, c1] * a * b)
    return np.array(out)


def classify(values):
    """Distribution over classes, or None when the winner is below tau."""
    total = values.sum()
    dist = values / total if total > 0 else np.zeros_like(values)
    best = int(np.argmax(values))
    return (None if values[best] < REJECT_TAU else classes[best]), dist


for name, values in (("nearest", lookup_nearest(QUERY)),
                      ("bilinear", lookup_bilinear(QUERY))):
    label, dist = classify(values)
    verdict = "unknown" if label is None else f"class {label}"
    print(f"{name:>9}: {verdict:>9} bytes=" +
          " ".join(f"{v * 255:6.1f}" for v in values) +
          " dist=" + " ".join(f"{d * 100:5.1f}%" for d in dist))

r, c = cell_of(QUERY)
print(f"\nquery {QUERY_RAW} -> cell [row {r}, col {c}] of {GRID_N}x{GRID_N}")
print("stored bytes in the 3x3 window around that cell:")
for k in classes:
    print(f" class {k}:")
    for rr in range(min(r + 1, GRID_N - 1), max(r - 2, -1), -1):
        print(" " + " ".join(f"{tables[k][rr, cc]:4d}"
                             for cc in range(max(c - 1, 0),
                                             min(c + 2, GRID_N))))

stack = np.stack([dequantize(tables[c]) for c in classes])
winning_value = stack.max(axis=0)
rejected = winning_value < REJECT_TAU
print(f"\nreject region: {rejected.sum()} of {rejected.size} cells "
      f"({100 * rejected.mean():.1f}%) below tau={REJECT_TAU}")

# The hollow centre of the ring: surrounded by data, yet no class was ever
# trained there. An honest classifier must answer "unknown", not guess.
RING_CENTRE_RAW = np.array([4.8, 1.6])
ring_centre, _ = scale_features(RING_CENTRE_RAW, bounds)
label, _ = classify(lookup_nearest(ring_centre))
print(f"ring centre {RING_CENTRE_RAW} -> "
      f"{'unknown' if label is None else f'class {label}'}")

# -- Figure: the one test query, the grid, and the class probabilities. --
# Left: where the query lands in the grid. Right: the probability read for
# each class at that cell; the tallest bar is the argmax decision.
_, dist = classify(lookup_nearest(QUERY))
winner = int(np.argmax(dist))
r, c = cell_of(QUERY)
edges = np.linspace(0, 1, GRID_N + 1)

fig, (axg, axp) = plt.subplots(1, 2, figsize=(7.2, 3.4))

# Left panel -- the grid with the query’s cell highlighted.
axg.add_patch(plt.Rectangle((edges[c], edges[r]),
                            1 / GRID_N, 1 / GRID_N,
                            facecolor="0.8",
                            edgecolor="black", lw=1.2, zorder=1))
for v in edges:
    axg.axvline(v, color="0.75", lw=0.2, zorder=0)
    axg.axhline(v, color="0.75", lw=0.2, zorder=0)
axg.plot(*QUERY, marker="*", markersize=18, markerfacecolor="white",
         markeredgecolor="black", markeredgewidth=1.5,
         linestyle="none", zorder=3)
axg.set_xlim(*VIEW_LIM)
axg.set_ylim(*VIEW_LIM)
axg.set_aspect("equal")
axg.set_xlabel("scaled feature $x_1$")
axg.set_ylabel("scaled feature $x_2$")
axg.set_title(f"Test query in cell (row {r}, col {c})", fontsize=9)

# Right panel -- one bar per class; the winner is filled dark.
xs = np.arange(len(classes))
shades = ["0.8"] * len(classes)
shades[winner] = "0.2"
axp.bar(xs, dist, width=0.6, color=shades,
        edgecolor="black", linewidth=0.8)
for x, p in zip(xs, dist):
    axp.text(x, p + 0.03, f"{p:.2f}", ha="center",
             va="bottom", fontsize=9)
axp.axhline(REJECT_TAU, color="black", ls="--", lw=0.8)
axp.text(len(classes) - 0.5, REJECT_TAU + 0.01, f"$\\tau={REJECT_TAU}$",
         ha="right", va="bottom", fontsize=8)
axp.set_xticks(xs)
axp.set_xticklabels([f"class {k}" for k in classes])
axp.set_ylim(0, 1.15)
axp.set_ylabel("probability $p_c$")
axp.set_title(f"argmax $\\rightarrow$ class {classes[winner]}", fontsize=9)
for side in ("top", "right"):
    axp.spines[side].set_visible(False)

fig.tight_layout()
fig.savefig("../figures/generated/decision_map.pdf",
            bbox_inches="tight", transparent=True)