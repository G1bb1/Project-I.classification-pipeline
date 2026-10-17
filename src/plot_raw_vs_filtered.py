'''
File:
plot_raw_vs_filtered.py

Description:
Part 4 (sensor) -- filter evidence: raw against EMA-filtered IR.
Plots a capture streamed by classify.ino's R command (raw and filtered
readings side by side, plus the class the device reported) against
sample time. The dashed lines are the IR values where the stored grid
switches class, computed from the shipped tables and scaling bounds, so
the figure shows directly when a dip in the signal crosses into another
class.

It also replays the raw readings offline and prints how many samples are
misclassified with no filter, with the EMA, and with the EMA plus an
output hold (a new class is reported only after it appears HOLD_N times
in a row).

Note:
- Run from src/ (paths are relative to it). Sensor dataset only.
- Reads datasets/sensor/data_raw_vs_filtered.csv
  (raw_uv,raw_ir,filt_uv,filt_ir,predicted), surfaces.npz and the
  training bounds.
- Writes outputs/sensor/figures/raw_vs_filtered.pdf.

___________________________________________________
Author      Date                Comment
Gib M.      Oct. 16th, 2026     rev 1
'''
import csv
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from pipeline_common import (DATA_DIR, FIG_DIR, GRID_N, REJECT_TAU,
                             SURFACES_NPZ, cell_of, dequantize,
                             load_split_scaled, quantize, scale_features,
                             transform_features)

CAPTURE_CSV = DATA_DIR + "data_raw_vs_filtered.csv"
TRUE_LABEL = "fluorescent"   # the light the capture was taken under
ALPHA = 0.2                  # must match classify.ino
SAMPLE_S = 0.1               # 10 Hz
HOLD_N = 5                   # output hold tested offline (0.5 s)

# --- Load the capture and the shipped model --------------------------------
with open(CAPTURE_CSV) as f:
    rows = list(csv.DictReader(f))
raw_uv = np.array([float(r["raw_uv"]) for r in rows])
raw_ir = np.array([float(r["raw_ir"]) for r in rows])
filt_uv = np.array([float(r["filt_uv"]) for r in rows])
filt_ir = np.array([float(r["filt_ir"]) for r in rows])
reported = [r["predicted"] for r in rows]
t = np.arange(len(rows)) * SAMPLE_S

bounds = load_split_scaled()["bounds"]
surfaces = np.load(SURFACES_NPZ)
classes = sorted(int(k[1:]) for k in surfaces.files)
tables = {c: quantize(surfaces[f"c{c}"]) for c in classes}


def stored_class(uv, ir):
    """Cluster number from the stored bytes, or -1 for unknown."""
    p, _ = scale_features(transform_features([uv, ir]), bounds)
    r, c = cell_of(p)
    v = np.array([dequantize(tables[k][r, c]) for k in classes])
    best = int(np.argmax(v))
    return -1 if v[best] < REJECT_TAU else classes[best]


# Cluster numbers -> names, by matching the replayed lookup on the filtered
# values against the names the device printed.
votes = Counter((stored_class(u, i), n) for u, i, n in zip(filt_uv, filt_ir, reported))
name_of = {}
for (cluster, name), _ in votes.most_common():
    name_of.setdefault(cluster, name)
name_of.setdefault(-1, "UNKNOWN")

# --- Class boundaries along IR (indoors, UV = 0 -> grid column 0) ----------
# A row edge k sits at log1p(IR) = ymin + k/GRID_N * (ymax - ymin).
lo, hi = bounds
row_class = [stored_class(0.0, np.expm1(lo[1] + (r + 0.5) / GRID_N * (hi[1] - lo[1])))
             for r in range(GRID_N)]
edges = [np.expm1(lo[1] + r / GRID_N * (hi[1] - lo[1]))
         for r in range(1, GRID_N) if row_class[r] != row_class[r - 1]]


# --- Offline replay: no filter, EMA, EMA + hold ----------------------------
def ema(x):
    out, y = [], None
    for v in x:
        y = v if y is None else ALPHA * v + (1 - ALPHA) * y
        out.append(y)
    return np.array(out)


def hold(seq, n):
    """Report a new class only after it appears n times in a row."""
    out, shown, cand, count = [], seq[0], None, 0
    for s in seq:
        if s == shown:
            cand, count = None, 0
        else:
            count = count + 1 if s == cand else 1
            cand = s
            if count >= n:
                shown, cand, count = s, None, 0
        out.append(shown)
    return out


def n_wrong(seq):
    return sum(name_of.get(s, "?") != TRUE_LABEL for s in seq)


no_filter = [stored_class(u, i) for u, i in zip(raw_uv, raw_ir)]
with_ema = [stored_class(u, i) for u, i in zip(ema(raw_uv), ema(raw_ir))]
print(f"samples: {len(rows)}, all {TRUE_LABEL}")
print(f"misclassified, no filter:        {n_wrong(no_filter)}")
print(f"misclassified, EMA:              {n_wrong(with_ema)}")
print(f"misclassified, EMA + {HOLD_N}-sample hold: {n_wrong(hold(with_ema, HOLD_N))}")
print(f"EMA replay vs device, max diff:  {np.abs(ema(raw_ir) - filt_ir).max():.2f} counts")

# --- Figure: signal on top, reported class as a strip below ----------------
plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False})
fig, (ax, strip) = plt.subplots(2, 1, figsize=(3.5, 2.8), sharex=True,
                                gridspec_kw={"height_ratios": [5, 1],
                                             "hspace": 0.12})

for e in edges:
    ax.axhline(e, color="0.6", lw=0.6, ls="--", zorder=0)
# Name each band at the right edge, by the class the grid gives inside it.
band_edges = [0.0] + edges + [400.0]
for a, b in zip(band_edges[:-1], band_edges[1:]):
    mid = np.sqrt(max(a, 1.0) * b) if a > 0 else (a + b) / 2
    ax.text(t[-1] + 0.15, min(mid, 330), name_of.get(stored_class(0.0, mid), "?"),
            fontsize=7, color="0.35", va="center", ha="left")
ax.plot(t, raw_ir, color="0.65", lw=0.8, label="raw IR")
ax.plot(t, filt_ir, color="0.05", lw=1.4, label=rf"EMA, $\alpha={ALPHA}$")
ax.set_ylabel("IR counts")
ax.set_ylim(0, 400)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False,
          fontsize=7, ncol=2)

shades = {"fluorescent": "0.85", "LED": "0.5", "shade": "0.1",
          "sun": "0.7", "UNKNOWN": "1.0"}
for ti, name in zip(t, reported):
    strip.axvspan(ti - SAMPLE_S / 2, ti + SAMPLE_S / 2, color=shades[name], lw=0)
strip.set_yticks([])
strip.spines["left"].set_visible(False)
strip.set_ylabel("output", rotation=0, ha="right", va="center")
strip.set_xlabel("time (s), 10 Hz")
strip.set_xlim(t[0], t[-1])
shown = [n for n in ("fluorescent", "LED", "shade", "sun", "UNKNOWN") if n in reported]
strip.legend(handles=[Patch(color=shades[n], label=n) for n in shown],
             loc="upper center", bbox_to_anchor=(0.5, -1.6), ncol=len(shown),
             frameon=False, fontsize=7)

fig.savefig(FIG_DIR + "raw_vs_filtered.pdf", bbox_inches="tight",
            transparent=True)
