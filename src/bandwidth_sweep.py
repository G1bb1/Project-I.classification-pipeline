'''
File:
bandwidth_sweep.py

Description:
Part 2 (tuning) -- choose the KDE bandwidth h on the VALIDATION set.

The bandwidth controls how smooth each class's density is: too small and
it only fits the exact training points, too large and the classes blur
together. For each candidate h, a KDE per class is fit on the training
data, and the held-out validation points are scored under the combined
model. The h with the highest mean log-likelihood generalises best.

This is how h gets frozen before Part 4 spends the test set -- the test
set is never touched here. The chosen value is then written into
BANDWIDTH in pipeline_common.py by hand.

Note:
- Run from src/ (paths are relative to it), after cluster_dbscan.py.
- Reads outputs/<DATASET>/labels.csv. Prints only; writes nothing.

___________________________________________________
Author      Date                Comment
Gib M.      Sept. 23rd, 2026    rev 1
Gib M.      Oct. 14th, 2026     rev 2 -- labels path from pipeline_common;
                                rewrote header and comments.
'''
import numpy as np
from pipeline_common import load_split_scaled, fit_density, LABELS_CSV

# Candidate bandwidths, in scaled-feature units (the space is [0, 1]).
SWEEP = [0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12]

data = load_split_scaled()
Xtr, Xval = data["train"], data["val"]
labels = np.loadtxt(LABELS_CSV, dtype=int)  # DBSCAN labels for Xtr
classes = sorted(set(labels) - {-1})        # noise (-1) is not a class

print(f"{'h':>6} {'val mean log-likelihood':>26}")
best_h, best_ll = None, -np.inf
for h in SWEEP:
    # Total density at each validation point = sum of the class KDEs.
    # score_samples() returns log density, hence the exp().
    density = np.zeros(len(Xval))
    for c in classes:
        density += np.exp(fit_density(Xtr[labels == c], h).score_samples(Xval))

    # Mean log-likelihood; the tiny constant avoids log(0) for points far
    # from every class.
    ll = np.log(density + 1e-12).mean()
    if ll > best_ll:
        best_h, best_ll = h, ll
    print(f"{h:>6.2f} {ll:>26.3f}")

print(f"\nvalidation selects h = {best_h:.2f} (highest log-likelihood)")
