'''
File:
bandwidth_sweep.py

Description:

All these 3 maps are saved on the class_surfaces_continuous.pdf.

Note:

___________________________________________________
Author      Date                Comment
Gib M.      Sept. 23th, 2026     rev 1
'''

"""Part 2 tuning -- choose the KDE bandwidth h on
the VALIDATION set.

The test set is never touched here. For each
candidate h we fit the
per-class KDE on training data, then score the
held-out validation points
under the fitted mixture: the h with the highest
validation mean
log-likelihood generalises best. This is how h is
frozen before Part 4
spends the test set once.
"""
import numpy as np
from pipeline_common import load_split_scaled, fit_density

SWEEP = [0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12]

# Gets data in the dataset
data = load_split_scaled()
Xtr, Xval = data["train"], data["val"]
labels = np.loadtxt("labels.csv", dtype=int)
classes = sorted(set(labels) - {-1})

# Computes multiple bandwidths to find the best one.
print(f"{'h':>6} {'val mean log-likelihood':>26}")
best_h, best_ll = None, -np.inf
for h in SWEEP:
    density = np.zeros(len(Xval))
    for c in classes:
        density += np.exp(fit_density(Xtr[labels == c], h).score_samples(Xval))
    ll = np.log(density + 1e-12).mean()
    if ll > best_ll:
        best_h, best_ll = h, ll
    print(f"{h:>6.2f} {ll:>26.3f}")
print(f"\nvalidation selects h = {best_h:.2f} (highest log-likelihood)")