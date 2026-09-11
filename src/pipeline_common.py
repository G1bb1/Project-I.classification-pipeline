'''
File:
pipeline_common.py

Description:

Note:
- At the very end of this file, be aware that the save file path
  is relative to the current working directory.

___________________________________________________
Author      Date                Comment
Gib M.      Sept. 11th, 2026     rev 1
'''

"""Shared setup: the dataset, feature scaling, the
lookup grid, and byte
storage. Defined once here so every stage agrees
on all four.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity

SEED = 20250903

# Flash cost is classes * GRID_N**2 bytes, so this
# is a memory decision.
GRID_N = 16

# Below this, the winning class is too weak to
# trust: answer "unknown".
REJECT_TAU = 0.12

# Plot axis limits: the [0,1] feature space padded
# 10% each side, so points
# on (or just past) the training range stay clear
# of the frame.
VIEW_LIM = (-0.1, 1.1)

# Three-way split. Train builds the model (
# clustering, KDE, discretization);
# validation tunes the design choices (bandwidth,
# grid); test is spent once,
# on the frozen design, to report an honest number
# on unseen data.
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.60, 0.20, 0.20


def _arc(rng, cx, cy, r, a0_deg, a1_deg, n, jitter):
    """n noisy points on the arc of a circle -- a
    crescent or a full ring."""
    a = np.linspace(np.radians(a0_deg), np.radians(a1_deg), n)
    pts = np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])
    return pts + rng.normal(0.0, jitter, pts.shape)


def load_dataset(csv_path=None):
    """Raw 2-D features: two crescents and a ring,
    plus uniform noise.

    The shapes are the point. A crescent’s
    centroid sits off the crescent
    and a ring’s sits in its hole, so no centre-
    based method can describe
    them -- which is why the pipeline starts with
    DBSCAN. The three shapes
    sit at well-separated centres so DBSCAN
    recovers them at a stable eps
    even after the 60% training split thins the
    data.
    """
    if csv_path is not None:
        return np.loadtxt(csv_path, delimiter=",")

    rng = np.random.default_rng(SEED)
    smile = _arc(rng, 0.0, 0.0, 1.8, 200, 340, 340, 0.11)  # opens upward
    frown = _arc(rng, 0.0, 3.2, 1.8, 20, 160, 300, 0.11)  # opens downward
    ring = _arc(rng, 4.8, 1.6, 2.0, 0, 360, 300, 0.11)  # closed loop

    X = np.vstack([smile, frown, ring])
    noise = rng.uniform(
        X.min(axis=0) - 0.4,
        X.max(axis=0) + 0.4,
        size=(90, 2)
    )
    return np.vstack([X, noise])


def scale_features(X, bounds=None):
    """Min-max each feature onto [0, 1]. Returns (X_scaled, bounds).

    Pass the returned bounds back to scale a query
    point the same way --
    they are part of the shipped model, not a
    training detail.
    """
    lo, hi = bounds if bounds is not None else (X.min(axis=0), X.max(axis=0))
    return (X - lo) / (hi - lo), (lo, hi)


def split_data(X, seed=SEED):
    """Split raw points into (train, val, test), 60/20/20, fixed seed.

    Reproducible because the seed is fixed: anyone
    re-running gets the same
    three subsets, which is what makes the
    reported test number checkable.
    """
    train, rest = train_test_split(
        X,
        train_size=TRAIN_FRAC,
        random_state=seed
    )
    val, test = train_test_split(
        rest,
        test_size=0.5,
        random_state=seed
    )
    return train, val, test


def load_split_scaled(seed=SEED):
    """Load raw points, split, and scale every
    subset by TRAIN’s bounds.

    The bounds come from training data alone --
    they ship with the model,
    so validation and test points are scaled by
    them, not by their own
    range. Held-out points may therefore land
    slightly outside [0, 1];
    that is honest, and the grid lookup clips them
    into the edge cells.
    Returns a dict: scaled train/val/test arrays
    plus the shared bounds.
    """
    train, val, test = split_data(load_dataset(), seed)
    train_s, bounds = scale_features(train)
    val_s, _ = scale_features(val, bounds)
    test_s, _ = scale_features(test, bounds)
    return {"train": train_s, "val": val_s, "test": test_s, "bounds": bounds}


def make_grid():
    """Cell centres of the GRID_N x GRID_N grid
    covering [0, 1]^2."""
    centres = (np.arange(GRID_N) + 0.5) / GRID_N
    xx, yy = np.meshgrid(centres, centres)
    return xx, yy, np.column_stack([xx.ravel(), yy.ravel()])


def cell_of(point):
    """The (row, col) cell holding a scaled point."""
    i = np.clip((np.asarray(point) * GRID_N).astype(int), 0, GRID_N - 1)
    return int(i[1]), int(i[0])


def fit_density(points, bandwidth):
    """Fit a Gaussian KDE. score_samples() returns
    LOG density."""
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    return kde.fit(points)


def evaluate_on_grid(kde, grid_points, xx):
    """Evaluate a fitted KDE at every cell centre."""
    return np.exp(kde.score_samples(grid_points)).reshape(xx.shape)


def normalise_peak(surface):
    """Scale a density map so its own peak is 1.0.

    Per class, so a small class is not dwarfed by
    a large one: the
    classifier compares "how close to this class’s
    own peak am I".
    """
    peak = surface.max()
    return surface / peak if peak > 0 else surface


def quantize(surface_01):
    """Store a peak-normalised map as one byte per
    cell."""
    return np.round(np.clip(surface_01, 0.0, 1.0) * 255).astype(np.uint8)


def dequantize(codes):
    """Read stored bytes back as values in [0, 1]."""
    return codes.astype(float) / 255.0