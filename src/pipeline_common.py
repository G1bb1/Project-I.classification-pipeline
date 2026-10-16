'''
File:
pipeline_common.py

Description:
Shared setup for every stage: which dataset to use, where files live, the
feature transform and scaling, the train/val/test split, the lookup grid,
and byte storage. Defining these once here guarantees every stage agrees
on them.

Switch datasets by changing DATASET below:
  "sim"    -- synthetic crescents + ring (the original R&D setup)
  "sensor" -- real UV/IR light readings (shade, LED, fluorescent, sun)

Note:
- All paths are relative to src/, so the stages must be run from there.
- Inputs are read from datasets/<DATASET>/; everything a stage produces
  goes to outputs/<DATASET>/ (labels.csv, surfaces.npz, figures/).

___________________________________________________
Author      Date                Comment
Gib M.      Sept. 11th, 2026    rev 1
Gib M.      Oct. 14th, 2026     rev 2 -- added the DATASET switch, sensor
                                data loading, the log1p feature transform,
                                per-dataset paths (datasets/ and outputs/
                                folders), shared BANDWIDTH and probe points;
                                rewrote header and comments.
Gib M.      Oct. 15th, 2026     rev 3 -- added CLASS_NAMES (cluster number
                                to name) for model.h and the device check.
'''
import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity

# ---------------------------------------------------------------------------
# Dataset selection and file locations
# ---------------------------------------------------------------------------

# Which data every stage runs on -- the one switch to flip.
DATASET = "sim"  # "sim" or "sensor"

# Inputs live in datasets/<DATASET>/, results in outputs/<DATASET>/.
DATA_DIR = f"../datasets/{DATASET}/"
OUT_DIR = f"../outputs/{DATASET}/"
DATASET_CSV = DATA_DIR + "dataset.csv"     # written by generate_dataset.py
LABELS_CSV = OUT_DIR + "labels.csv"        # DBSCAN labels (cluster_dbscan.py)
SURFACES_NPZ = OUT_DIR + "surfaces.npz"    # class maps (kde_class_contours.py)
FIG_DIR = OUT_DIR + "figures/"             # every report figure

# Settings that differ between the two datasets.
if DATASET == "sim":
    FEATURE_NAMES = ("x1", "x2")
    AXIS_LABELS = ("scaled feature $x_1$", "scaled feature $x_2$")
    BANDWIDTH = 0.05  # KDE bandwidth, frozen from bandwidth_sweep.py
    # Raw test points: one on each shape, then the ring's hollow centre.
    PROBES_RAW = ([0.0, -1.8], [0.0, 5.0], [6.8, 1.6], [4.8, 1.6])
    # A point that must come back "unknown": the ring centre has data all
    # around it, but no class was ever trained there.
    UNKNOWN_RAW = [4.8, 1.6]
    # DBSCAN only numbers its clusters; names by cluster number, read off
    # each cluster's centroid. Recheck if the DBSCAN settings change.
    CLASS_NAMES = ("lower crescent", "ring", "upper crescent")
else:
    # The peer's readings for all four light conditions with the label
    # column removed -- the pipeline's input, so it never sees the labels.
    RAW_CSV = DATA_DIR + "data_complete_no_labels.csv"
    FEATURE_NAMES = ("uv_sensor", "IR_sensor")
    AXIS_LABELS = ("scaled log UV", "scaled log IR")
    BANDWIDTH = 0.03  # KDE bandwidth, frozen from bandwidth_sweep.py
    # Raw [UV, IR] readings: typical shade, LED, fluorescent and sun.
    PROBES_RAW = ([0.0, 140.0], [0.0, 226.0], [0.0, 380.0], [3500.0, 28600.0])
    # Brighter than any indoor light but far dimmer than sun: "unknown".
    UNKNOWN_RAW = [1000.0, 2000.0]
    # Names by cluster number, read off each cluster's centroid (never from
    # the labels). They match the peer's labels so device results can be
    # scored against them.
    CLASS_NAMES = ("shade", "sun", "fluorescent", "LED")

os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Model and plotting constants
# ---------------------------------------------------------------------------

SEED = 20250903

# Flash cost is classes * GRID_N**2 bytes, so the grid size is a memory
# decision.
GRID_N = 16

# If even the winning class scores below this, it is too weak to trust and
# the answer is "unknown".
REJECT_TAU = 0.12

# Plot limits: the [0, 1] feature space padded 10% on each side, so points
# on (or just past) the training range stay clear of the frame.
VIEW_LIM = (-0.1, 1.1)

# Three-way split. Train builds the model (clustering, KDE, discretization);
# validation tunes the design choices (bandwidth, grid); test is spent once,
# on the frozen design, to report an honest number on unseen data.
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.60, 0.20, 0.20


# ---------------------------------------------------------------------------
# Loading the data
# ---------------------------------------------------------------------------

def _arc(rng, cx, cy, r, a0_deg, a1_deg, n, jitter):
    """n noisy points on the arc of a circle -- a crescent or a full ring."""
    a = np.linspace(np.radians(a0_deg), np.radians(a1_deg), n)
    pts = np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])
    return pts + rng.normal(0.0, jitter, pts.shape)


def load_dataset(csv_path=None):
    """Raw 2-D features for the current DATASET (or from csv_path).

    sim: two crescents and a ring, plus uniform noise. The shapes are the
    point: a crescent's centroid sits off the crescent and a ring's sits in
    its hole, so no centre-based method can describe them -- which is why
    the pipeline starts with DBSCAN. The shapes sit far enough apart that
    DBSCAN recovers them at a stable eps even after the 60% training split
    thins the data.

    sensor: the peer's [UV, IR] readings, without labels -- the pipeline
    must discover the classes on its own.
    """
    if csv_path is not None:
        return np.loadtxt(csv_path, delimiter=",")
    if DATASET == "sensor":
        return np.loadtxt(RAW_CSV, delimiter=",")

    rng = np.random.default_rng(SEED)
    smile = _arc(rng, 0.0, 0.0, 1.8, 200, 340, 340, 0.11)  # opens upward
    frown = _arc(rng, 0.0, 3.2, 1.8, 20, 160, 300, 0.11)   # opens downward
    ring = _arc(rng, 4.8, 1.6, 2.0, 0, 360, 300, 0.11)     # closed loop

    X = np.vstack([smile, frown, ring])
    # Background noise over the data's bounding box, padded by 0.4.
    noise = rng.uniform(X.min(axis=0) - 0.4, X.max(axis=0) + 0.4, size=(90, 2))
    return np.vstack([X, noise])


# ---------------------------------------------------------------------------
# Transform, scale and split
# ---------------------------------------------------------------------------

def transform_features(X):
    """Applied to RAW points before scaling -- query points included.

    Sensor IR spans ~130 (shade) to ~30000 (sun). Scaled linearly, shade,
    LED and fluorescent all land within 0.01 of each other, in the same
    grid cell, and DBSCAN merges them. log1p spreads them back out (log1p
    rather than log, because UV reads exactly 0 indoors). The sim data is
    left as it is.
    """
    X = np.asarray(X, dtype=float)
    return np.log1p(X) if DATASET == "sensor" else X


def untransform_features(X):
    """Inverse of transform_features, to print points in raw units."""
    X = np.asarray(X, dtype=float)
    return np.expm1(X) if DATASET == "sensor" else X


def scale_features(X, bounds=None):
    """Min-max each feature onto [0, 1]. Returns (X_scaled, bounds).

    Pass the returned bounds back in to scale a query point the same way:
    they are part of the shipped model, not just a training detail.
    """
    lo, hi = bounds if bounds is not None else (X.min(axis=0), X.max(axis=0))
    return (X - lo) / (hi - lo), (lo, hi)


def split_data(X, seed=SEED):
    """Split points into (train, val, test), 60/20/20.

    The seed is fixed, so anyone re-running gets the same three subsets --
    which is what makes the reported test number checkable.
    """
    train, rest = train_test_split(X, train_size=TRAIN_FRAC, random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, random_state=seed)
    return train, val, test


def load_split_scaled(seed=SEED, csv_path=None):
    """Load, transform, split, and scale every subset by TRAIN's bounds.

    The bounds come from training data alone and ship with the model, so
    validation and test points are scaled by them rather than by their own
    range. Held-out points may therefore land slightly outside [0, 1]; that
    is honest, and the grid lookup clips them into the edge cells.

    Returns a dict: scaled "train"/"val"/"test" arrays plus "bounds".
    """
    X = transform_features(load_dataset(csv_path))
    train, val, test = split_data(X, seed)
    train_s, bounds = scale_features(train)
    val_s, _ = scale_features(val, bounds)
    test_s, _ = scale_features(test, bounds)
    return {"train": train_s, "val": val_s, "test": test_s, "bounds": bounds}


# ---------------------------------------------------------------------------
# The lookup grid
# ---------------------------------------------------------------------------

def make_grid():
    """Cell centres of the GRID_N x GRID_N grid covering [0, 1]^2.

    Returns (xx, yy, points): the two meshgrid arrays and the same centres
    as an (N*N, 2) list of points.
    """
    centres = (np.arange(GRID_N) + 0.5) / GRID_N
    xx, yy = np.meshgrid(centres, centres)
    return xx, yy, np.column_stack([xx.ravel(), yy.ravel()])


def cell_of(point):
    """The (row, col) cell holding a scaled point; off-grid points clip to
    the nearest edge cell. Row follows feature 2, column feature 1."""
    i = np.clip((np.asarray(point) * GRID_N).astype(int), 0, GRID_N - 1)
    return int(i[1]), int(i[0])


# ---------------------------------------------------------------------------
# Density estimation
# ---------------------------------------------------------------------------

def fit_density(points, bandwidth):
    """Fit a Gaussian KDE. Note: score_samples() returns LOG density."""
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    return kde.fit(points)


def evaluate_on_grid(kde, grid_points, xx):
    """Evaluate a fitted KDE at every cell centre, shaped like the grid."""
    return np.exp(kde.score_samples(grid_points)).reshape(xx.shape)


def normalise_peak(surface):
    """Scale a density map so its own peak is 1.0.

    Done per class, so a small class is not dwarfed by a large one: the
    classifier asks "how close to this class's own peak am I?".
    """
    peak = surface.max()
    return surface / peak if peak > 0 else surface


# ---------------------------------------------------------------------------
# Byte storage
# ---------------------------------------------------------------------------

def quantize(surface_01):
    """Store a peak-normalised map as one byte (0-255) per cell."""
    return np.round(np.clip(surface_01, 0.0, 1.0) * 255).astype(np.uint8)


def dequantize(codes):
    """Read stored bytes back as values in [0, 1]."""
    return codes.astype(float) / 255.0
