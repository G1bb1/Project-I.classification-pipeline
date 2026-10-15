'''
File:
classify_skeleton.py

Description:
Part 4 (exercise) -- the classifier lookup, written by hand.

The tables, the grid and the scaling were provided; the exercise was to
fill in the lookup: turn a raw point into a grid cell, read one byte per
class, and return the winning class -- or None ("unknown") when even the
winner is below REJECT_TAU. Check the answers against classify_point.py.

Note:
- Run from src/ (paths are relative to it), after kde_class_contours.py.
- Reads outputs/<DATASET>/surfaces.npz. Prints only; writes nothing.

___________________________________________________
Author      Date                Comment
Gib M.      Sept. 23rd, 2026    rev 1
Gib M.      Oct. 14th, 2026     rev 2 -- applies the feature transform to
                                raw points (needed for sensor data); probe
                                points and paths from pipeline_common;
                                added header.
'''
import numpy as np
from pipeline_common import (load_split_scaled, scale_features, quantize,
                             dequantize, transform_features, GRID_N,
                             REJECT_TAU, SURFACES_NPZ, PROBES_RAW)

bounds = load_split_scaled()["bounds"]  # training bounds ship with the model
surfaces = np.load(SURFACES_NPZ)
classes = sorted(int(k[1:]) for k in surfaces.files)

# The shipped model. The classifier may read only these and `bounds`.
tables = {c: quantize(surfaces[f"c{c}"]) for c in classes}


def classify(raw_point):
    """Class label for a RAW point, or None for "unknown"."""
    # Same path as the training data: transform, then scale by the
    # training bounds.
    point, _ = scale_features(transform_features(raw_point), bounds)

    # Turn the scaled point into a grid cell. Points outside [0, 1] are
    # clipped into the edge cells.
    col = int(point[0] * GRID_N)
    row = int(point[1] * GRID_N)

    col = np.clip(col, 0, GRID_N - 1)
    row = np.clip(row, 0, GRID_N - 1)

    # Read one byte per class from that cell, back as values in [0, 1].
    values = np.array([
        dequantize(tables[c][row, col])
        for c in classes
    ])

    best = int(np.argmax(values))

    # Even the best class is too weak: refuse to guess.
    if values[best] < REJECT_TAU:
        return None

    return classes[best]


if __name__ == "__main__":
    # Raw probe points for the current DATASET (see pipeline_common).
    for q in PROBES_RAW:
        print(q, "->", classify(q))
