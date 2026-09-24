"""Part 4 skeleton -- YOU write the lookup.

The tables, the grid, and the scaling are done.
Fill in the two TODOs, then check against classify_point.py.
"""
import numpy as np
from pipeline_common import (load_split_scaled, scale_features, quantize,
                             dequantize, GRID_N, REJECT_TAU)

bounds = load_split_scaled()["bounds"]  # training bounds ship with the model
surfaces = np.load("surfaces.npz")
classes = sorted(int(k[1:]) for k in surfaces.files)

# The shipped model. Your code may read only these and `bounds`.
tables = {c: quantize(surfaces[f"c{c}"]) for c in classes}


def classify(raw_point):
    """Class label for a RAW point, or None for "unknown"."""
    point, _ = scale_features(np.asarray(raw_point), bounds)

    # turn `point` into a grid cell.
    # col = int(point[0] * GRID_N), row likewise with point[1],
    # then clip both into [0, GRID_N - 1].
    col = int(point[0] * GRID_N)
    row = int(point[1] * GRID_N)

    col = np.clip(col, 0, GRID_N - 1)
    row = np.clip(row, 0, GRID_N - 1)

    # read tables[c][row, col] for each class, convert with
    # dequantize(), and return the class with the largest value --
    # unless that value is below REJECT_TAU, then return None.
    values = np.array([
        dequantize(tables[c][row, col])
        for c in classes
    ])

    best = int(np.argmax(values))

    if values[best] < REJECT_TAU:
        return None

    return classes[best]


if __name__ == "__main__":
    # One raw point on each shape, then the ring's hollow centre (unknown).
    for q in ([0.0, -1.8], [0.0, 5.0], [6.8, 1.6], [4.8, 1.6]):
        print(q, "->", classify(q))