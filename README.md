# Density Lookup Classifier: Research (v1.0)

A staged machine-learning pipeline that turns **unlabelled 2-D data** into a tiny classifier a microcontroller can run. DBSCAN discovers the classes, a Gaussian kernel density estimate (KDE) maps each one, and each map is stored as **one byte per cell** of a 16×16 grid. Classifying a point is then a few byte reads plus a reject threshold that answers **UNKNOWN** outside every learned region.

This release is the **research stage**. The pipeline is built and validated on **synthetic data**, where the correct answer is known, before any real sensor is involved.

## Pipeline

**Offline** (runs once, on a workstation):

1. **Scale.** Min-max scale both features onto [0, 1] using the training range.
2. **Cluster.** DBSCAN (`eps = 0.06`, `min_samples = 8`) finds the classes; sparse points are labelled noise.
3. **Model.** A Gaussian KDE is fitted to each class and normalised to its own peak. The bandwidth is chosen on the validation split.
4. **Store.** Each map is sampled on a 16×16 grid and stored as `uint8`.

**Online** (what a device would run):

1. Scale the query with the training bounds.
2. Find its grid cell.
3. Read one byte per class and keep the largest.
4. Answer UNKNOWN if that byte is below `tau = 0.12`.

## Dataset

Generated, so that every run is reproducible (seed `20250903`):

- Two crescents opening in opposite directions (340 and 300 points)
- A ring (300 points)
- 90 uniform background-noise points

That makes 1030 points in total, split once into train, validation and test sets (60/20/20, so 618/206/206). The shapes are chosen on purpose: a crescent's centre sits off the crescent and a ring's centre sits in its hole, so no centre-based method can describe them.

## Results

| | |
|---|---|
| Classes found by DBSCAN | 3 (209 / 198 / 178 points), 33 noise |
| Bandwidth (validation log-likelihood) | h = 0.05, inside the Silverman bracket 0.034–0.068 |
| Model size | 3 × 16² = **768 bytes** + 16 bytes of scaling bounds |
| Worst-case `uint8` error | 0.195% of peak |
| Stored vs. continuous model, held-out test | **100%** agreement (206 / 206) |
| Ring centre (surrounded by data, contains none) | **UNKNOWN** |
| Reject region | 127 of 256 cells (49.6%) |

## Repository layout

```
src/                  one script per stage (run from here)
  pipeline_common.py  shared settings: dataset switch, paths, scaling, grid, KDE, byte storage
  generate_dataset.py     Part 0  write the dataset to disk
  dataset.csv             runs every stage in order
  cluster_dbscan.py       Part 1  discover classes (with a k-means comparison)
  bandwidth_sweep.py      Part 2  pick the KDE bandwidth on validation
  kde_class_contours.py   Part 2  per-class maps -> surfaces.npz
  kde_total_surface.py    Part 2b 3-D decision surface
  discretize_surfaces.py  Part 3  byte storage and flash cost
  classify_skeleton.py    Part 4  hand-written lookup
  classify_point.py       Part 4  the online path, nearest vs bilinear
  evaluate_test.py        Part 4  spend the test set once
figures/generated/          labels.csv, surfaces.npz, figures/*.pdf
paper/                LaTeX research paper (IEEE conference format)
```

## Running

Requires Python 3 with `numpy`, `scikit-learn` and `matplotlib`.

1. In `src/pipeline_common.py`, make sure the dataset switch is set to `DATASET = "sim"`.
2. Run the whole pipeline in order from `src/` (all paths are relative to it).

Each stage can also be run on its own, in the order listed above. Results go to `figures/generated/`. To build the paper, compile the `.tex` file in `paper/`; it reads its figures from `../figures/generated/`.

## Data discipline

- **Training data** builds everything: the clustering, the KDE and the scaling bounds.
- **Validation data** is used only to pick the bandwidth, which is then frozen by hand in `pipeline_common.py`.
- **Test data** is used **once**, in `evaluate_test.py`, on the frozen design, so the number it reports was not used to make any choice.

---
Gib Macarine, Vanier College, Computer Engineering Technology. Project I (247-519-VA).
