# classification-pipeline

A step-by-step pipeline that goes from **unlabeled 2-D data** to a **tiny
lookup-table classifier** that fits in a few hundred bytes of flash, as an R&D
step toward running it on an ESP32 / ATtiny prototype.

1. **Discover classes** (Part 1): DBSCAN finds the clusters (k-means is shown
   for contrast).
2. **Model each class** (Part 2): a Gaussian KDE per class. The bandwidth is
   chosen on a validation split.
3. **Store as bytes** (Part 3): each class density is sampled on a 16x16 grid
   and stored as `uint8`, so the whole model is `classes x 256` bytes plus the
   scaling bounds.
4. **Classify** (Part 4): transform and scale the query, look up its cell, pick
   the class with the largest value. If even that value is below
   `tau = 0.12`, answer **unknown**.
5. **Evaluate** (Part 4): on the held-out test set, check how often the byte
   lookup agrees with the full continuous KDE.

## Two datasets

| `DATASET` | Data | Notes |
|-----------|------|-------|
| `sim` | Synthetic crescents + ring + noise (instructor's R&D setup) | The original results |
| `sensor` | Real UV/IR readings under shade, LED, fluorescent and sun light, collected by a peer with physical components | Uses a `log1p` feature transform (see below) |

**Why the sensor data needs a log transform:** IR ranges from about 130 (shade)
to about 30,000 (sun), and UV reads 0 for every indoor source. With plain
linear min-max scaling, shade, LED and fluorescent all land within 0.01 of each
other, in the same grid cell, and DBSCAN merges them into one class. After
`log1p`, the four light conditions form four separate clusters.

Current sensor results: 4 clusters with no noise, one per light condition, and
the uint8 lookup matches the continuous KDE on 100% of the 480 test points.

## Requirements

Python 3 with `numpy`, `scikit-learn` and `matplotlib`:

```
pip install numpy scikit-learn matplotlib
```

## Usage

Choose the data in `src/pipeline_common.py`:

```python
DATASET = "sensor"   # or "sim"
```

Then, from `src/`, run everything at once or any stage on its own:

```
cd src
python run_all.py            # every stage, in order
python cluster_dbscan.py     # or any single stage
```

| Part | Script | Output (in `outputs/<DATASET>/`) |
|------|--------|-----------------------------------|
| 0 | `generate_dataset.py` | `datasets/<DATASET>/dataset.csv` |
| 1 | `cluster_dbscan.py` | `labels.csv`, `figures/clustered_space.pdf` |
| 2 | `bandwidth_sweep.py` | prints the best KDE bandwidth |
| 2 | `kde_class_contours.py` | `surfaces.npz`, `figures/class_surfaces_continuous.pdf` |
| 2b | `kde_total_surface.py` | `figures/total_distribution.pdf` |
| 3 | `discretize_surfaces.py` | storage-cost table, `figures/class_surfaces_discretized.pdf` |
| 4 | `classify_skeleton.py` | hand-written lookup (exercise) |
| 4 | `classify_point.py` | `figures/decision_map.pdf` |
| 4 | `evaluate_test.py` | test-set agreement and confusion table |

The stages must run in this order the first time: later ones read the
`labels.csv` and `surfaces.npz` written by earlier ones.

### Output files

- `labels.csv`: one DBSCAN cluster per training point (`-1` = noise). It starts
  with a `#` line recording the DBSCAN settings.
- `surfaces.npz`: one peak-normalised map per class, with keys `c0`, `c1`, ...
- `dataset.csv`: the raw points, with the column names on a `#` header line.

`np.loadtxt` skips the `#` lines automatically.

## Project layout

```
src/                  pipeline stages, pipeline_common.py (shared settings), run_all.py
src_original/         backup copy of src/ before the sensor-data edits
datasets/
  sim/                dataset.csv (synthetic points)
  sensor/             peer's raw readings:
                        data_<condition>.csv         one light condition each, labelled
                        data_complete.csv            all four conditions, labelled
                        data_complete_no_labels.csv  same readings, labels removed --
                                                     the pipeline's input
                        dataset.csv                  written by Part 0 from the above
outputs/
  sim/                labels.csv, surfaces.npz, figures/*.pdf
  sensor/             labels.csv, surfaces.npz, figures/*.pdf
paper/                LaTeX report
```

## How this was built with Claude Code

This project was adapted together with Claude Code (Anthropic's AI coding
assistant) working directly in the repository:

1. **Understanding the code.** I asked Claude to explain the project. It read
   every script, summarised the five-part pipeline, wrote a `CLAUDE.md`
   (project notes Claude reads at the start of each session), and pointed out
   problems: a crash in `generate_dataset.py`, the pipeline ignoring
   `dataset.csv`, every script having to be run separately, and `BANDWIDTH`
   being duplicated in three files.
2. **Explaining the goal.** I explained that `src/` is my instructor's
   simulated R&D setup for testing DBSCAN, that I now have real UV/IR sensor
   data from a peer, and that the code was built entirely around the simulated
   data.
3. **Checking the data before coding.** Claude analysed the sensor CSVs first
   and found the real reason the pipeline fails on them: with linear scaling,
   three of the four light conditions fall into one grid cell. It compared
   linear and `log1p` scaling under DBSCAN before changing any code.
4. **Adapting the instructor's code.** I wanted the work to stay inside `src/`
   with as few changes as possible, plus a backup (`src_original/`). Claude
   added a `DATASET` switch, the log transform and per-dataset paths to
   `pipeline_common.py`. It made each stage read its paths and bandwidth from
   there, and added `run_all.py` to run every stage with one command.
5. **Organising files and comments.** I asked for a cleaner layout. Claude
   moved inputs into `datasets/<DATASET>/` and results into
   `outputs/<DATASET>/`, and added self-describing `#` header lines to the CSV
   outputs. It merged the instructor's docstrings into my file headers and
   rewrote unevenly commented files so every stage explains what it does.
6. **Verification.** After each change, Claude re-ran the pipeline. It checked
   that `sim` mode still reproduces the original results exactly (same
   labels, same class maps, same dataset), and that `sensor` mode finds the
   four light conditions.

I reviewed the changes and decided what to keep and commit.

## Author

Gib M.
