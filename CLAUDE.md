# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

A staged ML pipeline that turns unlabeled 2-D data into a tiny classifier for a
microcontroller (later ESP32 PoC / ATtiny prototype, out of scope here):
DBSCAN discovers classes, a Gaussian KDE models each class, and the model ships
as one `uint8` lookup table per class (`GRID_N x GRID_N` bytes) with a reject
threshold `REJECT_TAU` for "unknown".

`src/` is the instructor's simulated R&D setup, adapted by the user (Gib M.)
for real sensor data. The teacher may want the work kept inside `src/`.

Two datasets, chosen by `DATASET` in `src/pipeline_common.py`:
- `sim`: synthetic crescents + ring + noise (original setup).
- `sensor`: a peer's UV/IR readings (shade, LED, fluorescent, sun). Needs the
  `log1p` transform, because IR spans about 130 to 30,000 and linear min-max
  scaling merges the three indoor classes into one grid cell.

## Layout

- `src/pipeline_common.py`: `DATASET` switch, paths (`DATA_DIR`, `OUT_DIR`,
  `LABELS_CSV`, `SURFACES_NPZ`, `FIG_DIR`), per-dataset settings
  (`BANDWIDTH`, `FEATURE_NAMES`, `AXIS_LABELS`, `PROBES_RAW`, `UNKNOWN_RAW`),
  shared constants and helpers (transform/scale/split, grid, KDE, quantize).
- `src/<stage>.py`: one script per Part, run top-to-bottom. `run_all.py` runs
  them all via `runpy`. All paths are relative to `src/`, so run from there.
- `src_original/`: backup of `src/` before any sensor edits. Do not modify.
- `datasets/<DATASET>/`: inputs. `outputs/<DATASET>/`: `labels.csv`,
  `surfaces.npz`, `figures/*.pdf`.
- Sensor input is `datasets/sensor/data_complete_no_labels.csv`, the peer's
  file, kept so the pipeline never sees labels. Formerly
  `dataset_complete_unlabeled.csv`; the user wants it kept. `data_complete.csv`
  is the same readings plus a label column (used only by `pipeline/` to score
  accuracy).
- `paper/paper.tex`: report (empty so far). `report.tex` and `reference.py`
  are style samples from an earlier RC-circuit report, not part of the
  pipeline.

### Not committed (the user keeps these local; do not mention them in README)

- `pipeline/` + `main.py`: an alternative standalone refactor (dataset
  `Profile`s, `GridClassifier`, true-label accuracy, `model.npz`). Writes to
  `pipeline_outputs/<profile>/`. Gives the same numbers as `src/`.
- `pipeline_outputs/`.

## Running

The venv in `.venv` has no packages installed. Use the system Python 3.13
(`C:\Users\gangs\AppData\Local\Programs\Python\Python313\python.exe`), which
has numpy, scikit-learn and matplotlib. Set `MPLBACKEND=Agg` when running
headless.

```
cd src && python run_all.py
```

There are no unit tests. The regression check is to run `sim` mode on a
**scratch copy** of `src/` and compare `labels.csv` / `surfaces.npz` /
`dataset.csv` numerically (`np.loadtxt`, `np.load`) against `outputs/sim/` and
`datasets/sim/`. They must match exactly. Rerunning in place only rewrites
PDF timestamps.

## Conventions

- **Data discipline**: train builds everything (clustering, KDE, scaling
  bounds); validation only picks the bandwidth (then frozen by hand in
  `pipeline_common.BANDWIDTH`); the test set is used once in
  `evaluate_test.py`.
- Raw query points go through `transform_features` and then
  `scale_features(..., bounds)`.
- CSV outputs carry a `#` header line (column names / DBSCAN settings);
  `np.loadtxt` skips it.
- Figures are grayscale, transparent PDF, `bbox_inches="tight"`.
- **File header** (the user's style, the only module docstring):
  ```
  '''
  File:
  name.py

  Description:
  Part N -- one-line summary.
  <prose>

  Note:
  - run/read/write notes

  ___________________________________________________
  Author      Date                Comment
  Gib M.      Sept. 11th, 2026    rev 1
  Gib M.      Oct. 14th, 2026     rev 2 -- what changed
  '''
  ```
  Add a new `rev N` row for each round of changes. `[DATE]` is a
  placeholder the user fills in. Comments should be balanced: a section
  comment per block, and explain *why*, not just what.

## Known quirks

- Sensor: validation picks the smallest candidate bandwidth (0.03). Going lower
  is pointless because a grid cell is 1/16 = 0.0625 wide. The classes occupy
  only a few cells, so about 98% of the grid is in the reject region.
- In sensor mode, indoor points sit on the grid's edge column (UV = 0);
  `classify_point.py` falls back to edge points when picking its query.
- `src/` prints cluster numbers only; for sensor they come out as 0 = shade,
  1 = sun, 2 = fluorescent, 3 = LED.
