# classification-pipeline: On-Device Light Classifier (v2.0)

A staged machine-learning pipeline that turns **unlabelled 2-D data** into a tiny classifier a microcontroller can run. DBSCAN discovers the classes, a Gaussian kernel density estimate (KDE) maps each one, and each map is stored as **one byte per cell** of a 16×16 grid. Classifying a reading then takes one byte read per class and a reject threshold, which answers **UNKNOWN** outside every learned region.

| Release | Stage | Data |
|---|---|---|
| v1.0 (git tag `v1.0`) | Research (Lab 3) | Synthetic crescents, ring and noise, where the right answer is known, to validate the pipeline |
| **v2.0** | **Deployment (Assignment 1)** | **Real UV/IR light-sensor readings; the lookup runs live on an ESP32** |

v2.0 retrains the same pipeline on real readings of four lighting conditions (shade, LED, fluorescent, sun) and runs the classifier on an ESP32. The `sim` dataset is still in the repository, so v1.0's results can be reproduced with the same code.

## Pipeline

**Offline** (runs once, on a workstation):

1. **Transform and scale.** Sensor features go through `log1p`, then min-max scaling with the training range.
2. **Split.** 60 / 20 / 20 into training, validation and test (fixed seed). Training builds the model, validation picks the bandwidth, and the test set is used once at the end.
3. **Cluster.** DBSCAN (`eps = 0.06`, `min_samples = 8`) finds the classes in the training data.
4. **Model.** A Gaussian KDE is fitted to each class and normalised to its own peak. The bandwidth is chosen on the validation split.
5. **Store.** Each map is sampled on a 16×16 grid and stored as `uint8`.
6. **Export.** `generate_model_header.py` writes everything the device needs to `model.h`.

**Online** (on the ESP32, about 10 Hz):

1. Read UV and IR, and apply the same EMA filter used during collection.
2. Apply `log1p`, then scale with the stored bounds.
3. Find the grid cell.
4. Read one byte per class and keep the largest.
5. Answer UNKNOWN if that byte is below `tau = 0.12` (31 of 255).

## Hardware

| Part | Connection |
|---|---|
| ESP32 DevKit V1 | — |
| GUVA-S12SD UV sensor (Octopus board, analog) | OUT → GPIO34 (ADC1), V+ → 3V3, GND → GND |
| Waveshare TSL25911 IR/light sensor (I²C, 0x29) | SDA → GPIO21, SCL → GPIO22, VCC → 3V3, GND → GND |
| Optional status LEDs (not fitted on the prototype) | GPIO32 shade, GPIO33 LED, GPIO25 fluorescent, GPIO26 sun, GPIO27 UNKNOWN; each through a 220–330 Ω resistor to GND |

**Why the TSL25911 and not a photodiode:** the assignment's EPD-712 photodiode with a 10 kΩ divider gave only millivolt-level signals under indoor light, in the noisiest part of the ESP32's ADC. On the instructor's recommendation, IR comes from the TSL25911 instead, which amplifies and digitises on-chip. The Adafruit TSL2591 library drives it: it's the same chip, at the same I²C address (0x29) and chip ID (0x50).

## Why the sensor data needs `log1p`

IR ranges from about 130 (shade) to about 30,000 (sun), and UV reads 0 for every indoor source. With plain linear min-max scaling, shade, LED and fluorescent all fall into **grid row 0**, and no byte table can separate them; DBSCAN finds only 2 clusters. After `log1p`, the indoor classes spread across rows 0–3, and DBSCAN finds the four conditions at every `eps` from 0.003 to 0.06. The device applies the same transform, because the bounds in `model.h` are in log units. `figures/log_vs_linear.pdf` shows both cases.

## Results

| | |
|---|---|
| DBSCAN vs. true labels (1440 training readings) | **4 / 4 conditions recovered exactly**, 0 noise |
| Bandwidth (validation log-likelihood) | h = 0.03 |
| Model size | 4 × 16² + 16 = **1040 bytes**, in flash (DROM), 0 B SRAM |
| Stored vs. continuous model, desktop test split | **100%** (480 / 480) |
| Grid cells answering UNKNOWN | 251 of 256 (98%) |
| Time per query on the ESP32 | 4.28 µs benchmark, 12.65 µs median in the live loop |
| Fresh held-out batch on the device (440 shade, 464 LED, 440 fluorescent) | **1344 / 1344 correct**, 100% agreement with the desktop model |
| Sketch size / heap in use | 332,800 B of 4 MB flash / 43,572 B |

**Known limitations** (full list in the paper):
- **Sun was not tested on the device.**
- **The EMA filter (α = 0.2) doesn't remove long glitches.** It keeps class levels, but doesn't suppress hand occlusions longer than about 0.3 s. On a 222-sample fluorescent capture with occlusions, 44 samples come out wrong with the EMA; holding the output for 5 samples would cut that to 11 (tested offline, not on the device).
- **Single-session data.** All readings come from one place and one sitting, at a fixed distance.
- **Readings below the training range clip to the edge cell.** So a covered sensor reads "shade", never UNKNOWN.

## Requirements

Python 3 with `numpy`, `scikit-learn` and `matplotlib`:

```
pip install numpy scikit-learn matplotlib
```

Firmware:
- **Board:** ESP32 Dev Module.
- **Libraries:** Adafruit TSL2591 and Adafruit Unified Sensor.

## Usage

Choose the data in `src/pipeline_common.py`:

```python
DATASET = "sensor"   # or "sim" to reproduce v1.0
```

Then, from `src/`, run every stage at once, or any stage on its own:

```
cd src
python run_all.py            # Parts 0 to 4, in order
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
| 3 | `generate_model_header.py` | `model.h` (in sensor mode, also copied to `src/classify/`) |
| 4 | `classify_skeleton.py` | hand-written lookup (exercise) |
| 4 | `classify_point.py` | `figures/decision_map.pdf` |
| 4 | `evaluate_test.py` | test-set agreement and confusion table |

The stages must run in this order the first time, because later ones read the `labels.csv` and `surfaces.npz` written by earlier ones. The bandwidth printed by `bandwidth_sweep.py` is copied into `BANDWIDTH` in `pipeline_common.py` by hand.

Two report figures are made separately, in sensor mode only, because they also need the labelled data or a capture from the device. `run_all.py` doesn't run them:

| Script | Reads | Output |
|--------|-------|--------|
| `plot_log_vs_linear.py` | `data_complete.csv` (labels used for grouping only) | `figures/log_vs_linear.pdf` |
| `plot_raw_vs_filtered.py` | `data_raw_vs_filtered.csv` (ESP32 `R` capture) | `figures/raw_vs_filtered.pdf` and the filter error counts |

## On the ESP32

`src/classify/classify.ino` runs the model from `src/classify/model.h`. Open the `src/classify/` folder in the Arduino IDE, or compile from the repository root with:

```
arduino-cli compile --fqbn esp32:esp32:esp32 src/classify
```

The sketch reports each reading and its class over serial at 115200 baud, and also drives the optional LEDs. At start-up, it prints the model's size, where it sits in memory, the free heap and stack, and the average time per query. If the TSL2591 isn't found, it prints an error and blinks all five LEDs.

It reads the sensors exactly as the collection sketch (`data_collection_UPDATED.ino`, see Credit) does: same TSL2591 gain (medium) and integration time (100 ms), same EMA filter (α = 0.2), same 10 Hz rate. **The model only works on readings taken the same way.**

Serial commands:

| Send | Effect | Output rows |
|---|---|---|
| `L:<label>` (e.g. `L:shade`) | Record a held-out batch under a known light | `uv,ir,predicted,label,us` |
| `L:` | Stop recording | — |
| `R` | Toggle raw-vs-filtered streaming (filter evidence) | `raw_uv,raw_ir,filt_uv,filt_ir,predicted` |

The recorded rows are saved as `datasets/sensor/data_esp_inference.csv` (held-out batches) and `datasets/sensor/data_raw_vs_filtered.csv` (`R` capture).

## Project layout

```
src/                  pipeline stages, pipeline_common.py (shared settings), run_all.py
  classify/           ESP32 inference sketch: classify.ino + model.h (generated)
datasets/
  sim/                dataset.csv                  synthetic points (Part 0)
  sensor/             data_<condition>.csv         one light condition each, labelled
                      data_complete.csv            all four conditions, labelled
                      data_complete_no_labels.csv  same readings, labels removed --
                                                   the pipeline's input
                      dataset.csv                  written by Part 0 from the above
                      data_esp_inference.csv       held-out batches recorded on the ESP32
                      data_raw_vs_filtered.csv     raw and filtered capture (R command)
outputs/
  sim/                labels.csv, surfaces.npz, figures/*.pdf
  sensor/             labels.csv, surfaces.npz, model.h, board_photo.jpg,
                      figures/*.pdf (wiring.pdf included)
paper/
  paper.tex           v2.0 proof-of-concept report (ESP32); the v1.0 research
                      paper is at the v1.0 tag
```

`paper.tex` takes its figures from `outputs/`, so build it from inside `paper/`.

## Proof of concept

- **Video of live classification:** https://youtu.be/epPjHrj3CWY
- **Prototype photo:** `outputs/sensor/board_photo.jpg`
- **Wiring diagram:** `outputs/sensor/figures/wiring.pdf`

## Credit

The data-collection firmware (`data_collection_UPDATED.ino`, including its EMA filter and sensor settings) and the labelled sensor dataset come from Gianluca D'Altorio's repository: https://github.com/GianlucaDAltorio/data-collection

## How this was built with Claude Code

This project was adapted together with Claude Code (Anthropic's AI coding assistant) working directly in the repository:

1. **Explaining the goal.** I explained that `src/` is my instructor's simulated R&D setup for testing DBSCAN, that I now have real UV/IR sensor data from a peer, and that the code was built entirely around the simulated data.
2. **Checking the data before coding.** Claude analysed the sensor CSVs first and found the real reason the pipeline fails on them: with linear scaling, three of the four light conditions fall into one grid cell. It compared linear and `log1p` scaling under DBSCAN before changing any code.
3. **Adapting the instructor's code.** I wanted the work to stay inside `src/` with as few changes as possible. Claude added a `DATASET` switch, the log transform and per-dataset paths to `pipeline_common.py`, and made each stage read its settings from there. It also added `run_all.py`, split inputs and results into `datasets/` and `outputs/`, and brought every file's comments into my header style.
4. **Verification.** After each change, Claude re-ran the pipeline. It checked that `sim` mode still reproduces the original results exactly (same labels, same class maps, same dataset), and that `sensor` mode finds the four light conditions.
5. **Deploying and testing on the ESP32.** Claude wrote `generate_model_header.py` in my instructor's `model.h` layout and an inference sketch built on my peer's collection code. It checked that the header gives the same answers as the Python lookup. I ran the sketch on the board and recorded fresh held-out batches and a raw-vs-filtered capture. Claude scored them against the desktop model, measured the filter's effect, and produced the figures and wiring diagram for the proof-of-concept report.

I reviewed the changes and decided what to keep and commit.

## Author

Gib Macarine.
