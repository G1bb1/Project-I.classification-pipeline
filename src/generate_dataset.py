'''
File:
generate_dataset.py

Description:
Part 0 -- write the dataset to disk, so every later stage works from the
same points and the file itself can be committed as evidence.

For "sim" this generates the synthetic shapes; for "sensor" it writes the
peer's unlabeled UV/IR readings (data_complete_no_labels.csv) in the
same format.

Note:
- Run from src/ (paths are relative to it). Run once per dataset.
- Writes datasets/<DATASET>/dataset.csv, with the column names as a
  "#" header line (np.loadtxt skips it automatically).

___________________________________________________
Author      Date                Comment
Gib M.      Sept. 11th, 2026    rev 1
Gib M.      Oct. 14th, 2026     rev 2 -- fixed load_dataset("") crash;
                                writes to datasets/<DATASET>/ with a
                                column-name header; rewrote header.
'''
import numpy as np
from pipeline_common import load_dataset, DATASET, DATASET_CSV, FEATURE_NAMES

X = load_dataset()  # synthetic or sensor, per DATASET in pipeline_common

np.savetxt(DATASET_CSV, X, delimiter=",", fmt="%.6f",
           header=",".join(FEATURE_NAMES))
print(f"wrote {DATASET_CSV} ({DATASET}): "
      f"{X.shape[0]} points, {X.shape[1]} features")
