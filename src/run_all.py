'''
File:
run_all.py

Description:
Runs every stage of the pipeline in order (Parts 0 to 4), so the whole
thing can be seen working with one command instead of running each file
by hand. Each stage file is executed exactly as if it were run on its
own, so the stages stay separate and can still be run individually.

Which data it runs on is set by DATASET in pipeline_common.py.

Note:
- Run from src/:  python run_all.py
- Results go to outputs/<DATASET>/.

___________________________________________________
Author      Date                Comment
Gib M.      Oct. 14th, 2026     rev 1
Gib M.      Oct. 15th, 2026     rev 2 -- added generate_model_header.py.
'''
import runpy

from pipeline_common import DATASET

STAGES = [
    "generate_dataset.py",     # Part 0  write the dataset
    "cluster_dbscan.py",       # Part 1  discover classes
    "bandwidth_sweep.py",      # Part 2  tune h on validation
    "kde_class_contours.py",   # Part 2  per-class maps -> surfaces.npz
    "kde_total_surface.py",    # Part 2b 3-D decision surface
    "discretize_surfaces.py",  # Part 3  bytes and flash cost
    "generate_model_header.py",  # Part 3  model.h for the firmware
    "classify_skeleton.py",    # Part 4  hand-written lookup
    "classify_point.py",       # Part 4  the online path
    "evaluate_test.py",        # Part 4  spend the test set once
]

print(f"DATASET = {DATASET!r}")
for stage in STAGES:
    print(f"\n=== {stage} " + "=" * (50 - len(stage)))
    # run_name="__main__" so `if __name__ == "__main__":` blocks run too.
    runpy.run_path(stage, run_name="__main__")
