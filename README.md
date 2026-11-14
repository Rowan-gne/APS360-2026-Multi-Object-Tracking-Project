# APS360-2026-Multi-Object-Tracking-Project

Deep learning model for multi-object tracking (MOT) on the MOT17 dataset (APS360, University of Toronto). Baseline: DeepSORT.

## Setup

```
pip install -r requirements.txt
```

## Where to put the MOT17 data

Download MOT17 from https://motchallenge.net/data/MOT17/ and unzip it so the folders sit here:

```
APS360-2026-Multi-Object-Tracking-Project/
  MOT17/MOT17/
    Exploratory data analysis.ipynb
    train/MOT17-02-FRCNN/ ...   (21 folders: 7 videos x 3 detector copies)
    test/MOT17-01-FRCNN/ ...    (21 folders)
```

The data (about 5.7 GB) is git-ignored and is never modified by the code.

## Phase 1: data cleaning and processing

1. Open `MOT17/MOT17/Exploratory data analysis.ipynb` and select a Python 3 kernel.
2. Restart the kernel and run all cells (takes roughly 5-10 minutes; the slow parts are hashing every frame and computing the pixel statistics).
3. The results are written to `outputs/` (git-ignored):
   - `outputs/data/frames.csv`, `boxes.csv`: the cleaned, model-ready dataset
   - `outputs/data/stats.csv`, `data_report.md`: statistics and the data report
   - `outputs/data/pixel_stats.json`: per-channel mean/std of the training frames
   - `outputs/data/problems.csv`: every problem found in the data
   - `outputs/figures/`: sample frames with their ground-truth boxes drawn on
