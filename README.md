# APS360-2026-Multi-Object-Tracking-Project

Deep learning model for multi-object tracking (MOT) on the MOT17 dataset (APS360, University of Toronto). Baseline: DeepSORT.

## Setup

```
pip install -r requirements.txt
```

Models run on the GPU by default (each notebook prints its `DEVICE` and warns if it falls back to the CPU).
Install the CUDA build of PyTorch first; the commands are at the top of `requirements.txt`.

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
   - `outputs/data/frames.csv`, `boxes.csv`: the cleaned, model-ready dataset (a `split` column marks train / val / test)
   - `outputs/data/splits/`: the same rows split into `train_`, `val_` and `test_` `frames.csv` / `boxes.csv`.
     Split by whole video: train = MOT17-02, 04, 05, 09, 11; val = MOT17-10; test = MOT17-13
   - `outputs/data/split_mix.csv`: % of small/large and hidden/visible people in each split
   - `outputs/data/videos.csv`: length, frame size and split of each video
   - `outputs/data/detections.csv`: MOT17's supplied Faster R-CNN detections (DeepSORT's input), values unchanged
   - `outputs/data/stats.csv`, `stats_before_after.csv`, `data_report.md`: statistics (after, and before vs after cleaning) and the data report
   - `outputs/data/pixel_stats.json`: per-channel mean/std of the training frames
   - `outputs/data/problems.csv`: every problem found in the data
   - `outputs/data/sample/`: one training item exactly as a model receives it (`sample_item.pt`) and its ground truth
   - `outputs/data/manifest.json`: MD5 of every prepared file and a short data version ID
   - `outputs/figures/`: sample frames with their ground-truth boxes drawn on, and the before/after cleaning figure

## How the notebooks share the data

All data preparation happens in `Exploratory data analysis.ipynb`. It also writes `MOT17/MOT17/mot17_data.py`,
the one shared definition of how frames are loaded (resize to 1920 x 1088, normalize), how boxes convert to pixels,
and the PyTorch datasets. Other notebooks in `MOT17/MOT17/` only read the prepared files, through that module:

```python
import mot17_data
print(mot17_data.data_version())                 # stops if the prepared data is missing or changed
train_set = mot17_data.MOT17Frames("train")      # or MOT17FramePairs for consecutive frames
train_loader = mot17_data.make_loader(train_set, "train")
```

Edit data loading in the notebook's section 0, not in `mot17_data.py` (it is overwritten on every run).
Run order: `Exploratory data analysis.ipynb` first, then `DeepSORT baseline.ipynb` or the model notebook.

## DeepSORT baseline

`MOT17/MOT17/DeepSORT baseline.ipynb` runs the full baseline on MOT17-10 (validation) and MOT17-13 (test):
our own Faster R-CNN (torchvision ResNet-50-FPN, COCO-pretrained, not fine-tuned) detects people on the raw frames,
DeepSORT tracks them, and TrackEval scores them. MOT17's supplied detections (`det/det.txt`) are not used.
The first run downloads the Faster R-CNN weights (about 160 MB) into PyTorch's cache folder.

Results go to `outputs/deepsort/` (git-ignored):

- `results.csv`, `deepsort_report.md`: HOTA, MOTA, IDF1, ID switches, false positives/negatives, speeds, detector recall
- `detections/`: the detector's boxes per video, reused on the next run if the weights, threshold and torchvision
  version match (set `RERUN_DETECTOR = True` to force a fresh run)
- `trackers/`, `eval/`: tracker output files and TrackEval's output; `figures/`: frames with track IDs drawn

The earlier run that used MOT17's supplied detections is kept in `outputs/archive/deepsort_supplied_detections/`.
