# APS360-2026-Multi-Object-Tracking-Project

Deep learning model for multi-object tracking (MOT) on the MOT17 dataset (APS360, University of Toronto). Baseline: DeepSORT.

## Setup

The project uses its own Python environment, `.venv` (Python 3.14, CUDA build of PyTorch). Create it once from the project folder:

```
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m ipykernel install --user --name mot17-venv --display-name "MOT17 (.venv)"
```

VS Code uses `.venv` automatically (`.vscode/settings.json`), and the notebooks use the "MOT17 (.venv)" kernel.
In a terminal outside VS Code, run `.venv\Scripts\activate` first, so `python` and `pip` are the project's.
Models run on the GPU by default.

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

## Notebooks (in `MOT17/MOT17/`, run in this order with Restart kernel -> Run all)

1. **`Exploratory data analysis.ipynb`**: cleans the ground truth (pedestrians with `class = 1`, `use = 1`), shows
   statistics before and after cleaning, splits by whole video (train = MOT17-02, 04, 05, 09, 11; val = MOT17-10;
   test = MOT17-13), computes the training pixel mean/std, and shows one sample exactly as a model receives it.
   It writes `outputs/data/` (`videos.csv`, `frames.csv`, `boxes.csv`, `pixel_stats.json`, `stats_before_after.csv`,
   `sample_item.pt`) and `mot17_data.py`, the shared code every notebook uses to load frames (resized to 1920 x 1088,
   normalized), convert boxes and build the PyTorch datasets. Edit `mot17_data.py` in the notebook's section 0, not by hand.
2. **`DeepSORT baseline.ipynb`**: our own Faster R-CNN (torchvision ResNet-50-FPN, COCO weights) detects people,
   DeepSORT tracks them, TrackEval scores MOT17-10 and MOT17-13. Results in `outputs/deepsort/`; the detector's boxes are
   saved in `outputs/deepsort/detections/` and reused (`RERUN_DETECTOR = True` forces a fresh run).
3. **`Model.ipynb`**: Rowan's model (CNN + fully connected layers + transformer), trained and compared with the
   baseline on MOT17-13. Results in `outputs/model/`.

```python
import mot17_data
train_pairs = mot17_data.MOT17FramePairs("train")      # or MOT17Frames for single frames
train_loader = mot17_data.make_loader(train_pairs, shuffle=True, batch_size=4)
```
