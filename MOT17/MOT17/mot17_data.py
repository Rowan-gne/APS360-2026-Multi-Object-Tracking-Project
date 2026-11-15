"""
mot17_data.py: the ONE definition of how every notebook loads the prepared MOT17 data.

WRITTEN BY "Exploratory data analysis.ipynb" (section 0). Don't edit this file by hand: edit that
cell and re-run the notebook, so all data preparation stays in one place.

Use it from any notebook in MOT17/MOT17/:
    import mot17_data
    print(mot17_data.data_version())                  # stops if the prepared data is missing or changed
    train_set = mot17_data.MOT17Frames("train")       # one item per frame (model input + ground truth)
    train_loader = mot17_data.make_loader(train_set, "train")
"""
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF

# ---- Paths, worked out from where this file is, so they're the same in every notebook ----
DATA_ROOT = Path(__file__).resolve().parent      # MOT17/MOT17/: holds train/ and test/ (read-only)
PROJECT_ROOT = DATA_ROOT.parents[1]
OUTPUT_DATA = PROJECT_ROOT / "outputs" / "data"  # the prepared files written by the EDA notebook
MANIFEST = OUTPUT_DATA / "manifest.json"         # MD5 of every prepared file (written last by the EDA notebook)

# ---- Settings every notebook shares ----
DETECTOR_COPY = "FRCNN"                          # the copy of each video used throughout the project
TARGET_WIDTH, TARGET_HEIGHT = 1920, 1088         # the model's input size (every frame is resized to this)
SPLITS = ["train", "val", "test"]


# ---------------------------------------------------------------- prepared files
def file_md5(path):
    """MD5 hash of a file's bytes: an unchanged file keeps the same hash."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def data_version():
    """Check every prepared file, and this file, are exactly what the EDA notebook wrote, and return a short ID for
    this version of the data (record it next to any results). Raises an error if a file is missing or changed."""
    if not MANIFEST.exists():
        raise FileNotFoundError(f"{MANIFEST} not found. Run 'Exploratory data analysis.ipynb' first (Restart -> Run all).")
    manifest = json.loads(MANIFEST.read_text())
    if file_md5(__file__) != manifest["code_md5"]:
        raise RuntimeError("mot17_data.py differs from the version the EDA notebook wrote. Make changes in the "
                           "notebook's section 0 and re-run 'Exploratory data analysis.ipynb'.")
    for name, md5 in manifest["files"].items():
        path = OUTPUT_DATA / name
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing. Re-run 'Exploratory data analysis.ipynb'.")
        if file_md5(path) != md5:
            raise RuntimeError(f"{path} changed after the EDA notebook wrote it. Re-run 'Exploratory data analysis.ipynb'.")
    return manifest["version"]


def read_prepared(name):
    """Read one prepared file from outputs/data/: "frames", "boxes", "videos", "detections" (tables) or
    "pixel_stats" (a dictionary). Split files are named like "splits/train_boxes"."""
    path = OUTPUT_DATA / (f"{name}.json" if name == "pixel_stats" else f"{name}.csv")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run 'Exploratory data analysis.ipynb' first.")
    return json.loads(path.read_text()) if path.suffix == ".json" else pd.read_csv(path)


# ---------------------------------------------------------------- frames
def open_original(image_path):
    """Open one frame (path relative to DATA_ROOT, as in frames.csv) as an RGB image at its ORIGINAL size.
    Used by DeepSORT, which works in original pixels."""
    with Image.open(DATA_ROOT / image_path) as img:
        return img.convert("RGB")


def open_resized(image_path):
    """Open one frame as an RGB image resized to 1920 x 1088 (the model's input size). Nothing is cropped or padded."""
    return open_original(image_path).resize((TARGET_WIDTH, TARGET_HEIGHT), Image.BILINEAR)


def load_frame(image_path, mean=None, std=None):
    """Turn one frame into the model's input: a float tensor [3, 1088, 1920].
    1. resize to 1920 x 1088
    2. scale pixels from 0-255 to 0-1 (what transforms.ToTensor() does)
    3. if mean and std are given, normalize each colour channel: (pixel - mean) / std
    """
    image = TF.to_tensor(open_resized(image_path))     # [3, 1088, 1920], values 0-1
    if mean is not None:
        image = TF.normalize(image, mean, std)
    return image


# ---------------------------------------------------------------- boxes
def boxes_to_pixels(boxes, image_width, image_height):
    """Boxes stored as (cx, cy, w, h) fractions of the ORIGINAL frame -> (left, top, width, height) in original pixels.
    `boxes` is a table with columns cx, cy, w, h; the sizes are numbers or columns of the same length.
    Rounded to 6 decimals: converting to fractions and back leaves noise of about 1e-13 px (e.g. 1288.9999999999995
    instead of 1289), and code that cuts boxes out with whole pixels (DeepSORT's appearance crops) would then be
    one pixel off. gt.txt values are whole numbers, so rounding gives back exactly the gt.txt values."""
    width = boxes["w"] * image_width
    height = boxes["h"] * image_height
    pixels = pd.DataFrame({"left": boxes["cx"] * image_width - width / 2,
                           "top": boxes["cy"] * image_height - height / 2,
                           "width": width, "height": height}, index=boxes.index)
    return pixels.round(6)


# ---------------------------------------------------------------- PyTorch datasets
class MOT17Frames(Dataset):
    """One item per frame of a split ("train", "val" or "test").

    Each item is a dictionary:
      image       float tensor [3, 1088, 1920]: resized, scaled to 0-1, normalized with pixel_stats.json
      orig_size   tensor [2] = (original width, original height), to turn outputs back into original pixels
      boxes       float tensor [N, 4] = (cx, cy, w, h) as 0-1 fractions of the original frame
      ids         int tensor [N] = person IDs (only meaningful within one video)
      video, frame  which frame this is
    """

    def __init__(self, split, normalize=True):
        self.data_version = data_version()            # stops here if the prepared data is missing or changed
        frames = read_prepared("frames")
        self.frames = frames[frames["split"] == split].sort_values(["video", "frame"]).reset_index(drop=True)
        boxes = read_prepared("boxes")
        boxes = boxes[boxes["split"] == split]
        # Group the boxes by frame once, so looking up a frame's boxes is fast.
        self.targets = {
            key: (torch.tensor(g[["cx", "cy", "w", "h"]].to_numpy(), dtype=torch.float32),
                  torch.tensor(g["id"].to_numpy(), dtype=torch.int64))
            for key, g in boxes.groupby(["video", "frame"])
        }
        stats = read_prepared("pixel_stats")
        self.mean = stats["mean"] if normalize else None
        self.std = stats["std"] if normalize else None

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        row = self.frames.iloc[index]
        key = (row["video"], int(row["frame"]))
        # A frame with no pedestrians gets empty tensors.
        boxes, ids = self.targets.get(key, (torch.zeros((0, 4)), torch.zeros(0, dtype=torch.int64)))
        return {
            "image": load_frame(row["image_path"], self.mean, self.std),
            "orig_size": torch.tensor([int(row["image_width"]), int(row["image_height"])]),
            "boxes": boxes,
            "ids": ids,
            "video": row["video"],
            "frame": int(row["frame"]),
        }


class MOT17FramePairs(Dataset):
    """One item per pair of consecutive frames (t-1, t) of the same video: returns (item at t-1, item at t).
    The IDs show which box in frame t is which person in frame t-1, which is what the transformer learns from."""

    def __init__(self, split, normalize=True):
        self.frames = MOT17Frames(split, normalize)
        self.data_version = self.frames.data_version
        table = self.frames.frames
        position = {(v, f): i for i, (v, f) in enumerate(zip(table["video"], table["frame"]))}
        # Pair each frame with the frame just before it, only if that frame exists (same video, frame number - 1).
        self.pairs = [(position[(v, f - 1)], i) for i, (v, f) in enumerate(zip(table["video"], table["frame"]))
                      if (v, f - 1) in position]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        previous, current = self.pairs[index]
        return self.frames[previous], self.frames[current]


def collate_frames(items):
    """Batch frames: images stack into [B, 3, 1088, 1920]; boxes and IDs stay as lists (frames have different numbers of boxes)."""
    return {
        "images": torch.stack([item["image"] for item in items]),
        "orig_sizes": torch.stack([item["orig_size"] for item in items]),
        "boxes": [item["boxes"] for item in items],
        "ids": [item["ids"] for item in items],
        "videos": [item["video"] for item in items],
        "frames": [item["frame"] for item in items],
    }


def collate_pairs(items):
    """Batch pairs: returns (batch of t-1 frames, batch of t frames). A pair's two frames stay together and in order."""
    return collate_frames([p[0] for p in items]), collate_frames([p[1] for p in items])


def make_loader(dataset, split, batch_size=2, num_workers=0):
    """Training data is shuffled; validation and test keep their order. num_workers=0 loads frames in the
    notebook's own process (simplest); a larger number loads them in parallel worker processes."""
    collate = collate_pairs if isinstance(dataset, MOT17FramePairs) else collate_frames
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), collate_fn=collate,
                      num_workers=num_workers)
