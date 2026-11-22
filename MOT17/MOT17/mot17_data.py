"""Shared MOT17 data code. WRITTEN BY 'Exploratory data analysis.ipynb': edit it there, then re-run that notebook."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF

DATA_ROOT = Path(__file__).resolve().parent        # MOT17/MOT17 (train/ and test/ are read-only)
PROJECT_ROOT = DATA_ROOT.parents[1]
PREPARED = PROJECT_ROOT / "outputs" / "data"        # written by the EDA notebook
WIDTH, HEIGHT = 1920, 1088                          # the model's input size


def read(name):
    """A prepared table: "videos", "frames" or "boxes"."""
    return pd.read_csv(PREPARED / f"{name}.csv")


def open_frame(image_path):
    """One frame at its original size, RGB."""
    with Image.open(DATA_ROOT / image_path) as image:
        return image.convert("RGB")


def boxes_to_pixels(boxes, image_width, image_height):
    """(cx, cy, w, h) fractions of the frame -> left, top, width, height in pixels.
    Rounded to 6 decimals, so the round trip gives back exactly the whole-pixel values of gt.txt."""
    w, h = boxes["w"] * image_width, boxes["h"] * image_height
    return pd.DataFrame({"left": boxes["cx"] * image_width - w / 2, "top": boxes["cy"] * image_height - h / 2,
                         "width": w, "height": h}).round(6)


class MOT17Frames(Dataset):
    """One item per frame of a split: image [3, 1088, 1920] (resized, 0-1, normalized), boxes [N, 4]
    (cx, cy, w, h fractions), ids [N], visibility [N], orig_size (width, height), video, frame."""

    def __init__(self, split):
        frames = read("frames")
        self.frames = frames[frames["split"] == split].sort_values(["video", "frame"]).reset_index(drop=True)
        boxes = read("boxes")
        boxes = boxes[boxes["split"] == split]
        self.boxes = {key: g[["cx", "cy", "w", "h", "id", "visibility"]].to_numpy()
                      for key, g in boxes.groupby(["video", "frame"])}
        stats = json.loads((PREPARED / "pixel_stats.json").read_text())
        self.mean, self.std = stats["mean"], stats["std"]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        row = self.frames.iloc[i]
        b = self.boxes.get((row["video"], row["frame"]), np.zeros((0, 6)))
        image = open_frame(row["image_path"]).resize((WIDTH, HEIGHT), Image.BILINEAR)
        return {"image": TF.normalize(TF.to_tensor(image), self.mean, self.std),
                "boxes": torch.tensor(b[:, :4], dtype=torch.float32),
                "ids": torch.tensor(b[:, 4], dtype=torch.int64),
                "visibility": torch.tensor(b[:, 5], dtype=torch.float32),
                "orig_size": (int(row["image_width"]), int(row["image_height"])),
                "video": row["video"], "frame": int(row["frame"])}


class MOT17FramePairs(Dataset):
    """One item per pair of consecutive frames (t-1, t) of the same video."""

    def __init__(self, split):
        self.frames = MOT17Frames(split)
        f = self.frames.frames
        self.pairs = [i for i in range(1, len(f))
                      if f["video"][i] == f["video"][i - 1] and f["frame"][i] == f["frame"][i - 1] + 1]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        j = self.pairs[i]
        return self.frames[j - 1], self.frames[j]


def collate(items):
    """Images stack into [B, 3, 1088, 1920]; everything else stays a list (frames have different numbers of boxes).
    Pairs give (batch of frames t-1, batch of frames t)."""
    if isinstance(items[0], tuple):
        return collate([p[0] for p in items]), collate([p[1] for p in items])
    batch = {key: [item[key] for item in items] for key in items[0]}
    batch["image"] = torch.stack(batch["image"])
    return batch


def make_loader(dataset, shuffle, batch_size=2, num_workers=0):
    """Shuffle only training data. pin_memory (on CUDA) copies batches to the GPU faster; persistent workers
    aren't restarted every epoch."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)


def to_device(batch, device):
    """Move a batch (or a (t-1, t) pair of batches) to the device: the images and every box / ID tensor."""
    if isinstance(batch, tuple):
        return tuple(to_device(b, device) for b in batch)
    move = lambda v: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return {k: move(v) if torch.is_tensor(v) else [move(x) for x in v] for k, v in batch.items()}
