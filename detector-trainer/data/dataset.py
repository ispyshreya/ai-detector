"""Torch Dataset over the shared manifest.

``VeilDataset`` reads rows for one split from ``manifest.csv`` and yields
``(image_tensor, label)`` pairs. The default transform matches the inference
contract in ``backend/app/signals/local_model.py`` exactly, so training and
serving see identical preprocessing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Must match backend/app/signals/local_model.py.
IMAGE_SIZE = 224
_NORM_MEAN = [0.485, 0.456, 0.406]
_NORM_STD = [0.229, 0.224, 0.225]

SPLITS = ("train", "val", "test_indist", "test_wild")


def default_transform() -> transforms.Compose:
    """Eval/inference transform matching ``local_model.py``.

    Resize(224, 224) → ToTensor → Normalize(ImageNet mean/std).
    """
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(_NORM_MEAN, _NORM_STD),
        ]
    )


def train_transform() -> transforms.Compose:
    """Training transform: default + horizontal flip (aug lives in trainer).

    Kept minimal here; JPEG/resize-jitter augmentation is applied by the
    trainer so this module stays a thin, deterministic reader.
    """
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_NORM_MEAN, _NORM_STD),
        ]
    )


class VeilDataset(Dataset):
    """Dataset of images for a single manifest split.

    Parameters
    ----------
    manifest:
        Either a path to ``manifest.csv`` or an already-loaded DataFrame.
    split:
        One of ``{train, val, test_indist, test_wild}``.
    transform:
        Optional callable applied to the PIL image; defaults to
        :func:`default_transform`.
    """

    def __init__(self, manifest, split: str, transform=None):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

        if isinstance(manifest, (str, Path)):
            df = pd.read_csv(manifest)
        elif isinstance(manifest, pd.DataFrame):
            df = manifest.copy()
        else:
            raise TypeError("manifest must be a path or a pandas DataFrame")

        required = {"path", "label", "split"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"manifest missing columns: {sorted(missing)}")

        self.df = df[df["split"] == split].reset_index(drop=True)
        self.split = split
        self.transform = transform if transform is not None else default_transform()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        with Image.open(row["path"]) as img:
            image = img.convert("RGB")
            tensor = self.transform(image)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return tensor, label
