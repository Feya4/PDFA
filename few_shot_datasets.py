"""
datasets/few_shot_datasets.py
==============================
Dataset implementations for:
    - miniImageNet
    - tieredImageNet
    - CIFAR-FS
    - CUB-200-2011

Expected directory structure:
    data/
    ├── miniImageNet/
    │   ├── images/
    │   └── split/  (train.csv, val.csv, test.csv)
    ├── tieredImageNet/
    │   ├── images/
    │   └── split/  (train.txt, val.txt, test.txt)
    ├── CIFAR-FS/
    │   ├── data/   (CIFAR-FS_train.pickle, ...)
    │   └── split/
    └── CUB-200/
        ├── images/
        └── split/  (train.txt, val.txt, test.txt)
"""

import os
import csv
import pickle
import numpy as np
from PIL import Image
from typing import List, Tuple

import torch
from torch import Tensor

from datasets.base_dataset import FewShotDataset, get_clip_transform


# ─────────────────────────────────────────────────────────────────────
# miniImageNet
# ─────────────────────────────────────────────────────────────────────

class MiniImageNet(FewShotDataset):
    """
    miniImageNet: 100 classes, 600 images/class, 84x84.
    Split: 64 train / 16 val / 20 test (Ravi & Larochelle, ICLR 2017).

    CSV format (each row): filename, label
    """

    def _load_data(self):
        split_file = os.path.join(
            self.root, "miniImageNet", "split", f"{self.split}.csv"
        )
        image_dir = os.path.join(self.root, "miniImageNet", "images")

        class_names = []
        rows = []
        with open(split_file, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for fname, label in reader:
                rows.append((fname, label))
                if label not in class_names:
                    class_names.append(label)

        self.classes = sorted(class_names)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for fname, label in rows:
            path = os.path.join(image_dir, fname)
            self.data.append((path, self.class_to_idx[label]))


# ─────────────────────────────────────────────────────────────────────
# tieredImageNet
# ─────────────────────────────────────────────────────────────────────

class TieredImageNet(FewShotDataset):
    """
    tieredImageNet: 608 classes in 34 semantic super-categories.
    Split: 351 train / 97 val / 160 test (Ren et al., ICLR 2018).

    Split files: one class name per line.
    Images organized as: images/<class_name>/<filename>
    """

    def _load_data(self):
        split_file = os.path.join(
            self.root, "tieredImageNet", "split", f"{self.split}.txt"
        )
        image_dir = os.path.join(self.root, "tieredImageNet", "images")

        with open(split_file, "r") as f:
            class_names = [line.strip() for line in f if line.strip()]

        self.classes = sorted(class_names)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for class_name in self.classes:
            class_dir = os.path.join(image_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    path = os.path.join(class_dir, fname)
                    self.data.append(
                        (path, self.class_to_idx[class_name])
                    )


# ─────────────────────────────────────────────────────────────────────
# CIFAR-FS
# ─────────────────────────────────────────────────────────────────────

class CIFARFS(FewShotDataset):
    """
    CIFAR-FS: derived from CIFAR-100.
    100 classes, 600 images/class, 32x32.
    Split: 64 train / 16 val / 20 test (Bertinetto et al., 2019).

    Data stored as pickle files with keys: 'data' (N,32,32,3), 'labels'.
    """

    def _load_data(self):
        pickle_file = os.path.join(
            self.root, "CIFAR-FS",
            f"CIFAR-FS_{self.split}.pickle"
        )
        with open(pickle_file, "rb") as f:
            pack = pickle.load(f, encoding="latin1")

        images = pack["data"]        # (N, 32, 32, 3) uint8
        labels = pack["labels"]      # list of int
        label2name = pack.get("label2name", {})

        unique_labels = sorted(set(labels))
        self.classes = [
            label2name.get(l, str(l)) for l in unique_labels
        ]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        local_map = {l: i for i, l in enumerate(unique_labels)}

        # store images in memory as PIL for transform compatibility
        self._images = images
        for idx, (img_arr, label) in enumerate(zip(images, labels)):
            self.data.append((idx, local_map[label]))

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        img_idx, label = self.data[idx]
        img = Image.fromarray(self._images[img_idx])
        return self.transform(img), label


# ─────────────────────────────────────────────────────────────────────
# CUB-200-2011
# ─────────────────────────────────────────────────────────────────────

class CUB200(FewShotDataset):
    """
    CUB-200-2011: 200 bird species, 11,788 images, 224x224.
    Split: 100 train / 50 val / 50 test.

    Split files: one <image_path> <label> per line.
    Images organized as: images/<species_folder>/<filename>
    """

    def _load_data(self):
        split_file = os.path.join(
            self.root, "CUB-200", "split", f"{self.split}.txt"
        )
        image_dir = os.path.join(self.root, "CUB-200", "images")

        class_names = []
        rows = []
        with open(split_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                rel_path = parts[0]
                label_name = rel_path.split("/")[0]
                rows.append((rel_path, label_name))
                if label_name not in class_names:
                    class_names.append(label_name)

        self.classes = sorted(class_names)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for rel_path, label_name in rows:
            path = os.path.join(image_dir, rel_path)
            self.data.append((path, self.class_to_idx[label_name]))


# ─────────────────────────────────────────────────────────────────────
# Dataset factory
# ─────────────────────────────────────────────────────────────────────

def get_dataset(
    name: str,
    root: str,
    split: str,
    image_size: int = 224,
    use_clip_transform: bool = True,
) -> FewShotDataset:
    """
    Factory function to instantiate the correct dataset class.

    Args:
        name      : one of 'miniImageNet', 'tieredImageNet',
                    'CIFAR-FS', 'CUB-200'
        root      : data root directory
        split     : 'train', 'val', or 'test'
        image_size: spatial resolution
        use_clip_transform: use CLIP-compatible normalization

    Returns:
        FewShotDataset instance
    """
    registry = {
        "miniImageNet":   MiniImageNet,
        "tieredImageNet": TieredImageNet,
        "CIFAR-FS":       CIFARFS,
        "CUB-200":        CUB200,
    }
    if name not in registry:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Choose from: {list(registry.keys())}"
        )
    return registry[name](
        root=root,
        split=split,
        image_size=image_size,
        use_clip_transform=use_clip_transform,
    )
