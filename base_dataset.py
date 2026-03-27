"""
datasets/base_dataset.py
========================
Base few-shot dataset class + episode sampler.
All dataset-specific classes inherit from FewShotDataset.
"""

import os
import json
import random
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms as T


# ─────────────────────────────────────────────────────────────────────
# Standard transforms
# ─────────────────────────────────────────────────────────────────────

def get_transform(split: str, image_size: int = 224) -> T.Compose:
    """
    Returns standard data augmentation transforms.
    Training: RandomCrop + HorizontalFlip + ColorJitter
    Val/Test: CenterCrop only
    """
    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if split == "train":
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.4, contrast=0.4,
                          saturation=0.4, hue=0.1),
            T.ToTensor(),
            normalize,
        ])
    else:
        resize = int(image_size * 1.15)
        return T.Compose([
            T.Resize(resize),
            T.CenterCrop(image_size),
            T.ToTensor(),
            normalize,
        ])


def get_clip_transform(image_size: int = 224) -> T.Compose:
    """
    CLIP-compatible transform (matches CLIP's preprocess pipeline).
    """
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)),
    ])


# ─────────────────────────────────────────────────────────────────────
# Base dataset
# ─────────────────────────────────────────────────────────────────────

class FewShotDataset(Dataset):
    """
    Base class for few-shot datasets.
    Subclasses must implement _load_data() which populates:
        self.data   : list of (image_path, class_id)
        self.classes: list of class name strings
        self.class_to_idx: {class_name: idx}
    """

    def __init__(
        self,
        root: str,
        split: str,
        image_size: int = 224,
        use_clip_transform: bool = True,
    ):
        assert split in ("train", "val", "test")
        self.root = root
        self.split = split
        self.image_size = image_size
        self.transform = (
            get_clip_transform(image_size) if use_clip_transform
            else get_transform(split, image_size)
        )
        self.data: List[Tuple[str, int]] = []
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        self._load_data()

    def _load_data(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        path, label = self.data[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

    @property
    def labels(self) -> List[int]:
        return [d[1] for d in self.data]

    @property
    def n_classes(self) -> int:
        return len(self.classes)


# ─────────────────────────────────────────────────────────────────────
# Episode Sampler
# ─────────────────────────────────────────────────────────────────────

class EpisodeSampler:
    """
    Samples N-way K-shot Q-query episodes from a FewShotDataset.

    Args:
        dataset    : FewShotDataset instance
        N_way      : number of classes per episode
        K_shot     : support samples per class
        Q_query    : query samples per class
    """

    def __init__(
        self,
        dataset: FewShotDataset,
        N_way: int   = 5,
        K_shot: int  = 1,
        Q_query: int = 15,
    ):
        self.dataset = dataset
        self.N = N_way
        self.K = K_shot
        self.Q = Q_query

        labels = np.array(dataset.labels)
        self.unique_classes = np.unique(labels)
        self.class_indices: Dict[int, np.ndarray] = {
            c: np.where(labels == c)[0]
            for c in self.unique_classes
        }
        # verify enough samples per class
        for c, idx in self.class_indices.items():
            assert len(idx) >= K_shot + Q_query, (
                f"Class {c} has only {len(idx)} samples; "
                f"need at least {K_shot + Q_query}."
            )

    def sample_episode(self) -> Tuple[List[int], List[int], List[int]]:
        """
        Returns:
            support_idx    : (N*K,) global dataset indices
            query_idx      : (N*Q,) global dataset indices
            episode_classes: (N,)   selected class ids
        """
        episode_classes = random.sample(
            list(self.unique_classes), self.N
        )
        support_idx, query_idx = [], []
        for c in episode_classes:
            chosen = np.random.choice(
                self.class_indices[c],
                self.K + self.Q,
                replace=False,
            )
            support_idx.extend(chosen[:self.K].tolist())
            query_idx.extend(chosen[self.K:].tolist())
        return support_idx, query_idx, episode_classes

    def get_episode_tensors(
        self,
        device: str = "cuda",
    ) -> Tuple[Tensor, Tensor, Tensor, List[int]]:
        """
        Convenience method: samples one episode and stacks tensors.

        Returns:
            support_imgs : (N*K, C, H, W)
            query_imgs   : (N*Q, C, H, W)
            query_labels : (N*Q,)  remapped to 0..N-1
            episode_classes: list of original class ids
        """
        s_idx, q_idx, ep_cls = self.sample_episode()
        label_map = {c: i for i, c in enumerate(ep_cls)}

        support_imgs = torch.stack(
            [self.dataset[i][0] for i in s_idx]
        ).to(device)
        query_imgs = torch.stack(
            [self.dataset[i][0] for i in q_idx]
        ).to(device)
        query_labels = torch.tensor(
            [label_map[self.dataset[i][1]] for i in q_idx],
            dtype=torch.long, device=device,
        )
        return support_imgs, query_imgs, query_labels, ep_cls
