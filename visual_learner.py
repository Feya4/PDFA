"""
models/visual_learner.py
========================
Visual Learner Z_mu.
Pre-trained on base classes, then frozen during episodic adaptation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VisualLearner(nn.Module):
    """
    Z_mu: projects raw CLIP visual features f_theta(I) into a stable,
    task-invariant representation space.

    Architecture:
        Linear(in_dim -> 2*out_dim) -> BatchNorm -> ReLU
        -> Linear(2*out_dim -> out_dim)

    Pre-trained on D_train base classes with cross-entropy loss,
    following Meta-Baseline (Chen et al., ICCV 2021).
    Frozen after pre-training for all episodic training and inference.

    Args:
        in_dim  : input dimension (CLIP embed_dim)
        out_dim : output dimension (proj_dim)
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.BatchNorm1d(out_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim * 2, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, in_dim)  raw CLIP visual features
        Returns:
            (B, out_dim)   projected visual features V
        """
        return self.net(x)

    def freeze(self):
        """Freeze all parameters after pre-training."""
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        return self

    def unfreeze(self):
        """Unfreeze for pre-training."""
        for p in self.parameters():
            p.requires_grad_(True)
        self.train()
        return self

    def extra_repr(self) -> str:
        return f"in_dim={self.in_dim}, out_dim={self.out_dim}"
