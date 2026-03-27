"""
models/task_adaptive_learner.py
================================
Semantic-Aware Task-Adaptive Feature Learner B_alpha.

Applied SYMMETRICALLY to both support and query samples.
This symmetric design is the core contribution of PDFA.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


class CrossModalAttention(nn.Module):
    """
    Scaled dot-product cross-modal attention.

    Visual features V attend to semantic features X:
        hat_V = Softmax( Q(V) · K(X)^T / sqrt(d_k) ) · V(X)

    Args:
        dim : feature dimension d
    """

    def __init__(self, dim: int):
        super().__init__()
        self.scale = math.sqrt(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, V: Tensor, X: Tensor) -> Tensor:
        """
        Args:
            V: (B, d)  visual features  (query in attention sense)
            X: (B, d)  semantic features (key and value)
        Returns:
            hat_V: (B, d)  semantically modulated visual features
        """
        Q = self.q_proj(V)              # (B, d)
        K = self.k_proj(X)              # (B, d)
        Vv = self.v_proj(X)             # (B, d)

        # element-wise dot then sigmoid gating
        attn = torch.sigmoid(
            (Q * K).sum(dim=-1, keepdim=True) / self.scale
        )                               # (B, 1)
        return self.out_proj(attn * Vv) # (B, d)


class TaskAdaptiveLearner(nn.Module):
    """
    B_alpha: Semantic-Aware Task-Adaptive Feature Learner.

    Pipeline per sample:
        1. LayerNorm on class embedding e_i
        2. Two-layer MLP (GELU) -> task-specific semantic X
        3. CrossModalAttention(V, X) -> hat_V
        4. InstanceNorm(hat_V)
        5. Concatenate [InstanceNorm(hat_V) || X] -> h in R^{2d}

    Applied identically to support (producing h_s, X_s)
    and query      (producing h_q, X_q).

    Args:
        in_dim     : input dimension (proj_dim from Z_mu)
        hidden_dim : hidden dimension of MLP (default 512)
    """

    def __init__(self, in_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.in_dim = in_dim

        # semantic transformation
        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, in_dim),
        )

        # cross-modal attention
        self.cross_attn = CrossModalAttention(in_dim)

        # instance normalisation over feature dimension
        self.inst_norm = nn.InstanceNorm1d(in_dim, affine=True)

    def forward(
        self,
        visual_feat: Tensor,    # V_s or V_q : (B, d)
        class_embed: Tensor,    # e_i (repeated to B): (B, d)
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            h : (B, 2d)  fused cross-modal representation
            X : (B, d)   task-specific semantic vector
        """
        # (1)+(2) semantic transformation
        X = self.mlp(self.norm(class_embed))        # (B, d)

        # (3) cross-modal attention
        hat_V = self.cross_attn(visual_feat, X)     # (B, d)

        # (4) instance normalisation
        hat_V_norm = self.inst_norm(
            hat_V.unsqueeze(1)
        ).squeeze(1)                                 # (B, d)

        # (5) fused representation h ∈ R^{2d}
        h = torch.cat([hat_V_norm, X], dim=-1)       # (B, 2d)
        return h, X

    def extra_repr(self) -> str:
        return f"in_dim={self.in_dim}"
