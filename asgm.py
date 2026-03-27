"""
models/asgm.py
==============
Adaptive Similarity Guided Module (ASGM).

Two mechanisms:
    1. Learnable instance-weighted prototype aggregation p_i
    2. Shared cross-modal similarity matrix S* (closed-form)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


class ASGM(nn.Module):
    """
    Adaptive Similarity Guided Module.

    Given fused support representations {h_s_ij}, semantic features
    X_s, and visual features V_s, ASGM computes:

        S_ij   = cosine(X_s^i, V_s^j)                    [Eq. sim]
        alpha_ij = softmax(S_ij)                          [Eq. softmax]
        p_i    = sum_j alpha_ij * h_s_ij                  [Eq. proto]
        S*_ij  = -1/(2*beta) * sum_m ||X_s^m - V_s^m||^2 [Eq. S*]

    Args:
        feat_dim  : dimension of fused h representations (2 * proj_dim)
        num_heads : number of MHA heads for feature refinement
        d_k       : key/query dimension per head
        beta      : regularisation coefficient for S*
    """

    def __init__(
        self,
        feat_dim:  int   = 1024,
        num_heads: int   = 8,
        d_k:       int   = 64,
        beta:      float = 0.01,
    ):
        super().__init__()
        self.beta = beta
        self.feat_dim = feat_dim

        # multi-head attention for feature refinement before prototype
        attn_dim = num_heads * d_k
        self.mha = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.feat_proj = nn.Linear(feat_dim, attn_dim, bias=False)
        self.out_proj  = nn.Linear(attn_dim, feat_dim, bias=False)
        self.norm      = nn.LayerNorm(feat_dim)

    def _cosine_sim(self, A: Tensor, B: Tensor) -> Tensor:
        """
        Row-wise cosine similarity matrix.
        A: (Na, d),  B: (Nb, d)  ->  S: (Na, Nb)
        """
        A_n = F.normalize(A, dim=-1)
        B_n = F.normalize(B, dim=-1)
        return A_n @ B_n.T

    def _compute_S_star(
        self, X_s_mean: Tensor, V_s_mean: Tensor, N: int
    ) -> Tensor:
        """
        Closed-form shared cross-modal similarity matrix S*.

        s*_ij = -1/(2*beta) * sum_m ||X_s^m - V_s^m||^2

        Args:
            X_s_mean: (N, d)  per-class mean semantic features
            V_s_mean: (N, d)  per-class mean visual features
            N       : number of classes
        Returns:
            S_star  : (N, N)
        """
        diff_sq = ((X_s_mean - V_s_mean) ** 2).sum(dim=-1)   # (N,)
        val = -diff_sq.sum() / (2.0 * self.beta)              # scalar
        return val.expand(N, N).clone()

    def forward(
        self,
        h_s: Tensor,    # (N*K, 2d)  support fused representations
        X_s: Tensor,    # (N*K, d)   support semantic features
        V_s: Tensor,    # (N*K, d)   support visual features
        N: int,         # number of classes
        K: int,         # shots per class
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            prototypes : (N, 2d)  adaptive weighted class prototypes p_i
            S_star     : (N, N)   shared cross-modal similarity matrix
        """
        # ── (1) cosine similarity S ───────────────────────────────────
        # S shape: (N*K, N*K)  — X_s as rows, V_s as columns
        S_full = self._cosine_sim(X_s, V_s)           # (N*K, N*K)

        # ── (2) per-class softmax weights alpha_ij ────────────────────
        # Reshape: for each class i, K rows of X_s attend to all N*K V_s
        S_per_class = S_full.view(N, K, N * K)
        alpha = F.softmax(S_per_class, dim=-1)         # (N, K, N*K)

        # Use only within-class attention weights: (N, K, K)
        # Extract block diagonal (class i attends to its own K V_s)
        alpha_within = torch.zeros(N, K, K, device=h_s.device)
        for i in range(N):
            alpha_within[i] = alpha[i, :, i*K:(i+1)*K]
        alpha_within = F.softmax(alpha_within.sum(dim=-1), dim=-1)  # (N, K)

        # ── (3) weighted prototype aggregation ───────────────────────
        h_s_3d = h_s.view(N, K, -1)                   # (N, K, 2d)
        # weighted sum over K support samples
        proto_raw = (
            alpha_within.unsqueeze(-1) * h_s_3d
        ).sum(dim=1)                                   # (N, 2d)

        # optional MHA refinement of prototypes
        proto_refined, _ = self.mha(
            proto_raw.unsqueeze(0),
            proto_raw.unsqueeze(0),
            proto_raw.unsqueeze(0),
        )                                              # (1, N, 2d)
        prototypes = self.norm(
            proto_raw + proto_refined.squeeze(0)
        )                                              # (N, 2d)

        # ── (4) closed-form S* ───────────────────────────────────────
        X_s_mean = X_s.view(N, K, -1).mean(dim=1)    # (N, d)
        V_s_mean = V_s.view(N, K, -1).mean(dim=1)    # (N, d)
        S_star = self._compute_S_star(X_s_mean, V_s_mean, N)

        return prototypes, S_star

    def extra_repr(self) -> str:
        return f"feat_dim={self.feat_dim}, beta={self.beta}"
