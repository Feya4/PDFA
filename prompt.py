"""
models/prompt.py
================
Learnable prompt token module.
M context tokens prepended to each class name token sequence.
"""

import torch
import torch.nn as nn
from torch import Tensor


class LearnablePrompt(nn.Module):
    """
    M learnable context tokens {t_1, ..., t_M} prepended to each
    class name embedding sequence.

    Initialized from N(0, sigma^2) following CoOp (Zhou et al., 2022).

    Args:
        M         : number of context tokens
        embed_dim : token embedding dimension (matches CLIP text encoder)
        init_std  : initialization standard deviation
    """

    def __init__(self, M: int, embed_dim: int, init_std: float = 0.02):
        super().__init__()
        self.M = M
        ctx = torch.empty(M, embed_dim)
        nn.init.normal_(ctx, std=init_std)
        self.context = nn.Parameter(ctx)   # (M, d)  — jointly trained

    def forward(self, token_embeddings: Tensor) -> Tensor:
        """
        Prepend context tokens to embedded class name tokens.

        Args:
            token_embeddings: (N, L, d)  embedded class tokens
        Returns:
            (N, M+L, d)  context-prepended sequence
        """
        N = token_embeddings.size(0)
        ctx = self.context.unsqueeze(0).expand(N, -1, -1)  # (N, M, d)
        return torch.cat([ctx, token_embeddings], dim=1)    # (N, M+L, d)

    def extra_repr(self) -> str:
        return f"M={self.M}, embed_dim={self.context.size(1)}"
