# models/film.py
import torch
import torch.nn as nn

class FiLM(nn.Module):
    """Feature-wise Linear Modulation"""
    def __init__(self, d: int):
        super().__init__()
        self.W_gamma = nn.Linear(d, d)
        self.W_zeta  = nn.Linear(d, d)
        self.b_gamma = nn.Parameter(torch.zeros(d))
        self.b_zeta  = nn.Parameter(torch.zeros(d))

    def forward(self, V: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """V: (..., d), X: (..., d)"""
        gamma = self.W_gamma(X) + self.b_gamma
        zeta  = self.W_zeta(X)  + self.b_zeta
        return gamma * V + zeta