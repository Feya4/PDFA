# models/prompt_learner.py
import torch
import torch.nn as nn

class PromptLearner(nn.Module):
    """Learnable context tokens"""
    def __init__(self, M: int, d: int):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(M, d) * 0.02)

    def forward(self, class_names, g_phi):
        # In real code, you would insert tokens before class names using tokenizer
        # Here we assume g_phi handles the prompt-augmented input
        return g_phi(class_names)  # placeholder for open_clip / CLIP text encoder