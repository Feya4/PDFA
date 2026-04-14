# models/asgm.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ASGM(nn.Module):
    """
    Adaptive Similarity Guided Module
    Calculates logits by comparing queries to prototypes in a weighted metric space.
    """
    def __init__(self, d: int, hidden_dim: int = 128):
        super().__init__()
        # The MLP should transform the FUSED features (2*d) 
        # to a weighting factor or a refined feature space.
        # Since PDFA uses h = [v, x], the input dim is 2*d.
        self.mlp = nn.Sequential(
            nn.Linear(2 * d, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Learnable temperature for cosine similarity
        self.scale = nn.Parameter(torch.tensor(10.0))

    def compute_prototypes(self, h_s: torch.Tensor, labels: torch.Tensor, n_way: int):
        """Computes class prototypes by averaging support features."""
        prototypes = []
        for i in range(n_way):
            mask = (labels == i)
            if mask.sum() > 0:
                proto = h_s[mask].mean(dim=0)
            else:
                # Fallback for empty classes (shouldn't happen in standard few-shot)
                proto = torch.zeros_like(h_s[0])
            prototypes.append(proto)
        return torch.stack(prototypes)  # (N, 2d)

    def forward(self, h_q_per_class: torch.Tensor, prototypes: torch.Tensor, S_star_ii: torch.Tensor):
        """
        Args:
            h_q_per_class: (NQ, N, 2d) - Query features fused with each class embedding
            prototypes:    (N, 2d)     - Support set prototypes
            S_star_ii:     (N,)        - Semantic guidance placeholder
        """
        # 1. Compute instance-level importance weights using the MLP
        # h_q_per_class has shape (NQ, N, 2d). MLP output: (NQ, N, 1)
        weights = self.mlp(h_q_per_class) 
        
        # 2. Refine features with weights
        h_q_weighted = h_q_per_class * weights

        # 3. Align prototypes for broadcasting
        # prototypes: (N, 2d) -> (1, N, 2d)
        proto = prototypes.unsqueeze(0)

        # 4. Compute Cosine Similarity
        # Result shape: (NQ, N)
        cos_sim = F.cosine_similarity(h_q_weighted, proto, dim=-1)
        
        # 5. Apply Semantic Guidance (S_star_ii) and Scaling
        # S_star_ii: (N,) -> (1, N)
        # We use a sigmoid-gated approach to scale the class similarities
        guidance = torch.sigmoid(S_star_ii).unsqueeze(0)
        
        # Final logits
        logits = cos_sim * guidance * self.scale
        
        return logits