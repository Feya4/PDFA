# utils/utils.py
import torch
import torch.nn.functional as F

def compute_alignment_loss(X_i: torch.Tensor, 
                          V_bar_i: torch.Tensor, 
                          e_i: torch.Tensor) -> torch.Tensor:
      # Normalize
    X_norm = F.normalize(X_i, dim=1)
    V_norm = F.normalize(V_bar_i, dim=1)
    e_norm = F.normalize(e_i, dim=1)
    
    # Cosine distance = 1 - cosine similarity
    loss_semantic = (1 - (X_norm * e_norm).sum(dim=1)).mean()
    loss_visual   = (1 - (V_norm * e_norm).sum(dim=1)).mean()
    
    return loss_semantic + loss_visual


def compute_S_star(X_centroids: torch.Tensor, 
                   V_centroids: torch.Tensor, 
                   beta: float = 0.5) -> torch.Tensor:
   
    # Cosine similarity
    cos_sim = F.cosine_similarity(X_centroids.unsqueeze(1), 
                                  V_centroids.unsqueeze(0), dim=-1)
    
    # Euclidean distance penalty
    dist = torch.cdist(X_centroids, V_centroids, p=2)
    
    S_star = cos_sim - beta * dist ** 2
    return S_star


def accuracy(output, target):
    """ Computes the accuracy over the k top predictions """
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        
        # FIX: Ensure target is on the same device as pred
        target = target.to(pred.device) 
        
        correct = (pred == target).sum().item()
        return correct / len(target)