
# config.py
from dataclasses import dataclass

@dataclass
class Config:
    # CLIP Model
    clip_model_name: str = "ViT-B-32"      # Important: Use "ViT-B-32", not "ViT-B/32"
    d: int = 512                           # embedding dimension
    M: int = 8                             # number of learnable prompt tokens

    # Few-shot settings
    n_way: int = 5
    k_shot: int = 1
    q_query: int = 15

    # Model architecture
    hidden_dim: int = 256

    # Training hyperparameters
    lambda_w: float = 0.5
    lr: float = 5e-4
    weight_decay: float = 0.01
    epochs: int = 100
    num_episodes_per_epoch: int = 200

    # Paths
    data_root = '/workdir1.8t/fei27/CGT/APDFA/data' 
    checkpoint_dir: str = "./checkpoints"
    results_dir: str = "./results"

    # Device
    device: str = "cuda"


# Global config instance
config = Config()
