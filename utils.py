"""
utils/utils.py
==============
General utilities: seeding, logging, checkpointing,
confidence intervals, metric tracking.
"""

import os
import json
import random
import logging
import numpy as np
from typing import List, Tuple, Dict, Any
from scipy import stats

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    """Fix random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────

def get_logger(name: str, log_dir: str = None,
               level=logging.INFO) -> logging.Logger:
    """
    Returns a logger that writes to console and optionally to a file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(
            os.path.join(log_dir, f"{name}.log")
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────────────────────

def save_checkpoint(
    state: Dict[str, Any],
    save_dir: str,
    filename: str = "checkpoint.pth",
    is_best: bool = False,
):
    """Save model checkpoint. Optionally saves a separate best copy."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(save_dir, "best_model.pth")
        torch.save(state, best_path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer=None,
    scheduler=None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Load checkpoint and restore model (and optionally optimizer) state."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt


# ─────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────

def mean_confidence_interval(
    accuracies: List[float],
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute mean and 95% confidence interval half-width.
    Standard reporting convention in few-shot learning.

    Returns:
        (mean_acc, ci_halfwidth)  both in [0, 1] range
    """
    n = len(accuracies)
    m = np.mean(accuracies)
    se = stats.sem(accuracies)
    h = se * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return float(m), float(h)


# ─────────────────────────────────────────────────────────────────────
# Metric tracker
# ─────────────────────────────────────────────────────────────────────

class AverageMeter:
    """Tracks a running average of a scalar metric."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


class MetricLogger:
    """Tracks multiple AverageMeters and logs them together."""

    def __init__(self):
        self.meters: Dict[str, AverageMeter] = {}

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.meters:
                self.meters[k] = AverageMeter(k)
            self.meters[k].update(v)

    def summary(self) -> str:
        return "  ".join(str(m) for m in self.meters.values())

    def to_dict(self) -> Dict[str, float]:
        return {k: m.avg for k, m in self.meters.items()}


# ─────────────────────────────────────────────────────────────────────
# Model summary
# ─────────────────────────────────────────────────────────────────────

def print_model_summary(model: nn.Module, logger=None):
    """Print parameter counts per module."""
    lines = ["\n── PDFA Parameter Summary ──────────────────"]
    total_train, total_frozen = 0, 0
    for name, module in model.named_children():
        n_train = sum(p.numel() for p in module.parameters()
                      if p.requires_grad)
        n_frozen = sum(p.numel() for p in module.parameters()
                       if not p.requires_grad)
        status = "frozen" if n_train == 0 else "trainable"
        lines.append(
            f"  {name:<20s}  {status:<10s}  "
            f"train={n_train:>8,}  frozen={n_frozen:>8,}"
        )
        total_train  += n_train
        total_frozen += n_frozen

    lines.append(f"{'─'*50}")
    lines.append(
        f"  {'TOTAL':<20s}  {'':10s}  "
        f"train={total_train:>8,}  frozen={total_frozen:>8,}"
    )
    lines.append("─────────────────────────────────────────────\n")
    msg = "\n".join(lines)
    if logger:
        logger.info(msg)
    else:
        print(msg)


# ─────────────────────────────────────────────────────────────────────
# CLIP loading utility
# ─────────────────────────────────────────────────────────────────────

def load_clip(backbone: str = "ViT-B/32", device: str = "cuda"):
    """
    Load and freeze a CLIP backbone.

    Returns:
        clip_model  : frozen CLIP model
        preprocess  : CLIP image preprocessing transform
        embed_dim   : visual/text embedding dimension
    """
    try:
        import clip
    except ImportError:
        raise ImportError(
            "Install CLIP: "
            "pip install git+https://github.com/openai/CLIP.git"
        )
    model, preprocess = clip.load(backbone, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    embed_dim = model.visual.output_dim
    return model, preprocess, embed_dim


def build_class_token_dict(
    class_names: List[str],
    class_ids: List[int],
    device: str = "cuda",
) -> Dict[int, torch.Tensor]:
    """
    Tokenize class names using CLIP's tokenizer.

    Returns:
        {class_id: token_tensor (L,)}
    """
    try:
        import clip
    except ImportError:
        raise ImportError(
            "Install CLIP: "
            "pip install git+https://github.com/openai/CLIP.git"
        )
    token_dict = {}
    for cid, name in zip(class_ids, class_names):
        # replace underscores and format naturally
        readable = name.replace("_", " ").replace("-", " ").lower()
        tokens = clip.tokenize([readable])[0]   # (L,)
        token_dict[cid] = tokens.to(device)
    return token_dict
