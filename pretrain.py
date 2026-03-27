"""
pretrain.py
===========
Stage 1: Pre-train the Visual Learner Z_mu on base classes.
After this script completes, Z_mu is saved and frozen for
all subsequent episodic training.

Usage:
    python pretrain.py --dataset miniImageNet --data_root ./data \
                       --backbone ViT-B/32 --pretrain_epochs 100
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import get_config, DATASET_DEFAULTS
from datasets import get_dataset
from models.visual_learner import VisualLearner
from utils import (
    set_seed, get_logger, save_checkpoint,
    load_clip, AverageMeter, print_model_summary,
)


def pretrain(cfg):
    # ── setup ──────────────────────────────────────────────────────
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)
    logger = get_logger("pretrain", cfg.log_dir)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    logger.info(f"Dataset: {cfg.dataset}  Backbone: {cfg.backbone}")

    # ── CLIP backbone ───────────────────────────────────────────────
    clip_model, _, embed_dim = load_clip(cfg.backbone, device)
    logger.info(f"CLIP loaded: {cfg.backbone} (embed_dim={embed_dim})")

    # ── dataset and loader ──────────────────────────────────────────
    train_dataset = get_dataset(
        cfg.dataset, cfg.data_root, "train",
        image_size=cfg.image_size, use_clip_transform=True,
    )
    n_base = train_dataset.n_classes
    logger.info(
        f"Train dataset: {len(train_dataset)} images, {n_base} classes"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.pretrain_bs,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ── build Z_mu + linear classification head ─────────────────────
    Z_mu = VisualLearner(embed_dim, cfg.proj_dim).to(device)
    head = nn.Linear(cfg.proj_dim, n_base).to(device)

    params = list(Z_mu.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(
        params, lr=cfg.pretrain_lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.pretrain_epochs
    )

    print_model_summary(Z_mu, logger)
    logger.info(
        f"Starting Z_mu pre-training for {cfg.pretrain_epochs} epochs"
    )

    best_acc = 0.0

    # ── training loop ───────────────────────────────────────────────
    for epoch in range(cfg.pretrain_epochs):
        Z_mu.train()
        head.train()
        loss_meter = AverageMeter("loss")
        acc_meter  = AverageMeter("acc")

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                raw = clip_model.encode_image(images).float()
                raw = F.normalize(raw, dim=-1)

            feat   = Z_mu(raw)
            logits = head(feat)
            loss   = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (logits.argmax(dim=1) == labels).float().mean()
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(acc.item(), images.size(0))

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch [{epoch+1:03d}/{cfg.pretrain_epochs}]  "
                f"loss={loss_meter.avg:.4f}  acc={acc_meter.avg*100:.2f}%  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if acc_meter.avg > best_acc:
            best_acc = acc_meter.avg
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state": Z_mu.state_dict(),
                    "acc": best_acc,
                    "cfg": vars(cfg),
                },
                save_dir=cfg.save_dir,
                filename="Zmu_best.pth",
                is_best=True,
            )

    logger.info(
        f"Z_mu pre-training complete. "
        f"Best train acc: {best_acc*100:.2f}%"
    )
    logger.info(
        f"Checkpoint saved to: {cfg.save_dir}/Zmu_best.pth"
    )


if __name__ == "__main__":
    cfg = get_config()
    pretrain(cfg)
