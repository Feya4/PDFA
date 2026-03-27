"""
train.py
========
Stage 2: Episodic training of PDFA.
Z_mu must be pre-trained and saved before running this script.

Usage:
    python train.py --dataset miniImageNet --data_root ./data \
                    --backbone ViT-B/32 --K_shot 1 \
                    --pretrain_ckpt ./checkpoints/Zmu_best.pth
"""

import os
import torch
import torch.nn.functional as F

from config import get_config
from datasets import get_dataset, EpisodeSampler
from models.pdfa import PDFA
from utils import (
    set_seed, get_logger,
    save_checkpoint, load_checkpoint,
    load_clip, build_class_token_dict,
    MetricLogger, print_model_summary,
)


def train_one_epoch(
    model, clip_model, sampler, class_token_dict,
    optimizer, device, n_episodes, logger, epoch,
):
    """Run one epoch of episodic training."""
    model.train()
    metrics = MetricLogger()

    for ep in range(n_episodes):
        s_imgs, q_imgs, q_labels, ep_cls = \
            sampler.get_episode_tensors(device)

        # gather class tokens for this episode
        c_tokens = torch.stack(
            [class_token_dict[c] for c in ep_cls]
        ).to(device)

        logits, loss = model(
            clip_model, s_imgs, q_imgs, c_tokens, q_labels
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.get_trainable_params(), max_norm=1.0
        )
        optimizer.step()

        acc = (logits.argmax(dim=1) == q_labels).float().mean()
        metrics.update(loss=loss.item(), acc=acc.item())

        if (ep + 1) % 50 == 0:
            logger.debug(
                f"  [ep {ep+1}/{n_episodes}]  {metrics.summary()}"
            )

    return metrics.to_dict()


@torch.no_grad()
def validate(
    model, clip_model, sampler, class_token_dict,
    device, n_episodes,
):
    """Run validation episodes and return mean accuracy."""
    model.eval()
    correct, total = 0, 0

    for _ in range(n_episodes):
        s_imgs, q_imgs, q_labels, ep_cls = \
            sampler.get_episode_tensors(device)

        c_tokens = torch.stack(
            [class_token_dict[c] for c in ep_cls]
        ).to(device)

        logits, _ = model(clip_model, s_imgs, q_imgs, c_tokens)
        preds = logits.argmax(dim=1)
        correct += (preds == q_labels).sum().item()
        total   += q_labels.size(0)

    return correct / total


def train(cfg):
    # ── setup ──────────────────────────────────────────────────────
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)
    logger = get_logger("train", cfg.log_dir)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    logger.info(f"{'='*55}")
    logger.info(f"PDFA Training")
    logger.info(f"  Dataset  : {cfg.dataset}")
    logger.info(f"  Backbone : {cfg.backbone}")
    logger.info(f"  Episode  : {cfg.N_way}-way {cfg.K_shot}-shot")
    logger.info(f"  Device   : {device}")
    logger.info(f"{'='*55}")

    # ── CLIP ────────────────────────────────────────────────────────
    clip_model, _, embed_dim = load_clip(cfg.backbone, device)
    logger.info(f"CLIP loaded: embed_dim={embed_dim}")

    # ── datasets ────────────────────────────────────────────────────
    train_set = get_dataset(cfg.dataset, cfg.data_root, "train",
                            cfg.image_size)
    val_set   = get_dataset(cfg.dataset, cfg.data_root, "val",
                            cfg.image_size)

    train_sampler = EpisodeSampler(
        train_set, cfg.N_way, cfg.K_shot, cfg.Q_query
    )
    val_sampler = EpisodeSampler(
        val_set, cfg.N_way, cfg.K_shot, cfg.Q_query
    )

    logger.info(
        f"Train: {len(train_set)} imgs / {train_set.n_classes} cls  |  "
        f"Val: {len(val_set)} imgs / {val_set.n_classes} cls"
    )

    # ── class token dictionaries ────────────────────────────────────
    train_token_dict = build_class_token_dict(
        train_set.classes,
        list(range(train_set.n_classes)),
        device,
    )
    val_token_dict = build_class_token_dict(
        val_set.classes,
        list(range(val_set.n_classes)),
        device,
    )

    # ── build PDFA ──────────────────────────────────────────────────
    model = PDFA(
        embed_dim   = embed_dim,
        proj_dim    = cfg.proj_dim,
        M           = cfg.M_prompt,
        N_way       = cfg.N_way,
        K_shot      = cfg.K_shot,
        Q_query     = cfg.Q_query,
        hidden_dim  = cfg.hidden_dim,
        asgm_heads  = cfg.asgm_heads,
        asgm_dk     = cfg.asgm_dk,
        beta        = cfg.beta,
        lam         = cfg.lam,
        mlp_hidden  = cfg.mlp_hidden,
        dropout     = cfg.dropout,
        prompt_std  = cfg.prompt_std,
    ).to(device)

    # ── load pre-trained Z_mu ────────────────────────────────────────
    if cfg.pretrain_ckpt is not None and \
       os.path.exists(cfg.pretrain_ckpt):
        ckpt = torch.load(cfg.pretrain_ckpt, map_location=device)
        model.Z_mu.load_state_dict(ckpt["model_state"])
        logger.info(f"Z_mu loaded from: {cfg.pretrain_ckpt}")
    else:
        logger.warning(
            "No pretrain_ckpt provided. Z_mu is randomly initialised."
        )
    model.Z_mu.freeze()
    logger.info("Z_mu frozen.")

    print_model_summary(model, logger)
    param_info = model.count_parameters()
    logger.info(
        f"Parameters — trainable: {param_info['trainable']:,}  "
        f"frozen: {param_info['frozen']:,}"
    )

    # ── optimizer + scheduler ────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.get_trainable_params(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.epochs
    )

    start_epoch = 0
    best_val_acc = 0.0

    # ── optional resume ─────────────────────────────────────────────
    if cfg.resume is not None and os.path.exists(cfg.resume):
        ckpt_info = load_checkpoint(
            cfg.resume, model, optimizer, scheduler, device
        )
        start_epoch  = ckpt_info.get("epoch", 0)
        best_val_acc = ckpt_info.get("best_val_acc", 0.0)
        logger.info(
            f"Resumed from epoch {start_epoch}, "
            f"best_val_acc={best_val_acc*100:.2f}%"
        )

    # ── training loop ────────────────────────────────────────────────
    logger.info(f"Starting episodic training for {cfg.epochs} epochs")

    for epoch in range(start_epoch, cfg.epochs):

        train_metrics = train_one_epoch(
            model, clip_model,
            train_sampler, train_token_dict,
            optimizer, device,
            cfg.n_train_episodes, logger, epoch,
        )

        val_acc = validate(
            model, clip_model,
            val_sampler, val_token_dict,
            device, cfg.n_val_episodes,
        )

        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        logger.info(
            f"Epoch [{epoch+1:03d}/{cfg.epochs}]  "
            f"loss={train_metrics['loss']:.4f}  "
            f"train_acc={train_metrics['acc']*100:.2f}%  "
            f"val_acc={val_acc*100:.2f}%  "
            f"best={best_val_acc*100:.2f}%  "
            f"lr={lr_now:.2e}"
            + ("  ← best" if is_best else "")
        )

        save_checkpoint(
            {
                "epoch":         epoch + 1,
                "model_state":   model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_acc":  best_val_acc,
                "cfg":           vars(cfg),
            },
            save_dir=cfg.save_dir,
            filename=f"epoch_{epoch+1:03d}.pth",
            is_best=is_best,
        )

    logger.info(
        f"\nTraining complete. "
        f"Best val acc: {best_val_acc*100:.2f}%"
    )


if __name__ == "__main__":
    cfg = get_config()
    train(cfg)
