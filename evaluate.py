"""
evaluate.py
===========
Stage 3: Final evaluation of PDFA on test episodes.
Reports mean accuracy ± 95% confidence interval over 2,000 episodes.
No test-time adaptation is performed.

Usage:
    python evaluate.py --dataset miniImageNet --data_root ./data \
                       --backbone ViT-B/32 --K_shot 1 \
                       --resume ./checkpoints/best_model.pth
"""

import os
import torch
import numpy as np

from config import get_config
from datasets import get_dataset, EpisodeSampler
from models.pdfa import PDFA
from utils import (
    set_seed, get_logger,
    load_checkpoint, load_clip,
    build_class_token_dict,
    mean_confidence_interval,
    print_model_summary,
)


@torch.no_grad()
def evaluate(cfg):
    # ── setup ──────────────────────────────────────────────────────
    set_seed(cfg.seed)
    logger = get_logger("evaluate", cfg.log_dir)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    logger.info(f"{'='*55}")
    logger.info(f"PDFA Evaluation")
    logger.info(f"  Dataset   : {cfg.dataset}")
    logger.info(f"  Backbone  : {cfg.backbone}")
    logger.info(f"  Episode   : {cfg.N_way}-way {cfg.K_shot}-shot")
    logger.info(f"  Episodes  : {cfg.n_test_episodes}")
    logger.info(f"  Device    : {device}")
    logger.info(f"{'='*55}")

    # ── CLIP ────────────────────────────────────────────────────────
    clip_model, _, embed_dim = load_clip(cfg.backbone, device)

    # ── test dataset ────────────────────────────────────────────────
    test_set = get_dataset(
        cfg.dataset, cfg.data_root, "test", cfg.image_size
    )
    test_sampler = EpisodeSampler(
        test_set, cfg.N_way, cfg.K_shot, cfg.Q_query
    )
    test_token_dict = build_class_token_dict(
        test_set.classes,
        list(range(test_set.n_classes)),
        device,
    )
    logger.info(
        f"Test: {len(test_set)} imgs / {test_set.n_classes} classes"
    )

    # ── build PDFA ──────────────────────────────────────────────────
    model = PDFA(
        embed_dim  = embed_dim,
        proj_dim   = cfg.proj_dim,
        M          = cfg.M_prompt,
        N_way      = cfg.N_way,
        K_shot     = cfg.K_shot,
        Q_query    = cfg.Q_query,
        hidden_dim = cfg.hidden_dim,
        asgm_heads = cfg.asgm_heads,
        asgm_dk    = cfg.asgm_dk,
        beta       = cfg.beta,
        lam        = cfg.lam,
        mlp_hidden = cfg.mlp_hidden,
        dropout    = cfg.dropout,
    ).to(device)

    # ── load checkpoint ─────────────────────────────────────────────
    assert cfg.resume is not None and os.path.exists(cfg.resume), \
        "Provide a valid --resume checkpoint path for evaluation."

    load_checkpoint(cfg.resume, model, device=device)
    model.Z_mu.freeze()
    model.eval()
    logger.info(f"Checkpoint loaded: {cfg.resume}")
    print_model_summary(model, logger)

    # ── episode-level evaluation ─────────────────────────────────────
    episode_accs = []

    for ep in range(cfg.n_test_episodes):
        s_imgs, q_imgs, q_labels, ep_cls = \
            test_sampler.get_episode_tensors(device)

        c_tokens = torch.stack(
            [test_token_dict[c] for c in ep_cls]
        ).to(device)

        logits, _ = model(clip_model, s_imgs, q_imgs, c_tokens)
        preds = logits.argmax(dim=1)
        ep_acc = (preds == q_labels).float().mean().item()
        episode_accs.append(ep_acc)

        if (ep + 1) % 200 == 0:
            mean, ci = mean_confidence_interval(episode_accs)
            logger.info(
                f"  [{ep+1}/{cfg.n_test_episodes}]  "
                f"running acc: {mean*100:.2f} ± {ci*100:.2f}%"
            )

    # ── final report ────────────────────────────────────────────────
    mean, ci = mean_confidence_interval(episode_accs)
    logger.info(f"\n{'='*55}")
    logger.info(
        f"FINAL RESULT  ({cfg.N_way}-way {cfg.K_shot}-shot):"
    )
    logger.info(
        f"  {mean*100:.2f}% ± {ci*100:.2f}%  "
        f"({cfg.n_test_episodes} episodes, 95% CI)"
    )
    logger.info(f"{'='*55}")

    return mean, ci


if __name__ == "__main__":
    cfg = get_config()
    evaluate(cfg)
