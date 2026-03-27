"""
config.py
=========
Central configuration for PDFA.
All hyperparameters, paths, and experiment settings are defined here.
"""

import argparse


def get_config():
    parser = argparse.ArgumentParser(
        description="Prompt-Driven Feature Adaptation (PDFA)"
    )

    # ── Dataset ──────────────────────────────────────────────────────
    parser.add_argument("--dataset", default="miniImageNet",
                        choices=["miniImageNet", "tieredImageNet",
                                 "CIFAR-FS", "CUB-200"])
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--image_size", type=int, default=224)

    # ── Episode ───────────────────────────────────────────────────────
    parser.add_argument("--N_way",            type=int, default=5)
    parser.add_argument("--K_shot",           type=int, default=1,
                        choices=[1, 5])
    parser.add_argument("--Q_query",          type=int, default=15)
    parser.add_argument("--n_train_episodes", type=int, default=200)
    parser.add_argument("--n_val_episodes",   type=int, default=600)
    parser.add_argument("--n_test_episodes",  type=int, default=2000)

    # ── Backbone ──────────────────────────────────────────────────────
    parser.add_argument("--backbone", default="ViT-B/32",
                        choices=["ViT-B/32", "ViT-B/16",
                                 "RN50", "RN101", "RN50x4"])

    # ── Prompt ────────────────────────────────────────────────────────
    parser.add_argument("--M_prompt",        type=int,   default=4)
    parser.add_argument("--prompt_init_std", type=float, default=0.02)

    # ── Visual Learner Z_mu ───────────────────────────────────────────
    parser.add_argument("--proj_dim",        type=int,   default=512)
    parser.add_argument("--pretrain_epochs", type=int,   default=100)
    parser.add_argument("--pretrain_lr",     type=float, default=1e-3)
    parser.add_argument("--pretrain_bs",     type=int,   default=128)

    # ── B_alpha ───────────────────────────────────────────────────────
    parser.add_argument("--hidden_dim", type=int, default=512)

    # ── ASGM ──────────────────────────────────────────────────────────
    parser.add_argument("--asgm_heads", type=int,   default=8)
    parser.add_argument("--asgm_dk",    type=int,   default=64)
    parser.add_argument("--beta",       type=float, default=0.01)

    # ── MLP Classifier ────────────────────────────────────────────────
    parser.add_argument("--mlp_hidden", type=int,   default=128)
    parser.add_argument("--dropout",    type=float, default=0.1)

    # ── Training ──────────────────────────────────────────────────────
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lam",          type=float, default=0.1)

    # ── Paths ─────────────────────────────────────────────────────────
    parser.add_argument("--save_dir",      default="./checkpoints")
    parser.add_argument("--log_dir",       default="./logs")
    parser.add_argument("--resume",        default=None)
    parser.add_argument("--pretrain_ckpt", default=None)

    # ── Misc ──────────────────────────────────────────────────────────
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_only",   action="store_true")
    parser.add_argument("--verbose",     action="store_true")

    return parser.parse_args()


DATASET_DEFAULTS = {
    "miniImageNet":   {"n_base": 64,  "n_val": 16, "n_test": 20},
    "tieredImageNet": {"n_base": 351, "n_val": 97, "n_test": 160},
    "CIFAR-FS":       {"n_base": 64,  "n_val": 16, "n_test": 20},
    "CUB-200":        {"n_base": 100, "n_val": 50, "n_test": 50},
}
