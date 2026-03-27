# Prompt-Driven Feature Adaptation (PDFA)
## Vision–Language Few-Shot Learning

---

## Project Structure

```
pdfa/
├── config.py                      # All hyperparameters and CLI arguments
├── pretrain.py                    # Stage 1: pre-train Z_mu on base classes
├── train.py                       # Stage 2: episodic training of PDFA
├── evaluate.py                    # Stage 3: final test evaluation
├── requirements.txt
│
├── models/
│   ├── __init__.py
│   ├── prompt.py                  # LearnablePrompt  {t_m}
│   ├── visual_learner.py          # VisualLearner    Z_mu
│   ├── task_adaptive_learner.py   # TaskAdaptiveLearner  B_alpha
│   ├── asgm.py                    # ASGM (prototypes + S*)
│   └── pdfa.py                    # Full PDFA model + MLP classifier
│
├── datasets/
│   ├── __init__.py
│   ├── base_dataset.py            # FewShotDataset base + EpisodeSampler
│   └── few_shot_datasets.py       # MiniImageNet, TieredImageNet,
│                                  # CIFARFS, CUB200, get_dataset()
│
└── utils/
    ├── __init__.py
    └── utils.py                   # seed, logger, checkpoint, CI, metrics
```

---

## Installation

```bash
git clone https://github.com/Feya4/PDFA.git
cd PDFA
pip install -r requirements.txt
```

---

## Data Preparation

```
data/
├── miniImageNet/
│   ├── images/          # all 60,000 images as .jpg
│   └── split/
│       ├── train.csv    # filename,label
│       ├── val.csv
│       └── test.csv
├── tieredImageNet/
│   ├── images/<class>/  # images grouped by class folder
│   └── split/
│       ├── train.txt    # one class name per line
│       ├── val.txt
│       └── test.txt
├── CIFAR-FS/
│   ├── CIFAR-FS_train.pickle
│   ├── CIFAR-FS_val.pickle
│   └── CIFAR-FS_test.pickle
└── CUB-200/
    ├── images/<species>/
    └── split/
        ├── train.txt    # relative_path label_name
        ├── val.txt
        └── test.txt
```

---

## Training Pipeline

### Stage 1 — Pre-train Z_mu

```bash
python pretrain.py \
    --dataset miniImageNet \
    --data_root ./data \
    --backbone ViT-B/32 \
    --pretrain_epochs 100 \
    --pretrain_lr 1e-3 \
    --pretrain_bs 128 \
    --save_dir ./checkpoints
```

### Stage 2 — Episodic training

```bash
# 1-shot
python train.py \
    --dataset miniImageNet \
    --data_root ./data \
    --backbone ViT-B/32 \
    --K_shot 1 \
    --pretrain_ckpt ./checkpoints/Zmu_best.pth \
    --epochs 100 \
    --lr 1e-3 \
    --lam 0.1 \
    --save_dir ./checkpoints

# 5-shot
python train.py --K_shot 5 [same args]
```

### Stage 3 — Evaluation

```bash
python evaluate.py \
    --dataset miniImageNet \
    --data_root ./data \
    --backbone ViT-B/32 \
    --K_shot 1 \
    --resume ./checkpoints/best_model.pth \
    --n_test_episodes 2000
```

---

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--M_prompt` | 4 | Learnable context tokens |
| `--lam` | 0.1 | Loss weight λ for L_w |
| `--beta` | 0.01 | ASGM S* regularisation |
| `--asgm_heads` | 8 | ASGM attention heads |
| `--hidden_dim` | 512 | B_alpha hidden dimension |

---

## Results

| Method | Backbone | mini 1s | mini 5s | tiered 1s | tiered 5s |
|--------|----------|---------|---------|-----------|-----------|
| PDFA (ours) | ResNet-12 | 72.16 | 83.79 | 72.34 | 85.97 |
| PDFA+CLIP (ours) | ViT-B/16 | **84.84** | **93.04** | **88.23** | **94.25** |

---

## Citation

```bibtex
@article{pdfa2025,
  title  = {Prompt-Driven Feature Adaptation for
             Vision-Language Few-Shot Learning},
  author = {Feidu Akmel, Xun Gong},
  year   = {2026}
}
```
