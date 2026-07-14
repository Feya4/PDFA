# ACAS-PT: Adaptive Cross-Modal Alignment via Symmetric Prompts for Few-Shot Vision–Language Learning 

**Abstract**
Few-shot learning with vision-language models suffers from a fundamental structural limitation: support and query samples are processed through independent and asymmetric encoding pipelines. This causes query-side semantic blindness, where the model lacks rich cross-modal interactions during query encoding. Consequently, it weakens vision-language alignment and creates a training-inference distribution gap, degrading generalization to novel categories. Existing prompt-based methods inherit this asymmetry and thus cannot leverage text-conditioned semantic context on the query side at inference time. To address this limitation, we propose an Adaptive Cross-Modal Alignment via Symmetric Prompt Tuning for Few-Shot Vision–Language Learning (ACAS-PT) a unified framework that resolves this issue via symmetric prompt tuning. ACAS-PT applies identical prompt-guided, text-conditioned feature transformations to both support and query samples in a shared multimodal space, eliminating the distribution gap by design. Specifically, we propose two modules. First, a Semantic-Aware Class-Embedding Learner transforms prompt-conditioned CLIP class embeddings into class-specific semantic vectors used to modulate both support and query visual features via FiLM-based affine transformation, ensuring that query samples receive the same class-specific semantic grounding as support prototypes at inference. Second, an Adaptive Similarity Guided Module (ASGM) replaces fragile equal-weight prototype averaging with learnable instance-weighted centroid aggregation and a per-class-pair cross-modal alignment matrix that gates classification scores by within-class semantic-visual alignment confidence, yielding robust prototype estimates even under extreme label scarcity. Extensive experiments on four benchmark datasets show ACAS-PT outperforms 16 state-of-the-art methods, with symmetric processing alone yielding up to a +2.5% improvement in 5-shot accuracy. These results highlight query-side semantic blindness as a critical bottleneck in vision-language few-shot learning.
---
<img width="822" height="652" alt="modelcopy" src="https://github.com/user-attachments/assets/6f30f57e-57dd-4104-b23d-e1fd8bd55420" />


## Project Structure

```
pdfa/
├── config.py                      # All hyperparameters and CLI arguments
├── pretrain.py                    # Stage 1: pre-train Z_mu on base classes
├── train.py                       # Stage 2: episodic training of PDFA
├── evaluation.py                    # Stage 3: final test evaluation
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
│   ├── dataset.py            # FewShotDataset base + EpisodeSampler
│  
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

<img width="736" height="440" alt="table1" src="https://github.com/user-attachments/assets/c4e68742-f14e-4461-a76d-6621a358b073" />
<img width="1783" height="1075" alt="PDFA-4" src="https://github.com/user-attachments/assets/c9488123-b8f6-47b9-87a7-cd2a81824129" />
<img width="2026" height="2547" alt="attention_visualization_1shot" src="https://github.com/user-attachments/assets/db42d7fc-5268-4789-a69b-8c49ccea5a9d" />


---

## Citation

```bibtex
@article{Akmel2026,
  author    = {Akmel, F. and Gong, X. and Hadabi, A. and others},
  title     = {Adaptive cross-modal alignment via symmetric prompt tuning for few-shot vision–language learning},
  journal   = {Journal of King Saud University - Computer and Information Sciences},
  year      = {2026},
  doi       = {10.1007/s44443-026-00952-8},
  url       = {https://doi.org/10.1007/s44443-026-00952-8}
}
```

