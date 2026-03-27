"""
models/pdfa.py
==============
Full PDFA model: end-to-end assembly of all modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List, Optional

from models.prompt import LearnablePrompt
from models.visual_learner import VisualLearner
from models.task_adaptive_learner import TaskAdaptiveLearner
from models.asgm import ASGM


# ─────────────────────────────────────────────────────────────────────
# Contrastive semantic alignment loss  L_w
# ─────────────────────────────────────────────────────────────────────

def contrastive_alignment_loss(
    X_pred:  Tensor,   # hat_w  (B, d)  B_alpha output
    V_pred:  Tensor,   # tilde_w (B, d)  Z_mu path
    w_gt:    Tensor,   # w       (B, d)  frozen CLIP class embedding
) -> Tensor:
    """
    L_w = mean_n [ (1 - cos(hat_w_n, w_n)) + (1 - cos(tilde_w_n, w_n)) ]

    Encourages both semantic prediction (X_pred) and visual prediction
    (V_pred) to remain close to the frozen CLIP class embedding w_gt.
    """
    w_n     = F.normalize(w_gt,    dim=-1)
    w_hat   = F.normalize(X_pred,  dim=-1)
    w_tilde = F.normalize(V_pred,  dim=-1)

    loss_hat   = 1.0 - (w_hat   * w_n).sum(dim=-1)   # (B,)
    loss_tilde = 1.0 - (w_tilde * w_n).sum(dim=-1)   # (B,)
    return (loss_hat + loss_tilde).mean()


# ─────────────────────────────────────────────────────────────────────
# MLP Classifier
# ─────────────────────────────────────────────────────────────────────

class MLPClassifier(nn.Module):
    """
    Two-layer MLP classifier.

    Input : cosine similarities cos(h_q, p_i) scaled by S*
    Output: class logits over N classes

    Args:
        N_way      : number of classes (input dimension)
        hidden_dim : hidden layer size
        dropout    : dropout rate
    """

    def __init__(self, N_way: int, hidden_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_way, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, N_way),
        )

    def forward(
        self,
        h_q:        Tensor,   # (Q, 2d)
        prototypes: Tensor,   # (N, 2d)
        S_star:     Tensor,   # (N, N)
    ) -> Tensor:
        """Returns logits (Q, N)."""
        # cosine similarity between each query and each prototype
        cos_sim = F.cosine_similarity(
            h_q.unsqueeze(1),          # (Q, 1, 2d)
            prototypes.unsqueeze(0),   # (1, N, 2d)
            dim=-1,
        )                              # (Q, N)

        # scale by global cross-modal alignment signal from S*
        s_scale = torch.sigmoid(
            S_star.mean(dim=0).unsqueeze(0)
        )                              # (1, N)
        scaled = cos_sim * s_scale     # (Q, N)

        return self.net(scaled)        # (Q, N)


# ─────────────────────────────────────────────────────────────────────
# Full PDFA Model
# ─────────────────────────────────────────────────────────────────────

class PDFA(nn.Module):
    """
    Prompt-Driven Feature Adaptation (PDFA).

    End-to-end vision-language framework for few-shot learning.
    Applies the same text-conditioned transformation symmetrically
    to both support and query samples within a unified pipeline.

    Args:
        embed_dim   : CLIP embedding dimension (512 for ViT-B/32)
        proj_dim    : Z_mu output dimension
        M           : number of learnable prompt tokens
        N_way       : classes per episode
        K_shot      : support shots per class
        Q_query     : query samples per class
        hidden_dim  : B_alpha hidden dimension
        asgm_heads  : ASGM attention heads
        asgm_dk     : ASGM key/query dim per head
        beta        : ASGM S* regularisation coefficient
        lam         : loss weighting coefficient lambda
        mlp_hidden  : MLP classifier hidden dim
        dropout     : MLP classifier dropout
        prompt_std  : prompt token init std
    """

    def __init__(
        self,
        embed_dim:  int   = 512,
        proj_dim:   int   = 512,
        M:          int   = 4,
        N_way:      int   = 5,
        K_shot:     int   = 1,
        Q_query:    int   = 15,
        hidden_dim: int   = 512,
        asgm_heads: int   = 8,
        asgm_dk:    int   = 64,
        beta:       float = 0.01,
        lam:        float = 0.1,
        mlp_hidden: int   = 128,
        dropout:    float = 0.1,
        prompt_std: float = 0.02,
    ):
        super().__init__()
        self.N = N_way
        self.K = K_shot
        self.Q = Q_query
        self.lam = lam
        self.embed_dim = embed_dim
        self.proj_dim  = proj_dim

        # ── modules ──────────────────────────────────────────────────
        self.prompt     = LearnablePrompt(M, embed_dim, prompt_std)
        self.Z_mu       = VisualLearner(embed_dim, proj_dim)
        self.B_alpha    = TaskAdaptiveLearner(proj_dim, hidden_dim)
        self.asgm       = ASGM(proj_dim * 2, asgm_heads, asgm_dk, beta)
        self.classifier = MLPClassifier(N_way, mlp_hidden, dropout)

        # optional linear to bridge embed_dim -> proj_dim
        self.embed_proj = (
            nn.Linear(embed_dim, proj_dim, bias=False)
            if embed_dim != proj_dim else nn.Identity()
        )

    # ── CLIP encoding helpers ─────────────────────────────────────────

    @torch.no_grad()
    def encode_images(self, clip_model, images: Tensor) -> Tensor:
        """(B, 3, H, W) -> (B, embed_dim) L2-normalised."""
        feat = clip_model.encode_image(images).float()
        return F.normalize(feat, dim=-1)

    @torch.no_grad()
    def encode_class_names(
        self, clip_model, class_tokens: Tensor
    ) -> Tensor:
        """(N, L) -> (N, embed_dim) L2-normalised frozen CLIP embeddings."""
        feat = clip_model.encode_text(class_tokens).float()
        return F.normalize(feat, dim=-1)

    # ── forward ───────────────────────────────────────────────────────

    def forward(
        self,
        clip_model,
        support_images: Tensor,          # (N*K, 3, H, W)
        query_images:   Tensor,          # (N*Q, 3, H, W)
        class_tokens:   Tensor,          # (N, L) tokenised class names
        query_labels:   Optional[Tensor] = None,  # (N*Q,) for training
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Full PDFA forward pass.

        Returns:
            logits : (N*Q, N)   class logits
            loss   : scalar     combined loss (None at inference)
        """
        N, K, Q = self.N, self.K, self.Q

        # ── Stage 1: frozen CLIP features ────────────────────────────
        raw_s = self.encode_images(clip_model, support_images)  # (N*K, d)
        raw_q = self.encode_images(clip_model, query_images)    # (N*Q, d)
        w_gt  = self.encode_class_names(clip_model, class_tokens)  # (N, d)

        # ground-truth class embeddings expanded to support size
        w_gt_s = w_gt.repeat_interleave(K, dim=0)   # (N*K, d)

        # ── Stage 2: visual projection Z_mu (frozen) ─────────────────
        V_s = self.Z_mu(raw_s)   # (N*K, proj_dim)
        V_q = self.Z_mu(raw_q)   # (N*Q, proj_dim)

        # ── Stage 3: class embeddings for B_alpha ────────────────────
        # project e_i from embed_dim to proj_dim
        e_i = self.embed_proj(w_gt)              # (N, proj_dim)

        # expand: each class embedding repeated for K support / Q query
        e_s = e_i.repeat_interleave(K, dim=0)   # (N*K, proj_dim)
        e_q = e_i.repeat_interleave(Q, dim=0)   # (N*Q, proj_dim)

        # ── Stage 4: symmetric B_alpha transformation ─────────────────
        h_s, X_s = self.B_alpha(V_s, e_s)   # (N*K, 2p), (N*K, p)
        h_q, X_q = self.B_alpha(V_q, e_q)   # (N*Q, 2p), (N*Q, p)

        # ── Stage 5: ASGM ─────────────────────────────────────────────
        prototypes, S_star = self.asgm(h_s, X_s, V_s, N, K)
        # (N, 2p),  (N, N)

        # ── Stage 6: MLP classifier ───────────────────────────────────
        logits = self.classifier(h_q, prototypes, S_star)  # (N*Q, N)

        # ── Stage 7: loss (training only) ────────────────────────────
        loss = None
        if query_labels is not None:
            L_ce = F.cross_entropy(logits, query_labels)

            # V_s path as tilde_w: project back to embed_dim
            V_s_emb = self.embed_proj(V_s) if isinstance(
                self.embed_proj, nn.Linear
            ) else V_s

            L_w = contrastive_alignment_loss(
                X_s,     # hat_w:   B_alpha semantic output
                V_s_emb, # tilde_w: Z_mu visual output (align dimension)
                w_gt_s,  # w:       frozen CLIP class embedding
            )
            loss = L_ce + self.lam * L_w

        return logits, loss

    def get_trainable_params(self):
        """Return only jointly-trained parameters (Z_mu excluded)."""
        return [
            p for n, p in self.named_parameters()
            if "Z_mu" not in n and p.requires_grad
        ]

    def count_parameters(self) -> dict:
        """Count trainable vs frozen parameters."""
        trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        frozen = sum(
            p.numel() for p in self.parameters() if not p.requires_grad
        )
        return {"trainable": trainable, "frozen": frozen,
                "total": trainable + frozen}
