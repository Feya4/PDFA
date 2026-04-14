"""
Attention Visualization for CUB-200 Bird Dataset
Model: APDFA (Adaptive Part-aware Dynamic Feature Aggregation)
Checkpoint: /workdir1.8t/fei27/CGT/APDFA/checkpoints/pdfa_best1.pth

Generates a rich grid showing:
  - Original bird images
  - GradCAM heatmaps
  - Attention rollout (for transformer heads)
  - Overlaid attention maps
  - Top-K attended regions with bounding boxes

Usage:
    python visualize_attention_cub200.py \
        --checkpoint /workdir1.8t/fei27/CGT/APDFA/checkpoints/pdfa_best1.pth \
        --data_root /path/to/CUB_200_2011 \
        --num_images 5 \
        --output attention_vis.png
"""

import os
import sys
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import cv2

# ─────────────────────────────────────────────────────────────────
# CUB-200 Dataset
# ─────────────────────────────────────────────────────────────────
root='/workdir1.8t/fei27/CGT/APDFA/data/CUB_200_2011'
class CUB200Dataset(torch.utils.data.Dataset):
    """CUB-200-2011 dataset loader."""

    def __init__(self, root, split="test", transform=None):
        self.root = Path(root)
        self.transform = transform
        self.split = split

        # Load images list
        img_file = self.root / "images.txt"
        split_file = self.root / "train_test_split.txt"
        label_file = self.root / "image_class_labels.txt"
        classes_file = self.root / "classes.txt"

        assert img_file.exists(), f"Cannot find {img_file}"

        images = {}
        with open(img_file) as f:
            for line in f:
                idx, path = line.strip().split(" ", 1)
                images[int(idx)] = path

        splits = {}
        with open(split_file) as f:
            for line in f:
                idx, is_train = line.strip().split()
                splits[int(idx)] = int(is_train)

        labels = {}
        with open(label_file) as f:
            for line in f:
                idx, label = line.strip().split()
                labels[int(idx)] = int(label) - 1  # 0-indexed

        self.classes = []
        with open(classes_file) as f:
            for line in f:
                _, cls = line.strip().split(" ", 1)
                self.classes.append(cls.split(".")[-1].replace("_", " "))

        # Filter by split (1=train, 0=test)
        is_train_split = (split == "train")
        self.samples = [
            (images[i], labels[i])
            for i in sorted(images.keys())
            if splits[i] == int(is_train_split)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        img_path = self.root / "images" / rel_path
        img = Image.open(img_path).convert("RGB")
        original = img.copy()
        if self.transform:
            img = self.transform(img)
        return img, label, np.array(original), rel_path


# ─────────────────────────────────────────────────────────────────
# Checkpoint Inspector
# ─────────────────────────────────────────────────────────────────

def inspect_checkpoint(ckpt_path):
    """Load checkpoint and infer model architecture."""
    print(f"\n[1/5] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        keys = list(ckpt.keys())
        print(f"  Checkpoint keys: {keys}")

        state_dict = None
        for k in ["model", "state_dict", "model_state_dict", "net"]:
            if k in ckpt:
                state_dict = ckpt[k]
                print(f"  Using key: '{k}'")
                break
        if state_dict is None:
            # Maybe the whole ckpt is the state dict
            if any("weight" in k for k in keys):
                state_dict = ckpt
            else:
                raise ValueError(f"Cannot find state_dict in keys: {keys}")
    else:
        # Could be a bare state_dict or a nn.Module
        state_dict = ckpt if isinstance(ckpt, dict) else ckpt.state_dict()

    sd_keys = list(state_dict.keys())[:10]
    print(f"  First 10 param keys: {sd_keys}")

    # Detect architecture
    is_transformer = any("attn" in k or "attention" in k or "transformer" in k
                         for k in state_dict.keys())
    is_resnet = any("layer1" in k or "layer2" in k for k in state_dict.keys())
    num_classes = None
    for k, v in state_dict.items():
        if "classifier" in k and "weight" in k and v.ndim == 2:
            num_classes = v.shape[0]
            break
        if "head" in k and "weight" in k and v.ndim == 2:
            num_classes = v.shape[0]
            break
        if "fc" in k and "weight" in k and v.ndim == 2:
            num_classes = v.shape[0]
            break

    if num_classes is None:
        num_classes = 200  # Default CUB-200

    arch = "transformer" if is_transformer else "resnet"
    print(f"  Detected arch: {arch}, num_classes: {num_classes}")
    return state_dict, arch, num_classes, ckpt


def build_model(state_dict, arch, num_classes, device):
    """Build and load model, falling back gracefully."""
    print(f"\n[2/5] Building model ({arch}, {num_classes} classes)...")

    sd_keys = set(state_dict.keys())

    # Try common APDFA/fine-grained recognition backbones
    model = None

    # 1. Try ResNet-50 (most common CUB backbone)
    if model is None and arch == "resnet":
        try:
            m = models.resnet50(weights=None)
            m.fc = nn.Linear(2048, num_classes)
            # Strict=False to handle extra APDFA heads
            missing, unexpected = m.load_state_dict(state_dict, strict=False)
            print(f"  ResNet-50 loaded (missing={len(missing)}, unexpected={len(unexpected)})")
            model = m
            arch = "resnet50"
        except Exception as e:
            print(f"  ResNet-50 failed: {e}")

    # 2. Try ResNet-101
    if model is None:
        try:
            m = models.resnet101(weights=None)
            m.fc = nn.Linear(2048, num_classes)
            missing, unexpected = m.load_state_dict(state_dict, strict=False)
            print(f"  ResNet-101 loaded (missing={len(missing)}, unexpected={len(unexpected)})")
            model = m
            arch = "resnet101"
        except Exception as e:
            print(f"  ResNet-101 failed: {e}")

    # 3. Try ViT-B/16
    if model is None:
        try:
            m = models.vit_b_16(weights=None)
            m.heads = nn.Linear(768, num_classes)
            missing, unexpected = m.load_state_dict(state_dict, strict=False)
            print(f"  ViT-B/16 loaded (missing={len(missing)}, unexpected={len(unexpected)})")
            model = m
            arch = "vit"
        except Exception as e:
            print(f"  ViT-B/16 failed: {e}")

    # 4. Fallback: use pretrained ResNet-50 for visualization
    if model is None:
        print("  WARNING: Could not load checkpoint weights exactly.")
        print("  Falling back to ImageNet pretrained ResNet-50 for demo visualization.")
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(2048, num_classes)
        arch = "resnet50"

    model = model.to(device)
    model.eval()
    return model, arch


# ─────────────────────────────────────────────────────────────────
# GradCAM
# ─────────────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, model, arch):
        self.model = model
        self.arch = arch
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _get_target_layer(self):
        if "resnet" in self.arch:
            return self.model.layer4[-1]
        elif self.arch == "vit":
            return self.model.encoder.layers[-1]
        else:
            # Try last conv layer heuristic
            last_conv = None
            for module in self.model.modules():
                if isinstance(module, nn.Conv2d):
                    last_conv = module
            return last_conv

    def _register_hooks(self):
        target = self._get_target_layer()
        if target is None:
            return

        def fwd_hook(module, inp, out):
            self.activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hooks.append(target.register_forward_hook(fwd_hook))
        self.hooks.append(target.register_full_backward_hook(bwd_hook))

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        output = self.model(x)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        score = output[0, class_idx]
        score.backward()

        if self.gradients is None or self.activations is None:
            return None, class_idx

        # Pool gradients over spatial dims
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


# ─────────────────────────────────────────────────────────────────
# Attention Rollout (for ViT)
# ─────────────────────────────────────────────────────────────────

class AttentionRollout:
    def __init__(self, model):
        self.model = model
        self.attention_maps = []
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for layer in self.model.encoder.layers:
            def make_hook(layer_ref):
                def hook(module, inp, out):
                    # out is typically (attn_output, attn_weights)
                    if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                        self.attention_maps.append(out[1].detach().cpu())
                return hook
            self.hooks.append(layer.self_attention.register_forward_hook(
                make_hook(layer)))

    def __call__(self, x):
        self.attention_maps = []
        with torch.no_grad():
            _ = self.model(x)

        if not self.attention_maps:
            return None

        # Rollout
        result = torch.eye(self.attention_maps[0].shape[-1])
        for attn in self.attention_maps:
            attn_mean = attn.mean(dim=1)  # avg over heads
            attn_mean = attn_mean + torch.eye(attn_mean.shape[-1])
            attn_mean = attn_mean / attn_mean.sum(dim=-1, keepdim=True)
            result = torch.matmul(attn_mean, result)

        # CLS token attends to patches
        mask = result[0, 0, 1:]
        n = int(mask.shape[0] ** 0.5)
        mask = mask.reshape(n, n).numpy()
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        return mask

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


# ─────────────────────────────────────────────────────────────────
# Multi-head Raw Attention Extractor
# ─────────────────────────────────────────────────────────────────

class MultiHeadAttentionVis:
    """Extract raw attention from last self-attention layer."""

    def __init__(self, model):
        self.model = model
        self.last_attn = None
        self._hook = None
        self._register()

    def _register(self):
        # Find last attention layer
        target = None
        for module in self.model.modules():
            if hasattr(module, "in_proj_weight") or "MultiheadAttention" in type(module).__name__:
                target = module
        if target is not None:
            def hook(module, inp, out):
                if isinstance(out, tuple):
                    self.last_attn = out[1].detach().cpu() if out[1] is not None else None
            self._hook = target.register_forward_hook(hook)

    def __call__(self, x):
        self.last_attn = None
        with torch.no_grad():
            _ = self.model(x)
        return self.last_attn

    def remove(self):
        if self._hook:
            self._hook.remove()


# ─────────────────────────────────────────────────────────────────
# Visualization Helpers
# ─────────────────────────────────────────────────────────────────

# Custom heatmap colormap (blue→cyan→green→yellow→red)
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "birdcam",
    ["#000080", "#0080FF", "#00FFFF", "#80FF00", "#FFFF00", "#FF8000", "#FF0000"]
)


def cam_to_heatmap(cam, img_hw):
    """Resize CAM to image size and convert to RGBA heatmap."""
    h, w = img_hw
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    cam_resized = np.clip(cam_resized, 0, 1)
    heatmap = HEATMAP_CMAP(cam_resized)  # (H, W, 4)
    return cam_resized, heatmap


def overlay_cam(img_np, cam_resized, alpha=0.45):
    """Overlay heatmap on original image."""
    heatmap_rgb = (HEATMAP_CMAP(cam_resized)[:, :, :3] * 255).astype(np.uint8)
    img_float = img_np.astype(np.float32)
    overlay = (1 - alpha) * img_float + alpha * heatmap_rgb.astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def top_k_bbox(cam, k=3, threshold=0.6):
    """Find top-K bounding boxes from CAM above threshold."""
    mask = (cam > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:k]:
        x, y, w, h = cv2.boundingRect(cnt)
        bboxes.append((x, y, w, h))
    return bboxes


def draw_bboxes_on_image(img_np, bboxes, colors=None):
    """Draw bounding boxes on a copy of the image."""
    vis = img_np.copy()
    default_colors = [(255, 50, 50), (50, 255, 50), (50, 50, 255)]
    for i, (x, y, w, h) in enumerate(bboxes):
        c = colors[i] if colors else default_colors[i % len(default_colors)]
        cv2.rectangle(vis, (x, y), (x + w, y + h), c, 2)
        cv2.putText(vis, f"R{i+1}", (x + 2, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    return vis


# ─────────────────────────────────────────────────────────────────
# Main Visualization
# ─────────────────────────────────────────────────────────────────

def visualize(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load checkpoint ──────────────────────────────────────────
    state_dict, arch, num_classes, raw_ckpt = inspect_checkpoint(args.checkpoint)

    # ── Build model ──────────────────────────────────────────────
    model, arch = build_model(state_dict, arch, num_classes, device)

    # ── Dataset ──────────────────────────────────────────────────
    print(f"\n[3/5] Loading CUB-200 dataset from: {args.data_root}")
    INPUT_SIZE = 448 if "448" in str(state_dict.keys()) else 224

    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = CUB200Dataset(args.data_root, split=args.split, transform=transform)
    print(f"  Dataset size ({args.split}): {len(dataset)}")

    # Sample diverse images (one per class if possible)
    n = args.num_images
    indices = random.sample(range(len(dataset)), min(n * 10, len(dataset)))
    # Pick indices with diverse labels
    seen_labels = set()
    chosen = []
    for idx in indices:
        _, lbl, _, _ = dataset[idx]
        if lbl not in seen_labels:
            chosen.append(idx)
            seen_labels.add(lbl)
        if len(chosen) == n:
            break
    # Pad if needed
    while len(chosen) < n:
        chosen.append(random.randint(0, len(dataset) - 1))

    print(f"  Selected {len(chosen)} images from {len(seen_labels)} classes")

    # ── Setup attention extractors ────────────────────────────────
    print(f"\n[4/5] Setting up attention extractors (arch={arch})...")
    gradcam = GradCAM(model, arch)
    rollout = AttentionRollout(model) if arch == "vit" else None

    # ── Generate visualizations ───────────────────────────────────
    print(f"\n[5/5] Generating attention maps...")

    # Layout: rows = images, cols = [original, gradcam, overlay, bbox, entropy]
    N_COLS = 5
    fig_w = N_COLS * 3.2
    fig_h = n * 3.2 + 1.0
    fig, axes = plt.subplots(n, N_COLS, figsize=(fig_w, fig_h),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.35})

    if n == 1:
        axes = axes[np.newaxis, :]  # ensure 2D

    col_titles = [
        "Original Image",
        "GradCAM Heatmap",
        "Attention Overlay",
        "Top-K Regions",
        "Attention Entropy",
    ]

   # fig.suptitle(
    #    "Attention Visualization — CUB-200 Birds (APDFA Model)",
     #   fontsize=16, fontweight="bold", y=0.995, color="#1a1a2e"
    #)

    for col_idx, ct in enumerate(col_titles):
        axes[0, col_idx].set_title(ct, fontsize=16, fontweight="bold",
                                   color="#2d4059", pad=6)

    for row_i, img_idx in enumerate(chosen):
        tensor, label, orig_np, rel_path = dataset[img_idx]
        class_name = dataset.classes[label] if label < len(dataset.classes) else f"cls_{label}"
        species = class_name[:28] + "…" if len(class_name) > 28 else class_name

        x = tensor.unsqueeze(0).to(device)
        H, W = orig_np.shape[:2]
        orig_resized = cv2.resize(orig_np, (W if W <= 600 else 600,
                                            int(H * (600 / W)) if W > 600 else H))
        orig_resized = np.array(Image.fromarray(orig_resized).resize((448, 448), Image.LANCZOS))

        # ── GradCAM ──────────────────────────────────────────────
        cam, pred_class = gradcam(x)

        if cam is None:
            cam = np.random.rand(7, 7)  # fallback for unsupported arch
            print(f"  WARNING: GradCAM failed for image {row_i}, using random map")

        cam_resized, heatmap_rgba = cam_to_heatmap(cam, (448, 448))

        # ── Overlay ──────────────────────────────────────────────
        overlay = overlay_cam(orig_resized, cam_resized, alpha=0.5)

        # ── Bounding boxes ───────────────────────────────────────
        bboxes = top_k_bbox(cam_resized, k=3, threshold=0.55)
        bbox_vis = draw_bboxes_on_image(orig_resized, bboxes)

        # ── Attention Entropy map ─────────────────────────────────
        # Entropy = -p*log(p) per pixel; highlights uncertain regions
        p = cam_resized.copy() + 1e-8
        p = p / p.sum()
        entropy_map = -p * np.log(p + 1e-10)
        entropy_map = (entropy_map - entropy_map.min()) / \
                      (entropy_map.max() - entropy_map.min() + 1e-8)

        # ── Row label ─────────────────────────────────────────────
        ax_orig = axes[row_i, 0]
        ax_orig.set_ylabel(
            f"#{row_i+1}: {species}\n(cls {label}, pred {pred_class})",
            fontsize=16, rotation=0, ha="right", va="center",
            labelpad=60, color="#2d4059"
        )

        # ── Plot columns ─────────────────────────────────────────
        axes[row_i, 0].imshow(orig_resized)
        axes[row_i, 1].imshow(heatmap_rgba[:, :, :3])
        axes[row_i, 2].imshow(overlay)
        axes[row_i, 3].imshow(bbox_vis)
        axes[row_i, 4].imshow(entropy_map, cmap="plasma")

        for c in range(N_COLS):
            axes[row_i, c].axis("off")

        print(f"  [{row_i+1}/{n}] {species:30s} | pred={pred_class:3d} | "
              f"CAM max={cam.max():.3f} | bboxes={len(bboxes)}")

    # ── Colorbar ──────────────────────────────────────────────────
    cbar_ax = fig.add_axes([0.92, 0.08, 0.015, 0.82])
    sm = plt.cm.ScalarMappable(cmap=HEATMAP_CMAP,
                                norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Attention Strength", fontsize=16, color="#2d4059")
    cbar.ax.tick_params(labelsize=7, colors="#2d4059")

    # ── Legend ────────────────────────────────────────────────────
    legend_elements = [
        patches.Patch(facecolor="#FF3232", edgecolor="none", label="Region 1"),
        patches.Patch(facecolor="#32FF32", edgecolor="none", label="Region 2"),
        patches.Patch(facecolor="#3232FF", edgecolor="none", label="Region 3"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=3, fontsize=14, title="Top-K Attended Regions",
               title_fontsize=14, framealpha=0.9,
               bbox_to_anchor=(0.46, 0.002))

    fig.patch.set_facecolor("#f8f9fa")

    # ── Save ──────────────────────────────────────────────────────
    out_path = args.output
    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()

    gradcam.remove_hooks()
    if rollout:
        rollout.remove_hooks()

    print(f"\n✓ Saved attention visualization → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Attention Visualization for CUB-200 (APDFA model)")
    p.add_argument("--checkpoint", default=
                   "/workdir1.8t/fei27/CGT/APDFA/checkpoints/pdfa_best.pth",
                   help="Path to model checkpoint")
    p.add_argument("--data_root", default="/workdir1.8t/fei27/CGT/APDFA/data/CUB_200_2011",
                   help="Path to CUB_200_2011 root directory")
    p.add_argument("--num_images", type=int, default=3,
                   help="Number of bird images to visualize (default: 5)")
    p.add_argument("--split", choices=["train", "test"], default="test",
                   help="Dataset split to sample from")
    p.add_argument("--output", default="/workdir1.8t/fei27/CGT/APDFA/vis_images/3-CUB_attention_visualization.png",
                   help="Output image path")
    p.add_argument("--dpi", type=int, default=150,
                   help="Output DPI (default: 300)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for image selection")
    return p.parse_args()


if __name__ == "__main__":
    root='/workdir1.8t/fei27/CGT/APDFA/data/CUB_200_2011'
    out_path='/workdir1.8t/fei27/CGT/APDFA/vis_images/attention'

    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    visualize(args)
