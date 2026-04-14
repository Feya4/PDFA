""" # train.py
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os

from config import config
from models.pdfa import PDFA
from data.dataset import FewShotDataset, episodic_sampler, get_transforms
from utils.utils import compute_alignment_loss, accuracy


def train():
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Training PDFA on device: {device}")

    # ==================== Datasets & Transforms ====================
    # Use image_size=32 for CIFAR-FS (original resolution is 32x32)
    transform = get_transforms(dataset_name="cifar_fs", image_size=224)

    train_dataset = FewShotDataset(
        root=config.data_root,
        dataset_name="cifar_fs",
        split="train",
        transform=transform
    )

    val_dataset = FewShotDataset(
        root=config.data_root,
        dataset_name="cifar_fs",
        split="val",
        transform=transform
    )

    # ==================== Model & Optimizer ====================
    model = PDFA(config).to(device)
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_val_acc = 0.0
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("Starting episodic training of PDFA...\n")

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_episodes = config.num_episodes_per_epoch if hasattr(config, 'num_episodes_per_epoch') else 100

        pbar = tqdm(range(num_episodes), desc=f"Epoch {epoch+1}/{config.epochs}")

        for _ in pbar:
            # Sample one episode
            support_imgs, query_imgs, support_labels, class_names, query_labels = episodic_sampler(
                train_dataset,
                n_way=config.n_way,
                k_shot=config.k_shot,
                q_query=config.q_query
            )

            support_imgs = support_imgs.to(device)
            query_imgs = query_imgs.to(device)
            support_labels = support_labels.to(device)
            query_labels = query_labels.to(device)

            # Forward pass with intermediates for L_w
            logits, X_i, V_bar_i, e_i = model(
                support_imgs, 
                query_imgs, 
                support_labels, 
                class_names,
                return_intermediates=True
            )

            # Classification loss
            loss_ce = F.cross_entropy(logits, query_labels)

            # Cross-modal alignment loss
            loss_w = compute_alignment_loss(X_i, V_bar_i, e_i)

            # Total loss
            loss = loss_ce + config.lambda_w * loss_w

            # Backward & update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            acc = accuracy(logits, query_labels)

            epoch_loss += loss.item()
            epoch_acc += acc

            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'CE': f"{loss_ce.item():.4f}",
                'L_w': f"{loss_w.item():.4f}",
                'Acc': f"{acc*100:.2f}%"
            })

        scheduler.step()

        avg_loss = epoch_loss / num_episodes
        avg_acc = epoch_acc / num_episodes

        print(f"Epoch {epoch+1:3d} | Avg Loss: {avg_loss:.4f} | Train Acc: {avg_acc*100:.2f}%")

        # Validation every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == config.epochs - 1:
            val_acc = evaluate(model, val_dataset, device, num_episodes=200)
            print(f"Validation {config.k_shot}-shot Acc: {val_acc*100:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, os.path.join(checkpoint_dir, "pdfa_best.pth"))
                print(f"→ Best model saved! Val Acc: {val_acc*100:.2f}%")

    print(f"\nTraining completed. Best validation accuracy: {best_val_acc*100:.2f}%")


# ====================== Evaluation Function ======================
@torch.no_grad()
def evaluate(model, dataset, device, num_episodes=200):
    model.eval()
    total_acc = 0.0

    for _ in range(num_episodes):
        support_imgs, query_imgs, support_labels, class_names, query_labels = episodic_sampler(
            dataset,
            n_way=config.n_way,
            k_shot=config.k_shot,
            q_query=config.q_query
        )

        support_imgs = support_imgs.to(device)
        query_imgs = query_imgs.to(device)
        support_labels = support_labels.to(device)
        query_labels = query_labels.to(device)

        logits, _, _, _ = model(
            support_imgs, 
            query_imgs, 
            support_labels, 
            class_names,
            return_intermediates=True
        )
       
        acc = accuracy(logits, query_labels)
        total_acc += acc

    return total_acc / num_episodes


if __name__ == "__main__":
    train() """
# train.py
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import json

from config import config
from models.pdfa import PDFA

from data.dataset import FewShotDataset, episodic_sampler, get_transforms
from utils.utils import compute_alignment_loss, accuracy
@torch.no_grad()
def visualize_attention_map(model, dataset, device, save_path="attention_map_final.png", num_samples=1):
    """
    Visualize and save an attention map at the end of training.
    Adapts to your PDFA model's intermediates (X_i, V_bar_i, e_i).
    """
    model.eval()
    
    # Sample one episode
    support_imgs, query_imgs, support_labels, class_names, query_labels = episodic_sampler(
        dataset, config.n_way, config.k_shot, config.q_query
    )
    
    support_imgs = support_imgs.to(device)
    query_imgs = query_imgs.to(device)
    support_labels = support_labels.to(device)
    
    # Forward with intermediates
    logits, X_i, V_bar_i, e_i = model(
        support_imgs, query_imgs, support_labels, class_names, return_intermediates=True
    )
    
    # === Example visualization using your alignment intermediates ===
    # You can replace this section with raw attention weights from your ViT encoder if available
    # For now, we create a simple heatmap from the alignment loss components (e.g., similarity between X_i and V_bar_i)
    
    # Compute a representative "attention-like" map (e.g., cosine similarity or alignment score per patch)
    # Assuming X_i and V_bar_i have shape [batch, seq_len, dim] or similar
    if X_i is not None and V_bar_i is not None:
        # Normalize and compute similarity (example: mean over batch and heads if multi-head)
        X_norm = F.normalize(X_i.mean(dim=0) if X_i.dim() > 2 else X_i, dim=-1)
        V_norm = F.normalize(V_bar_i.mean(dim=0) if V_bar_i.dim() > 2 else V_bar_i, dim=-1)
        
        attn_scores = torch.matmul(X_norm, V_norm.T)  # similarity matrix
        attn_map = attn_scores.softmax(dim=-1).mean(dim=0).cpu().numpy()  # average and to numpy
        
        # Reshape to 2D if it's patch-based (adjust patch size according to your ViT, e.g., 14x14 for 224/16)
        if len(attn_map.shape) == 1:
            patch_size = int(np.sqrt(len(attn_map)))  # assuming square grid
            if patch_size * patch_size == len(attn_map):
                attn_map = attn_map.reshape(patch_size, patch_size)
            else:
                attn_map = attn_map.reshape(-1, 1)  # fallback
        
        # Plot
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original query image (first one)
        img_to_show = query_imgs[0].cpu().permute(1, 2, 0).numpy()
        img_to_show = (img_to_show - img_to_show.min()) / (img_to_show.max() - img_to_show.min() + 1e-8)
        axs[0].imshow(img_to_show)
        axs[0].set_title("Query Image (Sample)")
        axs[0].axis('off')
        
        # Attention map
        im = axs[1].imshow(attn_map, cmap='viridis')
        axs[1].set_title("Attention / Alignment Map")
        axs[1].axis('off')
        fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
        
        plt.suptitle("Final Attention Map Visualization (PDFA)", fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"→ Attention map saved to: {save_path}")
    else:
        print("Warning: No suitable intermediates returned for attention visualization.")
def train():
    import os

    # Define the absolute path
    checkpoint_dir = "/workdir1.8t/fei27/CGT/APDFA/checkpoints"

    # Create the directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Checkpoint directory ready at: {checkpoint_dir}")
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Training PDFA on device: {device}")

    # ==================== Datasets & Transforms ====================
    # Mandatory 224 for CLIP ViT architectures
    
    # Example usage
    transform = get_transforms(image_size=224)

    train_dataset = FewShotDataset(
        root='/workdir1.8t/fei27/CGT/APDFA/data/miniImageNet',
        split='train',
        transform=transform
    )

    # For testing
    test_dataset = FewShotDataset(
        root='/workdir1.8t/fei27/CGT/APDFA/data/miniImageNet',
        split='test',
        transform=transform   # usually no RandomFlip for test, but ok for now
    )

    """ transform = get_transforms(dataset_name="miniImageNet", image_size=224)

    train_dataset = FewShotDataset(
        root=config.data_root,
        dataset_name="miniImageNet",
        split="train",
        transform=transform
    )

    val_dataset = FewShotDataset(
        root=config.data_root,
        dataset_name="miniImageNet",
        split="val",           # or "test"
        transform=transform
    ) """

    # ==================== Model & Optimizer ====================
    model = PDFA(config).to(device)
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # Metrics History for Journal Visualization
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'loss_ce': [],
        'loss_w': []
    }

    best_val_acc = 0.0
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Starting episodic training: {config.n_way}-way {config.k_shot}-shot\n")

    for epoch in range(config.epochs):
        model.train()
        epoch_loss, epoch_acc = 0.0, 0.0
        epoch_ce, epoch_lw = 0.0, 0.0
        
        num_episodes = config.num_episodes_per_epoch if hasattr(config, 'num_episodes_per_epoch') else 100
        pbar = tqdm(range(num_episodes), desc=f"Epoch {epoch+1}/{config.epochs}")

        for _ in pbar:
            # Sample Episode
            support_imgs, query_imgs, support_labels, class_names, query_labels = episodic_sampler(
                train_dataset,
                n_way=config.n_way,
                k_shot=config.k_shot,
                q_query=config.q_query
            )

            support_imgs, query_imgs = support_imgs.to(device), query_imgs.to(device)
            support_labels, query_labels = support_labels.to(device), query_labels.to(device)

            # Forward pass
            logits, X_i, V_bar_i, e_i = model(
                support_imgs, query_imgs, support_labels, class_names, return_intermediates=True
            )

            # Dual-Loss Computation
            loss_ce = F.cross_entropy(logits, query_labels)
            loss_w = compute_alignment_loss(X_i, V_bar_i, e_i)
            loss = loss_ce + config.lambda_w * loss_w

            # Optimization
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability in multi-modal fusion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Batch Metrics
            acc = accuracy(logits, query_labels)
            epoch_loss += loss.item()
            epoch_acc += acc
            epoch_ce += loss_ce.item()
            epoch_lw += loss_w.item()

            pbar.set_postfix({'L': f"{loss.item():.3f}", 'Acc': f"{acc*100:.1f}%"})

        scheduler.step()

        # Log Epoch Metrics
        avg_loss = epoch_loss / num_episodes
        avg_acc = epoch_acc / num_episodes
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(avg_acc)
        history['loss_ce'].append(epoch_ce / num_episodes)
        history['loss_w'].append(epoch_lw / num_episodes)

        print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Train Acc: {avg_acc*100:.2f}%")

        # Validation logic
        # Validation logic (inside your epoch loop)
        if (epoch + 1) % 5 == 0 or epoch == config.epochs - 1:
            val_acc = evaluate(model, test_dataset, device, num_episodes=200)
            history['val_acc'].append({'epoch': epoch + 1, 'acc': val_acc})
            print(f"→ Validation Acc: {val_acc*100:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                
                # 1. Define the absolute path clearly
                checkpoint_dir = "/workdir1.8t/fei27/CGT/APDFA/checkpoints"
                os.makedirs(checkpoint_dir, exist_ok=True)
                save_path = os.path.join(checkpoint_dir, "pdfa_best.pth")
                
                # 2. Save the full experimental state
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'history': history,    # Essential for your convergence plots
                    'val_acc': val_acc,
                    'config': vars(config) if not isinstance(config, dict) else config, # Hyperparameter tracking
                }, save_path)
                
                print(f"→ Best Checkpoint Saved at: {save_path}")

    # 3. Final safety save for the history JSON (after the epoch loop ends)
    history_json_path = os.path.join(checkpoint_dir, "history.json")
    with open(history_json_path, "w") as f:
        json.dump(history, f, indent=4) # indent=4 makes it readable for reviewers
    print(f"→ Training history exported to {history_json_path}")
    print("\nGenerating final attention map visualization...")
    
    # Use absolute checkpoint dir for consistency
    checkpoint_dir = "/workdir1.8t/fei27/CGT/APDFA/checkpoints"
    attn_save_path = os.path.join(checkpoint_dir, "attention_map_final.png")
    
    visualize_attention_map(model, test_dataset, device, save_path=attn_save_path)
    
    print("Training completed successfully!")

@torch.no_grad()
def evaluate(model, dataset, device, num_episodes=200):
    model.eval()
    total_acc = 0.0
    for _ in range(num_episodes):
        support_imgs, query_imgs, support_labels, class_names, query_labels = episodic_sampler(
            dataset, config.n_way, config.k_shot, config.q_query
        )
        logits = model(support_imgs.to(device), query_imgs.to(device), 
                       support_labels.to(device), class_names)
        total_acc += accuracy(logits, query_labels)
    return total_acc / num_episodes

if __name__ == "__main__":
    train()
