"""
train.py — SketchFill CVAE Training Loop
Trains the Class-Conditional VAE on QuickDraw sketch data.
Supports checkpointing, loss logging, early stopping, and loss curve plotting.

Usage:
    python models/train.py

Outputs:
    checkpoints/best_model.pt      — best validation loss checkpoint
    checkpoints/epoch_N.pt         — periodic checkpoints
    checkpoints/loss_log.csv       — per-epoch loss values
    checkpoints/loss_curves.png    — training curve plot
"""

import os
import sys
import csv
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Allow running from project root or models/ subdirectory
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.cvae_model import CVAE, cvae_loss

# ── Training Configuration ────────────────────────────────────────────────────
# All values are starting points — adjust freely based on training behaviour.
CONFIG = {
    "latent_dim":               128,        # 2x larger latent space
    "num_classes":              6,
    "image_size":               128,
    "batch_size":               64,         # Your GPU can handle this
    "learning_rate":            1e-4,       # Lower LR for better convergence
    "num_epochs":               150,
    "beta_max":                 1.0,        # Even stronger
    "beta_warmup_epochs":       80,         # Very slow warmup
    "free_bits":                0.5,        # Minimum KL per dimension (prevents collapse)
    "early_stopping_patience":  100,
    "checkpoint_every":         20,
    "val_split":                0.1,
    "num_workers":              0,
    "pin_memory":               True,
    "data_dir":                 str(PROJECT_ROOT / "data" / "processed"),
    "checkpoint_dir":           str(PROJECT_ROOT / "checkpoints"),
    "classes": ["car", "airplane", "cat", "dog", "mountain", "tree"],
}


# ── Dataset ───────────────────────────────────────────────────────────────────

class SketchDataset(Dataset):
    """
    Loads all QuickDraw class .npy files and returns (image, class_idx, class_onehot).
    """

    def __init__(self, data_dir: str, classes: list[str], split: str = "train", val_split: float = 0.1):
        self.classes     = classes
        self.num_classes = len(classes)
        self.split       = split

        images_list  = []
        labels_list  = []

        for idx, cls in enumerate(classes):
            path = Path(data_dir) / f"{cls}.npy"
            if not path.exists():
                raise FileNotFoundError(
                    f"Processed data not found: {path}\n"
                    f"Run 'python data/download_data.py' first."
                )
            arr = np.load(path)    # (N, 128, 128), float32, [0,1]
            n   = len(arr)

            # Deterministic train/val split
            cutoff = int(n * (1.0 - val_split))
            if split == "train":
                arr = arr[:cutoff]
            else:
                arr = arr[cutoff:]

            images_list.append(arr)
            labels_list.append(np.full(len(arr), idx, dtype=np.int64))

        self.images = np.concatenate(images_list, axis=0)   # (N_total, 128, 128)
        self.labels = np.concatenate(labels_list, axis=0)   # (N_total,)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        img   = torch.from_numpy(self.images[idx]).unsqueeze(0)  # (1, 128, 128)
        label = int(self.labels[idx])
        onehot = torch.zeros(self.num_classes)
        onehot[label] = 1.0
        return img, label, onehot


# ── Training Utilities ────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch: int, val_loss: float, path: str, config: dict) -> None:
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss":             val_loss,
        "config":               config,
    }, path)


def load_checkpoint(path: str, device: str) -> dict:
    return torch.load(path, map_location=device)


def plot_loss_curves(log_path: str, out_path: str) -> None:
    """Read loss_log.csv and save a matplotlib loss curve plot."""
    epochs, tr_total, tr_recon, tr_kl, va_total, va_recon, va_kl = ([] for _ in range(7))

    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            tr_total.append(float(row["train_total"]))
            tr_recon.append(float(row["train_recon"]))
            tr_kl.append(float(row["train_kl"]))
            va_total.append(float(row["val_total"]))
            va_recon.append(float(row["val_recon"]))
            va_kl.append(float(row["val_kl"]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (tr, va, title) in zip(
        axes,
        [(tr_total, va_total, "Total Loss"),
         (tr_recon, va_recon, "Reconstruction Loss (BCE)"),
         (tr_kl,    va_kl,    "KL Divergence")]
    ):
        ax.plot(epochs, tr, label="Train",      color="#2196F3")
        ax.plot(epochs, va, label="Validation", color="#FF5722", linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("SketchFill CVAE — Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Loss curves saved → {out_path}")


# ── Main Training Function ────────────────────────────────────────────────────

def train(config: dict = CONFIG) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    ckpt_dir = Path(config["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Datasets & DataLoaders ─────────────────────────────────────
    print("\nLoading datasets ...")
    train_ds = SketchDataset(config["data_dir"], config["classes"], split="train",  val_split=config["val_split"])
    val_ds   = SketchDataset(config["data_dir"], config["classes"], split="val",    val_split=config["val_split"])
    print(f"  Train samples: {len(train_ds)}")
    print(f"  Val   samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"] and device == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"] and device == "cuda",
    )

    # ── Model & Optimizer ─────────────────────────────────────────
    model = CVAE(
        latent_dim=config["latent_dim"],
        num_classes=config["num_classes"],
        image_size=config["image_size"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, verbose=True
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {total_params:,}")

    # ── Loss CSV ──────────────────────────────────────────────────
    log_path = ckpt_dir / "loss_log.csv"
    csv_file  = open(log_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_total", "train_recon", "train_kl",
                         "val_total",   "val_recon",   "val_kl"])

    # ── Training Loop ─────────────────────────────────────────────
    best_val_loss    = float("inf")
    patience_counter = 0
    start_time       = time.time()

    print(f"\nStarting training for up to {config['num_epochs']} epochs ...")
    print(f"KL warmup: 0 -> {config['beta_max']} over {config['beta_warmup_epochs']} epochs")
    print(f"Early stopping patience: {config['early_stopping_patience']} epochs\n")

    for epoch in range(1, config["num_epochs"] + 1):
        
        # ─ KL warmup schedule ────────────────────────────────────
        if epoch <= config["beta_warmup_epochs"]:
            current_beta = config["beta_max"] * (epoch / config["beta_warmup_epochs"])
        else:
            current_beta = config["beta_max"]

        # ─ Train phase ───────────────────────────────────────────
        model.train()
        tr_total_sum = tr_recon_sum = tr_kl_sum = 0.0
        tr_mu_sum = tr_logvar_sum = 0.0
        tr_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{config['num_epochs']}", leave=False)
        for imgs, _, onehots in pbar:
            imgs    = imgs.to(device)
            onehots = onehots.to(device)

            optimizer.zero_grad()
            recon, mu, logvar = model(imgs, onehots)
            total, recon_l, kl_l = cvae_loss(recon, imgs, mu, logvar, beta=current_beta, free_bits=config["free_bits"])
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            tr_total_sum += total.item()
            tr_recon_sum += recon_l.item()
            tr_kl_sum    += kl_l.item()
            tr_mu_sum    += mu.mean().item()
            tr_logvar_sum += logvar.mean().item()
            tr_count += 1

            pbar.set_postfix({"recon": f"{recon_l.item():.1f}", "kl": f"{kl_l.item():.1f}", "β": f"{current_beta:.3f}"})

        n_train_batches = len(train_loader)
        tr_total_avg = tr_total_sum / n_train_batches
        tr_recon_avg = tr_recon_sum / n_train_batches
        tr_kl_avg    = tr_kl_sum    / n_train_batches
        tr_mu_avg    = tr_mu_sum    / tr_count
        tr_logvar_avg = tr_logvar_sum / tr_count
        tr_std_avg   = np.exp(0.5 * tr_logvar_avg)

        # ─ Validation phase ──────────────────────────────────────
        model.eval()
        va_total_sum = va_recon_sum = va_kl_sum = 0.0

        with torch.no_grad():
            for imgs, _, onehots in val_loader:
                imgs    = imgs.to(device)
                onehots = onehots.to(device)
                recon, mu, logvar = model(imgs, onehots)
                total, recon_l, kl_l = cvae_loss(recon, imgs, mu, logvar, beta=current_beta, free_bits=config["free_bits"])
                va_total_sum += total.item()
                va_recon_sum += recon_l.item()
                va_kl_sum    += kl_l.item()

        n_val_batches = len(val_loader)
        va_total_avg = va_total_sum / n_val_batches
        va_recon_avg = va_recon_sum / n_val_batches
        va_kl_avg    = va_kl_sum    / n_val_batches

        # ─ Log ───────────────────────────────────────────────────
        csv_writer.writerow([epoch,
                             f"{tr_total_avg:.4f}", f"{tr_recon_avg:.4f}", f"{tr_kl_avg:.4f}",
                             f"{va_total_avg:.4f}", f"{va_recon_avg:.4f}", f"{va_kl_avg:.4f}"])
        csv_file.flush()

        elapsed = time.time() - start_time
        print(f"Epoch {epoch:03d} | β={current_beta:.3f} | "
              f"Train: {tr_total_avg:.1f} (recon={tr_recon_avg:.1f}, kl={tr_kl_avg:.1f}) | "
              f"Val: {va_total_avg:.1f} (recon={va_recon_avg:.1f}, kl={va_kl_avg:.1f}) | "
              f"μ={tr_mu_avg:.3f} σ={tr_std_avg:.3f} | "
              f"{elapsed/60:.1f}m")

        # ─ Scheduler step ────────────────────────────────────────
        scheduler.step(va_total_avg)

        # ─ Periodic checkpoint ───────────────────────────────────
        if epoch % config["checkpoint_every"] == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
            save_checkpoint(model, optimizer, epoch, va_total_avg, str(ckpt_path), config)
            print(f"  → Checkpoint saved: {ckpt_path}")

        # ─ Best model ────────────────────────────────────────────
        if va_total_avg < best_val_loss:
            best_val_loss    = va_total_avg
            patience_counter = 0
            best_path = ckpt_dir / "best_model.pt"
            save_checkpoint(model, optimizer, epoch, va_total_avg, str(best_path), config)
            print(f"  ✓ New best model (val_loss={best_val_loss:.4f}) → {best_path}")
        else:
            patience_counter += 1
            if patience_counter >= config["early_stopping_patience"]:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience_counter} epochs)")
                break

    # ── Post-training ─────────────────────────────────────────────
    csv_file.close()
    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time/60:.1f} minutes")
    print(f"Best val loss: {best_val_loss:.4f}")

    # Plot loss curves
    plot_loss_curves(str(log_path), str(ckpt_dir / "loss_curves.png"))

    # Save training summary with metrics
    summary = {
        "training_time_minutes": round(total_time / 60, 2),
        "total_epochs": epoch,
        "best_epoch": epoch if best_val_loss < float("inf") else None,
        "best_val_loss": round(best_val_loss, 4),
        "final_train_loss": round(tr_total_avg, 4),
        "config": config,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary_path = ckpt_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Training summary saved → {summary_path}")
    
    # Generate sample reconstructions for visual inspection
    print("\nGenerating sample reconstructions...")
    model.eval()
    with torch.no_grad():
        # Get one batch from validation
        val_batch = next(iter(val_loader))
        imgs, labels, onehots = val_batch
        imgs = imgs[:8].to(device)  # Take 8 samples
        onehots = onehots[:8].to(device)
        recons, _, _ = model(imgs, onehots)
        
        # Save side-by-side comparison
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        for i in range(8):
            # Original
            axes[0, i].imshow(imgs[i, 0].cpu(), cmap="gray")
            axes[0, i].axis("off")
            axes[0, i].set_title(f"{config['classes'][labels[i]]}", fontsize=9)
            # Reconstruction
            axes[1, i].imshow(recons[i, 0].cpu(), cmap="gray")
            axes[1, i].axis("off")
        axes[0, 0].set_ylabel("Original", fontsize=10)
        axes[1, 0].set_ylabel("Reconstructed", fontsize=10)
        plt.suptitle("Sample Reconstructions (Validation Set)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        recon_path = ckpt_dir / "sample_reconstructions.png"
        plt.savefig(recon_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Sample reconstructions saved → {recon_path}")


# ── Smoke Test (fast 2-epoch run on subset) ────────────────────────────────────

def run_smoke_test() -> None:
    """Quick 2-epoch test on 100 samples to verify the loop works end-to-end."""
    import tempfile
    from torch.utils.data import TensorDataset

    print("Running smoke test (2 epochs, synthetic data) ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Synthetic dataset
    N, C, H, W = 64, 1, 128, 128
    num_cls = 6
    imgs   = torch.rand(N, C, H, W)
    labels = torch.randint(0, num_cls, (N,))
    onehots = torch.zeros(N, num_cls).scatter_(1, labels.unsqueeze(1), 1.0)

    ds     = TensorDataset(imgs, labels, onehots)
    loader = DataLoader(ds, batch_size=16, shuffle=True)

    model  = CVAE(64, num_cls, H).to(device)
    optim  = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(2):
        model.train()
        for batch in loader:
            x, _, oh = batch
            x, oh = x.to(device), oh.to(device)
            optim.zero_grad()
            recon, mu, logvar = model(x, oh)
            loss, _, _ = cvae_loss(recon, x, mu, logvar)
            loss.backward()
            optim.step()
        print(f"  Smoke epoch {epoch+1}/2 loss={loss.item():.4f}")

    print("✓ Smoke test passed")


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        run_smoke_test()
    else:
        train()