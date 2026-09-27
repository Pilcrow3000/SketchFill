"""
demo_interpolation.py - Create impressive interpolation visualizations
Generates smooth transitions between different sketch classes.
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.cvae_model import CVAE
from models.train import SketchDataset

def create_interpolation_demo():
    """Create interpolation visualizations for all class pairs."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = PROJECT_ROOT / "checkpoints" / "best_model.pt"
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    config = checkpoint["config"]
    
    model = CVAE(
        latent_dim=config["latent_dim"],
        num_classes=config["num_classes"],
        image_size=config["image_size"],
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    train_ds = SketchDataset(
        config["data_dir"], 
        config["classes"], 
        split="train", 
        val_split=config["val_split"]
    )
    
    classes = config["classes"]
    steps = 10
    
    print("Creating interpolation demonstrations...")
    
    # Interesting pairs
    pairs = [
        (0, 1),  # car → airplane
        (2, 3),  # cat → dog
        (4, 5),  # mountain → tree
    ]
    
    for pair_idx, (cls_a_idx, cls_b_idx) in enumerate(pairs):
        cls_a = classes[cls_a_idx]
        cls_b = classes[cls_b_idx]
        
        print(f"\n{cls_a} → {cls_b}")
        
        # Get examples
        img_a, img_b = None, None
        onehot_a, onehot_b = None, None
        
        for i in range(len(train_ds)):
            img, label, onehot = train_ds[i]
            if label == cls_a_idx and img_a is None:
                img_a = img
                onehot_a = onehot
            if label == cls_b_idx and img_b is None:
                img_b = img
                onehot_b = onehot
            if img_a is not None and img_b is not None:
                break
        
        with torch.no_grad():
            img_a_batch = img_a.unsqueeze(0).to(device)
            img_b_batch = img_b.unsqueeze(0).to(device)
            onehot_a_batch = onehot_a.unsqueeze(0).to(device)
            onehot_b_batch = onehot_b.unsqueeze(0).to(device)
            
            mu_a, _ = model.encode(img_a_batch, onehot_a_batch)
            mu_b, _ = model.encode(img_b_batch, onehot_b_batch)
            
            # Interpolate
            interpolated = []
            for alpha in np.linspace(0, 1, steps):
                mu_interp = (1 - alpha) * mu_a + alpha * mu_b
                onehot_interp = (1 - alpha) * onehot_a_batch + alpha * onehot_b_batch
                
                recon = model.decode(mu_interp, onehot_interp)
                interpolated.append(recon[0, 0].cpu().numpy())
        
        # Visualize
        fig = plt.figure(figsize=(20, 3))
        gs = gridspec.GridSpec(1, steps, hspace=0.05, wspace=0.05)
        
        for i, img in enumerate(interpolated):
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            
            # Label
            alpha = i / (steps - 1)
            if i == 0:
                ax.set_title(cls_a, fontsize=12, fontweight="bold")
            elif i == steps - 1:
                ax.set_title(cls_b, fontsize=12, fontweight="bold")
            else:
                ax.set_title(f"{alpha:.1f}", fontsize=10)
        
        fig.suptitle(f"Latent Space Interpolation: {cls_a.title()} → {cls_b.title()}", 
                     fontsize=14, fontweight="bold")
        
        out_path = PROJECT_ROOT / "checkpoints" / f"interpolation_{cls_a}_{cls_b}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"  ✓ Saved: {out_path}")
    
    print("\n" + "="*70)
    print("All interpolations created!")
    print("Check checkpoints/ folder for results")
    print("="*70)

if __name__ == "__main__":
    create_interpolation_demo()
