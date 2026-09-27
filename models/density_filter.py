"""
density_filter.py - SketchFill Density-Based Latent Filtering
Fits a Kernel Density Estimator (KDE) over per-class latent codes from the
trained CVAE encoder, then uses it to filter generated samples - keeping only
latent points from high-density (well-supported) regions for better quality.

Usage (standalone - fits density model and generates test samples):
    python models/density_filter.py

Outputs:
    checkpoints/density_model.pkl - fitted KDE models (one per class)
    checkpoints/latent_codes.pkl - extracted per-class latent codes
    checkpoints/density_filter_test.png - 5 filtered "car" samples grid
"""

from __future__ import annotations

import sys
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.cvae_model import CVAE
from models.train import SketchDataset, CONFIG


# -- Latent Code Extraction ----------------------------------------------------

def extract_latent_codes(
    model: CVAE,
    dataloader: DataLoader,
    device: str = "cuda",
) -> dict[int, np.ndarray]:
    """
    Run all images through the encoder and collect per-class mu values.

    Returns:
        dict mapping class_idx -> np.array of shape (N_class, latent_dim)
    """
    model.eval()
    latent_dict: dict[int, list[np.ndarray]] = {}

    with torch.no_grad():
        for imgs, labels, onehots in tqdm(dataloader, desc="Extracting latent codes"):
            imgs    = imgs.to(device)
            onehots = onehots.to(device)
            mu, _   = model.encode(imgs, onehots)
            mu_np   = mu.cpu().numpy()

            for i, lbl in enumerate(labels.numpy()):
                if lbl not in latent_dict:
                    latent_dict[lbl] = []
                latent_dict[lbl].append(mu_np[i])

    # Concatenate per class
    return {cls: np.stack(codes, axis=0) for cls, codes in latent_dict.items()}


# -- KDE Fitting ---------------------------------------------------------------

def fit_and_save_density_model(
    latent_codes_dict: dict[int, np.ndarray],
    class_names: list[str],
    save_path: str,
) -> dict:
    """
    Fit a KernelDensity estimator per class over the latent mu values.
    Uses Scott's rule for bandwidth selection (automatic, data-adaptive).

    Saves: {'kde_models': {class_idx: kde}, 'class_names': [...]}
    """
    from sklearn.neighbors import KernelDensity

    kde_models: dict[int, KernelDensity] = {}

    print("\nFitting KDE density models per class:")
    for cls_idx, codes in sorted(latent_codes_dict.items()):
        kde = KernelDensity(kernel="gaussian", bandwidth="scott")
        kde.fit(codes)
        kde_models[cls_idx] = kde

        # Quick sanity: score the training data itself
        scores = kde.score_samples(codes)
        print(f"  [{class_names[cls_idx]:>10}] n={len(codes):5d}  "
              f"log-density: mean={scores.mean():.2f}, std={scores.std():.2f}")

    density_model = {
        "kde_models":   kde_models,
        "class_names":  class_names,
        "latent_dim":   next(iter(latent_codes_dict.values())).shape[1],
    }

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(density_model, f)
    print(f"\nDensity model saved ? {save_path}")
    return density_model


def load_density_model(path: str) -> dict:
    """Load a previously fitted density model from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


# -- Sample Generation ---------------------------------------------------------

def generate_samples(
    model: CVAE,
    class_label: int,
    num_samples: int,
    filtered: bool = True,
    density_model: dict | None = None,
    device: str = "cuda",
    oversample_factor: int = 5,
) -> np.ndarray:
    """
    Generate sketch images for a given class.

    Args:
        model:             trained CVAE
        class_label:       integer class index (0�5)
        num_samples:       how many images to return
        filtered:          if True, use KDE to keep high-density latent points only
        density_model:     dict from load_density_model(); required if filtered=True
        device:            'cuda' or 'cpu'
        oversample_factor: how many extra candidates to generate (5x by default)

    Returns:
        images: np.ndarray of shape (num_samples, 128, 128), values in [0, 1]
    """
    if filtered and density_model is None:
        raise ValueError("density_model must be provided when filtered=True")

    model.eval()
    num_classes = model.num_classes
    latent_dim  = model.latent_dim

    # Build class one-hot
    c_onehot = torch.zeros(1, num_classes, device=device)
    c_onehot[0, class_label] = 1.0

    with torch.no_grad():
        if filtered:
            # Sample a large pool of candidates
            n_candidates = num_samples * oversample_factor
            z_candidates = torch.randn(n_candidates, latent_dim, device=device)

            # Score by KDE
            kde    = density_model["kde_models"][class_label]
            z_np   = z_candidates.cpu().numpy()
            scores = kde.score_samples(z_np)

            # Take top-num_samples by density score
            top_indices = np.argsort(scores)[-num_samples:][::-1].copy()
            z_filtered  = z_candidates[torch.from_numpy(top_indices).to(device)]

            c_batch = c_onehot.expand(num_samples, -1)
            recon   = model.decode(z_filtered, c_batch)
        else:
            # Unfiltered: direct random sampling
            z       = torch.randn(num_samples, latent_dim, device=device)
            c_batch = c_onehot.expand(num_samples, -1)
            recon   = model.decode(z, c_batch)

    # Convert to numpy (H, W), values in [0,1]
    images = recon.cpu().numpy()[:, 0, :, :]   # (N, H, W)
    return images


# -- Grid Saver ----------------------------------------------------------------

def save_image_grid(images: np.ndarray, path: str, title: str = "") -> None:
    """Save a grid of generated images as a PNG."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    n = len(images)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(cols * 2, rows * 2 + 0.5))
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")

    gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.05, wspace=0.05)
    for i, img in enumerate(images):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        # Invert: white stroke on black ? black stroke on white (more natural)
        ax.imshow(1.0 - img, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Image grid saved ? {path}")


# -- Main (standalone test) ----------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fit density model and generate test samples")
    parser.add_argument("--fit-only", action="store_true", help="Only fit the density model, skip generation")
    args = parser.parse_args()

    device      = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir    = PROJECT_ROOT / "checkpoints"
    best_model  = ckpt_dir / "best_model.pt"
    density_pkl = ckpt_dir / "density_model.pkl"
    latent_pkl  = ckpt_dir / "latent_codes.pkl"

    if not best_model.exists():
        print(f"ERROR: {best_model} not found. Run 'python models/train.py' first.")
        sys.exit(1)

    # -- Load model -----------------------------------------------
    print(f"Loading model from {best_model} ...")
    ckpt   = torch.load(str(best_model), map_location=device)
    config = ckpt.get("config", CONFIG)

    model = CVAE(
        latent_dim=config["latent_dim"],
        num_classes=config["num_classes"],
        image_size=config["image_size"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Model loaded (epoch {ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', '?'):.4f})")

    classes = config.get("classes", CONFIG["classes"])

    # -- Extract latent codes -------------------------------------
    if latent_pkl.exists():
        print(f"\nLoading cached latent codes from {latent_pkl} ...")
        with open(latent_pkl, "rb") as f:
            latent_codes = pickle.load(f)
    else:
        print("\nExtracting latent codes from training data ...")
        train_ds = SketchDataset(
            config["data_dir"], classes, split="train", val_split=config["val_split"]
        )
        loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=0)
        latent_codes = extract_latent_codes(model, loader, device)

        with open(latent_pkl, "wb") as f:
            pickle.dump(latent_codes, f)
        print(f"Latent codes saved ? {latent_pkl}")

    # -- Fit density model ----------------------------------------
    if density_pkl.exists():
        print(f"\nLoading existing density model from {density_pkl} ...")
        density_model = load_density_model(str(density_pkl))
    else:
        density_model = fit_and_save_density_model(latent_codes, classes, str(density_pkl))

    if args.fit_only:
        print("\nDone (--fit-only mode).")
        sys.exit(0)

    # -- Generate test samples ------------------------------------
    print("\nGenerating 5 filtered samples for class 'car' ...")
    car_idx  = classes.index("car")
    filtered = generate_samples(model, car_idx, num_samples=5,
                                filtered=True, density_model=density_model, device=device)
    assert filtered.shape == (5, 128, 128), f"Shape error: {filtered.shape}"
    assert filtered.min() >= 0.0 and filtered.max() <= 1.0, "Value range error"

    out_path = str(ckpt_dir / "density_filter_test.png")
    save_image_grid(filtered, out_path, title="Density-Filtered 'car' Samples")

    # Also generate unfiltered for comparison
    unfiltered = generate_samples(model, car_idx, num_samples=5,
                                  filtered=False, density_model=density_model, device=device)
    save_image_grid(unfiltered, str(ckpt_dir / "density_filter_unfiltered_test.png"),
                    title="Unfiltered 'car' Samples")

    print("\n? Density filtering module OK")
    print("  Check checkpoints/density_filter_test.png for visual verification")
