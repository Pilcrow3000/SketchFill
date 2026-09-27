"""
download_data.py — SketchFill QuickDraw Data Downloader & Preprocessor
Downloads 6 QuickDraw categories as 28x28 .npy bitmaps from Google Cloud Storage,
upscales to 128x128 via bicubic interpolation, normalizes to [0,1], and saves
processed arrays alongside a metadata.json class mapping.

Usage:
    python data/download_data.py

Output:
    data/raw/<class>.npy       — raw 28x28 bitmaps (uint8)
    data/processed/<class>.npy — processed 128x128 float32 arrays, shape (N, 128, 128)
    data/processed/metadata.json
"""

import os
import json
import numpy as np
import requests
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────
NUM_SAMPLES_PER_CLASS = 5000   # how many drawings to keep per category
IMAGE_SIZE = 128               # output image resolution
CLASSES = ["car", "airplane", "cat", "dog", "mountain", "tree"]
BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
PROC_DIR     = PROJECT_ROOT / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)


def download_class(class_name: str) -> Path:
    """Download the raw .npy bitmap file for one QuickDraw class."""
    url      = f"{BASE_URL}/{class_name}.npy"
    out_path = RAW_DIR / f"{class_name}.npy"

    if out_path.exists():
        print(f"  [skip] {class_name}.npy already downloaded")
        return out_path

    print(f"  Downloading {class_name} from {url} ...")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(out_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=f"  {class_name}", leave=False
    ) as bar:
        for chunk in response.iter_content(chunk_size=65536):
            f.write(chunk)
            bar.update(len(chunk))

    return out_path


def resize_to_128(img_28: np.ndarray) -> np.ndarray:
    """Resize a single (28,28) uint8 image to (128,128) float32 in [0,1]."""
    pil_img = Image.fromarray(img_28.astype(np.uint8), mode="L")
    pil_img = pil_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    return np.array(pil_img, dtype=np.float32) / 255.0


def process_class(class_name: str, class_idx: int) -> int:
    """Load raw npy, take first NUM_SAMPLES_PER_CLASS, upscale, save processed."""
    raw_path  = RAW_DIR / f"{class_name}.npy"
    proc_path = PROC_DIR / f"{class_name}.npy"

    if proc_path.exists():
        existing = np.load(proc_path)
        print(f"  [skip] {class_name} already processed — {existing.shape[0]} samples")
        return existing.shape[0]

    print(f"  Processing {class_name} ...")
    raw = np.load(raw_path)                        # shape: (N_total, 784)
    raw = raw[:NUM_SAMPLES_PER_CLASS]              # trim to desired count
    n   = len(raw)

    processed = np.zeros((n, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    for i, flat in enumerate(tqdm(raw, desc=f"  resize {class_name}", leave=False)):
        img_28 = flat.reshape(28, 28)
        processed[i] = resize_to_128(img_28)

    # Validate
    assert processed.shape == (n, IMAGE_SIZE, IMAGE_SIZE), \
        f"Shape mismatch: {processed.shape}"
    assert processed.min() >= 0.0 and processed.max() <= 1.0, \
        f"Value range error: [{processed.min():.4f}, {processed.max():.4f}]"

    np.save(proc_path, processed)
    print(f"  Saved {n} samples → {proc_path}")
    return n


def save_metadata(class_counts: dict[str, int]) -> None:
    """Save class→index mapping and dataset statistics."""
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    meta = {
        "classes":               CLASSES,
        "class_to_idx":          class_to_idx,
        "num_samples_per_class": NUM_SAMPLES_PER_CLASS,
        "image_size":            IMAGE_SIZE,
        "sample_counts":         class_counts,
    }
    meta_path = PROC_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata saved → {meta_path}")


def main() -> None:
    print("=" * 60)
    print("  SketchFill — QuickDraw Data Download & Preprocessing")
    print("=" * 60)

    # ── Step 1: Download ──────────────────────────────────────────
    print("\n[1/2] Downloading raw .npy files ...")
    for cls in CLASSES:
        download_class(cls)

    # ── Step 2: Process ───────────────────────────────────────────
    print("\n[2/2] Upscaling to 128×128 and normalising ...")
    counts: dict[str, int] = {}
    for idx, cls in enumerate(CLASSES):
        counts[cls] = process_class(cls, idx)

    # ── Summary ───────────────────────────────────────────────────
    save_metadata(counts)

    print("\n" + "=" * 60)
    print("  Dataset Summary")
    print("=" * 60)
    print(f"  {'Class':<12} {'Idx':>4}  {'Samples':>8}")
    print(f"  {'-'*12} {'----':>4}  {'-------':>8}")
    for cls, cnt in counts.items():
        print(f"  {cls:<12} {CLASSES.index(cls):>4}  {cnt:>8}")
    total = sum(counts.values())
    print(f"  {'TOTAL':<12} {'':>4}  {total:>8}")
    print("=" * 60)
    print("\nAll done! Processed data is in data/processed/")


if __name__ == "__main__":
    main()