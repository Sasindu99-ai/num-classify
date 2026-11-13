"""
Feature Space 6 (FS6) Preprocessing Pipeline:
Contour (edge) images -> 14 Concentric-Ring Features

This script implements the complete, reproducible, end-to-end preprocessing pipeline
for Feature Space FS6 on MNIST handwritten digits.

Execution from project root:
    python src/FS6/0_preprocessing_pipeline.py --copies 2 --seed 42

Numbered Pipeline Stages:
    1. LOAD + SPLIT
       - Load raw MNIST training CSV (60,000 images, 28x28 grayscale, 0-255).
       - Stratified split: 90% train (54,000), 10% validation (6,000), seed=42.
       - Track original row index as `source_idx`.
    2. AUGMENTATION (Training split only)
       - Retain original images (is_augmented=0) and generate random copies (is_augmented=1).
       - Realistic, digit-preserving affine transforms (rotation, scale, shear, translation).
       - Compose into a single 2D affine matrix; single bilinear interpolation via Pillow.
    3. CONTOUR EXTRACTION (All splits, post-augmentation)
       - Fixed threshold (100) binarization.
       - Morphological erosion via MinFilter(3) by 1 pixel.
       - Outline extraction via subtraction (mask - inner), producing 0/1 binary contours.
    4. CONCENTRIC RING FEATURES (14 features)
       - Pixel-to-ring mapping: ring(r, c) = min(r, c, 27-r, 27-c) for r,c in [0..27].
       - Ring sizes: 108, 100, 92, ..., 4 (sum = 784).
       - Mean contour intensity per ring, computed via vectorized matrix projection.
    5. NORMALISATION
       - Unscaled ring means in [0, 1].
       - Standardized ring means via StandardScaler fitted STRICTLY on training split.
       - Zero-variance handling (outer rings) ensuring finite values (0.0).
    6. SAVE
       - Saved to `data/contour_concentric_rings/`.
       - NPZ image archives, unscaled CSVs, standardized CSVs, scaler.joblib, config.json, README.md.
       - Optional `--include-test` flag to transform test set without re-fitting.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageChops, ImageFilter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Resolve Project Root: two levels above this script (src/FS6/ -> src/ -> root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ==============================================================================
# SECTION 1: LOAD + SPLIT
# ==============================================================================

def load_raw_train_data(
    raw_train_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load raw MNIST training CSV.

    Structure of mnist_train.csv:
      - 60,000 rows, 785 columns.
      - Column 0: 'label' (integer digit 0-9).
      - Columns 1-784: pixel values '1x1' .. '28x28' (0-255 uint8).

    Returns:
      images: Array of shape (60000, 28, 28) with dtype uint8.
      labels: Array of shape (60000,) with dtype int64.
      source_idx: Array of shape (60000,) with original row index (0..59999).
    """
    if not raw_train_path.is_file():
        raise FileNotFoundError(f"Raw training file not found at: {raw_train_path}")

    print(f"[1/6] Loading raw training CSV from: {raw_train_path}")
    t0 = time.time()
    df = pd.read_csv(raw_train_path)

    # Validate schema
    if df.shape != (60000, 785):
        raise ValueError(
            f"Expected shape (60000, 785) for MNIST train CSV, got {df.shape}"
        )

    labels = df.iloc[:, 0].to_numpy(dtype=np.int64)
    pixels = df.iloc[:, 1:].to_numpy(dtype=np.uint8)
    images = pixels.reshape(-1, 28, 28)
    source_idx = np.arange(len(df), dtype=np.int64)

    print(f"      Loaded {len(df):,} images in {time.time() - t0:.2f}s "
          f"(min pixel={pixels.min()}, max pixel={pixels.max()}, classes={np.unique(labels)})")
    return images, labels, source_idx


def split_data(
    images: np.ndarray,
    labels: np.ndarray,
    source_idx: np.ndarray,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Perform stratified split into Train (90% = 54,000) and Validation (10% = 6,000).
    Preserves original row index in source_idx for traceability and leakage verification.
    """
    print(f"[1/6] Performing stratified split (val_size={val_size}, random_state={random_state})...")
    (
        train_img,
        val_img,
        train_lbl,
        val_lbl,
        train_src,
        val_src,
    ) = train_test_split(
        images,
        labels,
        source_idx,
        test_size=val_size,
        stratify=labels,
        random_state=random_state,
    )

    print(f"      Split complete: {len(train_img):,} train rows, {len(val_img):,} validation rows.")
    return (train_img, train_lbl, train_src), (val_img, val_lbl, val_src)


# ==============================================================================
# SECTION 2: AUGMENTATION (Training split only)
# ==============================================================================

# ------------------------------------------------------------------------------
# WHY FORBIDDEN TRANSFORMS ARE STRICTLY AVOIDED:
# 1. Horizontal / Vertical Flips:
#    Digits are directional glyphs. A horizontal flip turns a '3' or '7' into
#    non-digit garbage or invalid symbols. A vertical flip turns '6' into '9'
#    or '9' into '6', creating false ground-truth labels and ruining the model.
# 2. Large Rotations (> 15 degrees, e.g. 90 or 180 degrees):
#    Handwriting variation typically does not exceed +/- 10 degrees. Large rotations
#    lead to label ambiguity (e.g., rotating '6' by 180 degrees yields '9').
# 3. Aggressive Crops / Cutouts / Random Erasing:
#    MNIST digits are compact (28x28). Cutting holes can sever loops (turning '8'
#    into '3' or '0' into 'C'/'U'), destroying topological information essential
#    for contour and ring analysis.
# 4. Intensity Inversion:
#    MNIST digits consist of white ink on black background. Inverting pixel intensities
#    destroys the background semantics and breaks downstream thresholding (fixed at 100).
# 5. Strong Brightness / Contrast Alterations:
#    Heavy illumination shifts cause thin strokes to disappear under a fixed threshold
#    or create artificial edge noise across the bounding boundary.
# ------------------------------------------------------------------------------

def compute_inverse_affine_params(
    rotation_deg: float,
    scale: float,
    shear_x: float,
    tx: float,
    ty: float,
    cx: float = 13.5,
    cy: float = 13.5,
) -> Tuple[float, float, float, float, float, float]:
    """
    Build a single affine transformation matrix about the image center (cx, cy)
    combining:
      1. Center translation: (x - cx, y - cy)
      2. Horizontal shear: x' = x + shear_x * y
      3. Isotropic scale: x' = scale * x, y' = scale * y
      4. Rotation: standard 2D rotation matrix by angle rotation_deg
      5. Translation back: (x' + cx + tx, y' + cy + ty)

    Pillow's Image.transform with Image.AFFINE expects the INVERSE mapping:
      x_in = a * x_out + b * y_out + c
      y_in = d * x_out + e * y_out + f

    We compute the forward 3x3 homogeneous matrix M, invert it analytically,
    and return the 6 coefficients (a, b, c, d, e, f).
    """
    theta = np.deg2rad(rotation_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # 1. Translate center to origin
    T_to_orig = np.array([
        [1.0, 0.0, -cx],
        [0.0, 1.0, -cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # 2. Linear transforms: Shear, Scale, Rotation
    # H: horizontal shear
    H = np.array([
        [1.0, float(shear_x), 0.0],
        [0.0, 1.0,            0.0],
        [0.0, 0.0,            1.0],
    ], dtype=np.float64)

    # S: isotropic scale
    S = np.array([
        [float(scale), 0.0,          0.0],
        [0.0,          float(scale), 0.0],
        [0.0,          0.0,          1.0],
    ], dtype=np.float64)

    # R: 2D rotation
    R = np.array([
        [cos_t, -sin_t, 0.0],
        [sin_t,  cos_t, 0.0],
        [0.0,    0.0,   1.0],
    ], dtype=np.float64)

    # 3. Translate back to center and add translation (tx, ty)
    T_from_orig = np.array([
        [1.0, 0.0, cx + float(tx)],
        [0.0, 1.0, cy + float(ty)],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # Forward composite matrix M
    M_forward = T_from_orig @ R @ S @ H @ T_to_orig

    # Inverse mapping for Pillow
    M_inv = np.linalg.inv(M_forward)
    return (
        float(M_inv[0, 0]),
        float(M_inv[0, 1]),
        float(M_inv[0, 2]),
        float(M_inv[1, 0]),
        float(M_inv[1, 1]),
        float(M_inv[1, 2]),
    )


def augment_training_data(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    train_source_idx: np.ndarray,
    copies: int = 2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Augment training data by adding `copies` realistic random transforms per sample.
    Original samples are preserved with is_augmented=0.
    Augmented samples have is_augmented=1, matching label and source_idx.

    Transform parameter ranges (sampled independently per copy):
      - rotation:    uniform in [-10.0, +10.0] degrees
      - scale:       uniform in [0.90, 1.10]
      - shear:       horizontal, uniform in [-0.20, +0.20]
      - translation: integer pixels, uniform in [-2, +2] (x and y)

    Total output rows = N_train * (1 + copies). For N_train=54000, copies=2 -> 162,000.
    """
    n_train = len(train_images)
    total_samples = n_train * (1 + copies)
    print(f"[2/6] Augmenting training split: {n_train:,} originals + "
          f"{n_train * copies:,} copies ({copies} per sample) -> {total_samples:,} total rows...")
    t0 = time.time()

    rng = np.random.default_rng(seed)

    # Pre-allocate output arrays
    out_images = np.empty((total_samples, 28, 28), dtype=np.uint8)
    out_labels = np.empty(total_samples, dtype=np.int64)
    out_source_idx = np.empty(total_samples, dtype=np.int64)
    out_is_augmented = np.empty(total_samples, dtype=np.uint8)

    # 1. Fill originals first
    out_images[:n_train] = train_images
    out_labels[:n_train] = train_labels
    out_source_idx[:n_train] = train_source_idx
    out_is_augmented[:n_train] = 0

    if copies == 0:
        return out_images, out_labels, out_source_idx, out_is_augmented

    # 2. Generate augmented copies
    # Sample all parameters upfront for reproducible speed
    n_aug = n_train * copies
    rotations = rng.uniform(-10.0, 10.0, size=n_aug)
    scales = rng.uniform(0.90, 1.10, size=n_aug)
    shears = rng.uniform(-0.20, 0.20, size=n_aug)
    # integers in [-2, +2] inclusive -> high=3 (exclusive)
    txs = rng.integers(-2, 3, size=n_aug)
    tys = rng.integers(-2, 3, size=n_aug)

    write_idx = n_train
    param_idx = 0

    for copy_num in range(copies):
        for i in range(n_train):
            # Compute single affine inverse parameters
            affine_params = compute_inverse_affine_params(
                rotation_deg=rotations[param_idx],
                scale=scales[param_idx],
                shear_x=shears[param_idx],
                tx=txs[param_idx],
                ty=tys[param_idx],
            )
            param_idx += 1

            # Wrap in PIL image, apply single transform with bilinear filtering
            pil_img = Image.fromarray(train_images[i], mode="L")
            transformed_img = pil_img.transform(
                (28, 28),
                Image.AFFINE,
                affine_params,
                resample=Image.BILINEAR,
                fillcolor=0,
            )

            out_images[write_idx] = np.asarray(transformed_img, dtype=np.uint8)
            out_labels[write_idx] = train_labels[i]
            out_source_idx[write_idx] = train_source_idx[i]
            out_is_augmented[write_idx] = 1
            write_idx += 1

    print(f"      Augmentation complete in {time.time() - t0:.2f}s "
          f"({out_is_augmented.sum():,} augmented samples created).")
    return out_images, out_labels, out_source_idx, out_is_augmented


# ==============================================================================
# SECTION 3: CONTOUR EXTRACTION
# ==============================================================================

def extract_single_contour(image_arr: np.ndarray, threshold: int = 100) -> np.ndarray:
    """
    Extract a 1-pixel wide outer contour from a 28x28 grayscale image.

    Algorithm (Shared across all splits post-augmentation):
      1. Binarize at fixed threshold 100: mask = (pixel > 100) ? 255 : 0
      2. Erode by 1 pixel via 3x3 MinFilter: inner = mask.filter(ImageFilter.MinFilter(3))
      3. Subtract inner from mask: contour = ImageChops.subtract(mask, inner)
      4. Convert to uint8 binary image with values {0, 1}.

    Note: ImageFilter.FIND_EDGES is strictly NOT used, as it computes gradient filters
    rather than exact morphological 1-pixel boundary extraction.
    """
    img = Image.fromarray(image_arr, mode="L")

    # Step 1: Fixed-threshold binarization
    mask = img.point(lambda p: 255 if p > threshold else 0)

    # Step 2: 1-pixel morphological erosion (3x3 neighborhood minimum)
    inner = mask.filter(ImageFilter.MinFilter(3))

    # Step 3: Exact boundary subtraction
    contour = ImageChops.subtract(mask, inner)

    # Step 4: 0/1 binary array
    contour_arr = (np.asarray(contour) > 0).astype(np.uint8)
    return contour_arr


def batch_extract_contours(images: np.ndarray, threshold: int = 100) -> np.ndarray:
    """
    Extract 0/1 contours for an entire batch of (N, 28, 28) images.
    Returns (N, 28, 28) uint8 array containing only 0 and 1.
    """
    n_images = len(images)
    print(f"[3/6] Extracting 1-px contours for {n_images:,} images (fixed threshold={threshold})...")
    t0 = time.time()

    contours = np.empty((n_images, 28, 28), dtype=np.uint8)
    for i in range(n_images):
        contours[i] = extract_single_contour(images[i], threshold=threshold)

    print(f"      Contour extraction complete in {time.time() - t0:.2f}s "
          f"(unique values: {np.unique(contours).tolist()}).")
    return contours


# ==============================================================================
# SECTION 4: CONCENTRIC RING FEATURES (14 features)
# ==============================================================================

def compute_ring_projection_matrix() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the concentric ring index map and linear projection matrix.

    For pixel (r, c) in a 28x28 grid (r, c in 0..27):
      ring_index = min(r, c, 27 - r, 27 - c) -> values in 0..13.
      - Ring 0: outermost 1-pixel border (size 108)
      - Ring 13: innermost 2x2 square (size 4)

    Ring sizes arithmetic progression:
      size_k = 108 - 8 * k  for k = 0..13.
      Sizes: [108, 100, 92, 84, 76, 68, 60, 52, 44, 36, 28, 20, 12, 4]
      Sum of sizes = 784 (exact 28x28 pixel count).

    Returns:
      ring_map: (28, 28) int64 map of ring indices.
      ring_sizes: (14,) int64 count of pixels per ring.
      proj_matrix: (784, 14) float32 projection matrix where
                   proj_matrix[pixel_idx, k] = 1.0 / ring_sizes[k] if pixel in ring k, else 0.
                   Enables instantaneous vectorized calculation of ring means via:
                   features = contours.reshape(N, 784) @ proj_matrix
    """
    r, c = np.indices((28, 28))
    ring_map = np.minimum(np.minimum(r, c), np.minimum(27 - r, 27 - c))

    ring_sizes = np.bincount(ring_map.ravel(), minlength=14)
    expected_sizes = np.array([108 - 8 * k for k in range(14)], dtype=np.int64)

    assert len(ring_sizes) == 14, f"Expected 14 rings, got {len(ring_sizes)}"
    assert np.array_equal(ring_sizes, expected_sizes), "Ring sizes do not match formula 108 - 8*k"
    assert ring_sizes.sum() == 784, f"Ring sizes must sum to 784, got {ring_sizes.sum()}"

    # Build (784, 14) projection matrix
    flat_rings = ring_map.ravel()
    proj_matrix = np.zeros((784, 14), dtype=np.float32)
    for k in range(14):
        proj_matrix[flat_rings == k, k] = 1.0 / float(ring_sizes[k])

    return ring_map, ring_sizes, proj_matrix


def extract_concentric_ring_features(
    contour_images: np.ndarray,
    proj_matrix: np.ndarray,
) -> np.ndarray:
    """
    Vectorized extraction of 14 concentric ring mean features.
    contour_images: (N, 28, 28) uint8 array of 0/1 pixels.
    proj_matrix: (784, 14) float32 precomputed ring mean projection matrix.

    Returns:
      features: (N, 14) float32 matrix with ring mean contour densities in [0, 1].
    """
    n_images = len(contour_images)
    print(f"[4/6] Computing 14 concentric ring features for {n_images:,} images via matrix projection...")
    t0 = time.time()

    flat_contours = contour_images.reshape(n_images, 784).astype(np.float32)
    features = flat_contours @ proj_matrix

    print(f"      Ring feature extraction complete in {time.time() - t0:.2f}s "
          f"(feature shape: {features.shape}, min={features.min():.4f}, max={features.max():.4f}).")
    return features


# ==============================================================================
# SECTION 5: NORMALISATION
# ==============================================================================

def normalize_features(
    train_feats: np.ndarray,
    val_feats: np.ndarray,
    test_feats: Optional[np.ndarray] = None,
) -> Tuple[StandardScaler, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Fit a StandardScaler STRICTLY on the training split (originals + augmented).
    Apply the fitted scaler to validation (and optional test) splits.

    Zero-variance features (e.g. outer rings 0-3 that are always 0) are handled:
    StandardScaler sets scale_=1.0 for constant zero features, yielding 0.0.
    We also enforce np.nan_to_num to guarantee strictly finite values (no NaN/inf).
    """
    print("[5/6] Fitting StandardScaler on training set (never on validation/test)...")
    scaler = StandardScaler()
    scaler.fit(train_feats)

    # Verify zero-variance / constant handling
    constant_cols = np.where(scaler.var_ == 0.0)[0]
    if len(constant_cols) > 0:
        print(f"      Note: Detected {len(constant_cols)} zero-variance ring columns: "
              f"{[f'ring_{c:02d}' for c in constant_cols]}. Handled safely as 0.0.")

    train_scaled = scaler.transform(train_feats)
    val_scaled = scaler.transform(val_feats)

    # Ensure strictly finite output
    train_scaled = np.nan_to_num(train_scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    val_scaled = np.nan_to_num(val_scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    test_scaled: Optional[np.ndarray] = None
    if test_feats is not None:
        print("      Applying fitted scaler to test set...")
        test_scaled = scaler.transform(test_feats)
        test_scaled = np.nan_to_num(test_scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return scaler, train_scaled, val_scaled, test_scaled


# ==============================================================================
# SECTION 6: SAVE
# ==============================================================================

def save_dataframe_csv(
    labels: np.ndarray,
    features: np.ndarray,
    source_idx: np.ndarray,
    is_augmented: np.ndarray,
    csv_path: Path,
) -> None:
    """
    Assemble and save tabular feature CSV with standard FS6 layout:
      Columns: label, ring_00 .. ring_13, source_idx, is_augmented
    """
    col_names = ["label"] + [f"ring_{k:02d}" for k in range(14)] + ["source_idx", "is_augmented"]

    df = pd.DataFrame(features, columns=[f"ring_{k:02d}" for k in range(14)])
    df.insert(0, "label", labels)
    df["source_idx"] = source_idx
    df["is_augmented"] = is_augmented

    df.to_csv(csv_path, index=False, columns=col_names)
    print(f"      Saved CSV: {csv_path} ({len(df):,} rows, {csv_path.stat().st_size / (1024*1024):.2f} MB)")


def write_readme(output_dir: Path, config: dict) -> None:
    """Write descriptive README.md documenting files, schema, parameters, and usage warning."""
    readme_path = output_dir / "README.md"
    readme_content = f"""# FS6 Feature Representation: Contour Concentric Rings

## Overview
This directory contains preprocessed feature representations and compressed image archives for **Feature Space 6 (FS6)** of the MNIST handwritten digits dataset (EE5130 Machine Learning Mini-Project).

FS6 extracts 1-pixel binary outer contours using morphological erosion followed by concentric ring pooling across 14 nested square shells from the outer image border inward to the central 2x2 grid.

---

## Directory Contents

| File Name | Description | Shape / Rows | Format |
|---|---|---|---|
| `images_train.npz` | Training images (raw + contour), labels, source_idx, is_augmented | 162,000 samples | Compressed NPZ |
| `images_val.npz` | Validation images (raw + contour), labels, source_idx, is_augmented | 6,000 samples | Compressed NPZ |
| `train.csv` | Unscaled concentric ring features for training split | 162,000 x 17 | CSV |
| `val.csv` | Unscaled concentric ring features for validation split | 6,000 x 17 | CSV |
| `train_scaled.csv` | Standardized concentric ring features for training split | 162,000 x 17 | CSV |
| `val_scaled.csv` | Standardized concentric ring features for validation split | 6,000 x 17 | CSV |
| `scaler.joblib` | Scikit-learn StandardScaler fitted **ONLY** on the training set | - | Joblib pickle |
| `config.json` | Pipeline metadata, hyper-parameters, and split sizes | - | JSON |
| `images_test.npz` | (Optional) Test images archive if `--include-test` was set | 10,000 samples | Compressed NPZ |
| `test.csv` | (Optional) Unscaled test features if `--include-test` was set | 10,000 x 17 | CSV |
| `test_scaled.csv` | (Optional) Standardized test features if `--include-test` was set | 10,000 x 17 | CSV |

---

## CSV Column Specification (17 Columns)

1. `label` (int): True digit class (0 through 9).
2. `ring_00` .. `ring_13` (float): Mean contour intensity within concentric rings 0 through 13.
   - `ring_00`: Outermost 1-pixel border (size 108 pixels).
   - `ring_13`: Innermost central 2x2 square (size 4 pixels).
3. `source_idx` (int): Original row index from `data/raw/mnist_train.csv` (0 to 59,999) for traceability.
4. `is_augmented` (int): `0` for original sample, `1` for synthesized augmented copy.

> [!WARNING]
> **MODEL TRAINING WARNING**:
> Before training any classifier (SVM, Logistic Regression, Neural Network, etc.), **YOU MUST DROP** the columns `source_idx` and `is_augmented`!
>
> ```python
> X_train = train_df.drop(columns=["label", "source_idx", "is_augmented"]).to_numpy()
> y_train = train_df["label"].to_numpy()
> ```
> Including `source_idx` or `is_augmented` in feature vectors will cause artificial data leakage or spurious correlations.

---

## Pipeline Configuration & Hyper-parameters

- **Random Seed**: `{config.get('seed')}`
- **Augmentation Copies**: `{config.get('copies')}` per training sample (54,000 -> 162,000 rows)
- **Validation Split**: `{config.get('val_size') * 100:.1f}%` ({config.get('val_rows')} samples, unaugmented)
- **Rotation Range**: `[-10.0, +10.0]` degrees (uniform)
- **Scale Range**: `[0.90, 1.10]` (uniform)
- **Horizontal Shear**: `[-0.20, +0.20]` (uniform)
- **Translation Range**: `[-2, +2]` integer pixels for both x and y
- **Interpolation**: Bilinear with zero background fill
- **Contour Threshold**: Fixed constant `{config.get('contour_threshold')}`
- **Morphological Filter**: 3x3 MinFilter (erosion by 1 pixel)
- **Ring Mapping Formula**: `ring(r, c) = min(r, c, 27-r, 27-c)` for `r, c in [0..27]`
"""
    readme_path.write_text(readme_content, encoding="utf-8")
    print(f"      Saved documentation: {readme_path}")


# ==============================================================================
# MAIN PIPELINE ENTRY POINT & CLI
# ==============================================================================

def run_pipeline(
    raw_dir: Path,
    output_dir: Path,
    copies: int = 2,
    seed: int = 42,
    val_size: float = 0.1,
    threshold: int = 100,
    include_test: bool = False,
) -> None:
    """Execute the complete FS6 preprocessing pipeline."""
    total_start_time = time.time()
    print("=" * 80)
    print("STARTING FS6 PREPROCESSING PIPELINE (CONTOUR -> 14 CONCENTRIC RINGS)")
    print("=" * 80)
    print(f"Project root:  {PROJECT_ROOT}")
    print(f"Raw data dir:  {raw_dir}")
    print(f"Output dir:    {output_dir}")
    print(f"Copies:        {copies}")
    print(f"Seed:          {seed}")
    print(f"Include test:  {include_test}")
    print("-" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. LOAD + SPLIT
    raw_train_csv = raw_dir / "mnist_train.csv"
    raw_images, raw_labels, raw_source_idx = load_raw_train_data(raw_train_csv)

    (train_img_orig, train_lbl_orig, train_src_orig), (val_img, val_lbl, val_src) = split_data(
        raw_images, raw_labels, raw_source_idx, val_size=val_size, random_state=seed
    )

    # Validation is never augmented
    val_is_augmented = np.zeros(len(val_img), dtype=np.uint8)

    # 2. AUGMENTATION (Training split only)
    train_img, train_lbl, train_src, train_is_augmented = augment_training_data(
        train_img_orig,
        train_lbl_orig,
        train_src_orig,
        copies=copies,
        seed=seed,
    )

    # 3. CONTOUR EXTRACTION (Post-augmentation)
    train_contours = batch_extract_contours(train_img, threshold=threshold)
    val_contours = batch_extract_contours(val_img, threshold=threshold)

    # 4. CONCENTRIC RING FEATURES
    ring_map, ring_sizes, proj_matrix = compute_ring_projection_matrix()
    train_feats_unscaled = extract_concentric_ring_features(train_contours, proj_matrix)
    val_feats_unscaled = extract_concentric_ring_features(val_contours, proj_matrix)

    # Optional Test Set
    test_img: Optional[np.ndarray] = None
    test_contours: Optional[np.ndarray] = None
    test_feats_unscaled: Optional[np.ndarray] = None
    test_lbl: Optional[np.ndarray] = None
    test_src: Optional[np.ndarray] = None
    test_is_augmented: Optional[np.ndarray] = None

    if include_test:
        raw_test_csv = raw_dir / "mnist_test.csv"
        print(f"\n[Optional] Processing raw test CSV from: {raw_test_csv} (NO augmentation)")
        df_test = pd.read_csv(raw_test_csv)
        test_lbl = df_test.iloc[:, 0].to_numpy(dtype=np.int64)
        test_img = df_test.iloc[:, 1:].to_numpy(dtype=np.uint8).reshape(-1, 28, 28)
        test_src = np.arange(len(df_test), dtype=np.int64)
        test_is_augmented = np.zeros(len(df_test), dtype=np.uint8)

        test_contours = batch_extract_contours(test_img, threshold=threshold)
        test_feats_unscaled = extract_concentric_ring_features(test_contours, proj_matrix)

    # 5. NORMALISATION
    scaler, train_feats_scaled, val_feats_scaled, test_feats_scaled = normalize_features(
        train_feats_unscaled,
        val_feats_unscaled,
        test_feats=test_feats_unscaled,
    )

    # 6. SAVE
    print(f"\n[6/6] Saving all outputs into {output_dir}...")
    t0_save = time.time()

    # Save compressed image arrays (NPZ)
    print("      Saving images_train.npz...")
    np.savez_compressed(
        output_dir / "images_train.npz",
        raw=train_img,
        contour=train_contours,
        labels=train_lbl,
        source_idx=train_src,
        is_augmented=train_is_augmented,
    )

    print("      Saving images_val.npz...")
    np.savez_compressed(
        output_dir / "images_val.npz",
        raw=val_img,
        contour=val_contours,
        labels=val_lbl,
        source_idx=val_src,
        is_augmented=val_is_augmented,
    )

    # Save feature CSVs
    save_dataframe_csv(train_lbl, train_feats_unscaled, train_src, train_is_augmented, output_dir / "train.csv")
    save_dataframe_csv(val_lbl, val_feats_unscaled, val_src, val_is_augmented, output_dir / "val.csv")
    save_dataframe_csv(train_lbl, train_feats_scaled, train_src, train_is_augmented, output_dir / "train_scaled.csv")
    save_dataframe_csv(val_lbl, val_feats_scaled, val_src, val_is_augmented, output_dir / "val_scaled.csv")

    # Save scaler
    scaler_path = output_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"      Saved StandardScaler to {scaler_path}")

    # Save optional test artifacts
    if include_test and test_feats_unscaled is not None and test_feats_scaled is not None:
        print("      Saving test set outputs...")
        np.savez_compressed(
            output_dir / "images_test.npz",
            raw=test_img,
            contour=test_contours,
            labels=test_lbl,
            source_idx=test_src,
            is_augmented=test_is_augmented,
        )
        save_dataframe_csv(test_lbl, test_feats_unscaled, test_src, test_is_augmented, output_dir / "test.csv")
        save_dataframe_csv(test_lbl, test_feats_scaled, test_src, test_is_augmented, output_dir / "test_scaled.csv")

    # Save config.json
    config = {
        "dataset": "MNIST handwritten digits (FS6: Contour Concentric Rings)",
        "seed": seed,
        "copies": copies,
        "val_size": val_size,
        "contour_threshold": threshold,
        "raw_train_rows": len(raw_images),
        "train_rows": len(train_img),
        "train_original_rows": len(train_img_orig),
        "train_augmented_rows": len(train_img) - len(train_img_orig),
        "val_rows": len(val_img),
        "num_features": 14,
        "ring_sizes": ring_sizes.tolist(),
        "ring_sum_pixels": int(ring_sizes.sum()),
        "augmentation_ranges": {
            "rotation_deg": [-10.0, 10.0],
            "scale": [0.90, 1.10],
            "shear_x": [-0.20, 0.20],
            "translation_xy": [-2, 2],
        },
        "include_test": include_test,
    }
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"      Saved configuration: {config_path}")

    # Save README.md
    write_readme(output_dir, config)

    print(f"      All save operations finished in {time.time() - t0_save:.2f}s")
    print("=" * 80)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {time.time() - total_start_time:.2f}s")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FS6 Preprocessing Pipeline: Contour Images -> 14 Concentric Ring Features"
    )
    parser.add_argument(
        "--copies",
        type=int,
        default=2,
        help="Number of augmented copies per training sample (default: 2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split and augmentation reproducibility (default: 42)",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Fraction of training set for validation split (default: 0.1)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=100,
        help="Fixed binarization threshold for contour extraction (default: 100)",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        default=False,
        help="Optional flag to process the raw test set without fitting (default: False)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Path to raw data directory (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "contour_concentric_rings",
        help="Path to output directory (default: data/contour_concentric_rings)",
    )

    args = parser.parse_args()

    run_pipeline(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        copies=args.copies,
        seed=args.seed,
        val_size=args.val_size,
        threshold=args.threshold,
        include_test=args.include_test,
    )


if __name__ == "__main__":
    main()
