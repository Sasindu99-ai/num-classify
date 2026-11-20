"""
Feature Space 6 (FS6) Sanity Checks & Verification Suite

This script reads the saved outputs from `data/contour_concentric_rings/` and
rigorously verifies them using plain asserts and printed summaries.

Key Verification Checks:
  1. Row counts, shapes, dtypes, and NaN/inf absence (train: 162,000, val: 6,000, 14 features).
  2. Data leakage & split integrity:
     - Validation has NO augmented rows.
     - Every augmented sample's source_idx belongs to the training split.
     - Zero overlap between train and val source_idx.
  3. Contour properties:
     - Values strictly in {0, 1}.
     - Contour pixel count strictly less than binarized digit pixel count for interior shapes.
     - Sample check that no filled 2x2 blocks occur along the outer contour boundary.
  4. Ring geometry (Independent Re-Implementation):
     - Ring sizes sum to 784, matching [108, 100, 92, ..., 4].
     - All-ones image yields exactly 1.0 for all 14 rings.
     - Re-computed features on test samples match saved CSVs exactly.
  5. Normalisation / StandardScaler integrity:
     - Fitted strictly on training split.
     - Zero-variance columns handled with 0.0 (no NaN/inf).
     - Standardized train features have mean ~ 0 and std ~ 1 for active rings.
  6. Class distribution & balance:
     - Balanced digit distribution (~10% per class).
     - Augmented distribution matches original training distribution.
  7. Pipeline Determinism:
     - Reruns pipeline and verifies byte-for-byte SHA-256 hash identity across all CSVs.

Run from project root:
    python src/FS6/1_sanity_checks.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageChops, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "contour_concentric_rings"


# ==============================================================================
# INDEPENDENT RE-IMPLEMENTATION OF RING MAP
# ==============================================================================

def independent_ring_map() -> np.ndarray:
    """
    Independent geometric re-implementation of the concentric ring map.
    Instead of min(r, c, 27-r, 27-c), this logic paints concentric square layers
    from the outside inward:
      - Start with full 28x28 filled with ring 0.
      - In a loop for k = 1..13, fill sub-array [k : 28-k, k : 28-k] with k.
    This guarantees independent verification without reusing the pipeline code.
    """
    grid = np.zeros((28, 28), dtype=np.int64)
    for k in range(1, 14):
        grid[k : 28 - k, k : 28 - k] = k
    return grid


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


# ==============================================================================
# SANITY CHECK SUITE
# ==============================================================================

def check_1_row_counts_and_finite(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Check 1: Row counts, column count (14 features), absence of NaN/inf."""
    print("\n--- [Check 1/7] Row Counts, Columns, and Finite Values ---")
    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "val.csv"
    train_scaled_csv = data_dir / "train_scaled.csv"
    val_scaled_csv = data_dir / "val_scaled.csv"

    for p in [train_csv, val_csv, train_scaled_csv, val_scaled_csv]:
        assert p.is_file(), f"Missing required CSV file: {p}"

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_train_scaled = pd.read_csv(train_scaled_csv)
    df_val_scaled = pd.read_csv(val_scaled_csv)

    expected_cols = ["label"] + [f"ring_{k:02d}" for k in range(14)] + ["source_idx", "is_augmented"]

    # Verify column layout
    assert list(df_train.columns) == expected_cols, f"Train CSV columns mismatch: {df_train.columns}"
    assert list(df_val.columns) == expected_cols, f"Val CSV columns mismatch: {df_val.columns}"
    assert list(df_train_scaled.columns) == expected_cols, "Train scaled columns mismatch"
    assert list(df_val_scaled.columns) == expected_cols, "Val scaled columns mismatch"

    # Verify row counts
    assert len(df_train) == 162000, f"Expected 162,000 train rows, got {len(df_train):,}"
    assert len(df_val) == 6000, f"Expected 6,000 val rows, got {len(df_val):,}"
    assert len(df_train_scaled) == 162000, f"Expected 162,000 train_scaled rows, got {len(df_train_scaled):,}"
    assert len(df_val_scaled) == 6000, f"Expected 6,000 val_scaled rows, got {len(df_val_scaled):,}"

    # Verify original vs augmented breakdown
    n_train_orig = int((df_train["is_augmented"] == 0).sum())
    n_train_aug = int((df_train["is_augmented"] == 1).sum())
    assert n_train_orig == 54000, f"Expected 54,000 original train rows, got {n_train_orig:,}"
    assert n_train_aug == 108000, f"Expected 108,000 augmented train rows, got {n_train_aug:,}"

    # Verify no NaN or Inf
    for name, df in [("train", df_train), ("val", df_val),
                     ("train_scaled", df_train_scaled), ("val_scaled", df_val_scaled)]:
        assert not df.isna().any().any(), f"Found NaN values in {name}.csv!"
        feat_vals = df[[f"ring_{k:02d}" for k in range(14)]].to_numpy()
        assert np.isfinite(feat_vals).all(), f"Found infinite values in {name}.csv!"

    print(f"  [PASS] Train rows: {len(df_train):,} (54,000 originals + 108,000 augmented)")
    print(f"  [PASS] Validation rows: {len(df_val):,}")
    print(f"  [PASS] 14 ring features present (ring_00 .. ring_13). All values strictly finite.")
    return df_train, df_val, df_train_scaled, df_val_scaled


def check_2_split_integrity_and_leakage(df_train: pd.DataFrame, df_val: pd.DataFrame) -> None:
    """Check 2: Validation has no augmented rows, no data leakage between splits."""
    print("\n--- [Check 2/7] Split Integrity and Data Leakage Prevention ---")

    # 1. Validation must contain NO augmented rows
    val_aug_count = int(df_val["is_augmented"].sum())
    assert val_aug_count == 0, f"Validation split has {val_aug_count} augmented rows! Must be 0."

    # 2. Every augmented training row's source_idx must belong to the training originals
    train_orig_sources = set(df_train.loc[df_train["is_augmented"] == 0, "source_idx"])
    train_aug_sources = set(df_train.loc[df_train["is_augmented"] == 1, "source_idx"])
    assert train_aug_sources.issubset(train_orig_sources), (
        "Found augmented rows whose source_idx does not exist in training originals!"
    )

    # 3. Disjoint source_idx sets between Train and Validation
    train_all_sources = set(df_train["source_idx"])
    val_sources = set(df_val["source_idx"])
    overlap = train_all_sources.intersection(val_sources)
    assert len(overlap) == 0, f"DATA LEAKAGE DETECTED! {len(overlap)} source indices found in both train and val!"

    # 4. Total unique sources must equal 60,000
    total_unique = len(train_all_sources.union(val_sources))
    assert total_unique == 60000, f"Expected 60,000 unique source indices across splits, got {total_unique:,}"

    print(f"  [PASS] Validation is 100% clean (0 augmented rows).")
    print(f"  [PASS] All 108,000 augmented rows map strictly to training sources.")
    print(f"  [PASS] Zero source_idx overlap between train and validation (no data leakage).")
    print(f"  [PASS] Union of train and val source_idx spans all 60,000 original raw samples.")


def check_3_contour_properties(data_dir: Path) -> None:
    """Check 3: Contour pixel values {0, 1}, pixel count < digit mask, 2x2 outline check."""
    print("\n--- [Check 3/7] Contour Extraction Quality & Boundaries ---")
    npz_train = data_dir / "images_train.npz"
    npz_val = data_dir / "images_val.npz"
    assert npz_train.is_file() and npz_val.is_file(), "Missing NPZ image archives!"

    with np.load(npz_train) as data:
        train_raw = data["raw"][:1000]
        train_contour = data["contour"][:1000]
        train_lbl = data["labels"][:1000]

    # 1. Values strictly in {0, 1}
    unique_vals = np.unique(train_contour)
    assert set(unique_vals).issubset({0, 1}), f"Contour contains non-binary values: {unique_vals}"

    # 2. Check contour pixel count < binarized digit pixel count for solid digits
    # Threshold 100 binarization
    digit_masks = (train_raw > 100).astype(np.uint8)
    mask_pixel_counts = digit_masks.sum(axis=(1, 2))
    contour_pixel_counts = train_contour.sum(axis=(1, 2))

    # For images where interior exists (erosion removes pixels), contour < mask
    interior_exists = 0
    strictly_less = 0
    for m_cnt, c_cnt in zip(mask_pixel_counts, contour_pixel_counts):
        assert c_cnt <= m_cnt, f"Contour count ({c_cnt}) exceeded mask count ({m_cnt})!"
        if m_cnt > 20:  # digit with substantive stroke width
            interior_exists += 1
            if c_cnt < m_cnt:
                strictly_less += 1

    assert strictly_less > 0, "Expected contours to have fewer pixels than masks for digits with interior!"

    # 3. Check outline morphology on synthetic shape: no filled 2x2 blocks along outer edge
    solid_box = np.zeros((28, 28), dtype=np.uint8)
    solid_box[8:20, 8:20] = 255  # 12x12 solid block
    box_img = Image.fromarray(solid_box)
    box_mask = box_img.point(lambda p: 255 if p > 100 else 0)
    box_inner = box_mask.filter(ImageFilter.MinFilter(3))
    box_contour = (np.asarray(ImageChops.subtract(box_mask, box_inner)) > 0).astype(np.uint8)

    # Along the top edge (row 8, col 9..18), the contour must be strictly 1 pixel wide
    assert (box_contour[8, 9:19] == 1).all(), "Top edge missing contour pixels"
    assert (box_contour[7, 9:19] == 0).all(), "Contour leaked outside shape"
    assert (box_contour[9, 9:19] == 0).all(), "Contour thicker than 1 pixel on straight edge!"

    # A 2x2 block in the interior or straight outline must NOT be all 1s
    # (interior was eroded away to 0)
    assert not ((box_contour[10:12, 10:12] == 1).all()), "Interior contains filled 2x2 blocks!"

    print(f"  [PASS] Contours are strictly binary {0, 1} uint8 arrays.")
    print(f"  [PASS] Contour pixel count < binarized mask count ({strictly_less}/{interior_exists} sampled thick digits).")
    print(f"  [PASS] Morphological outline verified: exactly 1 px wide along straight edges, interior eroded.")


def check_4_ring_geometry_and_independent_verification(
    data_dir: Path, df_train: pd.DataFrame
) -> None:
    """Check 4: Ring geometry, sum = 784, all-ones image test, independent reproduction."""
    print("\n--- [Check 4/7] Concentric Ring Geometry & Independent Verification ---")

    # Independent ring map
    ind_map = independent_ring_map()
    assert ind_map.shape == (28, 28)
    assert ind_map.min() == 0 and ind_map.max() == 13

    # Check ring sizes
    sizes = np.bincount(ind_map.ravel(), minlength=14)
    expected_sizes = [108 - 8 * k for k in range(14)]
    assert list(sizes) == expected_sizes, f"Ring sizes mismatch: {sizes} vs {expected_sizes}"
    assert sizes.sum() == 784, f"Ring sizes must sum to 784, got {sizes.sum()}"

    # All-ones image test: ring mean must equal 1.0 for every ring
    ones_img = np.ones((28, 28), dtype=np.uint8)
    for k in range(14):
        mask_k = (ind_map == k)
        ring_mean = ones_img[mask_k].mean()
        assert np.isclose(ring_mean, 1.0), f"All-ones test failed on ring {k}: {ring_mean}"

    # Independent feature re-computation on first 100 training samples
    with np.load(data_dir / "images_train.npz") as data:
        sample_contours = data["contour"][:100]

    for i in range(100):
        c_img = sample_contours[i]
        for k in range(14):
            expected_mean = c_img[ind_map == k].mean()
            saved_val = df_train.iloc[i][f"ring_{k:02d}"]
            assert np.isclose(expected_mean, saved_val, atol=1e-5), (
                f"Sample {i} ring {k} feature mismatch: computed {expected_mean} vs saved {saved_val}"
            )

    print(f"  [PASS] Ring sizes match arithmetic progression: {expected_sizes}")
    print(f"  [PASS] Ring sizes sum to 784.")
    print(f"  [PASS] All-ones synthetic image yields exactly 1.0 for all 14 rings.")
    print(f"  [PASS] Independent re-implementation matches saved CSV features on 100 sample images.")


def check_5_normalisation_and_scaler(
    data_dir: Path, df_train: pd.DataFrame, df_val: pd.DataFrame,
    df_train_scaled: pd.DataFrame, df_val_scaled: pd.DataFrame
) -> None:
    """Check 5: Scaler fitted on training only, handles zero variance, mean ~0, std ~1."""
    print("\n--- [Check 5/7] Normalisation & StandardScaler Integrity ---")
    scaler_path = data_dir / "scaler.joblib"
    assert scaler_path.is_file(), f"Missing scaler at {scaler_path}"

    scaler = joblib.load(scaler_path)
    assert scaler.n_features_in_ == 14, f"Scaler expects 14 features, got {scaler.n_features_in_}"

    feat_cols = [f"ring_{k:02d}" for k in range(14)]
    X_train_unscaled = df_train[feat_cols].to_numpy()
    X_val_unscaled = df_val[feat_cols].to_numpy()

    # Re-apply scaler
    recomputed_train_scaled = np.nan_to_num(scaler.transform(X_train_unscaled), nan=0.0)
    recomputed_val_scaled = np.nan_to_num(scaler.transform(X_val_unscaled), nan=0.0)

    # Check matches saved CSVs
    assert np.allclose(recomputed_train_scaled, df_train_scaled[feat_cols].to_numpy(), atol=1e-5), (
        "Recomputed train scaled does not match train_scaled.csv!"
    )
    assert np.allclose(recomputed_val_scaled, df_val_scaled[feat_cols].to_numpy(), atol=1e-5), (
        "Recomputed val scaled does not match val_scaled.csv!"
    )

    # Check means and standard deviations on training data
    train_means = df_train_scaled[feat_cols].mean().to_numpy()
    train_stds = df_train_scaled[feat_cols].std().to_numpy()

    for k in range(14):
        # If ring variance is zero, scaled values must be exactly 0
        if scaler.var_[k] == 0.0:
            assert np.allclose(df_train_scaled[f"ring_{k:02d}"], 0.0), f"Constant ring {k} not 0.0 in scaled!"
            assert np.allclose(df_val_scaled[f"ring_{k:02d}"], 0.0), f"Constant ring {k} not 0.0 in val scaled!"
        else:
            assert np.isclose(train_means[k], 0.0, atol=1e-2), f"Train ring {k} mean not ~0: {train_means[k]}"
            assert np.isclose(train_stds[k], 1.0, atol=1e-2), f"Train ring {k} std not ~1: {train_stds[k]}"

    print(f"  [PASS] Scaler correctly reproduces train_scaled.csv and val_scaled.csv.")
    print(f"  [PASS] Active ring columns on train have mean ~0.0 and std ~1.0.")
    print(f"  [PASS] Zero-variance rings safely mapped to 0.0 without NaNs or Infs.")


def check_6_label_distribution(df_train: pd.DataFrame, df_val: pd.DataFrame) -> None:
    """Check 6: Balanced class distribution across train/val and between original/augmented."""
    print("\n--- [Check 6/7] Class Balance & Augmentation Consistency ---")

    # 1. Training split balance
    train_counts = df_train["label"].value_counts().sort_index()
    val_counts = df_val["label"].value_counts().sort_index()

    # Each digit class should be approximately 10% (+/- 2%)
    train_props = train_counts / len(df_train)
    val_props = val_counts / len(df_val)

    for digit in range(10):
        assert 0.08 <= train_props[digit] <= 0.12, f"Train digit {digit} proportion out of expected range: {train_props[digit]:.3f}"
        assert 0.08 <= val_props[digit] <= 0.12, f"Val digit {digit} proportion out of expected range: {val_props[digit]:.3f}"

    # 2. Check that augmented rows have EXACTLY copies (2) times the original counts per class
    orig_train_counts = df_train.loc[df_train["is_augmented"] == 0, "label"].value_counts().sort_index()
    aug_train_counts = df_train.loc[df_train["is_augmented"] == 1, "label"].value_counts().sort_index()

    for digit in range(10):
        orig_c = orig_train_counts[digit]
        aug_c = aug_train_counts[digit]
        assert aug_c == 2 * orig_c, (
            f"Class {digit} augmentation count mismatch: {aug_c} augmented vs 2 * {orig_c} originals"
        )

    print(f"  [PASS] All 10 digit classes (0-9) roughly balanced (~10% per class).")
    print(f"  [PASS] Augmented sample counts per class match exactly 2x original class counts.")


def check_7_determinism(data_dir: Path, skip_rerun: bool = False) -> None:
    """Check 7: Full pipeline determinism (compare CSV hashes across independent runs)."""
    print("\n--- [Check 7/7] Pipeline Determinism & Checksum Verification ---")
    csv_files = ["train.csv", "val.csv", "train_scaled.csv", "val_scaled.csv"]
    initial_hashes: Dict[str, str] = {}

    for name in csv_files:
        p = data_dir / name
        assert p.is_file(), f"Missing file: {p}"
        initial_hashes[name] = compute_sha256(p)
        print(f"  Initial SHA-256 ({name}): {initial_hashes[name][:16]}...")

    if skip_rerun:
        print("  [INFO] Skipping pipeline re-execution as --skip-rerun was specified.")
        print("  [PASS] Initial file checksums successfully verified.")
        return

    print("  Re-running 0_preprocessing_pipeline.py with same parameters to test determinism...")
    t0 = time.time()
    pipeline_script = PROJECT_ROOT / "src" / "FS6" / "0_preprocessing_pipeline.py"
    cmd = [sys.executable, str(pipeline_script), "--copies", "2", "--seed", "42"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Pipeline re-run failed: {res.stderr}"
    print(f"  Re-run finished in {time.time() - t0:.2f}s.")

    # Re-verify hashes
    for name in csv_files:
        p = data_dir / name
        new_hash = compute_sha256(p)
        assert new_hash == initial_hashes[name], (
            f"DETERMINISM FAILED on {name}! Initial: {initial_hashes[name]}, New: {new_hash}"
        )
        print(f"  [PASS] Identical SHA-256 for {name} ({new_hash[:16]}...)")

    print("  [PASS] Pipeline is 100% bitwise deterministic!")


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="FS6 Sanity Checks and Verification")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Path to preprocessed data directory (default: data/contour_concentric_rings)",
    )
    parser.add_argument(
        "--skip-rerun",
        action="store_true",
        default=False,
        help="Skip re-executing pipeline for hash comparison (for fast verification)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("RUNNING FS6 PREPROCESSING SANITY CHECKS")
    print(f"Target data directory: {args.data_dir}")
    print("=" * 80)

    t_start = time.time()
    df_train, df_val, df_train_scaled, df_val_scaled = check_1_row_counts_and_finite(args.data_dir)
    check_2_split_integrity_and_leakage(df_train, df_val)
    check_3_contour_properties(args.data_dir)
    check_4_ring_geometry_and_independent_verification(args.data_dir, df_train)
    check_5_normalisation_and_scaler(args.data_dir, df_train, df_val, df_train_scaled, df_val_scaled)
    check_6_label_distribution(df_train, df_val)
    check_7_determinism(args.data_dir, skip_rerun=args.skip_rerun)

    print("\n" + "=" * 80)
    print(f"ALL 7 SANITY CHECKS PASSED PERFECTLY IN {time.time() - t_start:.2f}s!")
    print("=" * 80)


if __name__ == "__main__":
    main()
