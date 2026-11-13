# FS6 Feature Representation: Contour Concentric Rings

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

- **Random Seed**: `42`
- **Augmentation Copies**: `2` per training sample (54,000 -> 162,000 rows)
- **Validation Split**: `10.0%` (6000 samples, unaugmented)
- **Rotation Range**: `[-10.0, +10.0]` degrees (uniform)
- **Scale Range**: `[0.90, 1.10]` (uniform)
- **Horizontal Shear**: `[-0.20, +0.20]` (uniform)
- **Translation Range**: `[-2, +2]` integer pixels for both x and y
- **Interpolation**: Bilinear with zero background fill
- **Contour Threshold**: Fixed constant `100`
- **Morphological Filter**: 3x3 MinFilter (erosion by 1 pixel)
- **Ring Mapping Formula**: `ring(r, c) = min(r, c, 27-r, 27-c)` for `r, c in [0..27]`
