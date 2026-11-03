"""Shared data loading and leakage-free splitting module for MNIST.

Reads raw CSV files (mnist_train.csv and mnist_test.csv), performs a stratified
train/validation split (54,000 / 6,000) on the training data, and serializes
the clean partitions as binary NumPy arrays (.npz) for all feature sets.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.common.utils import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    set_seed,
    ensure_a2_directories,
)


def load_raw_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read MNIST CSV file and return (images_2d, labels).

    Args:
        csv_path: Path to mnist_train.csv or mnist_test.csv.

    Returns:
        images: np.ndarray of shape (N, 28, 28) in float32 scaled to [0.0, 1.0].
        labels: np.ndarray of shape (N,) with integer class labels 0..9.
    """
    print(f"Loading raw data from: {csv_path.resolve()} ...")
    df = pd.read_csv(csv_path)

    # First column is label, subsequent 784 columns are pixels
    labels = df.iloc[:, 0].to_numpy(dtype=np.int64)
    pixels = df.iloc[:, 1:].to_numpy(dtype=np.float32)

    # Normalize pixel intensity from [0, 255] to [0.0, 1.0]
    pixels_normalized = pixels / 255.0

    # Reshape each 784-element row into a 28x28 2D grayscale image
    images_2d = pixels_normalized.reshape(-1, 28, 28)

    print(f"Loaded {len(labels):,} samples with shape {images_2d.shape}")
    return images_2d, labels


def split_and_save_partitions(
    val_size: int = 6000,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform stratified split on training set and persist all partitions to binary format.

    Guarantees strict anti-leakage: validation and test sets remain completely isolated.
    """
    set_seed(seed)
    ensure_a2_directories()

    train_csv = RAW_DATA_DIR / "mnist_train.csv"
    test_csv = RAW_DATA_DIR / "mnist_test.csv"

    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(
            f"Raw CSV files missing. Expected {train_csv} and {test_csv}."
        )

    # Load 60,000 raw training rows and 10,000 raw test rows
    X_train_full, y_train_full = load_raw_csv(train_csv)
    X_test, y_test = load_raw_csv(test_csv)

    # Stratified train/val split (54,000 / 6,000)
    print(f"Performing stratified split: 54,000 train / {val_size:,} validation ...")
    X_train_clean, X_val, y_train_clean, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_size,
        stratify=y_train_full,
        random_state=seed,
    )

    print(f"Partition summary:")
    print(f"  X_train_clean: {X_train_clean.shape}, dtype={X_train_clean.dtype}")
    print(f"  X_val:         {X_val.shape}, dtype={X_val.dtype}")
    print(f"  X_test:        {X_test.shape}, dtype={X_test.dtype}")

    # Save to compressed .npz archive in data/processed/
    output_path = PROCESSED_DATA_DIR / "clean_partitions.npz"
    np.savez_compressed(
        output_path,
        X_train_clean=X_train_clean,
        y_train_clean=y_train_clean,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )
    print(f"Saved clean partitions to: {output_path.resolve()}")

    return X_train_clean, y_train_clean, X_val, y_val, X_test, y_test


def load_clean_partitions() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load previously saved clean partitions from binary .npz file."""
    partition_file = PROCESSED_DATA_DIR / "clean_partitions.npz"
    if not partition_file.exists():
        return split_and_save_partitions()

    data = np.load(partition_file)
    return (
        data["X_train_clean"],
        data["y_train_clean"],
        data["X_val"],
        data["y_val"],
        data["X_test"],
        data["y_test"],
    )


if __name__ == "__main__":
    split_and_save_partitions()

