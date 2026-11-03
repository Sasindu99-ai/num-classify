"""Feature Reduction Layer: Representation 2 — Row + Column Sums (Feature Set A2).

Extracts 56 continuous features (28 row sums + 28 column sums) from [0, 1]-scaled
28x28 Path A images, preserving continuous grayscale intensities without binarisation.
Fits StandardScaler strictly on the Feature Set A2 training matrix.
Also prepares and persists the unreduced 2-D 28x28 Path A images for CNN input.
"""

from pathlib import Path
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

from src.common.utils import (
    FEATURES_A2_DIR,
    RANDOM_SEED,
    set_seed,
    ensure_a2_directories,
)
from src.common.data_loading import load_clean_partitions
from src.a2.augmentation import load_augmented_training_set


def extract_a2_features(images_2d: np.ndarray) -> np.ndarray:
    """Compute 56 continuous features from a batch of (N, 28, 28) grayscale images.

    - Features 0..27: Row sums across columns (horizontal ink distribution).
    - Features 28..55: Column sums across rows (vertical ink distribution).
    Preserves continuous [0, 1] intensities directly without thresholding/binarising.
    """
    if images_2d.ndim != 3 or images_2d.shape[1:] != (28, 28):
        raise ValueError(f"Expected input shape (N, 28, 28), got {images_2d.shape}")

    # Row sums: sum along columns for each row -> (N, 28)
    row_sums = images_2d.sum(axis=2, dtype=np.float32)

    # Column sums: sum along rows for each column -> (N, 28)
    col_sums = images_2d.sum(axis=1, dtype=np.float32)

    # Concatenate: [features 0..27 (row sums), features 28..55 (col sums)]
    features_56 = np.concatenate([row_sums, col_sums], axis=1).astype(np.float32)
    return features_56


def build_and_save_feature_sets() -> None:
    """Extract, standardise, and persist Feature Set A2 and 2-D CNN arrays."""
    set_seed(RANDOM_SEED)
    ensure_a2_directories()

    print("Loading data partitions for Feature Set A2 extraction ...")
    X_train_aug, y_train = load_augmented_training_set()
    _, _, X_val, y_val, X_test, y_test = load_clean_partitions()

    print(f"Extracting 56 A2 features for:")
    print(f"  Train:      {X_train_aug.shape[0]:,} images")
    print(f"  Validation: {X_val.shape[0]:,} images")
    print(f"  Test:       {X_test.shape[0]:,} images")

    X_train_A2_raw = extract_a2_features(X_train_aug)
    X_val_A2_raw = extract_a2_features(X_val)
    X_test_A2_raw = extract_a2_features(X_test)

    # 1. Save unscaled A2 matrices
    unscaled_path = FEATURES_A2_DIR / "A2_unscaled.npz"
    np.savez_compressed(
        unscaled_path,
        X_train=X_train_A2_raw,
        y_train=y_train,
        X_val=X_val_A2_raw,
        y_val=y_val,
        X_test=X_test_A2_raw,
        y_test=y_test,
    )
    print(f"Saved unscaled A2 feature matrices to: {unscaled_path.resolve()}")

    # 2. Fit StandardScaler strictly on training set
    print("Fitting StandardScaler on training Feature Set A2 matrix ...")
    scaler = StandardScaler()
    X_train_A2_std = scaler.fit_transform(X_train_A2_raw).astype(np.float32)
    X_val_A2_std = scaler.transform(X_val_A2_raw).astype(np.float32)
    X_test_A2_std = scaler.transform(X_test_A2_raw).astype(np.float32)

    # Save fitted scaler
    scaler_path = FEATURES_A2_DIR / "scaler_A2.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"Saved fitted StandardScaler to: {scaler_path.resolve()}")

    # 3. Save standardised A2 matrices
    standardized_path = FEATURES_A2_DIR / "A2_standardized.npz"
    np.savez_compressed(
        standardized_path,
        X_train=X_train_A2_std,
        y_train=y_train,
        X_val=X_val_A2_std,
        y_val=y_val,
        X_test=X_test_A2_std,
        y_test=y_test,
    )
    print(f"Saved standardised A2 feature matrices to: {standardized_path.resolve()}")

    # 4. Prepare and save unreduced 2-D Path A images for CNN input: (N, 1, 28, 28)
    print("Preparing 2-D Path A images for direct CNN input ...")
    X_train_2D = np.expand_dims(X_train_aug, axis=1)
    X_val_2D = np.expand_dims(X_val, axis=1)
    X_test_2D = np.expand_dims(X_test, axis=1)

    pathA_2D_path = FEATURES_A2_DIR / "pathA_2D.npz"
    np.savez_compressed(
        pathA_2D_path,
        X_train=X_train_2D,
        y_train=y_train,
        X_val=X_val_2D,
        y_val=y_val,
        X_test=X_test_2D,
        y_test=y_test,
    )
    print(f"Saved 2-D Path A CNN images to: {pathA_2D_path.resolve()}")


def load_a2_features(standardized: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Feature Set A2 matrices and labels."""
    filename = "A2_standardized.npz" if standardized else "A2_unscaled.npz"
    target_path = FEATURES_A2_DIR / filename
    if not target_path.exists():
        build_and_save_feature_sets()

    data = np.load(target_path)
    return (
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        data["X_test"],
        data["y_test"],
    )


def load_pathA_2D() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load unreduced 2-D Path A images and labels for CNN training."""
    target_path = FEATURES_A2_DIR / "pathA_2D.npz"
    if not target_path.exists():
        build_and_save_feature_sets()

    data = np.load(target_path)
    return (
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        data["X_test"],
        data["y_test"],
    )


if __name__ == "__main__":
    build_and_save_feature_sets()

