"""Path A raw image scaling and label-preserving geometric augmentation module.

Applies label-preserving transformations (small rotation ±10°, translation ±2px,
scaling 0.9x-1.1x, shear ±0.2) strictly to the clean Path A training set.
Validation and test sets are NEVER augmented.
"""

from pathlib import Path
import numpy as np
import scipy.ndimage as nd
from tqdm import tqdm

from src.common.utils import (
    AUGMENTED_RAW_DIR,
    RANDOM_SEED,
    set_seed,
    ensure_a2_directories,
)
from src.common.data_loading import load_clean_partitions


def apply_label_preserving_affine(
    image: np.ndarray,
    rng: np.random.Generator,
    max_rotation_deg: float = 10.0,
    max_translation_px: float = 2.0,
    scale_range: tuple[float, float] = (0.9, 1.1),
    shear_range: tuple[float, float] = (-0.2, 0.2),
) -> np.ndarray:
    """Apply a single random label-preserving affine transformation to a 28x28 image."""
    theta = np.deg2rad(rng.uniform(-max_rotation_deg, max_rotation_deg))
    dy = rng.uniform(-max_translation_px, max_translation_px)
    dx = rng.uniform(-max_translation_px, max_translation_px)
    scale = rng.uniform(*scale_range)
    shear = rng.uniform(*shear_range)

    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rot_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
    shear_mat = np.array([[1.0, 0.0], [shear, 1.0]], dtype=np.float64)
    scale_mat = np.array([[scale, 0.0], [0.0, scale]], dtype=np.float64)

    A_forward = rot_mat @ shear_mat @ scale_mat
    A_inv = np.linalg.inv(A_forward)

    center = np.array([13.5, 13.5], dtype=np.float64)
    trans = np.array([dy, dx], dtype=np.float64)
    offset = center - A_inv @ (center + trans)

    transformed = nd.affine_transform(
        image,
        A_inv,
        offset=offset,
        order=1,
        mode="constant",
        cval=0.0,
    )
    return np.clip(transformed, 0.0, 1.0).astype(np.float32)


def generate_augmented_training_set(
    copies_per_sample: int = 2,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate expanded Path A training set (162,000 samples) by combining clean originals with augmented copies."""
    set_seed(seed)
    ensure_a2_directories()
    rng = np.random.default_rng(seed)

    X_train_clean, y_train_clean, _, _, _, _ = load_clean_partitions()
    num_clean = len(X_train_clean)
    total_samples = num_clean * (1 + copies_per_sample)

    print(
        f"Augmenting Path A training set: {num_clean:,} clean originals + "
        f"{num_clean * copies_per_sample:,} augmented copies = {total_samples:,} total samples ..."
    )

    X_train_aug = np.empty((total_samples, 28, 28), dtype=np.float32)
    y_train_aug = np.empty((total_samples,), dtype=np.int64)

    # Place original clean images first
    X_train_aug[:num_clean] = X_train_clean
    y_train_aug[:num_clean] = y_train_clean

    idx = num_clean
    for copy_idx in range(copies_per_sample):
        print(f"  Generating copy batch {copy_idx + 1} of {copies_per_sample} ...")
        for i in tqdm(range(num_clean), desc=f"Augmentation Pass {copy_idx + 1}"):
            X_train_aug[idx] = apply_label_preserving_affine(X_train_clean[i], rng)
            y_train_aug[idx] = y_train_clean[i]
            idx += 1

    output_npz = AUGMENTED_RAW_DIR / "augmented_train.npz"
    np.savez_compressed(
        output_npz,
        X_train_aug=X_train_aug,
        y_train_aug=y_train_aug,
    )
    print(f"Saved augmented training set to: {output_npz.resolve()}")

    return X_train_aug, y_train_aug


def load_augmented_training_set() -> tuple[np.ndarray, np.ndarray]:
    """Load previously generated augmented training set from binary file."""
    output_npz = AUGMENTED_RAW_DIR / "augmented_train.npz"
    if not output_npz.exists():
        return generate_augmented_training_set()

    data = np.load(output_npz)
    return data["X_train_aug"], data["y_train_aug"]


if __name__ == "__main__":
    generate_augmented_training_set()

