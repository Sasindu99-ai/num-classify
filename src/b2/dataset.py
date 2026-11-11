
"""Dataset preparation and storage for B2 feature representation."""

import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler

from src.a2.augmentation import load_augmented_training_set
from src.common.data_loading import load_clean_partitions
from src.common.utils import PROJECT_ROOT

from src.b2.contours import extract_contours
from src.b2.features import extract_features


# B2 feature storage directory
FEATURES_B2_DIR = PROJECT_ROOT / "data" / "features" / "b2"


def ensure_b2_directories() -> None:
    """Create the required B2 data directory."""
    FEATURES_B2_DIR.mkdir(parents=True, exist_ok=True)


def process_images_in_batches(
    images: np.ndarray,
    batch_size: int = 1000
) -> np.ndarray:
    """
    Extract B2 features from images in batches.

    Each image is converted into a contour and then
    represented by 56 row and column sum features.
    """

    total_images = len(images)

    all_features = np.zeros(
        (total_images, 56),
        dtype=np.float32
    )

    for start in range(0, total_images, batch_size):

        end = min(start + batch_size, total_images)

        image_batch = images[start:end]

        # Extract contours
        contour_batch = extract_contours(image_batch)

        # Extract 56 features
        feature_batch = extract_features(contour_batch)

        # Store extracted features
        all_features[start:end] = feature_batch

        print(f"Processed {end}/{total_images} images")

    return all_features


def prepare_b2_datasets() -> None:
    """Prepare training, validation and test datasets for B2."""

    print("\nLoading datasets...")

    # Load augmented training images and labels
    X_train, y_train = load_augmented_training_set()

    # Load clean validation and test images
    _, _, X_val, y_val, X_test, y_test = load_clean_partitions()

    print("\nExtracting training features...")
    train_features = process_images_in_batches(X_train)

    print("\nExtracting validation features...")
    val_features = process_images_in_batches(X_val)

    print("\nExtracting test features...")
    test_features = process_images_in_batches(X_test)

    # Scale features using training data only
    print("\nScaling features...")

    scaler = StandardScaler()

    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)

    # Convert scaled features back to float32
    train_features = train_features.astype(np.float32)
    val_features = val_features.astype(np.float32)
    test_features = test_features.astype(np.float32)

    # Create B2 storage directory
    ensure_b2_directories()

    # Save training dataset
    np.savez_compressed(
        FEATURES_B2_DIR / "train_features.npz",
        X=train_features,
        y=y_train
    )

    # Save validation dataset
    np.savez_compressed(
        FEATURES_B2_DIR / "validation_features.npz",
        X=val_features,
        y=y_val
    )

    # Save test dataset
    np.savez_compressed(
        FEATURES_B2_DIR / "test_features.npz",
        X=test_features,
        y=y_test
    )

    # Save fitted scaler
    joblib.dump(
        scaler,
        FEATURES_B2_DIR / "scaler.joblib"
    )

    print("\nB2 datasets prepared successfully!")

    print("Training features:", train_features.shape)
    print("Validation features:", val_features.shape)
    print("Test features:", test_features.shape)


if __name__ == "__main__":
    prepare_b2_datasets()
