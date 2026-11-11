"""Verification of B2 contour extraction."""

import numpy as np
import matplotlib.pyplot as plt

from src.common.data_loading import load_clean_partitions
from src.b2.contours import extract_contour, extract_contours


def main():

    print("Loading MNIST dataset...")

    X_train, y_train, _, _, _, _ = load_clean_partitions()

    # Select one training image
    image = X_train[0]
    label = y_train[0]

    # Extract contour
    contour = extract_contour(image)

    # Verify single image
    assert contour.shape == (28, 28)
    assert np.all(np.isin(contour, [0.0, 1.0]))
    assert contour.sum() > 0

    print("Single image verification passed!")
    print("Original shape:", image.shape)
    print("Contour shape:", contour.shape)
    print("Unique values:", np.unique(contour))

    # Verify batch extraction
    sample_images = X_train[:10]
    contours = extract_contours(sample_images)

    assert contours.shape == (10, 28, 28)
    assert np.all(np.isin(contours, [0.0, 1.0]))

    print("Batch verification passed!")
    print("Batch output shape:", contours.shape)

    # Visual comparison
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title(f"Original Digit: {label}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(contour, cmap="gray")
    plt.title("B2 Contour")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
