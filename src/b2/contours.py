
"""Contour extraction for B2 feature representation."""

import numpy as np
from PIL import Image, ImageFilter, ImageChops


def extract_contour(image: np.ndarray, threshold: int = 100) -> np.ndarray:
    """
    Convert one 28x28 grayscale image into a one-pixel-wide contour.

    Input:
        image: 28x28 grayscale image with values between 0 and 1.

    Output:
        28x28 binary contour image with values 0 or 1.
    """

    if image.shape != (28, 28):
        raise ValueError(f"Expected (28, 28), got {image.shape}")

    # Convert pixel values from [0, 1] to [0, 255]
    image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    # Convert NumPy array into a grayscale image
    img = Image.fromarray(image_uint8)

    # Step 1: Binarisation
    mask = img.point(
        lambda p: 255 if p > threshold else 0
    )

    # Step 2: Erosion by one pixel
    eroded = mask.filter(ImageFilter.MinFilter(3))

    # Step 3: Subtract eroded image from original mask
    contour = ImageChops.subtract(mask, eroded)

    # Convert back into a NumPy array with values 0 or 1
    contour_array = np.asarray(contour, dtype=np.float32) / 255.0

    return contour_array


def extract_contours(images: np.ndarray) -> np.ndarray:
    """Extract contours from a batch of 28x28 images."""

    if images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError(
            f"Expected input shape (N, 28, 28), got {images.shape}"
        )

    contours = np.stack(
        [extract_contour(image) for image in images]
    )

    return contours.astype(np.float32)

