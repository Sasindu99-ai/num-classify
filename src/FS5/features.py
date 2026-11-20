"""Feature extraction from B2 contour images."""

import numpy as np


def extract_features(contours: np.ndarray) -> np.ndarray:
    """
    Convert contour images into 56 row and column sum features.

    Input:
        contours: Array with shape (N, 28, 28).

    Output:
        Feature array with shape (N, 56).
    """

    if contours.ndim != 3 or contours.shape[1:] != (28, 28):
        raise ValueError(
            f"Expected input shape (N, 28, 28), got {contours.shape}"
        )

    # Count white pixels in each row
    row_sums = np.sum(contours, axis=2)

    # Count white pixels in each column
    column_sums = np.sum(contours, axis=1)

    # Combine row and column features
    features = np.concatenate((row_sums, column_sums), axis=1)

    return features.astype(np.float32)
