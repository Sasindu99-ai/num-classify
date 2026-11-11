"""Dataset preparation and storage for B2 feature representation."""

from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.a2.augmentation import load_augmented_training_set
from src.common.data_loading import load_clean_partitions
from src.common.utils import PROJECT_ROOT, RANDOM_SEED

from src.b2.contours import extract_contours
from src.b2.features import extract_features


# B2 storage directory
FEATURES_B2_DIR = PROJECT_ROOT / "data" / "features" / "b2"


def ensure_b2_directories() -> None:
    """Create the required B2 data directory."""
    FEATURES_B2_DIR.mkdir(parents=True, exist_ok=True)
