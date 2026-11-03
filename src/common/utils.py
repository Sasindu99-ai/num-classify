"""Utility functions, configuration constants, and standard paths for EE5130 Mini Project.

Shared across all feature representation pipelines (A1, A2, A3, B1, B2, B3).
"""

from pathlib import Path
import random
import numpy as np

# Fixed random seed for complete reproducibility across the entire project
RANDOM_SEED = 42

# Base project paths (utils is located at src/common/utils.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
AUGMENTED_DIR = DATA_DIR / "augmented"
AUGMENTED_RAW_DIR = AUGMENTED_DIR / "raw"
AUGMENTED_CONTOUR_DIR = AUGMENTED_DIR / "contour"

# Feature Set A2 specific storage paths
FEATURES_A2_DIR = DATA_DIR / "features" / "a2"
RESULTS_A2_DIR = PROJECT_ROOT / "results" / "a2"
FIGURES_A2_DIR = RESULTS_A2_DIR / "figures"
TABLES_A2_DIR = RESULTS_A2_DIR / "tables"
CONFUSION_MATRICES_A2_DIR = RESULTS_A2_DIR / "confusion_matrices"
MODELS_A2_DIR = RESULTS_A2_DIR / "models"


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seed across Python, NumPy, and PyTorch (if available) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def ensure_a2_directories() -> None:
    """Ensure all required Feature Set A2 data and result directories exist."""
    for directory in [
        PROCESSED_DATA_DIR,
        AUGMENTED_RAW_DIR,
        FEATURES_A2_DIR,
        RESULTS_A2_DIR,
        FIGURES_A2_DIR,
        TABLES_A2_DIR,
        CONFUSION_MATRICES_A2_DIR,
        MODELS_A2_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

