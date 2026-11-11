"""Generate a k-distance plot to investigate DBSCAN eps values."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "b2"
    / "models"
    / "clustering_results.joblib"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "b2" / "figures"
OUTPUT_PATH = OUTPUT_DIR / "dbscan_k_distance.png"

# Match the min_samples value used in the baseline DBSCAN experiment
MIN_SAMPLES = 15

# Consistent B2 purple theme
DARK_PURPLE = "#542078"
LIGHT_PURPLE = "#B99AD5"


def main():
    # Load the same training sample used in the clustering experiment
    saved_data = joblib.load(INPUT_PATH)
    X = saved_data["X_data"]

    print(f"Loaded feature data: {X.shape}")

    # Calculate distances to the 15th nearest neighbour
    neighbours = NearestNeighbors(
        n_neighbors=MIN_SAMPLES,
        n_jobs=-1
    )

    neighbours.fit(X)

    distances, _ = neighbours.kneighbors(X)

    # Sort the distances to make the k-distance curve
    k_distances = np.sort(distances[:, -1])

    # Plot
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    plt.plot(
        range(1, len(k_distances) + 1),
        k_distances,
        color=DARK_PURPLE,
        linewidth=1.8
    )

    plt.xlabel("Points (sorted by distance)", fontsize=11)
    plt.ylabel(f"Distance to {MIN_SAMPLES}th nearest neighbour", fontsize=11)

    plt.title(
        "DBSCAN k-Distance Plot for B2 Features",
        fontsize=14,
        color=DARK_PURPLE,
        fontweight="bold"
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.3,
        color=LIGHT_PURPLE
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nK-distance plot saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()