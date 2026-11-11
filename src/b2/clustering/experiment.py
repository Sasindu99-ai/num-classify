
"""Run clustering experiments using B2 contour-based features."""

import joblib
import numpy as np

from src.common.utils import PROJECT_ROOT

from src.b2.clustering.models import (
    run_kmeans,
    run_minibatch_kmeans,
    run_gaussian_mixture,
    run_agglomerative,
    run_dbscan,
    run_spectral,
)


# Experiment configuration
RANDOM_SEED = 42
N_SAMPLES = 3000

FEATURES_PATH = (
    PROJECT_ROOT / "data" / "features" / "b2" / "train_features.npz"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "b2" / "models"


def load_experiment_data(n_samples: int = N_SAMPLES):
    """Load and randomly sample prepared B2 training features."""

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"B2 feature file not found: {FEATURES_PATH}"
        )

    data = np.load(FEATURES_PATH)

    X_train = data["X"]
    y_train = data["y"]

    if n_samples > len(X_train):
        raise ValueError("Requested sample exceeds training dataset size.")

    rng = np.random.default_rng(RANDOM_SEED)

    sample_indices = rng.choice(
        len(X_train),
        size=n_samples,
        replace=False,
    )

    X_sample = X_train[sample_indices]
    y_sample = y_train[sample_indices]

    print("B2 clustering dataset loaded.")
    print("Sample shape:", X_sample.shape)

    return X_sample, y_sample, sample_indices


def run_all_experiments():
    """Run all six clustering algorithms on the B2 feature sample."""

    X_sample, y_sample, sample_indices = load_experiment_data()

    print("\n" + "=" * 60)
    print("B2 UNSUPERVISED CLUSTERING EXPERIMENT")
    print("=" * 60)

    algorithms = [
        ("k-Means", run_kmeans),
        ("Mini-Batch k-Means", run_minibatch_kmeans),
        ("Gaussian Mixture Model", run_gaussian_mixture),
        ("Agglomerative Clustering", run_agglomerative),
        ("DBSCAN", run_dbscan),
        ("Spectral Clustering", run_spectral),
    ]

    experiment_results = {}

    for name, algorithm in algorithms:

        print(f"\nRunning {name}...")

        result = algorithm(X_sample)

        experiment_results[name] = result

        print("Clusters found:", result["n_clusters"])
        print(f"Fit time: {result['fit_time']:.3f} seconds")

        if name == "DBSCAN":
            print("Noise percentage:",
                  f"{result['noise_percentage']:.2f}%")

        if "inertia" in result:
            print("Inertia:", f"{result['inertia']:.2f}")

        if "bic" in result:
            print("BIC:", f"{result['bic']:.2f}")
            print("AIC:", f"{result['aic']:.2f}")

    # Create result directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save experiment results
    output_path = RESULTS_DIR / "clustering_results.joblib"

    saved_data = {
        "X_data": X_sample,
        "y_true": y_sample,
        "sample_indices": sample_indices,
        "models": experiment_results,
    }

    joblib.dump(saved_data, output_path)

    print("\n" + "=" * 60)
    print("B2 CLUSTERING EXPERIMENT COMPLETED")
    print("=" * 60)
    print("Results saved to:", output_path)


if __name__ == "__main__":
    run_all_experiments()