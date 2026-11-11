
"""Small-scale testing of B2 clustering algorithms."""

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


def test_clustering_models():

    # Load prepared B2 training features
    data_path = PROJECT_ROOT / "data" / "features" / "b2" / "train_features.npz"

    data = np.load(data_path)
    X_train = data["X"]

    # Select 100 random samples for initial testing
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(
        len(X_train),
        size=100,
        replace=False
    )

    X_sample = X_train[sample_indices]

    print("B2 sample shape:", X_sample.shape)
    print("\nTesting clustering algorithms...\n")

    algorithms = [
        ("k-Means", run_kmeans),
        ("Mini-Batch k-Means", run_minibatch_kmeans),
        ("Gaussian Mixture Model", run_gaussian_mixture),
        ("Agglomerative Clustering", run_agglomerative),
        ("DBSCAN", run_dbscan),
        ("Spectral Clustering", run_spectral),
    ]

    for name, algorithm in algorithms:

        print(f"Running {name}...")

        try:
            result = algorithm(X_sample)

            print("  Clusters found:", result["n_clusters"])
            print(f"  Execution time: {result['fit_time']:.4f} seconds")
            print("  Labels shape:", result["labels"].shape)
            print("  Test passed!\n")

        except Exception as error:
            print(f"  Test failed: {error}\n")
            raise

    print("All six clustering algorithms completed successfully!")


if __name__ == "__main__":
    test_clustering_models()
