
"""Investigate DBSCAN parameters for B2 contour-based features."""

import time
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import silhouette_score

from src.common.utils import PROJECT_ROOT
from src.b2.clustering.models import run_dbscan


# --------------------------------------------------
# Paths
# --------------------------------------------------

RESULTS_PATH = (
    PROJECT_ROOT / "results" / "b2" / "models" / "clustering_results.joblib"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "b2" / "tables" / "dbscan_tuning.csv"
)


# --------------------------------------------------
# Experiment configuration
# --------------------------------------------------

EPS_VALUES = [5, 6, 7, 8, 9, 10]
MIN_SAMPLES_VALUES = [5, 10, 15]

RANDOM_SEED = 42


def tune_dbscan():
    """Evaluate different DBSCAN parameter combinations."""

    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            "Original clustering results not found."
        )

    # Load the exact training sample used in the original experiment
    results = joblib.load(RESULTS_PATH)
    X = results["X_data"]

    print("=" * 60)
    print("B2 DBSCAN PARAMETER INVESTIGATION")
    print("=" * 60)

    print("Dataset shape:", X.shape)

    total_combinations = (
        len(EPS_VALUES) * len(MIN_SAMPLES_VALUES)
    )

    print("Total parameter combinations:", total_combinations)

    tuning_results = []

    # --------------------------------------------------
    # Test parameter combinations
    # --------------------------------------------------

    for eps in EPS_VALUES:

        for min_samples in MIN_SAMPLES_VALUES:

            print(
                f"\nTesting eps={eps}, min_samples={min_samples}"
            )

            start_time = time.perf_counter()

            result = run_dbscan(
                X,
                eps=eps,
                min_samples=min_samples,
            )

            fit_time = time.perf_counter() - start_time

            labels = result["labels"]

            # ------------------------------------------
            # Calculate cluster sizes
            # ------------------------------------------

            non_noise_mask = labels != -1

            cluster_ids, cluster_sizes = np.unique(
                labels[non_noise_mask],
                return_counts=True,
            )

            cluster_sizes_text = ", ".join(
                f"Cluster {cluster_id}: {size}"
                for cluster_id, size in zip(
                    cluster_ids,
                    cluster_sizes,
                )
            )

            if len(cluster_ids) == 0:
                cluster_sizes_text = "No clusters detected"

            # ------------------------------------------
            # Calculate silhouette score
            # ------------------------------------------

            X_clustered = X[non_noise_mask]
            labels_clustered = labels[non_noise_mask]

            n_clusters = result["n_clusters"]

            silhouette = np.nan

            # Silhouette requires at least two clusters
            # and fewer clusters than samples.
            if (
                n_clusters >= 2
                and len(np.unique(labels_clustered))
                < len(labels_clustered)
            ):

                silhouette = silhouette_score(
                    X_clustered,
                    labels_clustered,
                    sample_size=min(1000, len(X_clustered)),
                    random_state=RANDOM_SEED,
                )

            # ------------------------------------------
            # Store results
            # ------------------------------------------

            row = {
                "eps": eps,
                "min_samples": min_samples,
                "Clusters": n_clusters,
                "Cluster Sizes": cluster_sizes_text,
                "Noise Count": result["noise_count"],
                "Noise Percentage (%)": result["noise_percentage"],
                "Silhouette Score": silhouette,
                "Fit Time (s)": fit_time,
            }

            tuning_results.append(row)

            # ------------------------------------------
            # Display results
            # ------------------------------------------

            print("Clusters:", n_clusters)

            print(
                "Cluster sizes:",
                cluster_sizes_text,
            )

            print(
                "Noise percentage:",
                f"{result['noise_percentage']:.2f}%"
            )

            print(
                "Silhouette score:",
                silhouette,
            )

            print(
                f"Fit time: {fit_time:.3f} seconds"
            )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.DataFrame(tuning_results)

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 60)
    print("DBSCAN PARAMETER INVESTIGATION COMPLETED")
    print("=" * 60)

    print(df.to_string(index=False))

    print("\nResults saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    tune_dbscan()
