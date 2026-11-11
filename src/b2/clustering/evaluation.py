
"""Evaluation of unsupervised clustering algorithms on B2 features."""

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure,
)

from src.common.utils import PROJECT_ROOT


RESULTS_PATH = (
    PROJECT_ROOT / "results" / "b2" / "models" / "clustering_results.joblib"
)

TABLES_DIR = PROJECT_ROOT / "results" / "b2" / "tables"


def calculate_internal_metrics(X, labels):
    """Calculate internal clustering evaluation metrics."""

    unique_labels = np.unique(labels)

    # Internal metrics require at least two clusters
    if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
        return {
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
        }

    silhouette = silhouette_score(
        X,
        labels,
        sample_size=min(1000, len(X)),
        random_state=42,
    )

    davies_bouldin = davies_bouldin_score(X, labels)

    calinski_harabasz = calinski_harabasz_score(X, labels)

    return {
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "calinski_harabasz": calinski_harabasz,
    }


def calculate_external_metrics(y_true, labels):
    """Calculate external metrics using true labels after clustering."""

    ari = adjusted_rand_score(y_true, labels)

    nmi = normalized_mutual_info_score(y_true, labels)

    homogeneity, completeness, v_measure = (
        homogeneity_completeness_v_measure(y_true, labels)
    )

    return {
        "ari": ari,
        "nmi": nmi,
        "homogeneity": homogeneity,
        "completeness": completeness,
        "v_measure": v_measure,
    }


def evaluate_all_clustering():
    """Evaluate all saved B2 clustering experiments."""

    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            "Clustering results not found. Run experiment.py first."
        )

    data = joblib.load(RESULTS_PATH)

    X = data["X_data"]
    y_true = data["y_true"]
    models = data["models"]

    rows = []

    print("\nEvaluating B2 clustering algorithms...")
    print("=" * 60)

    for name, result in models.items():

        labels = result["labels"]

        print(f"\nEvaluating {name}...")

        # Handle DBSCAN noise separately for internal metrics
        if name == "DBSCAN":

            valid_mask = labels != -1

            X_internal = X[valid_mask]
            labels_internal = labels[valid_mask]

            noise_percentage = (
                np.sum(~valid_mask) / len(labels)
            ) * 100

            print(f"Noise percentage: {noise_percentage:.2f}%")

        else:
            X_internal = X
            labels_internal = labels
            noise_percentage = 0.0

        # Internal evaluation
        internal = calculate_internal_metrics(
            X_internal,
            labels_internal,
        )

        # External evaluation uses all samples, including DBSCAN noise
        external = calculate_external_metrics(
            y_true,
            labels,
        )

        rows.append({
            "Algorithm": name,
            "Clusters": result["n_clusters"],
            "Silhouette Score": internal["silhouette"],
            "Davies-Bouldin Index": internal["davies_bouldin"],
            "Calinski-Harabasz Index": internal["calinski_harabasz"],
            "Adjusted Rand Index (ARI)": external["ari"],
            "Normalized Mutual Info (NMI)": external["nmi"],
            "Homogeneity": external["homogeneity"],
            "Completeness": external["completeness"],
            "V-Measure": external["v_measure"],
            "Fit Time (s)": result["fit_time"],
        })

    # Create results directory
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Create and sort evaluation table
    columns = [
        "Algorithm",
        "Clusters",
        "Silhouette Score",
        "Davies-Bouldin Index",
        "Calinski-Harabasz Index",
        "Adjusted Rand Index (ARI)",
        "Normalized Mutual Info (NMI)",
        "Homogeneity",
        "Completeness",
        "V-Measure",
        "Fit Time (s)",
    ]

    df = pd.DataFrame(rows, columns=columns)

    df = df.sort_values(
        by="Adjusted Rand Index (ARI)",
        ascending=False,
    ).reset_index(drop=True)

    # Save evaluation results
    output_path = TABLES_DIR / "clustering_metrics.csv"

    df.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("B2 CLUSTERING EVALUATION RESULTS")
    print("=" * 60)

    print(df.to_string(index=False))

    print("\nResults saved to:", output_path)

    return df


if __name__ == "__main__":
    evaluate_all_clustering()
