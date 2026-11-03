"""Quantitative Evaluation of Clustering Models on Feature Set A2.

Computes:
  - Internal measures (unsupervised): Silhouette Score, Davies-Bouldin Index (DBI),
    Calinski-Harabasz Index (CHI), WCSS / Inertia.
  - External measures (post-hoc with true labels): Adjusted Rand Index (ARI),
    Normalized Mutual Information (NMI), Homogeneity, Completeness, V-measure.
"""

from pathlib import Path
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

from src.common.utils import (
    MODELS_A2_DIR,
    TABLES_A2_DIR,
    RANDOM_SEED,
    set_seed,
    ensure_a2_directories,
)


def evaluate_all_clustering():
    set_seed(RANDOM_SEED)
    ensure_a2_directories()

    print("=" * 70)
    print("QUANTITATIVE EVALUATION OF CLUSTERING ALGORITHMS ON FEATURE SET A2")
    print("=" * 70)

    results_file = MODELS_A2_DIR / "clustering_results.joblib"
    if not results_file.exists():
        raise FileNotFoundError(f"Missing {results_file}. Run cluster_models.py first.")

    data = joblib.load(results_file)
    X = data["X_data"]
    y_true = data["y_true"]
    models_dict = data["models"]

    rows = []

    for name, info in models_dict.items():
        labels = info["labels"]
        n_clusters = info["n_clusters"]
        fit_time = info.get("fit_time", np.nan)

        if name == "DBSCAN":
            valid_mask = labels != -1
            if np.sum(valid_mask) < 2 or len(set(labels[valid_mask])) < 2:
                print(f"Skipping internal metrics for DBSCAN: insufficient valid clusters.")
                continue
            X_eval = X[valid_mask]
            y_eval = y_true[valid_mask]
            labels_eval = labels[valid_mask]
        else:
            X_eval = X
            y_eval = y_true
            labels_eval = labels

        sil = silhouette_score(X_eval, labels_eval, sample_size=min(5000, len(X_eval)), random_state=RANDOM_SEED)
        dbi = davies_bouldin_score(X_eval, labels_eval)
        chi = calinski_harabasz_score(X_eval, labels_eval)

        ari = adjusted_rand_score(y_eval, labels_eval)
        nmi = normalized_mutual_info_score(y_eval, labels_eval)
        homo, comp, v_meas = homogeneity_completeness_v_measure(y_eval, labels_eval)

        rows.append({
            "Algorithm": name,
            "Clusters": n_clusters,
            "Silhouette Score": sil,
            "Davies-Bouldin Index": dbi,
            "Calinski-Harabasz Index": chi,
            "Adjusted Rand Index (ARI)": ari,
            "Normalized Mutual Info (NMI)": nmi,
            "Homogeneity": homo,
            "Completeness": comp,
            "V-Measure": v_meas,
            "Fit Time (s)": fit_time,
        })

    df = pd.DataFrame(rows).sort_values(by="Adjusted Rand Index (ARI)", ascending=False).reset_index(drop=True)
    csv_path = TABLES_A2_DIR / "clustering_metrics.csv"
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 70)
    print("CLUSTERING BENCHMARK LEADERBOARD (FEATURE SET A2)")
    print("=" * 70)
    print(df.to_string(index=False))
    print(f"\nSaved clustering benchmark table to: {csv_path.resolve()}")

    return df


if __name__ == "__main__":
    evaluate_all_clustering()

