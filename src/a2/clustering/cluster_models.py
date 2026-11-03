"""Unsupervised Learning and Clustering Models on Feature Set A2 (56 features).

Implements and evaluates 6 clustering algorithms strictly without class labels:
  1. k-Means Clustering
  2. Mini-Batch k-Means
  3. Gaussian Mixture Models (GMM)
  4. Agglomerative Hierarchical Clustering (Ward linkage)
  5. DBSCAN (Density-based clustering)
  6. Spectral Clustering

Also investigates PCA dimensionality reduction on top of Feature Set A2 (56 -> 15 features).
"""

from pathlib import Path
import time
import joblib
import numpy as np

from sklearn.cluster import (
    KMeans,
    MiniBatchKMeans,
    AgglomerativeClustering,
    DBSCAN,
    SpectralClustering,
)
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

from src.common.utils import (
    MODELS_A2_DIR,
    RANDOM_SEED,
    set_seed,
    ensure_a2_directories,
)
from src.a2.features import load_a2_features


def run_clustering_algorithms(n_samples: int = 10000) -> dict:
    """Fit all 6 clustering algorithms on standardized Feature Set A2."""
    set_seed(RANDOM_SEED)
    ensure_a2_directories()

    print("=" * 70)
    print(f"RUNNING UNSUPERVISED CLUSTERING ON FEATURE SET A2 (N = {n_samples:,})")
    print("=" * 70)

    _, _, _, _, X_data, y_true = load_a2_features(standardized=True)
    if len(X_data) > n_samples:
        X_data = X_data[:n_samples]
        y_true = y_true[:n_samples]

    clustering_results = {
        "X_data": X_data,
        "y_true": y_true,
        "models": {},
    }

    # 1. k-Means Clustering (k = 10 baseline)
    print("\n---> 1. Fitting k-Means (k = 10) ...")
    t0 = time.perf_counter()
    kmeans = KMeans(n_clusters=10, init="k-means++", n_init=10, random_state=RANDOM_SEED)
    labels_kmeans = kmeans.fit_predict(X_data)
    t_kmeans = time.perf_counter() - t0
    print(f"     Fit time: {t_kmeans:.2f} s | Inertia (WCSS): {kmeans.inertia_:.2f}")

    clustering_results["models"]["k-Means"] = {
        "labels": labels_kmeans,
        "model": kmeans,
        "n_clusters": 10,
        "wcss": kmeans.inertia_,
        "fit_time": t_kmeans,
    }

    # 2. Mini-Batch k-Means (k = 10)
    print("\n---> 2. Fitting Mini-Batch k-Means (k = 10) ...")
    t0 = time.perf_counter()
    mb_kmeans = MiniBatchKMeans(n_clusters=10, batch_size=1024, n_init=10, random_state=RANDOM_SEED)
    labels_mb = mb_kmeans.fit_predict(X_data)
    t_mb = time.perf_counter() - t0
    print(f"     Fit time: {t_mb:.2f} s | Inertia: {mb_kmeans.inertia_:.2f}")

    clustering_results["models"]["Mini-Batch k-Means"] = {
        "labels": labels_mb,
        "model": mb_kmeans,
        "n_clusters": 10,
        "wcss": mb_kmeans.inertia_,
        "fit_time": t_mb,
    }

    # 3. Gaussian Mixture Models (GMM, k = 10)
    print("\n---> 3. Fitting Gaussian Mixture Model (k = 10) ...")
    t0 = time.perf_counter()
    gmm = GaussianMixture(n_components=10, covariance_type="diag", reg_covar=1e-3, random_state=RANDOM_SEED)
    labels_gmm = gmm.fit_predict(X_data)
    t_gmm = time.perf_counter() - t0
    bic = gmm.bic(X_data)
    aic = gmm.aic(X_data)
    print(f"     Fit time: {t_gmm:.2f} s | BIC: {bic:.1f} | AIC: {aic:.1f}")

    clustering_results["models"]["Gaussian Mixture Model (GMM)"] = {
        "labels": labels_gmm,
        "model": gmm,
        "n_clusters": 10,
        "bic": bic,
        "aic": aic,
        "fit_time": t_gmm,
    }

    # 4. Agglomerative Hierarchical Clustering (Ward, k = 10)
    print("\n---> 4. Fitting Agglomerative Hierarchical Clustering (Ward, k = 10) ...")
    t0 = time.perf_counter()
    agg = AgglomerativeClustering(n_clusters=10, linkage="ward")
    labels_agg = agg.fit_predict(X_data)
    t_agg = time.perf_counter() - t0
    print(f"     Fit time: {t_agg:.2f} s")

    clustering_results["models"]["Agglomerative (Ward)"] = {
        "labels": labels_agg,
        "model": agg,
        "n_clusters": 10,
        "fit_time": t_agg,
    }

    # 5. DBSCAN
    print("\n---> 5. Fitting DBSCAN (eps = 4.2, min_samples = 15) ...")
    t0 = time.perf_counter()
    dbscan = DBSCAN(eps=4.2, min_samples=15, n_jobs=-1)
    labels_dbscan = dbscan.fit_predict(X_data)
    t_dbscan = time.perf_counter() - t0

    unique_db_labels = set(labels_dbscan)
    n_db_clusters = len(unique_db_labels - {-1})
    n_noise = np.sum(labels_dbscan == -1)
    pct_noise = (n_noise / len(labels_dbscan)) * 100
    print(f"     Fit time: {t_dbscan:.2f} s | Clusters Found: {n_db_clusters} | Noise: {n_noise} ({pct_noise:.1f}%)")

    clustering_results["models"]["DBSCAN"] = {
        "labels": labels_dbscan,
        "model": dbscan,
        "n_clusters": n_db_clusters,
        "noise_count": n_noise,
        "noise_pct": pct_noise,
        "fit_time": t_dbscan,
    }

    # 6. Spectral Clustering (k = 10)
    print("\n---> 6. Fitting Spectral Clustering (k = 10, nearest_neighbors) ...")
    t0 = time.perf_counter()
    spectral = SpectralClustering(
        n_clusters=10,
        affinity="nearest_neighbors",
        n_neighbors=15,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    labels_spectral = spectral.fit_predict(X_data)
    t_spectral = time.perf_counter() - t0
    print(f"     Fit time: {t_spectral:.2f} s")

    clustering_results["models"]["Spectral Clustering"] = {
        "labels": labels_spectral,
        "model": spectral,
        "n_clusters": 10,
        "fit_time": t_spectral,
    }

    # 7. PCA Dimensionality Reduction on A2 (56 -> 15 components)
    print("\n---> 7. Investigating PCA Dimensionality Reduction on A2 (56 -> 15 components) ...")
    pca = PCA(n_components=15, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_data)
    var_explained = np.sum(pca.explained_variance_ratio_) * 100
    print(f"     PCA explained variance ratio (15 components): {var_explained:.2f}%")

    kmeans_pca = KMeans(n_clusters=10, init="k-means++", n_init=10, random_state=RANDOM_SEED)
    labels_kmeans_pca = kmeans_pca.fit_predict(X_pca)

    clustering_results["models"]["k-Means (PCA-15 on A2)"] = {
        "labels": labels_kmeans_pca,
        "model": kmeans_pca,
        "n_clusters": 10,
        "pca_variance_ratio": var_explained,
        "fit_time": t_kmeans,
    }

    save_path = MODELS_A2_DIR / "clustering_results.joblib"
    joblib.dump(clustering_results, save_path)
    print(f"\nSaved all clustering results to: {save_path.resolve()}")
    return clustering_results


if __name__ == "__main__":
    run_clustering_algorithms()

