
"""Clustering algorithms for B2 contour-based features.

This module implements six unsupervised learning algorithms
using the 56-dimensional B2 contour feature representation.
"""

import time
import numpy as np

from sklearn.cluster import (
    KMeans,
    MiniBatchKMeans,
    AgglomerativeClustering,
    DBSCAN,
    SpectralClustering,
)
from sklearn.mixture import GaussianMixture


def validate_features(X: np.ndarray) -> None:
    """Check whether the input feature matrix is valid."""

    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be a NumPy array.")

    if X.ndim != 2:
        raise ValueError("Expected a 2D feature matrix.")

    if X.shape[0] < 2:
        raise ValueError("At least two samples are required.")

    if not np.isfinite(X).all():
        raise ValueError("Feature matrix contains invalid values.")


def run_kmeans(
    X: np.ndarray,
    n_clusters: int = 10,
    random_state: int = 42,
) -> dict:
    """Perform k-Means clustering on B2 features."""

    validate_features(X)

    if not 2 <= n_clusters <= len(X):
        raise ValueError("Invalid number of clusters.")

    model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=20,
        max_iter=300,
        random_state=random_state,
    )

    start_time = time.perf_counter()
    labels = model.fit_predict(X)
    fit_time = time.perf_counter() - start_time

    return {
        "algorithm": "k-Means",
        "model": model,
        "labels": labels,
        "n_clusters": len(np.unique(labels)),
        "inertia": model.inertia_,
        "fit_time": fit_time,
    }


def run_minibatch_kmeans(
    X: np.ndarray,
    n_clusters: int = 10,
    random_state: int = 42,
) -> dict:
    """Perform Mini-Batch k-Means clustering on B2 features."""

    validate_features(X)

    if not 2 <= n_clusters <= len(X):
        raise ValueError("Invalid number of clusters.")

    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=512,
        n_init=10,
        max_iter=300,
        random_state=random_state,
    )

    start_time = time.perf_counter()
    labels = model.fit_predict(X)
    fit_time = time.perf_counter() - start_time

    return {
        "algorithm": "Mini-Batch k-Means",
        "model": model,
        "labels": labels,
        "n_clusters": len(np.unique(labels)),
        "inertia": model.inertia_,
        "fit_time": fit_time,
    }


def run_gaussian_mixture(
    X: np.ndarray,
    n_components: int = 10,
    random_state: int = 42,
) -> dict:
    """Perform Gaussian Mixture Model clustering on B2 features."""

    validate_features(X)

    if not 2 <= n_components <= len(X):
        raise ValueError("Invalid number of components.")

    model = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        reg_covar=1e-3,
        n_init=3,
        max_iter=200,
        random_state=random_state,
    )

    start_time = time.perf_counter()
    labels = model.fit_predict(X)
    fit_time = time.perf_counter() - start_time

    return {
        "algorithm": "Gaussian Mixture Model",
        "model": model,
        "labels": labels,
        "n_clusters": len(np.unique(labels)),
        "bic": model.bic(X),
        "aic": model.aic(X),
        "fit_time": fit_time,
    }


def run_agglomerative(
    X: np.ndarray,
    n_clusters: int = 10,
) -> dict:
    """Perform Ward hierarchical clustering on B2 features."""

    validate_features(X)

    if not 2 <= n_clusters <= len(X):
        raise ValueError("Invalid number of clusters.")

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="ward",
    )

    start_time = time.perf_counter()
    labels = model.fit_predict(X)
    fit_time = time.perf_counter() - start_time

    return {
        "algorithm": "Agglomerative Clustering",
        "model": model,
        "labels": labels,
        "n_clusters": len(np.unique(labels)),
        "fit_time": fit_time,
    }


def run_dbscan(
    X: np.ndarray,
    eps: float = 5.0,
    min_samples: int = 15,
) -> dict:
    """Perform density-based clustering on B2 features."""

    validate_features(X)

    model = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        n_jobs=-1,
    )

    start_time = time.perf_counter()
    labels = model.fit_predict(X)
    fit_time = time.perf_counter() - start_time

    n_clusters = len(set(labels) - {-1})
    noise_count = int(np.sum(labels == -1))
    noise_percentage = (noise_count / len(labels)) * 100

    return {
        "algorithm": "DBSCAN",
        "model": model,
        "labels": labels,
        "n_clusters": n_clusters,
        "noise_count": noise_count,
        "noise_percentage": noise_percentage,
        "fit_time": fit_time,
    }


def run_spectral(
    X: np.ndarray,
    n_clusters: int = 10,
    n_neighbors: int = 15,
    random_state: int = 42,
) -> dict:
    """Perform spectral clustering on B2 features."""

    validate_features(X)

    if not 2 <= n_clusters <= len(X):
        raise ValueError("Invalid number of clusters.")

    if not 1 <= n_neighbors < len(X):
        raise ValueError("Invalid number of neighbors.")

    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        n_neighbors=n_neighbors,
        assign_labels="kmeans",
        random_state=random_state,
        n_jobs=-1,
    )

    start_time = time.perf_counter()
    labels = model.fit_predict(X)
    fit_time = time.perf_counter() - start_time

    return {
        "algorithm": "Spectral Clustering",
        "model": model,
        "labels": labels,
        "n_clusters": len(np.unique(labels)),
        "fit_time": fit_time,
    }
