"""Visualization and Structural Interpretation of Feature Set A2 Clustering.

Generates:
  1. 2-D Projections using PCA, t-SNE, and UMAP (Cluster ID vs. True Label).
  2. Contingency Heatmaps (distribution of true digits 0-9 within each cluster).
  3. Cluster Size Distribution plots.
  4. Representative Digits: Grids of cluster medoids and ambiguous boundary samples.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from src.common.utils import (
    MODELS_A2_DIR,
    FIGURES_A2_DIR,
    RANDOM_SEED,
    set_seed,
    ensure_a2_directories,
)
from src.a2.features import load_pathA_2D


def generate_all_clustering_visualizations():
    set_seed(RANDOM_SEED)
    ensure_a2_directories()

    print("=" * 70)
    print("GENERATING CLUSTERING VISUALIZATIONS & STRUCTURAL PROJECTIONS")
    print("=" * 70)

    results_path = MODELS_A2_DIR / "clustering_results.joblib"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing {results_path}. Run cluster_models.py first.")

    cluster_data = joblib.load(results_path)
    X = cluster_data["X_data"]
    y_true = cluster_data["y_true"]
    models = cluster_data["models"]

    kmeans_labels = models["k-Means"]["labels"]
    kmeans_model = models["k-Means"]["model"]

    n_plot = min(4000, len(X))
    X_sub = X[:n_plot]
    y_sub = y_true[:n_plot]
    kmeans_sub = kmeans_labels[:n_plot]

    # 1. 2-D Projections
    print("\n[Plot 1/4] Computing PCA, t-SNE, and UMAP 2-D projections ...")
    pca_2d = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(X_sub)
    tsne_2d = TSNE(n_components=2, perplexity=30, random_state=RANDOM_SEED).fit_transform(X_sub)
    umap_2d = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=RANDOM_SEED).fit_transform(X_sub)

    fig, axes = plt.subplots(3, 2, figsize=(13, 16))
    projections = [
        ("PCA (Linear Projection)", pca_2d),
        ("t-SNE (Non-Linear Manifold)", tsne_2d),
        ("UMAP (Topological Manifold)", umap_2d),
    ]

    for row_idx, (title, coords) in enumerate(projections):
        sc1 = axes[row_idx, 0].scatter(
            coords[:, 0], coords[:, 1], c=kmeans_sub, cmap="tab10", s=10, alpha=0.7
        )
        axes[row_idx, 0].set_title(f"{title} — Discovered Clusters (k-Means)", fontsize=11, fontweight="bold")
        axes[row_idx, 0].axis("off")
        fig.colorbar(sc1, ax=axes[row_idx, 0], ticks=range(10), label="Cluster ID", fraction=0.046, pad=0.04)

        sc2 = axes[row_idx, 1].scatter(
            coords[:, 0], coords[:, 1], c=y_sub, cmap="tab10", s=10, alpha=0.7
        )
        axes[row_idx, 1].set_title(f"{title} — Ground Truth Digit Labels", fontsize=11, fontweight="bold")
        axes[row_idx, 1].axis("off")
        fig.colorbar(sc2, ax=axes[row_idx, 1], ticks=range(10), label="Digit Class", fraction=0.046, pad=0.04)

    plt.suptitle("2-D Manifold Projections of Feature Set A2: Discovered Clusters vs. Ground Truth", fontsize=13, fontweight="bold", y=1.005)
    plt.tight_layout()
    proj_path = FIGURES_A2_DIR / "clustering_manifold_projections.png"
    plt.savefig(proj_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved manifold projections to: {proj_path.resolve()}")

    # 2. Contingency Matrix Heatmaps
    print("\n[Plot 2/4] Generating Contingency Matrix Heatmaps ...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, (alg_name, labels_arr) in enumerate([("k-Means", kmeans_labels), ("Agglomerative", models["Agglomerative (Ward)"]["labels"])]):
        contingency = pd.crosstab(
            pd.Series(y_true, name="True Digit (0–9)"),
            pd.Series(labels_arr, name="Discovered Cluster ID"),
            normalize="index",
        )
        sns.heatmap(
            contingency,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            cbar=True,
            ax=axes[ax_idx],
        )
        axes[ax_idx].set_title(f"Contingency Matrix: {alg_name} on Feature Set A2", fontsize=11, fontweight="bold")

    plt.tight_layout()
    contingency_path = FIGURES_A2_DIR / "cluster_contingency_heatmaps.png"
    plt.savefig(contingency_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved contingency heatmaps to: {contingency_path.resolve()}")

    # 3. Cluster Size Distribution
    print("\n[Plot 3/4] Generating Cluster Size Distributions ...")
    plt.figure(figsize=(10, 5))
    cluster_counts = np.bincount(kmeans_labels, minlength=10)
    bars = plt.bar(range(10), cluster_counts, color="#2ca02c", edgecolor="black", alpha=0.85)
    plt.title("Cluster Size Distribution (k-Means on Feature Set A2, N = 10,000)", fontsize=12, fontweight="bold")
    plt.xlabel("Cluster ID (0–9)", fontsize=11)
    plt.ylabel("Number of Assigned Samples", fontsize=11)
    plt.xticks(range(10))
    plt.grid(axis="y", linestyle=":", alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, h + 20, f"{h:,}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    size_path = FIGURES_A2_DIR / "cluster_size_distributions.png"
    plt.savefig(size_path, dpi=300)
    plt.close()
    print(f"  -> Saved cluster size distributions to: {size_path.resolve()}")

    # 4. Representative Digit Samples
    print("\n[Plot 4/4] Extracting Cluster Medoids and Ambiguous Boundary Digits ...")
    _, _, _, _, X_test_2d, _ = load_pathA_2D()
    X_test_2d = X_test_2d[:len(X)]

    centroids = kmeans_model.cluster_centers_
    medoid_indices = []
    boundary_indices = []

    for k in range(10):
        members = np.where(kmeans_labels == k)[0]
        if len(members) == 0:
            continue
        member_features = X[members]
        dists = np.linalg.norm(member_features - centroids[k], axis=1)
        medoid_indices.append(members[np.argmin(dists)])
        boundary_indices.append(members[np.argmax(dists)])

    fig, axes = plt.subplots(2, 10, figsize=(16, 4))
    for k in range(10):
        m_idx = medoid_indices[k]
        axes[0, k].imshow(X_test_2d[m_idx, 0], cmap="gray", interpolation="nearest")
        axes[0, k].set_title(f"Clust {k} Medoid\n(True: {y_true[m_idx]})", fontsize=8, fontweight="bold")
        axes[0, k].axis("off")

        b_idx = boundary_indices[k]
        axes[1, k].imshow(X_test_2d[b_idx, 0], cmap="gray", interpolation="nearest")
        axes[1, k].set_title(f"Clust {k} Boundary\n(True: {y_true[b_idx]})", fontsize=8, color="darkred")
        axes[1, k].axis("off")

    plt.suptitle("Representative Digits per Discovered Cluster: Centroids/Medoids (Top) vs. Boundary Outliers (Bottom)", fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    rep_path = FIGURES_A2_DIR / "cluster_representative_samples.png"
    plt.savefig(rep_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved representative samples to: {rep_path.resolve()}")

    print("\nALL CLUSTERING VISUALIZATIONS GENERATED SUCCESSFULLY!")


if __name__ == "__main__":
    generate_all_clustering_visualizations()

