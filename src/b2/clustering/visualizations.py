
"""Professional visualisations for the B2 clustering experiments."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap

from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.common.utils import PROJECT_ROOT
from src.a2.augmentation import load_augmented_training_set
from src.b2.contours import extract_contours


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

RESULTS_PATH = (
    PROJECT_ROOT / "results" / "b2" / "models" / "clustering_results.joblib"
)

METRICS_PATH = (
    PROJECT_ROOT / "results" / "b2" / "tables" / "clustering_metrics.csv"
)

FIGURES_DIR = PROJECT_ROOT / "results" / "b2" / "figures"


# ---------------------------------------------------------
# VISUAL THEME
# ---------------------------------------------------------

BACKGROUND = "#FFFFFF"
DARK_PURPLE = "#542078"
DEEP_PURPLE = "#895BB5"
MEDIUM_PURPLE = "#B99AD5"
LIGHT_PURPLE = "#DCCCEB"
VERY_LIGHT_PURPLE = "#F5F0FA"
GRID_GREY = "#D9D9D9"
TEXT_COLOUR = "#30243A"

PURPLE_GRADIENT = sns.color_palette(
    [
        VERY_LIGHT_PURPLE,
        LIGHT_PURPLE,
        MEDIUM_PURPLE,
        DEEP_PURPLE,
        DARK_PURPLE,
    ],
    as_cmap=True,
)

# Ten distinct categorical colours.
# The same colours are used for actual digits and k-Means clusters.
CLASS_COLOURS = [
    "#542078",
    "#895BB5",
    "#B99AD5",
    "#D17A9A",
    "#E6A15C",
    "#D7BD58",
    "#7EAD8B",
    "#5A9CA5",
    "#6486B8",
    "#A18AB8",
]

CLASS_CMAP = ListedColormap(CLASS_COLOURS)
CLASS_NORM = BoundaryNorm(np.arange(-0.5, 10.5, 1), 10)


def apply_theme():
    """Apply the common B2 visualisation theme."""

    sns.set_theme(
        style="whitegrid",
        context="notebook",
        font_scale=1.0,
    )

    plt.rcParams.update({
        "figure.facecolor": BACKGROUND,
        "axes.facecolor": BACKGROUND,
        "axes.edgecolor": GRID_GREY,
        "axes.labelcolor": TEXT_COLOUR,
        "text.color": TEXT_COLOUR,
        "xtick.color": TEXT_COLOUR,
        "ytick.color": TEXT_COLOUR,
        "axes.titlecolor": DARK_PURPLE,
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "grid.color": GRID_GREY,
        "grid.linestyle": "--",
        "grid.alpha": 0.45,
        "legend.frameon": False,
        "savefig.facecolor": BACKGROUND,
    })


def save_figure(fig, filename):
    """Save a figure using consistent formatting."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        FIGURES_DIR / filename,
        dpi=300,
        bbox_inches="tight",
        facecolor=BACKGROUND,
    )

    plt.close(fig)


# ---------------------------------------------------------
# ARI COMPARISON
# ---------------------------------------------------------

def plot_ari_comparison(metrics):
    """Compare clustering algorithms using Adjusted Rand Index."""

    df = metrics.sort_values(
        "Adjusted Rand Index (ARI)",
        ascending=False,
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        df["Algorithm"],
        df["Adjusted Rand Index (ARI)"],
        color=MEDIUM_PURPLE,
        edgecolor=DARK_PURPLE,
        linewidth=0.7,
    )

    if len(bars):
        bars[0].set_color(DARK_PURPLE)

    for bar in bars:
        value = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title("B2 Clustering: Adjusted Rand Index Comparison")
    ax.set_xlabel("Clustering Algorithm")
    ax.set_ylabel("Adjusted Rand Index (ARI)")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    fig.tight_layout()
    save_figure(fig, "ari_comparison.png")


# ---------------------------------------------------------
# SILHOUETTE COMPARISON
# ---------------------------------------------------------

def plot_silhouette_comparison(metrics):
    """Compare internal cluster separation using Silhouette Score."""

    df = metrics.dropna(
        subset=["Silhouette Score"]
    ).sort_values(
        "Silhouette Score",
        ascending=False,
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        df["Algorithm"],
        df["Silhouette Score"],
        color=MEDIUM_PURPLE,
        edgecolor=DARK_PURPLE,
        linewidth=0.7,
    )

    if len(bars):
        bars[0].set_color(DARK_PURPLE)

    for bar in bars:
        value = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title("B2 Clustering: Silhouette Score Comparison")
    ax.set_xlabel("Clustering Algorithm")
    ax.set_ylabel("Silhouette Score")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    fig.tight_layout()
    save_figure(fig, "silhouette_comparison.png")


# ---------------------------------------------------------
# ALL ALGORITHM CLUSTER-SIZE DISTRIBUTIONS
# ---------------------------------------------------------

def plot_all_cluster_size_distributions(results):
    """Compare cluster-size distributions across all six algorithms."""

    models = results["models"]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(17, 10),
    )

    axes = axes.flatten()

    for ax, (algorithm_name, result) in zip(axes, models.items()):

        labels = result["labels"]

        cluster_ids, cluster_sizes = np.unique(
            labels,
            return_counts=True,
        )

        display_labels = [
            "Noise" if cluster_id == -1 else str(cluster_id)
            for cluster_id in cluster_ids
        ]

        bar_colours = [
            GRID_GREY if cluster_id == -1 else MEDIUM_PURPLE
            for cluster_id in cluster_ids
        ]

        bars = ax.bar(
            display_labels,
            cluster_sizes,
            color=bar_colours,
            edgecolor=DARK_PURPLE,
            linewidth=0.5,
        )

        for bar, size in zip(bars, cluster_sizes):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{size:,}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.set_title(algorithm_name)
        ax.set_xlabel("Cluster ID")
        ax.set_ylabel("Number of Samples")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)

    fig.suptitle(
        "B2: Cluster Size Distribution Across Algorithms",
        fontsize=16,
        fontweight="bold",
        color=DARK_PURPLE,
    )

    fig.tight_layout()
    save_figure(fig, "all_cluster_size_distributions.png")


# ---------------------------------------------------------
# CONTINGENCY HEATMAPS
# ---------------------------------------------------------

def plot_contingency_heatmap(y_true, labels, algorithm_name):
    """Plot the digit composition of clusters as percentages."""

    composition = pd.crosstab(
        pd.Series(y_true, name="Actual Digit"),
        pd.Series(labels, name="Predicted Cluster"),
    )

    # Normalise each cluster column to percentages.
    composition_percentage = composition.div(
        composition.sum(axis=0),
        axis=1,
    ) * 100

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(
        composition_percentage,
        cmap=PURPLE_GRADIENT,
        vmin=0,
        vmax=100,
        annot=True,
        fmt=".1f",
        annot_kws={"size": 9},
        linewidths=0.6,
        linecolor="white",
        cbar_kws={
            "label": "Samples within cluster (%)",
            "shrink": 0.85,
        },
        ax=ax,
    )

    ax.set_title(
        f"Cluster Composition – {algorithm_name}",
        fontsize=15,
        fontweight="bold",
        color=DARK_PURPLE,
        pad=18,
    )

    ax.set_xlabel("Predicted Cluster", labelpad=10)
    ax.set_ylabel("Actual Digit", labelpad=10)
    ax.tick_params(axis="both", labelsize=10)

    fig.tight_layout()

    filename = (
        algorithm_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
        + "_contingency_matrix.png"
    )

    save_figure(fig, filename)


def plot_all_contingency_matrices(results):
    """Generate contingency heatmaps for all six algorithms."""

    y_true = results["y_true"]

    for algorithm_name, result in results["models"].items():

        print(f"Generating contingency matrix: {algorithm_name}")

        plot_contingency_heatmap(
            y_true,
            result["labels"],
            algorithm_name,
        )


# ---------------------------------------------------------
# PCA, t-SNE AND UMAP
# ---------------------------------------------------------

def plot_dimensionality_reduction(results):
    """Visualise B2 features using PCA, t-SNE and UMAP."""

    y_true = results["y_true"]
    kmeans_labels = results["models"]["k-Means"]["labels"]
    X = results["X_data"]

    methods = {
        "PCA": PCA(
            n_components=2,
            random_state=42,
        ),

        "t-SNE": TSNE(
            n_components=2,
            perplexity=30,
            init="pca",
            learning_rate="auto",
            max_iter=1000,
            random_state=42,
        ),

        "UMAP": umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            random_state=42,
        ),
    }

    for method_name, reducer in methods.items():

        print(f"Generating {method_name} projection...")

        embedding = reducer.fit_transform(X)

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(15, 6),
        )

        # Actual digit labels.
        actual_plot = axes[0].scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=y_true,
            cmap=CLASS_CMAP,
            norm=CLASS_NORM,
            s=10,
            alpha=0.65,
        )

        axes[0].set_title("Actual Digit Labels")
        axes[0].set_xlabel("Component 1")
        axes[0].set_ylabel("Component 2")

        # k-Means cluster assignments.
        cluster_plot = axes[1].scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=kmeans_labels,
            cmap=CLASS_CMAP,
            norm=CLASS_NORM,
            s=10,
            alpha=0.65,
        )

        axes[1].set_title("k-Means Cluster Assignments")
        axes[1].set_xlabel("Component 1")
        axes[1].set_ylabel("Component 2")

        for ax in axes:
            ax.grid(False)

        # Consistent digit legend.
        fig.colorbar(
            actual_plot,
            ax=axes[0],
            ticks=range(10),
            label="Actual Digit",
            fraction=0.046,
            pad=0.04,
        )

        fig.colorbar(
            cluster_plot,
            ax=axes[1],
            ticks=range(10),
            label="k-Means Cluster",
            fraction=0.046,
            pad=0.04,
        )

        fig.suptitle(
            f"B2 Feature Visualisation using {method_name}",
            fontsize=15,
            fontweight="bold",
            color=DARK_PURPLE,
        )

        fig.tight_layout()

        filename = (
            method_name.lower().replace("-", "_")
            + "_visualization.png"
        )

        save_figure(fig, filename)

        print(f"{method_name} visualisation saved.")


# ---------------------------------------------------------
# REPRESENTATIVE CONTOUR SAMPLES
# ---------------------------------------------------------

def plot_representative_samples(results):
    """Display three representative contour images from each k-Means cluster."""

    X = results["X_data"]
    y_true = results["y_true"]
    sample_indices = results["sample_indices"]

    kmeans = results["models"]["k-Means"]["model"]
    labels = results["models"]["k-Means"]["labels"]

    # Load the original augmented training images.
    X_augmented, _ = load_augmented_training_set()

    # Retrieve the exact images used in the clustering experiment.
    selected_images = X_augmented[sample_indices]

    fig, axes = plt.subplots(
        10,
        3,
        figsize=(9, 22),
    )

    for cluster_id in range(kmeans.n_clusters):

        cluster_indices = np.where(labels == cluster_id)[0]

        if len(cluster_indices) == 0:
            continue

        # Calculate distances from the cluster centre.
        distances = np.linalg.norm(
            X[cluster_indices] - kmeans.cluster_centers_[cluster_id],
            axis=1,
        )

        # Select the three closest samples.
        nearest_positions = np.argsort(distances)[:3]
        representative_indices = cluster_indices[nearest_positions]

        # Extract contours from the selected original images.
        representative_images = extract_contours(
            selected_images[representative_indices]
        )

        for column, sample_index in enumerate(representative_indices):

            axes[cluster_id, column].imshow(
                representative_images[column],
                cmap="gray",
                vmin=0,
                vmax=1,
            )

            axes[cluster_id, column].set_title(
                f"Digit: {y_true[sample_index]}",
                fontsize=10,
                color=DARK_PURPLE,
            )

            axes[cluster_id, column].axis("off")

        axes[cluster_id, 0].set_ylabel(
            f"Cluster {cluster_id}",
            fontsize=11,
            fontweight="bold",
            color=DARK_PURPLE,
        )

    fig.suptitle(
        "B2: Representative Contour Samples from k-Means Clusters",
        fontsize=15,
        fontweight="bold",
        color=DARK_PURPLE,
        y=0.995,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "kmeans_representative_samples.png",
    )

    print("Representative cluster samples saved.")


# ---------------------------------------------------------
# MAIN VISUALISATION PIPELINE
# ---------------------------------------------------------

def generate_visualizations():
    """Generate all B2 clustering visualisations."""

    if not RESULTS_PATH.exists():
        raise FileNotFoundError("Clustering results not found.")

    if not METRICS_PATH.exists():
        raise FileNotFoundError("Clustering metrics not found.")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    apply_theme()

    results = joblib.load(RESULTS_PATH)
    metrics = pd.read_csv(METRICS_PATH)

    print("=" * 65)
    print("GENERATING B2 CLUSTERING VISUALISATIONS")
    print("=" * 65)

    plot_ari_comparison(metrics)
    print("ARI comparison saved.")

    plot_silhouette_comparison(metrics)
    print("Silhouette comparison saved.")

    plot_all_cluster_size_distributions(results)

    plot_all_contingency_matrices(results)

    plot_dimensionality_reduction(results)
    print("Dimensionality-reduction visualisations saved.")

    plot_representative_samples(results)

    # Remove the two obsolete individual k-Means figures.
    obsolete_files = [
        "kmeans_cluster_composition.png",
        "cluster_size_distribution.png",
    ]

    for filename in obsolete_files:
        obsolete_path = FIGURES_DIR / filename

        if obsolete_path.exists():
            obsolete_path.unlink()
            print(f"Removed obsolete figure: {filename}")

    print("\nAll B2 visualisations generated successfully.")
    print("Output directory:", FIGURES_DIR)


if __name__ == "__main__":
    generate_visualizations()
