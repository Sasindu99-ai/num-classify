"""
Feature Space 6 (FS6) Figure Generation Suite

This script reads the saved outputs from `data/contour_concentric_rings/` and
generates publication-ready PNG figures in `results/FS6/`:
  1. original_vs_augmented.png:
     - 10 digits (0-9) x 3 columns (Original, Augmented Copy 1, Augmented Copy 2).
  2. raw_vs_contour.png:
     - Side-by-side comparison of raw grayscale digits vs 1-pixel outer contours.
  3. ring_feature_profile.png:
     - Heatmap of mean ring feature activations across 10 digits and 14 concentric rings.
  4. class_balance.png:
     - Grouped bar chart showing label balance across original train, augmented train, and validation splits.
  5. ring_geometry_map.png:
     - Educational schematic showing the 14 concentric rings and pixel counts.

Run from project root:
    python src/FS6/2_make_figures.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "contour_concentric_rings"
RESULTS_DIR = PROJECT_ROOT / "results" / "FS6"


def plot_1_original_vs_augmented(
    train_raw: np.ndarray,
    train_labels: np.ndarray,
    train_source_idx: np.ndarray,
    train_is_augmented: np.ndarray,
    save_path: Path,
) -> None:
    """
    Figure 1: Original vs 2 augmented copies for each digit class (0 through 9).
    Organized as a 10 x 3 grid: 10 rows (digits 0-9), 3 columns (Original, Copy 1, Copy 2).
    """
    print("[1/5] Generating Figure 1: original_vs_augmented.png...")
    fig, axes = plt.subplots(10, 3, figsize=(7.5, 20), dpi=300)

    # Find one representative sample for each digit 0..9 in the original training split
    for digit in range(10):
        # Find first occurrence of this digit in originals
        orig_indices = np.where((train_labels == digit) & (train_is_augmented == 0))[0]
        chosen_orig_idx = orig_indices[0]
        orig_source_idx = train_source_idx[chosen_orig_idx]

        # Find the 2 augmented copies sharing this exact source_idx
        aug_indices = np.where((train_source_idx == orig_source_idx) & (train_is_augmented == 1))[0]

        # Col 0: Original
        axes[digit, 0].imshow(train_raw[chosen_orig_idx], cmap="gray", vmin=0, vmax=255)
        axes[digit, 0].set_ylabel(f"Digit {digit}", fontsize=12, fontweight="bold", rotation=0, labelpad=25, va="center")
        if digit == 0:
            axes[digit, 0].set_title("Original\n(is_augmented=0)", fontsize=11, fontweight="bold", pad=10)

        # Col 1: Augmented Copy 1
        if len(aug_indices) > 0:
            axes[digit, 1].imshow(train_raw[aug_indices[0]], cmap="gray", vmin=0, vmax=255)
        if digit == 0:
            axes[digit, 1].set_title("Augmented Copy 1\n(is_augmented=1)", fontsize=11, fontweight="bold", pad=10)

        # Col 2: Augmented Copy 2
        if len(aug_indices) > 1:
            axes[digit, 2].imshow(train_raw[aug_indices[1]], cmap="gray", vmin=0, vmax=255)
        if digit == 0:
            axes[digit, 2].set_title("Augmented Copy 2\n(is_augmented=1)", fontsize=11, fontweight="bold", pad=10)

        for col in range(3):
            axes[digit, col].set_xticks([])
            axes[digit, col].set_yticks([])

    plt.suptitle("FS6 Training Augmentation: Original vs Augmented Copies (Digits 0-9)", fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved: {save_path}")


def plot_2_raw_vs_contour(
    train_raw: np.ndarray,
    train_contour: np.ndarray,
    train_labels: np.ndarray,
    train_is_augmented: np.ndarray,
    save_path: Path,
) -> None:
    """
    Figure 2: Raw vs Contour for each digit class (0 through 9).
    Organized as a 2-row by 10-column comparison (Top: Raw, Bottom: Contour).
    """
    print("[2/5] Generating Figure 2: raw_vs_contour.png...")
    fig, axes = plt.subplots(2, 10, figsize=(16, 4.2), dpi=300)

    for digit in range(10):
        # Pick first clean original sample for this digit
        idx = np.where((train_labels == digit) & (train_is_augmented == 0))[0][0]

        # Top row: Raw grayscale digit
        axes[0, digit].imshow(train_raw[idx], cmap="gray", vmin=0, vmax=255)
        axes[0, digit].set_title(f"Digit {digit}", fontsize=11, fontweight="bold")
        axes[0, digit].set_xticks([])
        axes[0, digit].set_yticks([])

        # Bottom row: Binary 1-px contour (inverted grayscale for crisp outline visibility)
        axes[1, digit].imshow(train_contour[idx], cmap="gray_r", vmin=0, vmax=1)
        axes[1, digit].set_xticks([])
        axes[1, digit].set_yticks([])

    axes[0, 0].set_ylabel("Raw Image\n(28x28, 0-255)", fontsize=11, fontweight="bold", rotation=0, labelpad=55, va="center")
    axes[1, 0].set_ylabel("1-px Contour\n(0/1 Binary)", fontsize=11, fontweight="bold", rotation=0, labelpad=55, va="center")

    plt.suptitle("FS6 Contour Extraction: Raw Grayscale Image vs Morphological Boundary (Threshold=100, Erode=3x3)",
                 fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved: {save_path}")


def plot_3_ring_feature_profile(
    df_train: pd.DataFrame,
    save_path: Path,
) -> None:
    """
    Figure 3: Mean ring-feature profile per digit class (0 to 9) across the 14 concentric rings.
    Includes a 2D Heatmap (Top) and Line Profiles (Bottom) for direct interpretability.
    """
    print("[3/5] Generating Figure 3: ring_feature_profile.png...")
    feat_cols = [f"ring_{k:02d}" for k in range(14)]
    grouped = df_train.groupby("label")[feat_cols].mean()

    fig, (ax_heat, ax_line) = plt.subplots(2, 1, figsize=(12, 10), dpi=300, gridspec_kw={"height_ratios": [1, 1.2]})

    # 1. Heatmap
    cax = ax_heat.imshow(grouped.to_numpy(), cmap="viridis", aspect="auto")
    fig.colorbar(cax, ax=ax_heat, orientation="vertical", label="Mean Contour Density", pad=0.02)

    ax_heat.set_title("Mean Concentric Ring Contour Density Heatmap (10 Digits x 14 Rings)", fontsize=12, fontweight="bold")
    ax_heat.set_xlabel("Concentric Ring Index (0 = Outer Border, 13 = Center 2x2)", fontsize=11)
    ax_heat.set_ylabel("Digit Class", fontsize=11, fontweight="bold")
    ax_heat.set_xticks(range(14))
    ax_heat.set_xticklabels([f"R{k:02d}" for k in range(14)])
    ax_heat.set_yticks(range(10))
    ax_heat.set_yticklabels(range(10))

    # Add numeric annotations inside heatmap cells
    for d in range(10):
        for k in range(14):
            val = grouped.iloc[d, k]
            color = "white" if val < 0.12 else "black"
            ax_heat.text(k, d, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

    # 2. Line Profile comparing structural topologies
    rings = range(14)
    # Highlight distinctive digits: '0' (hollow center), '1' (narrow center stroke), '8' (dense loops)
    markers = ["o", "s", "^", "v", "<", ">", "p", "*", "D", "h"]
    for digit in range(10):
        ax_line.plot(
            rings,
            grouped.iloc[digit].to_numpy(),
            label=f"Digit {digit}",
            marker=markers[digit],
            markersize=5,
            linewidth=1.8,
            alpha=0.85,
        )

    ax_line.set_title("Concentric Ring Density Profiles by Digit Class", fontsize=12, fontweight="bold")
    ax_line.set_xlabel("Concentric Ring Index (0 = Outer Border, 13 = Center 2x2)", fontsize=11)
    ax_line.set_ylabel("Mean Contour Density (Fraction of White Edge Pixels)", fontsize=11)
    ax_line.set_xticks(range(14))
    ax_line.set_xticklabels([f"R{k:02d}" for k in range(14)])
    ax_line.grid(True, linestyle="--", alpha=0.5)
    ax_line.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Digit", frameon=True)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved: {save_path}")


def plot_4_class_balance(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    save_path: Path,
) -> None:
    """
    Figure 4: Class distribution across training originals (54k), training augmented (108k), and val (6k).
    """
    print("[4/5] Generating Figure 4: class_balance.png...")
    train_orig = df_train.loc[df_train["is_augmented"] == 0, "label"].value_counts().sort_index()
    train_aug = df_train.loc[df_train["is_augmented"] == 1, "label"].value_counts().sort_index()
    val_counts = df_val["label"].value_counts().sort_index()

    digits = np.arange(10)
    width = 0.26

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    bars1 = ax.bar(digits - width, train_orig.values, width, label="Train Originals (N=54,000)", color="#2b5c8f")
    bars2 = ax.bar(digits, train_aug.values, width, label="Train Augmented (N=108,000)", color="#469990")
    bars3 = ax.bar(digits + width, val_counts.values, width, label="Validation (N=6,000)", color="#e6ab02")

    ax.set_title("Class Balance Across Dataset Splits (EE5130 FS6 Preprocessing)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Digit Class", fontsize=11, fontweight="bold")
    ax.set_ylabel("Sample Count", fontsize=11, fontweight="bold")
    ax.set_xticks(digits)
    ax.set_xticklabels(digits, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10)

    # Annotate values on top of bars
    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 100, f"{yval:,}", ha="center", va="bottom", fontsize=7, rotation=90)
    for bar in bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 200, f"{yval:,}", ha="center", va="bottom", fontsize=7, rotation=90)
    for bar in bars3:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 50, f"{yval:,}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_ylim(0, max(train_aug.values) * 1.25)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved: {save_path}")


def plot_5_ring_geometry_map(save_path: Path) -> None:
    """
    Figure 5 (Supplementary): Visual diagram of the 14 concentric rings on the 28x28 grid
    along with their exact pixel counts (108, 100, ..., 4).
    """
    print("[5/5] Generating Supplementary Figure: ring_geometry_map.png...")
    r, c = np.indices((28, 28))
    ring_map = np.minimum(np.minimum(r, c), np.minimum(27 - r, 27 - c))
    ring_sizes = np.bincount(ring_map.ravel(), minlength=14)

    fig, (ax_grid, ax_sizes) = plt.subplots(1, 2, figsize=(13, 6), dpi=300, gridspec_kw={"width_ratios": [1, 1.2]})

    # Grid visualization
    cax = ax_grid.imshow(ring_map, cmap="tab20b", vmin=0, vmax=13)
    ax_grid.set_title("14 Concentric Rings Geometry (28x28)", fontsize=12, fontweight="bold")
    ax_grid.set_xlabel("Pixel Column (0..27)", fontsize=10)
    ax_grid.set_ylabel("Pixel Row (0..27)", fontsize=10)
    ax_grid.set_xticks(range(0, 28, 4))
    ax_grid.set_yticks(range(0, 28, 4))

    # Bar chart of ring sizes
    bars = ax_sizes.bar(range(14), ring_sizes, color="#386cb0", edgecolor="black", alpha=0.85)
    ax_sizes.set_title("Ring Pixel Capacity: size(k) = 108 - 8*k (Sum = 784 px)", fontsize=12, fontweight="bold")
    ax_sizes.set_xlabel("Ring Index k (0=Outer, 13=Center)", fontsize=10)
    ax_sizes.set_ylabel("Pixel Count in Ring", fontsize=10)
    ax_sizes.set_xticks(range(14))
    ax_sizes.set_xticklabels([f"R{k:02d}" for k in range(14)], rotation=45)
    ax_sizes.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, size in zip(bars, ring_sizes):
        ax_sizes.text(bar.get_x() + bar.get_width() / 2, size + 2, str(size), ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved: {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FS6 Figure Generation Suite")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Path to preprocessed data directory (default: data/contour_concentric_rings)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Path to results directory (default: results/FS6)",
    )
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 80)
    print("STARTING FS6 FIGURE GENERATION")
    print(f"Data source:    {args.data_dir}")
    print(f"Output figures: {args.results_dir}")
    print("=" * 80)

    # Load required data
    npz_train = args.data_dir / "images_train.npz"
    csv_train = args.data_dir / "train.csv"
    csv_val = args.data_dir / "val.csv"

    for path in [npz_train, csv_train, csv_val]:
        if not path.is_file():
            raise FileNotFoundError(f"Required artifact not found: {path}. Run 0_preprocessing_pipeline.py first!")

    with np.load(npz_train) as data:
        train_raw = data["raw"]
        train_contour = data["contour"]
        train_labels = data["labels"]
        train_source_idx = data["source_idx"]
        train_is_augmented = data["is_augmented"]

    df_train = pd.read_csv(csv_train)
    df_val = pd.read_csv(csv_val)

    # Generate Figures
    plot_1_original_vs_augmented(
        train_raw, train_labels, train_source_idx, train_is_augmented,
        args.results_dir / "original_vs_augmented.png"
    )
    plot_2_raw_vs_contour(
        train_raw, train_contour, train_labels, train_is_augmented,
        args.results_dir / "raw_vs_contour.png"
    )
    plot_3_ring_feature_profile(
        df_train,
        args.results_dir / "ring_feature_profile.png"
    )
    plot_4_class_balance(
        df_train, df_val,
        args.results_dir / "class_balance.png"
    )
    plot_5_ring_geometry_map(
        args.results_dir / "ring_geometry_map.png"
    )

    print("=" * 80)
    print(f"ALL 5 FIGURES GENERATED SUCCESSFULLY IN {time.time() - t0:.2f}s")
    print(f"Figures saved in: {args.results_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
