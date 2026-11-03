"""Phase 1 Verification Gate and Sanity Suite for Feature Set A2.

Verifies:
  1. Clean partitions and intensity ranges in [0, 1].
  2. Augmented Path A training set (162,000 samples).
  3. Feature Set A2 (56 features) shape and standardization without leakage.
  4. 2-D CNN tensor shapes (N, 1, 28, 28).
  5. Generates results/a2/phase1_verification_figure.png.
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.common.utils import (
    AUGMENTED_RAW_DIR,
    FEATURES_A2_DIR,
    RESULTS_A2_DIR,
    RANDOM_SEED,
    set_seed,
    ensure_a2_directories,
)
from src.common.data_loading import load_clean_partitions
from src.a2.features import load_a2_features, load_pathA_2D


def verify_phase1_pipeline() -> bool:
    """Execute all assertions for Phase 1 verification."""
    print("=" * 70)
    print("RUNNING PHASE 1 VERIFICATION GATE (FEATURE SET A2)")
    print("=" * 70)
    set_seed(RANDOM_SEED)
    ensure_a2_directories()

    # 1. Verify Clean Partitions
    print("\n[Check 1/5] Verifying clean partitions...")
    X_train_clean, y_train_clean, X_val, y_val, X_test, y_test = load_clean_partitions()

    assert X_train_clean.shape == (54000, 28, 28)
    assert y_train_clean.shape == (54000,)
    assert X_val.shape == (6000, 28, 28)
    assert y_val.shape == (6000,)
    assert X_test.shape == (10000, 28, 28)
    assert y_test.shape == (10000,)
    assert 0.0 <= X_train_clean.min() and X_train_clean.max() <= 1.0
    print("  -> Clean partitions: 54,000 train, 6,000 val, 10,000 test. Intensities in [0, 1]. PASS.")

    # 2. Verify Augmented Path A Training Set
    print("\n[Check 2/5] Verifying augmented Path A training set...")
    aug_path = AUGMENTED_RAW_DIR / "augmented_train.npz"
    assert aug_path.exists(), f"Augmented file missing: {aug_path}"
    aug_data = np.load(aug_path)
    X_train_aug = aug_data["X_train_aug"]
    y_train_aug = aug_data["y_train_aug"]

    assert X_train_aug.shape == (162000, 28, 28)
    assert y_train_aug.shape == (162000,)
    assert np.allclose(X_train_aug[:54000], X_train_clean)
    assert 0.0 <= X_train_aug.min() and X_train_aug.max() <= 1.0
    print(f"  -> Path A augmented set: {len(X_train_aug):,} samples (3x expansion: 54k clean + 108k aug). PASS.")

    # 3. Verify Feature Set A2 (56 features)
    print("\n[Check 3/5] Verifying Feature Set A2 matrices...")
    X_tr_std, y_tr, X_va_std, y_va, X_te_std, y_te = load_a2_features(standardized=True)
    X_tr_raw, _, X_va_raw, _, X_te_raw, _ = load_a2_features(standardized=False)

    assert X_tr_std.shape == (162000, 56)
    assert X_va_std.shape == (6000, 56)
    assert X_te_std.shape == (10000, 56)

    sample_img = X_train_aug[0]
    expected_sample_feat = np.concatenate([sample_img.sum(axis=1), sample_img.sum(axis=0)])
    assert np.allclose(X_tr_raw[0], expected_sample_feat, atol=1e-5)

    train_mean = np.mean(X_tr_std, axis=0)
    train_std = np.std(X_tr_std, axis=0)
    assert np.allclose(train_mean, 0.0, atol=1e-3)
    valid_std_mask = train_std > 1e-4
    assert np.allclose(train_std[valid_std_mask], 1.0, atol=1e-2)
    print("  -> Feature Set A2: (N, 56) verified. StandardScaler trained without leakage. PASS.")

    # 4. Verify 2-D CNN Input Arrays
    print("\n[Check 4/5] Verifying 2-D Path A CNN images...")
    X_tr_2d, _, X_va_2d, _, X_te_2d, _ = load_pathA_2D()
    assert X_tr_2d.shape == (162000, 1, 28, 28)
    assert X_va_2d.shape == (6000, 1, 28, 28)
    assert X_te_2d.shape == (10000, 1, 28, 28)
    print("  -> 2-D Path A images: (N, 1, 28, 28) verified for direct CNN intake. PASS.")

    # 5. Generate and Save Phase 1 Verification Figure
    print("\n[Check 5/5] Generating Phase 1 verification figure...")
    sample_idx = 0
    digit_img = X_train_aug[sample_idx]
    digit_label = y_train_aug[sample_idx]
    digit_a2 = X_tr_raw[sample_idx]

    row_sums = digit_a2[:28]
    col_sums = digit_a2[28:]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 2]})

    # Left: 28x28 grayscale image
    im = axes[0].imshow(digit_img, cmap="gray", interpolation="nearest")
    axes[0].set_title(f"Sample Path A Image (Digit Label: {digit_label})", fontsize=12, fontweight="bold")
    axes[0].axis("off")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="Pixel Intensity [0, 1]")

    # Right: Two-tone bar chart of 56 features
    feat_indices = np.arange(56)
    axes[1].bar(feat_indices[:28], row_sums, color="#ff7f0e", width=0.8, label="Features 0–27: Row Sums (Horizontal Profile)")
    axes[1].bar(feat_indices[28:], col_sums, color="#1f77b4", width=0.8, label="Features 28–55: Column Sums (Vertical Profile)")
    axes[1].axvline(x=27.5, color="black", linestyle="--", alpha=0.6, label="Boundary (Row vs Col Sums)")
    axes[1].set_xlabel("Feature Index (0–55)", fontsize=11)
    axes[1].set_ylabel("Pixel Intensity Sum", fontsize=11)
    axes[1].set_title(f"Feature Set A2: 56 Row & Column Sums for Digit '{digit_label}'", fontsize=12, fontweight="bold")
    axes[1].set_xticks(np.arange(0, 56, 4))
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    verification_fig_path = RESULTS_A2_DIR / "phase1_verification_figure.png"
    plt.savefig(verification_fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    assert verification_fig_path.exists()
    print(f"  -> Verification figure saved successfully to: {verification_fig_path.resolve()}")

    print("\n" + "=" * 70)
    print("ALL PHASE 1 VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    verify_phase1_pipeline()

