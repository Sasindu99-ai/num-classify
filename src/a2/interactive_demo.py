"""Interactive Prediction and Clustering Demonstration Tool for Mini Exhibition (Feature Set A2).

Enables visitors or examiners to supply a sample digit image or select an index:
  1. Computes the 56 continuous Feature Set A2 features (row & col ink profiles).
  2. Runs classification via the best Feature Set A2 model (SVM).
  3. Runs classification via the 2-D Path A Convolutional Neural Network (CNN).
  4. Predicts the unsupervised cluster assignment via k-Means.
  5. Displays side-by-side terminal metrics and saves a visual demonstration snapshot.
"""

from pathlib import Path
import argparse
import sys
import numpy as np
import joblib
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.common.utils import (
    MODELS_A2_DIR,
    FEATURES_A2_DIR,
    RESULTS_A2_DIR,
    set_seed,
    RANDOM_SEED,
)
from src.a2.features import extract_a2_features, load_pathA_2D
from src.a2.classification.neural_models import CNNClassifier


def run_demo(sample_index: int = 0, save_snapshot: bool = True):
    """Run interactive exhibition demo on a specific test sample."""
    set_seed(RANDOM_SEED)

    _, _, _, _, X_test_2d, y_test = load_pathA_2D()
    if sample_index < 0 or sample_index >= len(y_test):
        print(f"Error: sample_index must be between 0 and {len(y_test) - 1}.")
        return

    raw_2d = X_test_2d[sample_index, 0]
    true_label = int(y_test[sample_index])

    print("=" * 70)
    print("EE5130 MINI PROJECT: INTERACTIVE DEMONSTRATION (FEATURE SET A2)")
    print(f"Evaluating Test Sample #{sample_index:,} (Ground Truth Digit: '{true_label}')")
    print("=" * 70)

    # 1. Extract 56 Continuous A2 Features
    features_56_raw = extract_a2_features(np.expand_dims(raw_2d, axis=0))[0]
    row_sums = features_56_raw[:28]
    col_sums = features_56_raw[28:]

    # 2. Standardize Features
    scaler_path = FEATURES_A2_DIR / "scaler_A2.joblib"
    if not scaler_path.exists():
        print(f"Error: Scaler not found at {scaler_path}.")
        return
    scaler = joblib.load(scaler_path)
    features_56_std = scaler.transform(features_56_raw.reshape(1, -1))

    # 3. Predict with Best Classical A2 Classifier (SVM)
    svm_path = MODELS_A2_DIR / "support_vector_machine_svm.joblib"
    if svm_path.exists():
        svm = joblib.load(svm_path)
        a2_pred_class = int(svm.predict(features_56_std)[0])
        a2_probas = svm.predict_proba(features_56_std)[0] if hasattr(svm, "predict_proba") else None
        a2_confidence = (a2_probas[a2_pred_class] * 100) if a2_probas is not None else 100.0
    else:
        a2_pred_class = "N/A"
        a2_confidence = 0.0

    # 4. Predict with 2-D CNN (using CUDA if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_path = MODELS_A2_DIR / "cnn_best.pt"
    if cnn_path.exists():
        cnn = CNNClassifier(in_channels=1, num_classes=10).to(device)
        cnn.load_state_dict(torch.load(cnn_path, map_location=device))
        cnn.eval()

        with torch.no_grad():
            tensor_in = torch.from_numpy(raw_2d).unsqueeze(0).unsqueeze(0).to(device)
            logits = cnn(tensor_in)
            probas = torch.softmax(logits, dim=1).cpu().numpy()[0]
            cnn_pred_class = int(probas.argmax())
            cnn_confidence = probas[cnn_pred_class] * 100
    else:
        cnn_pred_class = "N/A"
        cnn_confidence = 0.0

    # 5. Predict Unsupervised Cluster Assignment (k-Means)
    cluster_results_path = MODELS_A2_DIR / "clustering_results.joblib"
    if cluster_results_path.exists():
        cluster_data = joblib.load(cluster_results_path)
        kmeans = cluster_data["models"]["k-Means"]["model"]
        assigned_cluster = int(kmeans.predict(features_56_std)[0])
    else:
        assigned_cluster = "N/A"

    print("\n[PREDICTION RESULTS]")
    print(f"  * Ground Truth Label:          {true_label}")
    print(f"  * Feature Set A2 (SVM) Class:  {a2_pred_class} (Confidence: {a2_confidence:.1f}%)")
    print(f"  * 2-D Path A CNN Class:        {cnn_pred_class} (Confidence: {cnn_confidence:.1f}%)")
    print(f"  * Unsupervised Cluster ID:     Cluster #{assigned_cluster} (k-Means, k=10)")

    if a2_pred_class == true_label and cnn_pred_class == true_label:
        status_msg = "CONVERGENT SUCCESS: Both representations correctly identify the digit."
    elif a2_pred_class != true_label and cnn_pred_class == true_label:
        status_msg = "PROJECTION AMBIGUITY: A2 fails due to 1D projection loss; 2-D CNN resolves spatial topology!"
    else:
        status_msg = f"A2 Result: {a2_pred_class == true_label} | CNN Result: {cnn_pred_class == true_label}"
    print(f"\n  Diagnosis: {status_msg}")

    # 6. Generate Snapshot Visualization
    if save_snapshot:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1, 1.8]})

        axes[0].imshow(raw_2d, cmap="gray", interpolation="nearest")
        axes[0].set_title(
            f"Test Sample #{sample_index} (True: '{true_label}')\n"
            f"A2 Pred: '{a2_pred_class}' | CNN Pred: '{cnn_pred_class}'",
            fontsize=10,
            fontweight="bold",
        )
        axes[0].axis("off")

        axes[1].bar(range(28), row_sums, color="#ff7f0e", width=0.8, label="Features 0–27: Row Sums")
        axes[1].bar(range(28, 56), col_sums, color="#1f77b4", width=0.8, label="Features 28–55: Col Sums")
        axes[1].set_title(
            f"56-Feature Ink Profile (Cluster #{assigned_cluster})",
            fontsize=10,
            fontweight="bold",
        )
        axes[1].set_xlabel("Feature Index (0–55)")
        axes[1].set_ylabel("Pixel Sum")
        axes[1].legend(loc="upper right", fontsize=8)
        axes[1].grid(axis="y", linestyle=":", alpha=0.5)

        plt.tight_layout()
        demo_path = RESULTS_A2_DIR / "demo_snapshot.png"
        plt.savefig(demo_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"\nVisual demonstration snapshot saved to: {demo_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EE5130 Mini Project Exhibition Demo (Feature Set A2)")
    parser.add_argument("--index", type=int, default=0, help="Test sample index (0 to 9999)")
    args = parser.parse_args()
    run_demo(args.index)

