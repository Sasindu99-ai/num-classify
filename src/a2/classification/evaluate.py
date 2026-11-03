"""Comprehensive Classifier Evaluation, Comparison, and Error Analysis for Feature Set A2.

Computes:
  - Macro, Weighted, and Micro Precision, Recall, and F1-Scores.
  - Per-digit (0-9) performance breakdowns.
  - Normalized confusion matrix heatmaps for all 10 models.
  - One-vs-Rest ROC curves and multi-class ROC-AUC.
  - Model leaderboard comparing accuracy, training time, latency, and memory.
  - Detailed projection-loss error analysis contrasting A2 failure modes against 2-D CNN.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from src.common.utils import (
    MODELS_A2_DIR,
    TABLES_A2_DIR,
    FIGURES_A2_DIR,
    CONFUSION_MATRICES_A2_DIR,
    RANDOM_SEED,
    set_seed,
    ensure_a2_directories,
)
from src.a2.features import load_a2_features, load_pathA_2D


def evaluate_all_classifiers():
    set_seed(RANDOM_SEED)
    ensure_a2_directories()

    print("=" * 70)
    print("EVALUATING ALL 10 CLASSIFIERS ON ISOLATED TEST SET (10,000 SAMPLES)")
    print("=" * 70)

    _, _, _, _, _, y_test = load_a2_features(standardized=True)
    y_test_bin = label_binarize(y_test, classes=list(range(10)))

    classical_pred_path = MODELS_A2_DIR / "classical_predictions.joblib"
    neural_pred_path = MODELS_A2_DIR / "neural_predictions.npz"

    if not classical_pred_path.exists():
        raise FileNotFoundError(f"Missing {classical_pred_path}. Run classical_models.py first.")
    if not neural_pred_path.exists():
        raise FileNotFoundError(f"Missing {neural_pred_path}. Run neural_models.py first.")

    classical_data = joblib.load(classical_pred_path)
    neural_data = np.load(neural_pred_path)

    all_models = {}
    for name, data in classical_data.items():
        all_models[name] = {
            "type": "Classical (Feature Set A2)",
            "y_pred": data["y_test_pred"],
            "y_proba": data["y_test_proba"],
            "val_acc": data["val_acc"],
            "train_time": data["train_time"],
            "latency": data["inf_latency_per_1k_ms"],
            "size_mb": data["model_size_mb"],
        }

    all_models["Multilayer Perceptron (MLP)"] = {
        "type": "Neural (Feature Set A2)",
        "y_pred": neural_data["mlp_preds"],
        "y_proba": neural_data["mlp_probas"],
        "val_acc": float(neural_data["mlp_val_acc"]),
        "train_time": float(neural_data["mlp_train_time"]),
        "latency": float(neural_data["mlp_inf_latency"]),
        "size_mb": float(neural_data["mlp_size_mb"]),
    }

    all_models["Convolutional Neural Network (CNN)"] = {
        "type": "Neural (2-D Path A Images)",
        "y_pred": neural_data["cnn_preds"],
        "y_proba": neural_data["cnn_probas"],
        "val_acc": float(neural_data["cnn_val_acc"]),
        "train_time": float(neural_data["cnn_train_time"]),
        "latency": float(neural_data["cnn_inf_latency"]),
        "size_mb": float(neural_data["cnn_size_mb"]),
    }

    summary_rows = []
    per_digit_reports = {}

    for name, info in all_models.items():
        y_pred = info["y_pred"]
        y_proba = info["y_proba"]

        test_acc = accuracy_score(y_test, y_pred)
        macro_prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        macro_rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        weighted_prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        weighted_rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        micro_f1 = f1_score(y_test, y_pred, average="micro", zero_division=0)

        try:
            roc_auc = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")
        except Exception:
            roc_auc = np.nan

        summary_rows.append({
            "Model": name,
            "Representation": info["type"],
            "Val Accuracy (%)": info["val_acc"] * 100,
            "Test Accuracy (%)": test_acc * 100,
            "Macro F1": macro_f1,
            "Weighted F1": weighted_f1,
            "Micro F1": micro_f1,
            "Macro Precision": macro_prec,
            "Macro Recall": macro_rec,
            "ROC-AUC (Macro OVR)": roc_auc,
            "Train Time (s)": info["train_time"],
            "Inference Latency (ms/1k)": info["latency"],
            "Model Size (MB)": info["size_mb"],
        })

        rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        per_digit_reports[name] = rep

        cm = confusion_matrix(y_test, y_pred, normalize="true")
        plt.figure(figsize=(7, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            cbar=True,
            xticklabels=range(10),
            yticklabels=range(10),
        )
        plt.title(f"Normalized Confusion Matrix: {name}\n(Test Accuracy: {test_acc*100:.2f}%)", fontsize=11, fontweight="bold")
        plt.xlabel("Predicted Class", fontsize=10)
        plt.ylabel("True Class", fontsize=10)
        plt.tight_layout()

        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        cm_path = CONFUSION_MATRICES_A2_DIR / f"cm_{safe_name}.png"
        plt.savefig(cm_path, dpi=200)
        plt.close()

    df_summary = pd.DataFrame(summary_rows).sort_values(by="Test Accuracy (%)", ascending=False).reset_index(drop=True)
    csv_summary_path = TABLES_A2_DIR / "classification_summary.csv"
    df_summary.to_csv(csv_summary_path, index=False)
    print(f"\nLeaderboard exported to: {csv_summary_path.resolve()}")
    print("\n" + df_summary.to_string(index=False))

    print("\nPlotting Multi-Class One-vs-Rest ROC Curves ...")
    plt.figure(figsize=(10, 8))
    for name in ["Convolutional Neural Network (CNN)", "Support Vector Machine (SVM)", "Random Forest", "Multilayer Perceptron (MLP)", "Logistic Regression"]:
        if name in all_models:
            y_proba = all_models[name]["y_proba"]
            fpr_grid = np.linspace(0.0, 1.0, 200)
            tpr_list = []
            for c in range(10):
                fpr_c, tpr_c, _ = roc_curve(y_test_bin[:, c], y_proba[:, c])
                tpr_interp = np.interp(fpr_grid, fpr_c, tpr_c)
                tpr_interp[0] = 0.0
                tpr_list.append(tpr_interp)
            mean_tpr = np.mean(tpr_list, axis=0)
            mean_tpr[-1] = 1.0
            macro_auc = auc(fpr_grid, mean_tpr)
            plt.plot(fpr_grid, mean_tpr, label=f"{name} (Macro AUC = {macro_auc:.4f})", linewidth=2.0)

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC = 0.5000)")
    plt.xlabel("False Positive Rate (FPR)", fontsize=11)
    plt.ylabel("True Positive Rate (TPR)", fontsize=11)
    plt.title("One-vs-Rest Macro ROC Curves Comparison", fontsize=13, fontweight="bold")
    plt.grid(alpha=0.4, linestyle=":")
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    roc_plot_path = FIGURES_A2_DIR / "roc_curves_comparison.png"
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()
    print(f"ROC Curves figure saved to: {roc_plot_path.resolve()}")

    print("\nGenerating Projection Loss Failure Mode Analysis ...")
    svm_preds = all_models["Support Vector Machine (SVM)"]["y_pred"]
    cnn_preds = all_models["Convolutional Neural Network (CNN)"]["y_pred"]
    _, _, _, _, X_test_2d, _ = load_pathA_2D()

    a2_fail_cnn_pass = np.where((svm_preds != y_test) & (cnn_preds == y_test))[0]
    print(f"Found {len(a2_fail_cnn_pass):,} samples where Feature Set A2 fails but 2-D CNN succeeds.")

    if len(a2_fail_cnn_pass) >= 4:
        selected_indices = a2_fail_cnn_pass[:4]
        fig, axes = plt.subplots(2, 4, figsize=(14, 7))

        X_tr_raw, _, _, _, X_te_raw, _ = load_a2_features(standardized=False)

        for col_idx, s_idx in enumerate(selected_indices):
            true_lbl = y_test[s_idx]
            a2_pred_lbl = svm_preds[s_idx]
            cnn_pred_lbl = cnn_preds[s_idx]
            raw_img = X_test_2d[s_idx, 0]
            a2_vector = X_te_raw[s_idx]

            axes[0, col_idx].imshow(raw_img, cmap="gray", interpolation="nearest")
            axes[0, col_idx].set_title(
                f"True: {true_lbl}\nA2 Pred: {a2_pred_lbl} | CNN: {cnn_pred_lbl}",
                fontsize=10,
                fontweight="bold",
                color="red" if a2_pred_lbl != true_lbl else "green",
            )
            axes[0, col_idx].axis("off")

            axes[1, col_idx].bar(range(28), a2_vector[:28], color="#ff7f0e", width=0.8, label="Row Sums")
            axes[1, col_idx].bar(range(28, 56), a2_vector[28:], color="#1f77b4", width=0.8, label="Col Sums")
            axes[1, col_idx].set_title("56-Feature Ink Profile", fontsize=9)
            axes[1, col_idx].set_xticks([0, 28, 55])
            axes[1, col_idx].grid(axis="y", linestyle=":", alpha=0.5)
            if col_idx == 0:
                axes[1, col_idx].set_ylabel("Sum")
                axes[1, col_idx].legend(loc="upper right", fontsize=8)

        plt.suptitle(
            "Feature Set A2 Projection Ambiguity: Samples Misclassified by A2 but Correctly Resolved by 2-D CNN",
            fontsize=12,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        err_analysis_path = FIGURES_A2_DIR / "misclassification_projection_analysis.png"
        plt.savefig(err_analysis_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Projection Loss Error Analysis saved to: {err_analysis_path.resolve()}")

    return df_summary


if __name__ == "__main__":
    evaluate_all_classifiers()

