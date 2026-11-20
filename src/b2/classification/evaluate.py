
"""Evaluation, comparison and error analysis for B2 classifiers."""

import json
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
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from src.common.utils import PROJECT_ROOT, RANDOM_SEED, set_seed


MODELS_DIR = PROJECT_ROOT / "results" / "b2" / "models"
TABLES_DIR = PROJECT_ROOT / "results" / "b2" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "b2" / "figures"
CONFUSION_DIR = PROJECT_ROOT / "results" / "b2" / "confusion_matrices"

FEATURES_DIR = PROJECT_ROOT / "data" / "features" / "b2"

# Purple/lavender visual theme
BACKGROUND = "#F5F0FA"
LIGHT_PURPLE = "#DCCCEB"
MID_PURPLE = "#B99AD5"
DARK_PURPLE = "#542078"
PURPLE = "#895BB5"


def ensure_directories():
    """Create output directories."""

    for directory in [
        TABLES_DIR,
        FIGURES_DIR,
        CONFUSION_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def load_test_labels():
    """Load clean test labels."""

    test_data = np.load(FEATURES_DIR / "test_features.npz")
    return test_data["y"]


def load_all_predictions():
    """Load predictions and metadata from all nine classifiers."""

    classical_path = MODELS_DIR / "classical_predictions_b2.joblib"
    neural_path = MODELS_DIR / "neural_predictions_b2.npz"

    if not classical_path.exists():
        raise FileNotFoundError(
            "Classical predictions missing. "
            "Run classical_models.py first."
        )

    if not neural_path.exists():
        raise FileNotFoundError(
            "MLP predictions missing. "
            "Run neural_models.py first."
        )

    classical_data = joblib.load(classical_path)
    neural_data = np.load(neural_path)

    all_models = {}

    for name, data in classical_data.items():

        all_models[name] = {
            "representation": "B2 Contour Row + Column Sums",
            "y_pred": data["y_test_pred"],
            "y_proba": data["y_test_proba"],
            "val_acc": data["val_acc"],
            "train_time": data["train_time"],
            "latency": data["inf_latency_per_1k_ms"],
            "size_mb": data["model_size_mb"],
        }

    all_models["Multilayer Perceptron (MLP)"] = {
        "representation": "B2 Contour Row + Column Sums",
        "y_pred": neural_data["mlp_preds"],
        "y_proba": neural_data["mlp_probas"],
        "val_acc": float(neural_data["mlp_val_acc"]),
        "train_time": float(neural_data["mlp_train_time"]),
        "latency": float(neural_data["mlp_inf_latency"]),
        "size_mb": float(neural_data["mlp_size_mb"]),
    }

    return all_models


def create_confusion_matrix(name, y_true, y_pred, test_accuracy):
    """Generate and save a normalized confusion matrix."""

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(10)),
        normalize="true",
    )

    plt.figure(figsize=(8, 7))

    ax = sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap=sns.color_palette(
            ["#F5F0FA", "#DCCCEB", "#B99AD5", "#895BB5", "#542078"],
            as_cmap=True,
        ),
        vmin=0,
        vmax=1,
        cbar=True,
        xticklabels=range(10),
        yticklabels=range(10),
        linewidths=0.3,
        linecolor="white",
    )

    ax.set_facecolor(BACKGROUND)

    plt.title(
        f"B2 Normalized Confusion Matrix: {name}\n"
        f"Test Accuracy: {test_accuracy * 100:.2f}%",
        color=DARK_PURPLE,
        fontsize=12,
        fontweight="bold",
    )

    plt.xlabel("Predicted Digit", color=DARK_PURPLE)
    plt.ylabel("Actual Digit", color=DARK_PURPLE)

    plt.tight_layout()

    safe_name = (
        name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
    )

    output_path = CONFUSION_DIR / f"cm_{safe_name}_b2.png"

    plt.savefig(
        output_path,
        dpi=250,
        facecolor=BACKGROUND,
        bbox_inches="tight",
    )

    plt.close()


def create_roc_comparison(all_models, y_test, y_test_bin):
    """Plot comparative one-vs-rest ROC curves."""

    selected_models = [
        "Gradient Boosting",
        "Multilayer Perceptron (MLP)",
        "Random Forest",
        "Support Vector Machine (SVM)",
        "Logistic Regression",
    ]

    plt.figure(figsize=(10, 8))
    plt.gca().set_facecolor(BACKGROUND)

    colors = [
        DARK_PURPLE,
        PURPLE,
        "#B99AD5",
        "#A17BC4",
        "#C6A9DD",
    ]

    fpr_grid = np.linspace(0, 1, 200)

    for name, color in zip(selected_models, colors):

        if name not in all_models:
            continue

        probabilities = all_models[name]["y_proba"]

        if probabilities.shape != (len(y_test), 10):
            continue

        interpolated_tprs = []

        for digit in range(10):

            fpr, tpr, _ = roc_curve(
                y_test_bin[:, digit],
                probabilities[:, digit],
            )

            interpolated = np.interp(
                fpr_grid,
                fpr,
                tpr,
            )

            interpolated[0] = 0.0
            interpolated_tprs.append(interpolated)

        mean_tpr = np.mean(interpolated_tprs, axis=0)
        mean_tpr[-1] = 1.0

        # Use the same macro OVR AUC calculation as the summary CSV.
        macro_auc = roc_auc_score(
            y_test_bin,
            probabilities,
            average="macro",
        )

        plt.plot(
            fpr_grid,
            mean_tpr,
            color=color,
            linewidth=2,
            label=f"{name} (AUC={macro_auc:.4f})",
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#777777",
        alpha=0.7,
        label="Random Guess",
    )

    plt.xlabel("False Positive Rate", color=DARK_PURPLE)
    plt.ylabel("True Positive Rate", color=DARK_PURPLE)

    plt.title(
        "B2 Multiclass ROC Curve Comparison",
        color=DARK_PURPLE,
        fontsize=14,
        fontweight="bold",
    )

    plt.grid(alpha=0.25, linestyle=":")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()

    output_path = FIGURES_DIR / "roc_curves_comparison_b2.png"

    plt.savefig(
        output_path,
        dpi=300,
        facecolor=BACKGROUND,
        bbox_inches="tight",
    )

    plt.close()

    print(f"ROC comparison saved: {output_path}")


def evaluate_all_classifiers():
    """Evaluate all nine B2 classifiers and export results."""

    set_seed(RANDOM_SEED)
    ensure_directories()

    print("=" * 70)
    print("EVALUATING NINE CLASSIFIERS ON B2 TEST SET")
    print("=" * 70)

    y_test = load_test_labels()
    y_test_bin = label_binarize(
        y_test,
        classes=list(range(10)),
    )

    all_models = load_all_predictions()

    summary_rows = []
    per_digit_rows = []

    for name, info in all_models.items():

        print(f"\nEvaluating: {name}")

        y_pred = info["y_pred"]
        y_proba = info["y_proba"]

        if len(y_pred) != len(y_test):
            raise ValueError(
                f"Prediction count mismatch for {name}."
            )

        test_acc = accuracy_score(y_test, y_pred)

        macro_precision = precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )

        macro_recall = recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )

        macro_f1 = f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )

        weighted_f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        micro_f1 = f1_score(
            y_test,
            y_pred,
            average="micro",
            zero_division=0,
        )

        # Calculate ROC-AUC only when valid class probabilities exist.
        try:
            if (
                y_proba.shape == (len(y_test), 10)
                and np.all(np.isfinite(y_proba))
                and np.all(y_proba.sum(axis=1) > 0)
            ):
                roc_auc = roc_auc_score(
                    y_test_bin,
                    y_proba,
                    average="macro",
                )
            else:
                roc_auc = np.nan

        except (ValueError, TypeError):
            roc_auc = np.nan

        summary_rows.append({
            "Model": name,
            "Representation": info["representation"],
            "Val Accuracy (%)": info["val_acc"] * 100,
            "Test Accuracy (%)": test_acc * 100,
            "Macro F1": macro_f1,
            "Weighted F1": weighted_f1,
            "Micro F1": micro_f1,
            "Macro Precision": macro_precision,
            "Macro Recall": macro_recall,
            "ROC-AUC (Macro OVR)": roc_auc,
            "Train Time (s)": info["train_time"],
            "Inference Latency (ms/1k)": info["latency"],
            "Model Size (MB)": info["size_mb"],
        })

        # Per-digit precision, recall and F1.
        report = classification_report(
            y_test,
            y_pred,
            labels=list(range(10)),
            output_dict=True,
            zero_division=0,
        )

        for digit in range(10):

            digit_result = report[str(digit)]

            per_digit_rows.append({
                "Model": name,
                "Digit": digit,
                "Precision": digit_result["precision"],
                "Recall": digit_result["recall"],
                "F1-Score": digit_result["f1-score"],
                "Support": digit_result["support"],
            })

        create_confusion_matrix(
            name,
            y_test,
            y_pred,
            test_acc,
        )

        print(f"     Test Accuracy: {test_acc * 100:.2f}%")
        print(f"     Macro F1:      {macro_f1:.4f}")
        print(f"     ROC-AUC:       {roc_auc:.4f}")

    # Export classification summary.
    summary_df = pd.DataFrame(summary_rows)

    summary_df = summary_df.sort_values(
        by="Test Accuracy (%)",
        ascending=False,
    ).reset_index(drop=True)

    summary_path = TABLES_DIR / "classification_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # Export per-digit performance.
    per_digit_df = pd.DataFrame(per_digit_rows)

    per_digit_path = TABLES_DIR / "classification_per_digit.csv"
    per_digit_df.to_csv(per_digit_path, index=False)

    # Export a machine-readable copy of the per-digit reports.
    report_path = TABLES_DIR / "classification_per_digit.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(
            per_digit_rows,
            file,
            indent=4,
        )

    # Generate ROC comparison.
    create_roc_comparison(
        all_models,
        y_test,
        y_test_bin,
    )

    print("\n" + "=" * 70)
    print("B2 CLASSIFICATION EVALUATION COMPLETED")
    print("=" * 70)

    print("\nCLASSIFICATION LEADERBOARD")
    print(
        summary_df[
            ["Model", "Val Accuracy (%)", "Test Accuracy (%)", "Macro F1"]
        ].to_string(index=False)
    )

    print(f"\nClassification summary: {summary_path}")
    print(f"Per-digit metrics:       {per_digit_path}")
    print(f"Per-digit JSON:          {report_path}")
    print(f"Confusion matrices:      {CONFUSION_DIR}")
    print("=" * 70)

    return summary_df


if __name__ == "__main__":
    evaluate_all_classifiers()
