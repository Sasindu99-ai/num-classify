
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    # Project paths
    project_root = Path(__file__).resolve().parents[3]

    predictions_path = (
        project_root
        / "results"
        / "b2"
        / "models"
        / "neural_predictions_b2.npz"
    )

    output_path = (
        project_root
        / "results"
        / "b2"
        / "figures"
        / "mlp_learning_curves_b2.png"
    )

    # Load saved training history
    data = np.load(predictions_path)

    train_accuracy = data["train_accuracy"] * 100
    val_accuracy = data["val_accuracy"] * 100

    train_loss = data["train_loss"]
    val_loss = data["val_loss"]

    epochs = np.arange(1, len(train_accuracy) + 1)

    # Purple/lavender theme
    purple = "#6A3D9A"
    lavender = "#B39DDB"
    light_purple = "#D1C4E9"
    dark_purple = "#45236B"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Accuracy plot
    axes[0].plot(
        epochs,
        train_accuracy,
        color=purple,
        marker="o",
        linewidth=2,
        label="Training Accuracy",
    )

    axes[0].plot(
        epochs,
        val_accuracy,
        color=lavender,
        marker="s",
        linewidth=2,
        label="Validation Accuracy",
    )

    axes[0].set_title("MLP Accuracy", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_xticks(epochs)
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    # Loss plot
    axes[1].plot(
        epochs,
        train_loss,
        color=purple,
        marker="o",
        linewidth=2,
        label="Training Loss",
    )

    axes[1].plot(
        epochs,
        val_loss,
        color=lavender,
        marker="s",
        linewidth=2,
        label="Validation Loss",
    )

    axes[1].set_title("MLP Loss", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_xticks(epochs)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.suptitle(
        "B2 – MLP Training and Validation Performance",
        fontsize=14,
        fontweight="bold",
        color=dark_purple,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Learning curves saved to: {output_path}")


if __name__ == "__main__":
    main()