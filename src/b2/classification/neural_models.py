"""Train an MLP neural network on B2's 56 contour features."""

from pathlib import Path
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.common.utils import PROJECT_ROOT, RANDOM_SEED, set_seed
from src.b2.classification.classical_models import load_b2_features


MODELS_DIR = PROJECT_ROOT / "results" / "b2" / "models"


class MLPClassifier(nn.Module):
    """Multilayer Perceptron for the 56 B2 contour features."""

    def __init__(self, in_features=56, num_classes=10):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_epoch(model, loader, criterion, optimizer, device):
    """Train the model for one epoch."""

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        predictions = outputs.argmax(dim=1)

        correct += (predictions == y_batch).sum().item()
        total += len(y_batch)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    """Evaluate model performance on validation data."""

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        total_loss += loss.item() * len(y_batch)
        predictions = outputs.argmax(dim=1)

        correct += (predictions == y_batch).sum().item()
        total += len(y_batch)

    return total_loss / total, correct / total


@torch.no_grad()
def predict_full(model, X_array, device, batch_size=512):
    """Generate predictions and class probabilities in batches."""

    model.eval()

    dataset = TensorDataset(torch.from_numpy(X_array))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    predictions_list = []
    probabilities_list = []

    for (X_batch,) in loader:

        X_batch = X_batch.to(device)

        outputs = model(X_batch)
        probabilities = torch.softmax(outputs, dim=1)

        predictions_list.append(
            probabilities.argmax(dim=1).cpu().numpy()
        )

        probabilities_list.append(
            probabilities.cpu().numpy()
        )

    return (
        np.concatenate(predictions_list),
        np.concatenate(probabilities_list),
    )


def synchronize_device(device):
    """Synchronize MPS operations before measuring elapsed time."""

    if device.type == "mps":
        torch.mps.synchronize()


def run_neural_model():
    """Train, validate and evaluate the B2 MLP."""

    set_seed(RANDOM_SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("=" * 70)
    print("TRAINING MULTILAYER PERCEPTRON ON B2 FEATURES")
    print(f"Compute device: {device}")
    print("=" * 70)

    X_train, y_train, X_val, y_val, X_test, y_test = load_b2_features()

    print(f"Training samples:   {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    print(f"Test samples:       {len(X_test):,}")
    print(f"Input features:     {X_train.shape[1]}")

    # Convert arrays to PyTorch tensors.
    X_train_tensor = torch.from_numpy(
        X_train.astype(np.float32)
    )
    y_train_tensor = torch.from_numpy(
        y_train.astype(np.int64)
    )

    X_val_tensor = torch.from_numpy(
        X_val.astype(np.float32)
    )
    y_val_tensor = torch.from_numpy(
        y_val.astype(np.int64)
    )

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=256,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(X_val_tensor, y_val_tensor),
        batch_size=512,
        shuffle=False,
    )

    model = MLPClassifier(
        in_features=56,
        num_classes=10,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=15,
    )

    best_val_acc = 0.0

    model_path = MODELS_DIR / "mlp_best_b2.pt"

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    print("\nStarting training...")

    synchronize_device(device)
    training_start = time.perf_counter()

    for epoch in range(1, 16):

        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_acc = evaluate_epoch(
            model,
            val_loader,
            criterion,
            device,
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        # Save the model with the best validation accuracy.
        if val_acc > best_val_acc:

            best_val_acc = val_acc

            torch.save(
                model.state_dict(),
                model_path,
            )

        print(
            f"Epoch {epoch:2d}/15 | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}%"
        )

    synchronize_device(device)
    train_time = time.perf_counter() - training_start

    # Load the best validation checkpoint.
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
            weights_only=True,
        )
    )

    # Final test evaluation.
    synchronize_device(device)
    inference_start = time.perf_counter()

    test_predictions, test_probabilities = predict_full(
        model,
        X_test.astype(np.float32),
        device,
    )

    synchronize_device(device)
    inference_time = time.perf_counter() - inference_start

    test_accuracy = np.mean(test_predictions == y_test)

    inference_latency = (
        inference_time / len(X_test)
    ) * 1000 * 1000

    model_size_mb = (
        os.path.getsize(model_path) / (1024 * 1024)
    )

    # Save predictions and training information.
    predictions_path = MODELS_DIR / "neural_predictions_b2.npz"

    np.savez_compressed(
        predictions_path,
        mlp_preds=test_predictions,
        mlp_probas=test_probabilities,
        mlp_val_acc=best_val_acc,
        mlp_test_acc=test_accuracy,
        mlp_train_time=train_time,
        mlp_inf_latency=inference_latency,
        mlp_size_mb=model_size_mb,
        train_loss=np.array(history["train_loss"]),
        train_accuracy=np.array(history["train_accuracy"]),
        val_loss=np.array(history["val_loss"]),
        val_accuracy=np.array(history["val_accuracy"]),
    )

    print("\n" + "=" * 70)
    print("B2 MLP TRAINING COMPLETED")
    print(f"Best validation accuracy: {best_val_acc * 100:.2f}%")
    print(f"Test accuracy:            {test_accuracy * 100:.2f}%")
    print(f"Training time:            {train_time:.2f} s")
    print(f"Inference latency:        {inference_latency:.2f} ms / 1,000 samples")
    print(f"Model size:               {model_size_mb:.2f} MB")
    print(f"Best model saved to:      {model_path}")
    print(f"Predictions saved to:     {predictions_path}")
    print("=" * 70)

    return history


if __name__ == "__main__":
    run_neural_model()
