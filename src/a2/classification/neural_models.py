"""PyTorch Neural Network Classifiers: MLP (on 56 A2 Features) and CNN (on 2-D Path A Images).

Features:
  - MLP trained on standardized Feature Set A2 (56 features).
  - CNN trained directly on continuous [0, 1] 2-D 28x28 Path A grayscale images.
  - Full NVIDIA CUDA GPU acceleration (utilising NVIDIA GeForce RTX 5060).
  - Validation checkpointing, learning curves, and inference latency profiling.
"""

from pathlib import Path
import time
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.common.utils import (
    MODELS_A2_DIR,
    RANDOM_SEED,
    set_seed,
    ensure_a2_directories,
)
from src.a2.features import load_a2_features, load_pathA_2D


class MLPClassifier(nn.Module):
    """Multilayer Perceptron for 56-feature Feature Set A2."""

    def __init__(self, in_features: int = 56, num_classes: int = 10):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNClassifier(nn.Module):
    """Convolutional Neural Network for 2-D 28x28 Path A Grayscale Images."""

    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        out = model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        preds = out.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        out = model(X_batch)
        loss = criterion(out, y_batch)

        total_loss += loss.item() * len(y_batch)
        preds = out.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


@torch.no_grad()
def predict_full(model, X_arr, device, batch_size=512):
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(X_arr)), batch_size=batch_size, shuffle=False)
    preds_list, probas_list = [], []
    for (X_batch,) in loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        probas = torch.softmax(logits, dim=1)
        preds_list.append(probas.argmax(dim=1).cpu().numpy())
        probas_list.append(probas.cpu().numpy())
    return np.concatenate(preds_list), np.concatenate(probas_list)


def run_neural_models():
    set_seed(RANDOM_SEED)
    ensure_a2_directories()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"TRAINING NEURAL NETWORK CLASSIFIERS")
    print(f"Hardware Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 70)

    # 1. Multilayer Perceptron (MLP) on 56 A2 Features
    print("\n---> Training Multilayer Perceptron (MLP) on 56 A2 Features ...")
    X_tr_a2, y_tr_a2, X_va_a2, y_va_a2, X_te_a2, y_te_a2 = load_a2_features(standardized=True)

    mlp_train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr_a2), torch.from_numpy(y_tr_a2)),
        batch_size=256,
        shuffle=True,
    )
    mlp_val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_va_a2), torch.from_numpy(y_va_a2)),
        batch_size=512,
        shuffle=False,
    )

    mlp = MLPClassifier(in_features=56, num_classes=10).to(device)
    mlp_criterion = nn.CrossEntropyLoss()
    mlp_optimizer = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    mlp_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(mlp_optimizer, T_max=15)

    best_mlp_val_acc = 0.0
    mlp_best_path = MODELS_A2_DIR / "mlp_best.pt"

    t0_mlp = time.perf_counter()
    for epoch in range(1, 16):
        tr_loss, tr_acc = train_epoch(mlp, mlp_train_loader, mlp_criterion, mlp_optimizer, device)
        va_loss, va_acc = evaluate_epoch(mlp, mlp_val_loader, mlp_criterion, device)
        mlp_scheduler.step()

        if va_acc > best_mlp_val_acc:
            best_mlp_val_acc = va_acc
            torch.save(mlp.state_dict(), mlp_best_path)

        if epoch % 3 == 0 or epoch == 15:
            print(f"     Epoch {epoch:2d}/15: Train Loss={tr_loss:.4f}, Train Acc={tr_acc*100:.2f}% | Val Loss={va_loss:.4f}, Val Acc={va_acc*100:.2f}%")

    mlp_train_time = time.perf_counter() - t0_mlp

    mlp.load_state_dict(torch.load(mlp_best_path, map_location=device))
    t1_mlp = time.perf_counter()
    mlp_preds, mlp_probas = predict_full(mlp, X_te_a2, device)
    mlp_inf_latency = (time.perf_counter() - t1_mlp) / len(X_te_a2) * 1000 * 1000
    mlp_test_acc = np.mean(mlp_preds == y_te_a2)
    mlp_size_mb = os.path.getsize(mlp_best_path) / (1024 * 1024)

    print(f"     [MLP Final] Best Val Acc: {best_mlp_val_acc*100:.2f}% | Test Acc: {mlp_test_acc*100:.2f}%")

    # 2. Convolutional Neural Network (CNN) on 2-D 28x28 Path A Images
    print("\n---> Training Convolutional Neural Network (CNN) on 2-D 28x28 Path A Images ...")
    X_tr_2d, y_tr_2d, X_va_2d, y_va_2d, X_te_2d, y_te_2d = load_pathA_2D()

    cnn_train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr_2d), torch.from_numpy(y_tr_2d)),
        batch_size=256,
        shuffle=True,
    )
    cnn_val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_va_2d), torch.from_numpy(y_va_2d)),
        batch_size=512,
        shuffle=False,
    )

    cnn = CNNClassifier(in_channels=1, num_classes=10).to(device)
    cnn_criterion = nn.CrossEntropyLoss()
    cnn_optimizer = torch.optim.AdamW(cnn.parameters(), lr=1e-3, weight_decay=1e-4)
    cnn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(cnn_optimizer, T_max=12)

    best_cnn_val_acc = 0.0
    cnn_best_path = MODELS_A2_DIR / "cnn_best.pt"

    t0_cnn = time.perf_counter()
    for epoch in range(1, 13):
        tr_loss, tr_acc = train_epoch(cnn, cnn_train_loader, cnn_criterion, cnn_optimizer, device)
        va_loss, va_acc = evaluate_epoch(cnn, cnn_val_loader, cnn_criterion, device)
        cnn_scheduler.step()

        if va_acc > best_cnn_val_acc:
            best_cnn_val_acc = va_acc
            torch.save(cnn.state_dict(), cnn_best_path)

        print(f"     Epoch {epoch:2d}/12: Train Loss={tr_loss:.4f}, Train Acc={tr_acc*100:.2f}% | Val Loss={va_loss:.4f}, Val Acc={va_acc*100:.2f}%")

    cnn_train_time = time.perf_counter() - t0_cnn

    cnn.load_state_dict(torch.load(cnn_best_path, map_location=device))
    t1_cnn = time.perf_counter()
    cnn_preds, cnn_probas = predict_full(cnn, X_te_2d, device)
    cnn_inf_latency = (time.perf_counter() - t1_cnn) / len(X_te_2d) * 1000 * 1000
    cnn_test_acc = np.mean(cnn_preds == y_te_2d)
    cnn_size_mb = os.path.getsize(cnn_best_path) / (1024 * 1024)

    print(f"     [CNN Final] Best Val Acc: {best_cnn_val_acc*100:.2f}% | Test Acc: {cnn_test_acc*100:.2f}%")

    neural_results_path = MODELS_A2_DIR / "neural_predictions.npz"
    np.savez_compressed(
        neural_results_path,
        mlp_preds=mlp_preds,
        mlp_probas=mlp_probas,
        mlp_val_acc=best_mlp_val_acc,
        mlp_test_acc=mlp_test_acc,
        mlp_train_time=mlp_train_time,
        mlp_inf_latency=mlp_inf_latency,
        mlp_size_mb=mlp_size_mb,
        cnn_preds=cnn_preds,
        cnn_probas=cnn_probas,
        cnn_val_acc=best_cnn_val_acc,
        cnn_test_acc=cnn_test_acc,
        cnn_train_time=cnn_train_time,
        cnn_inf_latency=cnn_inf_latency,
        cnn_size_mb=cnn_size_mb,
    )
    print(f"\nSaved neural predictions and metadata to: {neural_results_path.resolve()}")


if __name__ == "__main__":
    run_neural_models()

