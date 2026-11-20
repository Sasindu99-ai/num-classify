"""Train eight classical classifiers on B2 contour row/column features."""

from pathlib import Path
import os
import time

import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from src.common.utils import PROJECT_ROOT, RANDOM_SEED, set_seed


FEATURES_DIR = PROJECT_ROOT / "data" / "features" / "b2"
MODELS_DIR = PROJECT_ROOT / "results" / "b2" / "models"


def load_b2_features():
    """Load already-standardized B2 training, validation and test features."""

    train = np.load(FEATURES_DIR / "train_features.npz")
    val = np.load(FEATURES_DIR / "validation_features.npz")
    test = np.load(FEATURES_DIR / "test_features.npz")

    return (
        train["X"], train["y"],
        val["X"], val["y"],
        test["X"], test["y"],
    )


def get_classical_models(seed=RANDOM_SEED):
    """Return the eight classical classifiers used by the A2 pipeline."""

    return {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=seed,
            n_jobs=-1,
        ),
        "Naive Bayes": GaussianNB(
            var_smoothing=1e-8,
        ),
        "Linear Discriminant Analysis": LinearDiscriminantAnalysis(
            solver="svd",
        ),
        "Decision Tree": DecisionTreeClassifier(
            criterion="gini",
            max_depth=16,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=seed,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=22,
            min_samples_split=5,
            random_state=seed,
            n_jobs=-1,
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_iter=150,
            max_leaf_nodes=31,
            random_state=seed,
        ),
        "k-Nearest Neighbours (k-NN)": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            metric="euclidean",
            n_jobs=-1,
        ),
        "Support Vector Machine (SVM)": SVC(
            C=5.0,
            kernel="rbf",
            gamma="scale",
            probability=True,
            random_state=seed,
        ),
    }


def train_and_evaluate_classical_models(
    knn_sample_limit=40000,
    svm_sample_limit=30000,
):
    """Train and evaluate all eight classical B2 classifiers."""

    set_seed(RANDOM_SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TRAINING 8 CLASSICAL CLASSIFIERS ON B2 (56 CONTOUR FEATURES)")
    print("=" * 70)

    X_train, y_train, X_val, y_val, X_test, y_test = load_b2_features()

    print(f"Training samples:   {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    print(f"Test samples:       {len(X_test):,}")
    print(f"Number of features: {X_train.shape[1]}")

    models = get_classical_models()
    results = {}

    # Random subsets avoid favouring the original samples that appear first
    # in the augmented training dataset.
    rng = np.random.default_rng(RANDOM_SEED)

    for name, model in models.items():

        print(f"\n---> Training: {name}")

        if name == "k-Nearest Neighbours (k-NN)":
            indices = rng.choice(
                len(X_train),
                size=min(knn_sample_limit, len(X_train)),
                replace=False,
            )
            X_current = X_train[indices]
            y_current = y_train[indices]
            print(f"     Random training subset: {len(X_current):,} samples")

        elif name == "Support Vector Machine (SVM)":
            indices = rng.choice(
                len(X_train),
                size=min(svm_sample_limit, len(X_train)),
                replace=False,
            )
            X_current = X_train[indices]
            y_current = y_train[indices]
            print(f"     Random training subset: {len(X_current):,} samples")

        else:
            X_current = X_train
            y_current = y_train
            print(f"     Full training set: {len(X_current):,} samples")

        # Train
        start_time = time.perf_counter()
        model.fit(X_current, y_current)
        train_time = time.perf_counter() - start_time

        # Validation accuracy
        val_acc = model.score(X_val, y_val)

        # Test predictions and inference timing
        start_time = time.perf_counter()
        y_test_pred = model.predict(X_test)
        inference_time = time.perf_counter() - start_time

        latency_per_1k = (
            inference_time / len(X_test)
        ) * 1000 * 1000

        # Class probabilities for ROC-AUC evaluation
        if hasattr(model, "predict_proba"):
            y_test_proba = model.predict_proba(X_test)
        else:
            y_test_proba = np.zeros(
                (len(X_test), 10),
                dtype=np.float32,
            )

        test_acc = np.mean(y_test_pred == y_test)

        # Save individual model
        safe_name = (
            name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "_")
        )

        model_path = MODELS_DIR / f"{safe_name}_b2.joblib"
        joblib.dump(model, model_path)

        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

        print(f"     Validation Accuracy: {val_acc * 100:.2f}%")
        print(f"     Test Accuracy:       {test_acc * 100:.2f}%")
        print(f"     Training Time:       {train_time:.2f} s")
        print(f"     Inference Latency:   {latency_per_1k:.2f} ms / 1,000 samples")
        print(f"     Model Size:          {model_size_mb:.2f} MB")

        results[name] = {
            "model": model,
            "y_test_pred": y_test_pred,
            "y_test_proba": y_test_proba,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "train_time": train_time,
            "inf_latency_per_1k_ms": latency_per_1k,
            "model_size_mb": model_size_mb,
            "safe_name": safe_name,
        }

    # Save predictions and evaluation metadata separately from model files.
    predictions_path = MODELS_DIR / "classical_predictions_b2.joblib"

    predictions_data = {
        name: {
            "y_test_pred": result["y_test_pred"],
            "y_test_proba": result["y_test_proba"],
            "val_acc": result["val_acc"],
            "test_acc": result["test_acc"],
            "train_time": result["train_time"],
            "inf_latency_per_1k_ms": result["inf_latency_per_1k_ms"],
            "model_size_mb": result["model_size_mb"],
        }
        for name, result in results.items()
    }

    joblib.dump(predictions_data, predictions_path)

    print("\n" + "=" * 70)
    print("CLASSICAL CLASSIFICATION COMPLETED")
    print(f"Predictions saved to: {predictions_path}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    train_and_evaluate_classical_models()
