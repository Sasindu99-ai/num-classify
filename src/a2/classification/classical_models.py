"""Classical Classifiers on Feature Set A2 (56 features).

Implements, trains, and evaluates 8 classical machine learning models:
  1. Logistic Regression
  2. k-Nearest Neighbours (k-NN)
  3. Support Vector Machine (SVM)
  4. Decision Tree
  5. Random Forest
  6. Naive Bayes (Gaussian)
  7. Linear Discriminant Analysis (LDA)
  8. Gradient Boosting (HistGradientBoosting)
"""

from pathlib import Path
import time
import os
import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from src.common.utils import (
    MODELS_A2_DIR,
    RANDOM_SEED,
    set_seed,
    ensure_a2_directories,
)
from src.a2.features import load_a2_features


def get_classical_models(seed: int = RANDOM_SEED) -> dict:
    """Return dictionary of configured candidate classical classifiers."""
    models = {
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
    return models


def train_and_evaluate_classical_models(
    knn_sample_limit: int = 40000,
    svm_sample_limit: int = 30000,
) -> dict:
    """Train all 8 classical classifiers on Feature Set A2, evaluate on validation and test sets."""
    set_seed(RANDOM_SEED)
    ensure_a2_directories()

    print("=" * 70)
    print("TRAINING 8 CLASSICAL CLASSIFIERS ON FEATURE SET A2 (56 FEATURES)")
    print("=" * 70)

    X_train, y_train, X_val, y_val, X_test, y_test = load_a2_features(standardized=True)
    models = get_classical_models(RANDOM_SEED)
    results = {}

    for name, model in models.items():
        print(f"\n---> Training: {name} ...")
        if name == "k-Nearest Neighbours (k-NN)":
            X_tr_curr = X_train[:knn_sample_limit]
            y_tr_curr = y_train[:knn_sample_limit]
            print(f"     (Using documented representative subset of {len(X_tr_curr):,} samples for O(N) k-NN)")
        elif name == "Support Vector Machine (SVM)":
            X_tr_curr = X_train[:svm_sample_limit]
            y_tr_curr = y_train[:svm_sample_limit]
            print(f"     (Using documented representative subset of {len(X_tr_curr):,} samples for O(N^2) RBF-SVM)")
        else:
            X_tr_curr = X_train
            y_tr_curr = y_train
            print(f"     (Using full augmented training set: {len(X_tr_curr):,} samples)")

        t0 = time.perf_counter()
        model.fit(X_tr_curr, y_tr_curr)
        train_time = time.perf_counter() - t0

        val_acc = model.score(X_val, y_val)

        t1 = time.perf_counter()
        y_test_pred = model.predict(X_test)
        total_inf_time = time.perf_counter() - t1
        inf_latency_per_1k = (total_inf_time / len(X_test)) * 1000.0

        if hasattr(model, "predict_proba"):
            y_test_proba = model.predict_proba(X_test)
        elif hasattr(model, "decision_function"):
            df = model.decision_function(X_test)
            exp_df = np.exp(df - np.max(df, axis=1, keepdims=True))
            y_test_proba = exp_df / exp_df.sum(axis=1, keepdims=True)
        else:
            y_test_proba = np.zeros((len(X_test), 10), dtype=np.float32)

        test_acc = np.mean(y_test_pred == y_test)

        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        model_save_path = MODELS_A2_DIR / f"{safe_name}.joblib"
        joblib.dump(model, model_save_path)
        model_size_mb = os.path.getsize(model_save_path) / (1024 * 1024)

        print(f"     Validation Accuracy: {val_acc * 100:.2f}%")
        print(f"     Test Accuracy:       {test_acc * 100:.2f}%")
        print(f"     Training Duration:   {train_time:.2f} s")
        print(f"     Inference Latency:   {inf_latency_per_1k * 1000:.2f} ms / 1,000 samples")
        print(f"     Model Size on Disk:  {model_size_mb:.2f} MB")

        results[name] = {
            "model": model,
            "y_test_pred": y_test_pred,
            "y_test_proba": y_test_proba,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "train_time": train_time,
            "inf_latency_per_1k_ms": inf_latency_per_1k * 1000,
            "model_size_mb": model_size_mb,
            "safe_name": safe_name,
        }

    predictions_path = MODELS_A2_DIR / "classical_predictions.joblib"
    joblib.dump(
        {
            name: {
                "y_test_pred": res["y_test_pred"],
                "y_test_proba": res["y_test_proba"],
                "val_acc": res["val_acc"],
                "test_acc": res["test_acc"],
                "train_time": res["train_time"],
                "inf_latency_per_1k_ms": res["inf_latency_per_1k_ms"],
                "model_size_mb": res["model_size_mb"],
            }
            for name, res in results.items()
        },
        predictions_path,
    )
    print(f"\nAll classical model predictions saved to: {predictions_path.resolve()}")
    return results


if __name__ == "__main__":
    train_and_evaluate_classical_models()

