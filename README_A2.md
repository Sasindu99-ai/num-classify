# EE5130 Mini Project: Feature Set A2 & Path A 2-D CNN Documentation

> **Module:** `src.a2` — Path A (Raw $28 \times 28$ Grayscale) & Feature Set A2 (56 Row + Column Sums)  
> **Course:** EE5130 Artificial Intelligence and Machine Learning  
> **Author / Assigned Member:** Rachani Perera - Assigned Portion (Path A & Feature Set A2)

---

## 📌 Assigned Scope Overview

This package encapsulates the complete implementation, systematic experimental pipeline, quantitative evaluation, and mini exhibition demonstration strictly restricted to:

1. **Augmentation Layer (Path A)**: Intensity normalization to $[0.0, 1.0]$ and label-preserving geometric transformations (small rotation $\pm 10^\circ$, translation $\pm 2\text{ px}$, scaling $0.9\text{--}1.1\times$, shear $\pm 0.2$) strictly applied to the training set, expanding it $3\times$ ($54,000 \rightarrow 162,000$ samples). Validation ($6,000$) and test ($10,000$) sets remain unaugmented.
2. **Feature Reduction Layer (Feature Set A2)**: Continuous projection into a 56-feature vector ($28 \text{ row sums} + 28 \text{ column sums}$), preserving continuous grayscale intensities without binarisation. Standardized via `StandardScaler` fitted strictly on training data without leakage.
3. **Part A (Supervised Classification)**: 8 Classical Classifiers + Multilayer Perceptron (MLP) trained on the 56 A2 features, contrasted against a Deep Convolutional Neural Network (CNN) trained directly on unreduced 2-D $28 \times 28$ images with **NVIDIA CUDA GPU acceleration** on an RTX 5060.
4. **Part B (Unsupervised Clustering)**: 6 clustering algorithms (k-Means, Mini-Batch k-Means, GMM, Agglomerative Hierarchical, DBSCAN, Spectral Clustering) evaluated on standardized Feature Set A2 with internal (Silhouette, DBI, CHI) and external post-hoc metrics (ARI, NMI, V-measure).
5. **Mini Exhibition Deliverable**: Interactive CLI/GUI demonstration tool enabling live inference, 56-feature ink profile visualization, and projection failure diagnosis.

---

## 📁 Feature Set A2 Directory Structure

```text
num-classify/
├── data/
│   ├── raw/                      # Shared: mnist_train.csv, mnist_test.csv (untouched)
│   ├── processed/                # Shared: clean 54k/6k/10k stratified split
│   ├── augmented/
│   │   └── raw/                  # 162k Path A training images (augmented_train.npz)
│   └── features/
│       └── a2/                   # A2_standardized.npz, scaler_A2.joblib, pathA_2D.npz
├── src/
│   ├── common/                   # Shared team utilities
│   │   ├── __init__.py
│   │   ├── utils.py              # Global SEED=42, paths, device detection
│   │   └── data_loading.py       # Stratified 54k/6k train-val split
│   └── a2/                       # === FEATURE SET A2 PACKAGE ===
│       ├── __init__.py
│       ├── augmentation.py       # Path A geometric augmentations (162k train set)
│       ├── features.py           # Feature Set A2 extraction (56 continuous sums)
│       ├── verify_phase1.py      # Phase 1 verification assertions & plot
│       ├── classification/
│       │   ├── __init__.py
│       │   ├── classical_models.py # 8 classical models tuned on A2
│       │   ├── neural_models.py    # PyTorch MLP (56 A2) & CNN (2-D Path A) with CUDA
│       │   └── evaluate.py         # Full classification evaluation & error analysis
│       ├── clustering/
│       │   ├── __init__.py
│       │   ├── cluster_models.py   # 6 clustering models + PCA ablation on A2
│       │   ├── evaluate.py         # Internal & external clustering metrics
│       │   └── visualize.py        # PCA, t-SNE, UMAP, medoids, contingency heatmaps
│       └── interactive_demo.py   # Interactive exhibition live demo tool
├── results/
│   └── a2/                       # All generated A2 assets
│       ├── phase1_verification_figure.png
│       ├── demo_snapshot.png
│       ├── figures/              # ROC curves, manifold projections, failure analysis
│       ├── confusion_matrices/   # Normalized CM heatmaps for 10 models
│       ├── tables/               # classification_summary.csv, clustering_metrics.csv
│       └── models/               # cnn_best.pt, mlp_best.pt, *.joblib
├── README_A2.md
└── requirements.txt
```

---

## 🚀 Execution & Reproduction Commands

All commands are executed from the project root (`num-classify/`):

### 1. Phase 1: Preprocessing & Feature Extraction
```bash
# 1. Stratified 54k/6k split and binary conversion (shared)
python -m src.common.data_loading

# 2. Path A training set augmentation (162,000 samples)
python -m src.a2.augmentation

# 3. Feature Set A2 extraction (56 continuous row/col sums) and standardization
python -m src.a2.features

# 4. Phase 1 Verification Gate
python -m src.a2.verify_phase1
```

### 2. Phase 2: Supervised Classification (Part A)
```bash
# 5. Train 8 classical classifiers on Feature Set A2
python -m src.a2.classification.classical_models

# 6. Train PyTorch MLP (A2) and CNN (2-D images) with NVIDIA CUDA GPU acceleration
python -m src.a2.classification.neural_models

# 7. Comprehensive evaluation, leaderboard export, ROC curves, and error analysis
python -m src.a2.classification.evaluate
```

### 3. Phase 3: Unsupervised Clustering (Part B)
```bash
# 8. Fit 6 clustering algorithms on standardized Feature Set A2
python -m src.a2.clustering.cluster_models

# 9. Compute internal and external quantitative clustering metrics
python -m src.a2.clustering.evaluate

# 10. Generate PCA, t-SNE, UMAP projections, contingency heatmaps, and medoids
python -m src.a2.clustering.visualize
```

### 4. Phase 4: Interactive Exhibition Demonstration
```bash
# Test sample #0 (Convergent success: A2 and CNN both predict '7')
python -m src.a2.interactive_demo --index 0

# Test sample #7 (Demonstrates projection ambiguity: A2 predicts '4', CNN resolves '9')
python -m src.a2.interactive_demo --index 7
```

---

## 📊 Summary Leaderboards (Feature Set A2)

### Classification Leaderboard (Isolated 10,000-Sample Test Set)

| Rank | Model | Representation | Test Acc (%) | Macro F1 | Weighted F1 | Micro F1 | ROC-AUC | Train Time (s) | Latency (ms/1k) | Size |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Convolutional Neural Network (CNN)** | **Neural (2-D Path A Images)** | **99.61%** | **0.9961** | **0.9961** | **0.9961** | **0.99999** | 144.56 s | 25.60 ms | 3.44 MB |
| 🥈 | **Gradient Boosting** | Classical (Feature Set A2) | **93.96%** | **0.9386** | **0.9395** | **0.9396** | **0.9975** | 17.23 s | 16.92 ms | 5.25 MB |
| 🥉 | **Multilayer Perceptron (MLP)** | Neural (Feature Set A2) | **93.59%** | **0.9347** | **0.9358** | **0.9359** | **0.9973** | 43.02 s | 4.29 ms | 0.07 MB |
| 4 | **Support Vector Machine (SVM)** | Classical (Feature Set A2) | **92.44%** | **0.9229** | **0.9241** | **0.9244** | **0.9953** | 40.52 s | 788.15 ms | 5.35 MB |
| 5 | **Random Forest** | Classical (Feature Set A2) | **91.86%** | **0.9174** | **0.9186** | **0.9186** | **0.9956** | 20.17 s | 6.25 ms | 502.99 MB |
| 6 | **k-Nearest Neighbours (k-NN)** | Classical (Feature Set A2) | **88.42%** | **0.8810** | **0.8825** | **0.8842** | **0.9724** | 0.00 s | 46.19 ms | 8.85 MB |
| 7 | **Decision Tree** | Classical (Feature Set A2) | **83.47%** | **0.8325** | **0.8348** | **0.8347** | **0.9278** | 9.17 s | 0.19 ms | 1.73 MB |
| 8 | **Logistic Regression** | Classical (Feature Set A2) | **78.69%** | **0.7774** | **0.7807** | **0.7869** | **0.9658** | 2.99 s | 0.12 ms | 3.05 KB |
| 9 | **Linear Discriminant Analysis** | Classical (Feature Set A2) | **72.65%** | **0.7187** | **0.7230** | **0.7265** | **0.9538** | 0.34 s | 0.11 ms | 7.56 KB |
| 10 | **Naive Bayes** | Classical (Feature Set A2) | **60.21%** | **0.5707** | **0.5754** | **0.6021** | **0.9303** | 0.07 s | 2.18 ms | 5.13 KB |

### Clustering Benchmark ($N = 10,000$, $k=10$ Baseline)

| Algorithm | Clusters | Silhouette | Davies-Bouldin | Calinski-Harabasz | ARI | NMI | V-Measure | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Spectral Clustering** | 10 | 0.0668 | **1.9049** | 683.73 | **0.3537** | **0.5133** | **0.5133** | 8.65 s |
| **Agglomerative (Ward)** | 10 | 0.0646 | 2.4124 | 729.25 | **0.2997** | **0.4517** | **0.4517** | 9.31 s |
| **Mini-Batch k-Means** | 10 | 0.1018 | 2.1421 | 868.74 | 0.2719 | 0.3757 | 0.3757 | **0.27 s** |
| **Gaussian Mixture (GMM)** | 10 | 0.0544 | 2.6312 | 588.90 | 0.2486 | 0.3716 | 0.3716 | 0.31 s |
| **k-Means (PCA-15 on A2)** | 10 | 0.1121 | 1.9433 | 927.51 | 0.2339 | 0.3691 | 0.3691 | 0.37 s |
| **k-Means** | 10 | **0.1143** | 1.9391 | **928.75** | 0.2320 | 0.3696 | 0.3696 | 0.37 s |
| **DBSCAN** | 1 (11.6% noise) | — | — | — | — | — | — | 0.15 s |
