# EE5130 Mini Project: Feature Set B2 – Contour Row & Column Sums Documentation

> **Module:** `src.b2` — Path B (Binary Contours) & Feature Set B2 (56 Contour Row + Column Sums)  
> **Course:** EE5130 Artificial Intelligence and Machine Learning  
> **Assigned Portion:** Path B – Feature Set B2  
> **Dataset:** MNIST Handwritten Digits

---

## 📌 Assigned Scope Overview

This package implements the complete experimental pipeline for **Feature Set B2**, which represents MNIST handwritten digits using contour-based features.

The implementation covers:

1. **Contour Extraction Layer (Path B):** Conversion of grayscale MNIST images into binary foreground masks, followed by morphological erosion and boundary extraction to obtain contour representations.

2. **Feature Reduction Layer (Feature Set B2):** Projection of each 28 × 28 binary contour image into a 56-dimensional feature vector consisting of 28 row-wise contour sums and 28 column-wise contour sums.

3. **Feature Scaling:** Standardization using `StandardScaler`, fitted strictly on the training feature matrix and subsequently applied to validation and test features without refitting.

4. **Part A (Supervised Classification):** Evaluation of eight classical machine learning classifiers and a PyTorch Multilayer Perceptron (MLP) using the extracted 56-dimensional B2 features.

5. **Part B (Unsupervised Clustering):** Comparison of six clustering algorithms using internal and external evaluation metrics, including Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index, Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), Homogeneity, Completeness, and V-measure.

6. **Experimental Analysis:** Classification performance comparison, confusion analysis, model efficiency evaluation, clustering visualizations, and DBSCAN parameter investigation.

---

## 📁 Feature Set B2 Directory Structure

```text
num-classify/
├── data/
│   ├── raw/                      # Shared MNIST datasets
│   ├── processed/                # Shared stratified 54k/6k/10k split
│   └── features/
│       └── b2/                   # B2 extracted features and scaler
├── src/
│   ├── common/                   # Shared team utilities
│   │   ├── utils.py
│   │   └── data_loading.py
│   └── b2/                       # === FEATURE SET B2 PACKAGE ===
│       ├── contours.py           # Binary contour extraction
│       ├── features.py           # 56-dimensional contour feature extraction
│       ├── dataset.py            # B2 dataset preparation
│       ├── verify_phase1.py      # Phase 1 verification
│       ├── classification/
│       │   ├── __init__.py
│       │   ├── classical_models.py
│       │   ├── neural_models.py
│       │   ├── evaluate.py
│       │   └── learning_curves.py
│       └── clustering/
│           ├── dbscan_k_distance.py
│           ├── dbscan_tuning.py
│           ├── evaluation.py
│           ├── experiment.py
│           ├── models.py
│           ├── test_models.py
│           └── visualizations.py
├── results/
│   └── b2/                       # Generated B2 experimental outputs
│       ├── baseline/
│       ├── classification/
│       ├── clustering/
│       ├── figures/
│       ├── confusion_matrices/
│       ├── tables/
│       └── models/
├── docs/
│   └── technical_report_B2.md
└── requirements.txt
```

---

## 🔬 B2 Feature Extraction Methodology

### 1. Binary Contour Extraction

The original MNIST image is a 28 × 28 grayscale image. A binary foreground mask is generated using an intensity threshold of 100.

Morphological erosion is then applied using a 3 × 3 minimum filter. The contour is obtained by subtracting the eroded foreground mask from the original binary mask.

The resulting binary contour image represents the boundary structure of the handwritten digit.

### 2. Contour Row and Column Sums

For a binary contour image \(C\), the row-wise and column-wise sums are calculated as follows:

**Row-wise contour sums:**

\[
R_i = \sum_{j=1}^{28} C_{ij},
\quad i=1,\ldots,28
\]

**Column-wise contour sums:**

\[
C_j = \sum_{i=1}^{28} C_{ij},
\quad j=1,\ldots,28
\]

The final B2 feature vector is:

\[
X_{B2} =
[R_1,\ldots,R_{28},C_1,\ldots,C_{28}]
\]

Therefore:

\[
28+28=\boxed{56\text{ features}}
\]

This approach reduces the original 784-dimensional contour representation to 56 features while retaining information about the distribution of contour pixels across rows and columns.

Unlike Feature Set A2, which uses continuous grayscale intensity sums, B2 uses binary contour information.

---

## 🚀 Execution & Reproduction Commands

All commands are executed from the project root (`num-classify/`).

### 1. Phase 1: Contour Extraction & Feature Preparation

```bash
# 1. Generate shared stratified training, validation and test partitions
python -m src.common.data_loading

# 2. Prepare the B2 contour dataset
python -m src.b2.dataset

# 3. Verify contour extraction and feature preparation
python -m src.b2.verify_phase1
```

### 2. Phase 2: Supervised Classification (Part A)

```bash
# 4. Train eight classical classification models
python -m src.b2.classification.classical_models

# 5. Train the B2 Multilayer Perceptron
python -m src.b2.classification.neural_models

# 6. Evaluate classification models and generate performance outputs
python -m src.b2.classification.evaluate

# 7. Generate MLP learning curves
python -m src.b2.classification.learning_curves
```

### 3. Phase 3: Unsupervised Clustering (Part B)

```bash
# 8. Run clustering experiments
python -m src.b2.clustering.experiment

# 9. Evaluate clustering performance
python -m src.b2.clustering.evaluation

# 10. Generate clustering visualizations
python -m src.b2.clustering.visualizations
```

### 4. Phase 4: DBSCAN Parameter Investigation

```bash
# 11. Generate the k-distance plot
python -m src.b2.clustering.dbscan_k_distance

# 12. Investigate DBSCAN parameter combinations
python -m src.b2.clustering.dbscan_tuning
```

---

## 📊 Summary Leaderboards (Feature Set B2)

### Classification Leaderboard (Isolated 10,000-Sample Test Set)

| Rank | Model | Representation | Test Acc (%) | Macro F1 | ROC-AUC | Train Time (s) | Latency (ms/1k) | Model Size (MB) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Gradient Boosting** | Classical (Feature Set B2) | **92.89%** | **0.9279** | **0.9967** | 55.38 | 37.74 | 5.19 |
| 🥈 | **Multilayer Perceptron (MLP)** | Neural (Feature Set B2) | **91.37%** | **0.9122** | **0.9952** | 45.80 | 6.48 | 0.07 |
| 🥉 | **Random Forest** | Classical (Feature Set B2) | **90.52%** | **0.9037** | **0.9939** | 12.27 | 6.80 | 656.34 |
| 4 | **Support Vector Machine (SVM)** | Classical (Feature Set B2) | **89.48%** | **0.8927** | **0.9920** | 85.36 | 1,879.96 | 8.90 |
| 5 | **k-Nearest Neighbours (k-NN)** | Classical (Feature Set B2) | **84.82%** | **0.8436** | **0.9669** | 0.01 | 63.08 | 8.85 |
| 6 | **Decision Tree** | Classical (Feature Set B2) | **80.27%** | **0.7989** | **0.9202** | 2.96 | 0.19 | 2.00 |
| 7 | **Logistic Regression** | Classical (Feature Set B2) | **77.35%** | **0.7610** | **0.9641** | 2.03 | 0.13 | 0.003 |
| 8 | **Linear Discriminant Analysis (LDA)** | Classical (Feature Set B2) | **74.10%** | **0.7258** | **0.9582** | 0.31 | 0.13 | 0.008 |
| 9 | **Gaussian Naive Bayes** | Classical (Feature Set B2) | **60.73%** | **0.5702** | **0.9448** | 0.06 | 1.21 | 0.005 |

### Clustering Benchmark (3,000-Sample Training Subset)

| Algorithm | Clusters | Silhouette | Davies-Bouldin | Calinski-Harabasz | ARI | NMI | V-Measure | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **k-Means** | 10 | 0.0746 | 2.3250 | 159.69 | **0.2521** | **0.3211** | **0.3211** | 0.376 |
| **Agglomerative (Ward)** | 10 | 0.0361 | 2.6834 | 130.13 | 0.2285 | 0.3131 | 0.3131 | 0.310 |
| **Mini-Batch k-Means** | 10 | 0.0538 | 2.6229 | 141.64 | 0.1813 | 0.2737 | 0.2737 | **0.034** |
| **Spectral Clustering** | 10 | 0.0338 | 2.1375 | 135.83 | 0.1710 | 0.3109 | 0.3109 | 1.392 |
| **Gaussian Mixture Model (GMM)** | 10 | 0.0316 | 2.9437 | 112.99 | 0.1615 | 0.2473 | 0.2473 | 0.243 |
| **DBSCAN** | 1* | — | — | — | 0.0463 | 0.0848 | 0.0848 | 0.017 |

*DBSCAN produced one non-noise cluster, with approximately 53.03% of samples labelled as noise. Internal clustering scores are not meaningful for this one-cluster result.*

---

## 📈 Experimental Findings & Observations

### 1. Supervised Classification

- Gradient Boosting achieved the highest classification accuracy of **92.89%** and macro F1-score of **0.9279**.
- The MLP achieved **91.37%** test accuracy while maintaining a compact model size of approximately 0.07 MB.
- Random Forest achieved 90.52% accuracy but required substantially more storage than the other models.
- SVM achieved 89.48% accuracy but had comparatively high inference latency.

### 2. Unsupervised Clustering

- k-Means achieved the highest ARI (0.2521) and NMI (0.3211) among the evaluated baseline clustering algorithms.
- Spectral Clustering achieved higher completeness, but its overall external agreement was lower than k-Means.
- Mini-Batch k-Means provided the fastest clustering execution.
- DBSCAN did not identify a meaningful multi-cluster structure under the tested parameter settings.

### 3. Overall Interpretation

The B2 representation provides a compact description of handwritten digit boundaries. However, reducing the contour image to row and column sums removes some spatial information.

This loss of detailed spatial relationships can cause different handwritten digits to produce similar feature vectors, which may affect classification and clustering performance.

The supervised results demonstrate that the extracted contour features retain useful information for digit recognition, while the clustering results indicate that separating all ten digit classes without labels remains challenging.

---

## ⚠️ Limitations

1. **Spatial Information Loss:** Row and column summation removes exact pixel-position information within each row and column.
2. **Contour Dependency:** The representation depends on the quality of the binary contour extraction process.
3. **Reduced Representation:** The 56-dimensional feature vector cannot preserve all structural details of the original 28 × 28 image.
4. **Clustering Performance:** The clustering algorithms show limited agreement with the true digit classes.
5. **DBSCAN Limitations:** The tested DBSCAN configurations failed to produce a useful ten-cluster structure.
6. **Subset-Based Clustering:** Clustering experiments were conducted using a 3,000-sample subset rather than the complete augmented training dataset.

---

## 📚 Related Documentation

- **[B2 Technical Report](docs/technical_report_B2.md)** — Detailed methodology, experimental procedures, results, and analysis.
- **[Main Project README](README.md)** — Overall project architecture and team integration.
- **[Team Instructions](TeamInstructions.md)** — Shared development conventions, data handling, and evaluation requirements.

---

## 🏁 Conclusion

Feature Set B2 provides a contour-based dimensionality reduction approach for handwritten digit recognition. By converting binary contour images into 56 row and column sum features, the representation significantly reduces dimensionality while retaining useful boundary-distribution information.

Gradient Boosting achieved the best supervised classification performance, with a test accuracy of 92.89%. The MLP also demonstrated competitive performance with a considerably smaller model size.

For unsupervised clustering, k-Means achieved the highest external agreement with the true digit labels among the evaluated baseline algorithms. However, the clustering results indicate that the B2 representation does not completely separate the ten digit classes.

Overall, B2 demonstrates the usefulness of contour-based feature extraction for compact handwritten digit representation, while also highlighting the limitations of losing detailed two-dimensional spatial information.
