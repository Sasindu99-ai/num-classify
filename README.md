# EE5130 Mini Project: MNIST Classification & Clustering

> **Course:** EE5130 Artificial Intelligence and Machine Learning  
> **Topic:** Comparative Study of Supervised Classification and Unsupervised Clustering Techniques Using the MNIST Dataset  
> **Repository Architecture:** Representation-First Modular Design (Zero Merge Conflicts Across Assigned Portions)

---

## 📌 Executive Summary & Architecture

This repository contains the complete experimental framework for evaluating **Supervised Classification (Part A)** and **Unsupervised Clustering (Part B)** across multiple feature representations of the standard MNIST digit dataset.

To ensure clean isolation between team members working on different representations, the codebase adheres to a **Representation-First Modular Architecture**:

```text
num-classify/
├── data/
│   ├── raw/                      # Shared raw datasets (mnist_train.csv, mnist_test.csv - tracked via Git LFS)
│   ├── processed/                # Shared stratified 54k/6k train/val split (clean_partitions.npz)
│   ├── augmented/                # Augmented training copies (e.g. data/augmented/raw/ for Path A)
│   └── features/                 # Extracted feature matrices organized by representation
│       ├── a1/                   # (Assigned Member: 784 Raw Flattened Pixels)
│       ├── a2/                   # (Assigned: 56 Continuous Row + Col Sums + 2-D CNN)
│       ├── a3/                   # (Assigned Member: 14 Concentric Rings)
│       ├── b1/                   # (Assigned Member: Flattened Contour Pixels)
│       ├── b2/                   # (Assigned Member: Contour Row + Col Sums)
│       └── b3/                   # (Assigned Member: Contour Concentric Rings)
├── src/
│   ├── common/                   # Shared infrastructure across all team members
│   │   ├── utils.py              # Fixed SEED=42, path resolvers, CUDA device configuration
│   │   └── data_loading.py       # Anti-leakage stratified train/val split
│   ├── a1/                       # Submodule for Feature Set A1
│   ├── a2/                       # === FEATURE SET A2 PACKAGE (Implemented & Verified) ===
│   │   ├── augmentation.py       # Label-preserving Path A affine transformations (162k train set)
│   │   ├── features.py           # 56 continuous row/col sums extraction & StandardScaler
│   │   ├── verify_phase1.py      # Phase 1 verification assertions & diagnostic plots
│   │   ├── classification/       # Part A: 8 Classical + PyTorch MLP (56 A2) & CNN (2-D Path A)
│   │   ├── clustering/           # Part B: 6 Clustering algorithms + PCA ablation
│   │   └── interactive_demo.py   # Live CLI/GUI exhibition demonstration tool
│   ├── a3/                       # Submodule for Feature Set A3
│   ├── b1/                       # Submodule for Feature Set B1
│   ├── b2/                       # Submodule for Feature Set B2
│   ├── b3/                       # Submodule for Feature Set B3
│   └── compare_all.py            # Master cross-representation aggregator & leaderboard generator
├── results/
│   ├── a2/                       # Results, figures, CMs, and tables for Feature Set A2
│   └── master_tables/            # Unified cross-representation leaderboards
├── docs/                         # Assignment specifications and technical reports
├── README.md                     # Repository overview & team integration guidelines
├── README_A2.md                  # Comprehensive documentation for Feature Set A2
└── requirements.txt              # Standard Python scientific stack + PyTorch with CUDA
```

---

## 👥 Module Allocation & Scope

The EE5130 project investigates two preprocessing paths and three feature reductions:

| Path | Representation | Description | Features | Package Directory | Status |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Path A** (Raw Grayscale) | **A1** | Flattened Pixel Intensities | 784 | `src/a1/` | Teammate Module |
| **Path A** (Raw Grayscale) | **A2** | **Row + Column Sums & 2-D CNN** | **56** | **`src/a2/`** | **✅ Completed & Verified** |
| **Path A** (Raw Grayscale) | **A3** | Concentric Ring Densities | 14 | `src/a3/` | Teammate Module |
| **Path B** (Contours) | **B1** | Flattened Contour Pixels | 784 | `src/b1/` | Teammate Module |
| **Path B** (Contours) | **B2** | Contour Row + Column Sums | 56 | `src/b2/` | Teammate Module |
| **Path B** (Contours) | **B3** | Contour Concentric Rings | 14 | `src/b3/` | Teammate Module |

👉 For in-depth documentation, mathematical derivations, failure mode analysis, and reproduction details of **Feature Set A2**, please refer to **[`README_A2.md`](README_A2.md)**.

---

## 🤝 Team Integration & Collaboration Guidelines

To allow multiple team members to push and merge independently without conflicts:

### 1. Zero-Overlap File Structure
- Never edit files outside your assigned module directory (`src/<module>/`, `data/features/<module>/`, `results/<module>/`).
- Common functions (seeds, device resolution, data splitting) must remain in `src/common/`.

### 2. Strict Anti-Leakage Protocol
All modules must adhere to the shared experimental safeguards:
- **Never touch the test set** during preprocessing, scaler fitting, or hyperparameter selection.
- **Never augment the validation or test sets**.
- Run `python -m src.common.data_loading` once to populate `data/processed/clean_partitions.npz` (54,000 train, 6,000 val, 10,000 test).
- Fit scalers (`StandardScaler`, etc.) strictly on the training partition and save them to `data/features/<module>/scaler_<module>.joblib`.

### 3. Master Aggregator Compatibility
When your experiments finish, save your evaluation tables to:
- `results/<module>/tables/classification_summary.csv`
- `results/<module>/tables/clustering_metrics.csv`

Running the cross-representation comparator:
```bash
python -m src.compare_all
```
will automatically discover all completed modules and build the unified master classification and clustering leaderboards in `results/master_tables/`.

---

## ⚡ Quickstart & Setup

### 1. Prerequisites
- Python 3.12 (or compatible 3.12–3.15)
- Git & Git LFS (`git lfs install; git lfs pull`)
- NVIDIA GPU with CUDA support (tested on RTX 5060 Laptop GPU with CUDA 13.2 / `sm_120`)

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/rachani-perera/num-classify.git
cd num-classify

# Create and activate virtual environment
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install required scientific packages
pip install -r requirements.txt
```

### 3. Reproduce Feature Set A2 End-to-End
```bash
# Step 1: Preprocessing & Data Partitions
python -m src.common.data_loading

# Step 2: Path A Training Augmentations (162,000 samples)
python -m src.a2.augmentation

# Step 3: Feature Set A2 Extraction (56 Continuous Row/Col Sums)
python -m src.a2.features

# Step 4: Verification Gate
python -m src.a2.verify_phase1

# Step 5: Part A Supervised Classification (8 Classical Models + MLP & CNN)
python -m src.a2.classification.classical_models
python -m src.a2.classification.neural_models
python -m src.a2.classification.evaluate

# Step 6: Part B Unsupervised Clustering (6 Algorithms + Projections)
python -m src.a2.clustering.cluster_models
python -m src.a2.clustering.evaluate
python -m src.a2.clustering.visualize

# Step 7: Cross-Representation Master Leaderboard
python -m src.compare_all
```

### 4. Interactive Live Exhibition Demonstration
To inspect live model predictions, ink profiles, and spatial projection loss on test digits:
```bash
# Run on sample #0 (Success case: A2 and CNN both predict '7')
python -m src.a2.interactive_demo --index 0

# Run on sample #7 (Projection failure case: A2 predicts '4', CNN resolves '9')
python -m src.a2.interactive_demo --index 7
```

---

## 🏆 Current Benchmark (Feature Set A2 & Path A CNN)

### Supervised Classification (Isolated 10,000 Test Samples)

| Rank | Model | Input Representation | Test Acc (%) | Macro F1 | ROC-AUC | Inference (ms/1k) | Model Size |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Convolutional Neural Network (CNN)** | **2-D Path A Images ($28 \times 28$)** | **99.61%** | **0.9961** | **0.99999** | 25.60 ms | 3.44 MB |
| 🥈 | **Gradient Boosting** | Feature Set A2 (56 features) | **93.96%** | **0.9386** | **0.9975** | 16.92 ms | 5.25 MB |
| 🥉 | **Multilayer Perceptron (MLP)** | Feature Set A2 (56 features) | **93.59%** | **0.9347** | **0.9973** | 4.29 ms | 0.07 MB |
| 4 | **Support Vector Machine (SVM)** | Feature Set A2 (56 features) | **92.44%** | **0.9229** | **0.9953** | 788.15 ms | 5.35 MB |
| 5 | **Random Forest** | Feature Set A2 (56 features) | **91.86%** | **0.9174** | **0.9956** | 6.25 ms | 502.99 MB |
| 6 | **k-Nearest Neighbours (k-NN)** | Feature Set A2 (56 features) | **88.42%** | **0.8810** | **0.9724** | 46.19 ms | 8.85 MB |
| 7 | **Decision Tree** | Feature Set A2 (56 features) | **83.47%** | **0.8325** | **0.9278** | 0.19 ms | 1.73 MB |
| 8 | **Logistic Regression** | Feature Set A2 (56 features) | **78.69%** | **0.7774** | **0.9658** | 0.12 ms | 3.05 KB |
| 9 | **Linear Discriminant Analysis** | Feature Set A2 (56 features) | **72.65%** | **0.7187** | **0.9538** | 0.11 ms | 7.56 KB |
| 10 | **Naive Bayes** | Feature Set A2 (56 features) | **60.21%** | **0.5707** | **0.9303** | 2.18 ms | 5.13 KB |

### Unsupervised Clustering ($N = 10,000$, $k=10$ Baseline on Standardized A2)

| Algorithm | Clusters | Silhouette | Davies-Bouldin | Calinski-Harabasz | ARI | NMI | V-Measure |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Spectral Clustering** | 10 | 0.0668 | **1.9049** | 683.73 | **0.3537** | **0.5133** | **0.5133** |
| **Agglomerative (Ward)** | 10 | 0.0646 | 2.4124 | 729.25 | **0.2997** | **0.4517** | **0.4517** |
| **Mini-Batch k-Means** | 10 | 0.1018 | 2.1421 | 868.74 | 0.2719 | 0.3757 | 0.3757 |
| **Gaussian Mixture (GMM)** | 10 | 0.0544 | 2.6312 | 588.90 | 0.2486 | 0.3716 | 0.3716 |
| **k-Means (PCA-15 on A2)** | 10 | 0.1121 | 1.9433 | 927.51 | 0.2339 | 0.3691 | 0.3691 |
| **k-Means** | 10 | **0.1143** | 1.9391 | **928.75** | 0.2320 | 0.3696 | 0.3696 |
| **DBSCAN** | 1 (11.6% noise) | — | — | — | — | — | — |

---

## 🔍 Key Theoretical Finding: Spatial Projection Ambiguity

A foundational insight of our investigation is the mathematical characterization of **Spatial Projection Ambiguity** inherent in Feature Set A2:
$$\mathbf{f}_{\text{row}}[i] = \sum_{j=1}^{28} I[i, j], \quad \mathbf{f}_{\text{col}}[j] = \sum_{i=1}^{28} I[i, j]$$

While 1-D row and column marginal projections achieve an impressive **93.96% accuracy** with non-linear ensemble models (Gradient Boosting), they inherently lose the **2-D joint spatial distribution** $P(x, y)$ of pixel mass. As a result:
- Pairs of digits with similar horizontal and vertical mass densities (e.g., $3 \leftrightarrow 5$, $4 \leftrightarrow 9$, $1 \leftrightarrow 7$) exhibit high overlap in 56-D feature space.
- The 2-D Convolutional Neural Network (CNN) preserves local $3 \times 3$ receptive fields, resolving these spatial ambiguities and boosting accuracy from **93.96%** to **99.61%** (achieving a **$15.5\times$ error reduction**, from 604 down to just 39 errors across 10,000 test digits).

---

## 📄 Documentation & Reports

Detailed technical documentation and submission drafts are available in `docs/`:
- **[`README_A2.md`](README_A2.md)**: Full implementation reference for Feature Set A2.
- **[`docs/technical_report.md`](docs/technical_report.md)**: Comprehensive technical report draft with methodology, mathematical formulations, metric tables, and academic discussion.
