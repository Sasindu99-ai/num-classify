# EE5130 Mini Project: Team Instructions & Integration Guide

> **Course:** EE5130 Artificial Intelligence and Machine Learning  
> **Topic:** Comparative Study of Supervised Classification and Unsupervised Clustering Techniques Using the MNIST Dataset  
> **Repository Architecture:** Representation-First Modular Design (Zero Merge Conflicts Across Assigned Portions)

---

## 📌 1. Repository Architecture & Directory Layout

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
│   ├── a1/                       # Results, figures, CMs, and tables for Feature Set A1
│   ├── a2/                       # Results, figures, CMs, and tables for Feature Set A2
│   ├── a3/                       # Results, figures, CMs, and tables for Feature Set A3
│   ├── b1/                       # Results, figures, CMs, and tables for Feature Set B1
│   ├── b2/                       # Results, figures, CMs, and tables for Feature Set B2
│   ├── b3/                       # Results, figures, CMs, and tables for Feature Set B3
│   └── master_tables/            # Unified cross-representation master leaderboards
├── docs/                         # Assignment specifications and technical reports
├── README.md                     # Root project overview
├── README_A2.md                  # Comprehensive documentation for Feature Set A2
├── TeamInstructions.md           # This document (team integration guide)
└── requirements.txt              # Standard Python scientific stack + PyTorch with CUDA
```

---

## 👥 2. Module Allocation & Scope

The EE5130 project investigates two preprocessing paths and three feature representations:

| Path | Representation | Description | Features | Package Directory | Results Directory | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Path A** (Raw Grayscale) | **A1** | Flattened Pixel Intensities | 784 | `src/a1/` | `results/a1/` | Teammate Module |
| **Path A** (Raw Grayscale) | **A2** | **Row + Column Sums & 2-D CNN** | **56** | **`src/a2/`** | **`results/a2/`** | **✅ Completed & Verified** |
| **Path A** (Raw Grayscale) | **A3** | Concentric Ring Densities | 14 | `src/a3/` | `results/a3/` | Teammate Module |
| **Path B** (Contours) | **B1** | Flattened Contour Pixels | 784 | `src/b1/` | `results/b1/` | Teammate Module |
| **Path B** (Contours) | **B2** | Contour Row + Column Sums | 56 | `src/b2/` | `results/b2/` | Teammate Module |
| **Path B** (Contours) | **B3** | Contour Concentric Rings | 14 | `src/b3/` | `results/b3/` | Teammate Module |

👉 For an end-to-end reference implementation, mathematical formulations, and execution examples, see **[`README_A2.md`](README_A2.md)**.

---

## 🤝 3. Team Integration & Collaboration Guidelines

To allow multiple team members to work on branches, commit, and merge independently without conflicts, follow these rules:

### Rule 1: Zero-Overlap File Boundaries
- Each team member works strictly inside their assigned submodule folders:
  - Source code: `src/<module>/` (e.g. `src/a1/`, `src/a3/`, etc.)
  - Feature matrices: `data/features/<module>/`
  - Results, figures, tables: `results/<module>/`
- **Never modify code inside another member's submodule.**
- Shared utilities live in `src/common/`. If a shared utility needs an update, discuss with the team before modifying.

### Rule 2: Strict Anti-Leakage Protocol
All modules must adhere to identical anti-leakage safeguards:
1. **Never touch the test set** during preprocessing, feature scaler fitting, or hyperparameter selection.
2. **Never augment the validation or test sets**. Augmentation must be applied *strictly* to the training partition.
3. Use the shared split generated by `src/common/data_loading.py`:
   - Training partition: 54,000 samples
   - Validation partition: 6,000 samples
   - Test partition: 10,000 samples
   - Stored in: `data/processed/clean_partitions.npz`
4. Any scalers (e.g., `StandardScaler`, `MinMaxScaler`) must be **fitted strictly on the training partition** and transformed on validation and test sets without refitting:
   ```python
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_val_scaled = scaler.transform(X_val)
   X_test_scaled = scaler.transform(X_test)
   ```
5. Save your fitted scaler to `data/features/<module>/scaler_<module>.joblib`.

### Rule 3: Git & File Size Etiquette
- Raw CSVs (`mnist_train.csv`, `mnist_test.csv`) are tracked via Git LFS. Run `git lfs install` and `git lfs pull` once.
- **Do not commit large binary checkpoints or `.npz` arrays to Git.** The `.gitignore` is already configured to ignore `.npz`, `.npy`, `*.pt`, and `*.joblib` inside `results/*/models/` to prevent exceeding GitHub file size limits.
- Commit only your source code, figure PNGs, markdown documentation, and summary CSV tables.

---

## 📊 4. Master Aggregator Compatibility

To automatically generate unified cross-representation comparison tables and master leaderboards across all team modules, adhere to the standard table output formats:

### Step 1: Save Your Evaluation CSVs
When your classification and clustering experiments finish, export your results to these exact paths:
- `results/<module>/tables/classification_summary.csv`
- `results/<module>/tables/clustering_metrics.csv`

*(Replace `<module>` with your module name: `a1`, `a2`, `a3`, `b1`, `b2`, or `b3`.)*

### Step 2: Expected Column Headers

#### `classification_summary.csv`
```csv
Model,Representation,Val Accuracy (%),Test Accuracy (%),Macro F1,Weighted F1,Micro F1,Macro Precision,Macro Recall,ROC-AUC (Macro OVR),Train Time (s),Inference Latency (ms/1k),Model Size (MB)
```

#### `clustering_metrics.csv`
```csv
Algorithm,Clusters,Silhouette Score,Davies-Bouldin Index,Calinski-Harabasz Index,Adjusted Rand Index (ARI),Normalized Mutual Info (NMI),Homogeneity,Completeness,V-Measure,Fit Time (s)
```

### Step 3: Run the Master Comparator
Execute from the project root:
```bash
python -m src.compare_all
```
This script automatically:
1. Discovers all available `results/*/tables/*.csv` files.
2. Compiles unified master classification and clustering leaderboards.
3. Ranks models by test accuracy, macro F1, and clustering NMI/ARI.
4. Exports merged tables to `results/master_tables/master_classification_leaderboard.csv` and `results/master_tables/master_clustering_leaderboard.csv`.

---

## ⚡ 5. Quickstart & Environment Setup

### 1. Prerequisites
- Python 3.12 (or compatible 3.12–3.15)
- Git & Git LFS (`git lfs install; git lfs pull`)
- Optional: NVIDIA GPU with CUDA for PyTorch acceleration

### 2. Setup Virtual Environment
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

# Install dependencies
pip install -r requirements.txt
```

### 3. Generate Shared Data Partitions (Run Once)
```bash
python -m src.common.data_loading
```
This produces `data/processed/clean_partitions.npz` with clean 54k/6k/10k stratified splits for all representations.
