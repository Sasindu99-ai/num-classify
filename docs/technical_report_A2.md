# EE5130 Mini Project Technical Report: Comparative Study of Supervised Classification and Unsupervised Clustering on MNIST

**Course:** EE5130 Artificial Intelligence and Machine Learning  
**Assigned Scope:** Path A (Raw Grayscale $28 \times 28$ Images), Feature Set A2 (56 Continuous Row + Column Sums), and 2-D Path A CNN  
**Hardware Platform:** NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM, `sm_120`), CUDA 13.2  

---

## Abstract

This report presents an empirical and theoretical investigation into supervised classification and unsupervised clustering on the MNIST benchmark dataset under **Path A (Raw Grayscale Images)** and **Feature Set A2 (56 Row and Column Sums)**, alongside an unreduced **2-D Convolutional Neural Network (CNN)** baseline. A rigorous, anti-leakage preprocessing pipeline was implemented, performing a stratified $54,000 / 6,000$ train-validation split and expanding the training partition $3\times$ ($162,000$ samples) using label-preserving affine transformations ($\pm 10^\circ$ rotation, $\pm 2\text{ px}$ translation, $0.9\text{--}1.1\times$ scaling, and $\pm 0.2$ shear). Feature Set A2 compresses the $784$-dimensional pixel space down to $56$ continuous features ($14\times$ dimensionality reduction) by taking marginal row and column intensity sums.

In **Part A (Supervised Classification)**, ten distinct architectures were evaluated on an isolated $10,000$-sample test set. Among classical classifiers operating on the $56$ A2 features, **Gradient Boosting** achieved peak performance with **93.96% accuracy** (Macro F1: 0.9386, ROC-AUC: 0.9975), followed closely by a PyTorch **Multilayer Perceptron (MLP)** at **93.59% accuracy** (Macro F1: 0.9347) and an **RBF SVM** at **92.44% accuracy**. However, the 2-D CNN operating directly on spatial images attained an extraordinary **99.61% accuracy** (Macro F1: 0.9961, ROC-AUC: 0.99999), reducing classification errors by $15.5\times$ (from $604$ down to $39$ misclassifications). We mathematically demonstrate that this gap is attributable to **Spatial Projection Ambiguity**: marginal 1-D projections preserve ink mass per axis but discard the joint spatial distribution $P(x, y)$, confounding digits with overlapping marginal profiles ($3 \leftrightarrow 5$, $4 \leftrightarrow 9$, $1 \leftrightarrow 7$).

In **Part B (Unsupervised Clustering)**, six clustering algorithms were benchmarked on standardized Feature Set A2 ($N = 10,000, k = 10$) strictly without label exposure during training. **Spectral Clustering** achieved the highest alignment with true digit morphology (**NMI: 0.5133, ARI: 0.3537**), followed by **Agglomerative Hierarchical Clustering (Ward's linkage)** (**NMI: 0.4517, ARI: 0.2997**). Density-based DBSCAN revealed that MNIST digits form continuous, high-density bridges in 56-D space rather than isolated density islands, yielding a single macro-cluster with $11.6\%$ noise points. Manifold projections (PCA, t-SNE, UMAP) corroborate that Feature Set A2 captures global digit elongation and loop mass while leaving fine topological loops entangled.

---

## 1. Introduction & Assigned Scope

The primary objective of the EE5130 Mini Project is to systematically contrast supervised pattern recognition and unsupervised cluster discovery across multiple input representations of the MNIST dataset. To prevent merge collisions and isolate methodological variables, this work focuses exclusively on:
1. **Augmentation Layer**: Path A (Raw $28 \times 28$ grayscale images with intensities in $[0.0, 1.0]$).
2. **Feature Reduction Layer**: Representation 2 — Continuous Row and Column Sums ($28 + 28 = 56$ features, designated as **Feature Set A2**).
3. **Supervised Classification (Part A)**: 8 Classical Classifiers + PyTorch MLP on the 56 A2 features, contrasted with a Deep CNN receiving unreduced 2-D Path A images.
4. **Unsupervised Clustering (Part B)**: 6 clustering models evaluated on standardized Feature Set A2.

---

## 2. Preprocessing & Feature Extraction Pipeline

### 2.1 Anti-Leakage Partitioning
The original dataset comprises $60,000$ training rows (`mnist_train.csv`) and $10,000$ testing rows (`mnist_test.csv`). To guarantee experimental integrity and zero test-set contamination:
- The $60,000$ training samples were partitioned into a **stratified training set ($54,000$ samples)** and a **stratified validation set ($6,000$ samples)** using fixed random state $42$.
- The validation and test sets were strictly held out from augmentation, feature scaler fitting, and model selection.
- All partitions were serialized into binary `.npz` format (`data/processed/clean_partitions.npz`) to eliminate redundant CSV parsing overhead.

### 2.2 Path A Label-Preserving Data Augmentation
To improve generalization against handwriting variability, the training set was augmented using affine transformations directly on the $28 \times 28$ image grid:
- **Rotations:** Uniformly sampled within $[-10^\circ, +10^\circ]$.
- **Translations:** Horizontal and vertical shifts within $[-2\text{ px}, +2\text{ px}]$.
- **Scaling:** Uniform zoom factor between $0.9\times$ and $1.1\times$.
- **Shearing:** Slant distortion factor within $[-0.2, +0.2]$.
- **Forbidden Operations:** Horizontal/vertical flips, inversions, and rotations $> 20^\circ$ were strictly excluded as they invert digit semantics (e.g., flipping a $6$ yields a $9$).

For each clean training sample, two distinct augmented variants were generated alongside the original image, yielding an augmented training set of:
$$N_{\text{train, aug}} = 54,000 \times 3 = 162,000 \text{ samples}$$

### 2.3 Mathematical Formulation of Feature Set A2
Let an input grayscale image be represented as an intensity matrix $\mathbf{I} \in [0, 1]^{28 \times 28}$. The Feature Set A2 projection maps $\mathbf{I} \mapsto \mathbf{x} \in \mathbb{R}^{56}$ via continuous discrete marginal sums:
$$\mathbf{x}_{\text{row}}[i] = \sum_{j=1}^{28} \mathbf{I}[i, j], \quad i \in \{1, 2, \dots, 28\}$$
$$\mathbf{x}_{\text{col}}[j] = \sum_{i=1}^{28} \mathbf{I}[i, j], \quad j \in \{1, 2, \dots, 28\}$$
$$\mathbf{x} = \left[ \mathbf{x}_{\text{row}}[1], \dots, \mathbf{x}_{\text{row}}[28], \mathbf{x}_{\text{col}}[1], \dots, \mathbf{x}_{\text{col}}[28] \right]^T$$

Continuous pixel values are summed without binarization, preserving stroke thickness and anti-aliasing energy. The feature dimensionality is reduced by $14\times$ ($784 \rightarrow 56$).

A `StandardScaler` was fitted strictly on the augmented training matrix:
$$\mu_k = \frac{1}{N} \sum_{n=1}^N x_{n, k}, \quad \sigma_k = \sqrt{\frac{1}{N} \sum_{n=1}^N (x_{n, k} - \mu_k)^2}$$
$$z_{n, k} = \frac{x_{n, k} - \mu_k}{\sigma_k + \epsilon}$$
The fitted scaler was saved to `data/features/a2/scaler_A2.joblib` and applied to validation and test matrices without re-estimation.

---

## 3. Supervised Classification (Part A)

### 3.1 Model Architectures & Hyperparameter Optimization
Ten classification models were implemented and systematically tuned:

1. **Logistic Regression (Multinomial)**: L2 regularization ($C=1.0$), L-BFGS solver, max iterations $1,000$.
2. **k-Nearest Neighbours (k-NN)**: Grid search over $k \in \{1, 3, 5, 7, 9\}$ with Euclidean and Manhattan metrics. Optimal: $k=3$, Euclidean distance.
3. **Support Vector Machine (SVM)**: RBF kernel with $C=10.0, \gamma=\text{'scale'}$.
4. **Decision Tree (CART)**: Maximum depth $15$, Gini impurity criterion, min samples split $10$.
5. **Random Forest**: $200$ estimators, maximum depth $25$, bootstrap aggregation.
6. **Naive Bayes**: Gaussian Naive Bayes with variance smoothing $\text{var\_smoothing}=10^{-9}$.
7. **Linear Discriminant Analysis (LDA)**: SVD solver, multi-class discriminant subspace projection.
8. **Gradient Boosting**: $150$ boosting stages, learning rate $0.1$, max depth $4$, sub-sample $0.8$.
9. **Multilayer Perceptron (MLP)**: Fully connected architecture ($56 \rightarrow 256 \rightarrow 128 \rightarrow 64 \rightarrow 10$), ReLU activations, Dropout ($p=0.2$), Adam optimizer ($\eta = 10^{-3}$, weight decay $10^{-4}$), batch size $128$, early stopping patience $5$.
10. **2-D Convolutional Neural Network (CNN)**: Architecture:
    - Conv2D($1 \rightarrow 32$, kernel $3 \times 3$, padding $1$) $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ MaxPool($2 \times 2$)
    - Conv2D($32 \rightarrow 64$, kernel $3 \times 3$, padding $1$) $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ MaxPool($2 \times 2$)
    - Conv2D($64 \rightarrow 128$, kernel $3 \times 3$, padding $1$) $\rightarrow$ BatchNorm $\rightarrow$ ReLU
    - AdaptiveAvgPool2d($4 \times 4$) $\rightarrow$ Flatten $\rightarrow$ Linear($2048 \rightarrow 128$) $\rightarrow$ Dropout($0.3$) $\rightarrow$ Linear($128 \rightarrow 10$)
    - Hardware: Accelerated via PyTorch CUDA on RTX 5060 Laptop GPU.

### 3.2 Quantitative Results & Classification Leaderboard

The performance on the isolated $10,000$-sample test set is summarized below:

| Model | Representation | Test Acc (%) | Macro F1 | Weighted F1 | Micro F1 | ROC-AUC | Train Time (s) | Latency (ms/1k) | Size |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Convolutional Neural Network (CNN)** | **2-D Path A Images** | **99.61%** | **0.9961** | **0.9961** | **0.9961** | **0.99999** | 144.56 s | 25.60 ms | 3.44 MB |
| **Gradient Boosting** | Feature Set A2 (56D) | **93.96%** | **0.9386** | **0.9395** | **0.9396** | **0.9975** | 17.23 s | 16.92 ms | 5.25 MB |
| **Multilayer Perceptron (MLP)** | Feature Set A2 (56D) | **93.59%** | **0.9347** | **0.9358** | **0.9359** | **0.9973** | 43.02 s | 4.29 ms | 0.07 MB |
| **Support Vector Machine (SVM)** | Feature Set A2 (56D) | **92.44%** | **0.9229** | **0.9241** | **0.9244** | **0.9953** | 40.52 s | 788.15 ms | 5.35 MB |
| **Random Forest** | Feature Set A2 (56D) | **91.86%** | **0.9174** | **0.9186** | **0.9186** | **0.9956** | 20.17 s | 6.25 ms | 502.99 MB |
| **k-Nearest Neighbours (k-NN)** | Feature Set A2 (56D) | **88.42%** | **0.8810** | **0.8825** | **0.8842** | **0.9724** | 0.00 s | 46.19 ms | 8.85 MB |
| **Decision Tree** | Feature Set A2 (56D) | **83.47%** | **0.8325** | **0.8348** | **0.8347** | **0.9278** | 9.17 s | 0.19 ms | 1.73 MB |
| **Logistic Regression** | Feature Set A2 (56D) | **78.69%** | **0.7774** | **0.7807** | **0.7869** | **0.9658** | 2.99 s | 0.12 ms | 3.05 KB |
| **Linear Discriminant Analysis** | Feature Set A2 (56D) | **72.65%** | **0.7187** | **0.7230** | **0.7265** | **0.9538** | 0.34 s | 0.11 ms | 7.56 KB |
| **Naive Bayes** | Feature Set A2 (56D) | **60.21%** | **0.5707** | **0.5754** | **0.6021** | **0.9303** | 0.07 s | 2.18 ms | 5.13 KB |

### 3.3 Receiver Operating Characteristic (ROC) Analysis
All non-linear models on Feature Set A2 achieved Macro One-vs-Rest ROC-AUC scores exceeding $0.995$. Even though Decision Trees and Linear Classifiers exhibited moderate accuracy ($72\%\text{--}83\%$), their ROC-AUC scores remained high ($0.927\text{--}0.965$), demonstrating that class probability ranking is robust, but fixed threshold argmax decision boundaries are constrained by the linear separability of marginal sums. The 2-D CNN achieved a near-perfect ROC-AUC of **0.999992**.

---

## 4. Deep Failure Analysis: Spatial Projection Ambiguity

### 4.1 Theoretical Mechanism
The fundamental limitation of Feature Set A2 is that discrete 1-D projections are non-injective:
$$\mathcal{P}_{\text{row}}: \mathbb{R}^{28 \times 28} \rightarrow \mathbb{R}^{28}, \quad \mathcal{P}_{\text{col}}: \mathbb{R}^{28 \times 28} \rightarrow \mathbb{R}^{28}$$
Given row sums $\mathbf{r}$ and column sums $\mathbf{c}$, there exists a high-dimensional affine subspace of matrices $\mathbf{M}$ satisfying:
$$\sum_j M_{ij} = r_i \quad \text{and} \quad \sum_i M_{ij} = c_j$$
This creates **Spatial Projection Ambiguity**: row and column sums record *how much ink* exists along each horizontal and vertical slice, but completely lose *where along the slice* that ink is located.

### 4.2 Empirical Confusion Patterns
Confusion matrix analysis on Feature Set A2 classifiers reveals persistent error modes that reflect this spatial ambiguity:
- **Digit $3$ vs. Digit $5$:** Both digits possess identical horizontal strokes (top bar, middle prong, bottom curve) and symmetric column distributions. In Feature Set A2, the row and column sums are nearly indistinguishable, causing $78$ misclassifications in the MLP and $54$ in Gradient Boosting.
- **Digit $4$ vs. Digit $9$:** Both digits have strong vertical stems and a concentrated horizontal crossbar. In test sample #7 (a handwritten $9$), the upper loop is compressed, producing marginal sums that Gradient Boosting and MLP misclassified as a $4$.
- **Digit $1$ vs. Digit $7$:** A slanted $1$ matches the vertical column distribution and row spread of an uncrossed $7$.

### 4.3 Resolution by the 2-D CNN
In stark contrast, the 2-D CNN evaluates localized $3 \times 3$ convolutional receptive fields:
$$(\mathbf{I} * \mathbf{K})_{i, j} = \sum_{u=-1}^1 \sum_{v=-1}^1 I[i+u, j+v] K[u, v]$$
This maintains the 2-D relative coordinate grid. Consequently:
- The CNN correctly distinguishes the top-left vertical bar of a $5$ from the top-right curve of a $3$.
- The CNN correctly resolves test sample #7 as digit $9$ with $99.98\%$ posterior confidence.
- The total test errors plummet from **$604$ (Gradient Boosting)** down to **$39$ (CNN)**, achieving a **$15.5\times$ error reduction**.

---

## 5. Unsupervised Clustering (Part B)

### 5.1 Experimental Setup & Evaluation Protocol
Clustering experiments were conducted on $10,000$ standardized Feature Set A2 test samples. To ensure strictly unsupervised learning:
- Models were fitted without access to digit labels.
- Evaluation was performed post-hoc using both internal criteria (no ground truth required) and external criteria (comparing clusters against true digit classes).

### 5.2 Quantitative Clustering Results ($k=10$ Baseline)

| Algorithm | Clusters | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index | ARI | NMI | V-Measure | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Spectral Clustering** | 10 | 0.0668 | **1.9049** | 683.73 | **0.3537** | **0.5133** | **0.5133** | 8.65 s |
| **Agglomerative (Ward)** | 10 | 0.0646 | 2.4124 | 729.25 | **0.2997** | **0.4517** | **0.4517** | 9.31 s |
| **Mini-Batch k-Means** | 10 | 0.1018 | 2.1421 | 868.74 | 0.2719 | 0.3757 | 0.3757 | **0.27 s** |
| **Gaussian Mixture (GMM)** | 10 | 0.0544 | 2.6312 | 588.90 | 0.2486 | 0.3716 | 0.3716 | 0.31 s |
| **k-Means** | 10 | **0.1143** | 1.9391 | **928.75** | 0.2320 | 0.3696 | 0.3696 | 0.37 s |
| **k-Means (PCA-15 on A2)** | 10 | 0.1121 | 1.9433 | 927.51 | 0.2339 | 0.3691 | 0.3691 | 0.37 s |
| **DBSCAN** | 1 (11.6% noise) | — | — | — | — | — | — | 0.15 s |

### 5.3 Methodological Analysis
1. **Spectral Clustering Superiority:** Spectral Clustering constructs an affinity graph and projects data into the normalized Laplacian eigenspace. By capturing non-convex, manifold-shaped cluster boundaries, Spectral Clustering achieved the highest external agreement (**NMI: 0.5133, ARI: 0.3537**), outperforming standard centroid-based k-Means (NMI: 0.3696) by over $38\%$.
2. **Agglomerative Hierarchical Clustering:** Ward's minimum variance criterion produced the second-highest clustering alignment (**NMI: 0.4517, ARI: 0.2997**), successfully isolating dense, distinct digits (such as digit $1$ and digit $0$) into pure hierarchical leaves.
3. **Internal vs. External Discrepancy:** While k-Means scored the highest internal cohesion (**Silhouette: 0.1143, CHI: 928.75**), its external NMI was substantially lower than Spectral and Agglomerative clustering. This demonstrates that Euclidean spherical partitions in 56-D space do not naturally correspond to semantic digit classes.
4. **DBSCAN Density Structure:** DBSCAN ($\epsilon=3.0, \text{min\_samples}=10$) identified a single massive connected component containing $88.4\%$ of samples, with $11.6\%$ flagged as noise. In 56-D continuous space, digit classes form a continuous manifold bridge rather than isolated density islands.
5. **PCA Dimensionality Reduction Ablation:** Projecting the 56 A2 features onto $15$ principal components prior to k-Means preserved $98.2\%$ of total variance, slightly improved ARI ($0.2320 \rightarrow 0.2339$), and maintained parity with raw A2 k-Means, proving that the top orthogonal eigenvectors retain almost all cluster-discriminative ink variance.

---

## 6. Computational Efficiency & Latency Trade-offs

A comprehensive comparison across model complexity, storage footprint, and real-time execution cost:
- **Inference Latency Leader:** Linear Discriminant Analysis ($0.11\text{ ms} / 1\text{k}$ samples) and Logistic Regression ($0.12\text{ ms} / 1\text{k}$ samples) provide sub-millisecond throughput, suitable for ultra-low-power embedded edge devices.
- **Throughput-Accuracy Sweet Spot:** The PyTorch **MLP** on Feature Set A2 achieves **93.59% accuracy** with an inference latency of just **4.29 ms per 1,000 digits** and a compact model footprint of **70 KB**.
- **Memory Overhead:** While Random Forest achieves $91.86\%$ accuracy, its 200 deep unpruned trees require **502.99 MB** of disk storage. In contrast, Gradient Boosting achieves higher accuracy ($93.96\%$) with only **5.25 MB** of storage.
- **CNN GPU Acceleration:** Training the 2-D CNN on $162,000$ augmented images required $144.56$ seconds on the NVIDIA GeForce RTX 5060 Laptop GPU. Inference latency on the GPU was $25.60\text{ ms} / 1\text{k}$ images with a model size of $3.44\text{ MB}$, proving highly viable for desktop and server deployment.

---

## 7. Conclusions & Cross-Representation Integration

1. **Efficacy of Feature Set A2:** Reducing $784$ grayscale pixels down to $56$ continuous row/column sums ($14\times$ compression) preserves sufficient semantic energy for non-linear models to achieve **$93.96\%$ classification accuracy** and an **NMI of $0.5133$ in spectral clustering**.
2. **Limit of 1-D Projections:** The fundamental ceiling of Feature Set A2 is bounded by Spatial Projection Ambiguity. Preserving 2-D local topology via a CNN provides a dramatic boost to **$99.61\%$ accuracy**, effectively eliminating ambiguity between topologically similar digits.
3. **Cross-Representation Collaboration:** The modular Option A repository architecture guarantees that team members working on A1, A3, B1, B2, and B3 can independently push features and results. Executing `python -m src.compare_all` automatically aggregates all team modules into unified master leaderboards.

