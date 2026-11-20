# EE5130 Mini Project Technical Report
## Comparative Study of Supervised Classification and Unsupervised Clustering on MNIST

**Assigned Scope:** Path B – Contour Images, Feature Set B2 – Contour Row and Column Sums
**Course:** EE5130 – Artificial Intelligence and Machine Learning
**Dataset:** MNIST Handwritten Digits
**Feature Dimensionality:** 56

---

# Abstract

This report presents the B2 component of the EE5130 MNIST mini project. B2 belongs to Path B, which represents each handwritten digit using a binary contour image. The contour image is compressed into 56 numerical features by calculating 28 row-wise contour-pixel counts and 28 column-wise contour-pixel counts. The resulting representation is evaluated using supervised classification and unsupervised clustering.

The preprocessing pipeline uses the shared stratified partition of 54,000 training images, 6,000 validation images and 10,000 test images. The training partition is expanded to 162,000 samples through augmentation, while validation and test data remain unaugmented. A StandardScaler is fitted using training features only.

Nine classifiers were evaluated. Gradient Boosting achieved the strongest B2 classification result, with 92.89% test accuracy, 0.9279 macro F1 and 0.9967 macro one-vs-rest ROC-AUC. The MLP achieved 91.37% test accuracy and a compact model size of approximately 0.07 MB. Across the classifiers, the digit pair 3 and 5 was the most consistently confused.

Six clustering algorithms were evaluated on a reproducible 3,000-sample subset of the augmented training features. k-Means achieved the highest ARI (0.2521) and NMI (0.3211) among the tested methods. DBSCAN returned one non-noise cluster and classified 53.03% of the sample as noise. Overall, the results show that B2 provides useful information for classification, while its compressed contour projections limit the recovery of the ten semantic digit groups through clustering.

---

# 1. Introduction and Assigned Scope

The MNIST dataset contains 28 × 28 grayscale images of handwritten digits from 0 to 9. The project compares alternative feature representations and evaluates both supervised classification and unsupervised clustering.

This report focuses exclusively on the B2 representation:

- **Path B:** Contour-based representation of the original handwritten digit images.
- **Feature Set B2:** 28 row-wise contour sums concatenated with 28 column-wise contour sums.
- **Feature dimensionality:** 56 values per image.
- **Supervised learning:** Eight classical classifiers and one multilayer perceptron.
- **Unsupervised learning:** Six clustering algorithms.

Unlike A2, which sums grayscale intensity values, B2 counts pixels in a binary contour image. B2 therefore describes the distribution of detected boundaries rather than the total grayscale ink intensity.

---

# 2. Data Preparation and Feature Extraction

## 2.1 Dataset Partitioning and Leakage Prevention

The shared dataset partition contains 54,000 training samples, 6,000 validation samples and 10,000 test samples. The split is stratified and uses random seed 42. Validation and test partitions are kept separate from training augmentation and scaler fitting.

The training set is expanded from 54,000 original images to 162,000 images by retaining the originals and adding two augmented variants per original. The validation and test partitions remain clean and unaugmented. This prevents augmented versions of validation or test images from entering training.

| Partition | Number of Samples | Use |
|---|---:|---|
| Training (original) | 54,000 | Source partition for model training |
| Training (after augmentation) | 162,000 | Used to train classifiers; clustering uses a 3,000-sample subset |
| Validation | 6,000 | Model selection and MLP validation monitoring |
| Test | 10,000 | Final supervised evaluation only |

## 2.2 Contour Extraction

For each 28 × 28 input image, B2 first creates a binary foreground mask using an intensity threshold of 100. A 3 × 3 minimum filter is applied to obtain an eroded version of the foreground mask. The eroded mask is subtracted from the original binary mask, leaving the boundary pixels as the contour representation.

The resulting contour image is binary: contour pixels are represented by 1 and background pixels by 0. This operation reduces the original stroke regions to their detected boundaries. The implementation uses a 3 × 3 neighbourhood for erosion.

The contour extraction stage is applied to the augmented training images and to the clean validation and test images independently. The same contour-extraction procedure is used for all partitions.

## 2.3 Mathematical Formulation of B2 Features

Let \(C\) be the binary contour image of size \(28 \times 28\), where \(C_{ij}\) is 1 for a contour pixel and 0 otherwise.

The row feature at row \(i\) is the number of contour pixels in that row:

\[
r_i = \sum_{j=1}^{28} C_{ij}, \quad i=1,\ldots,28
\]

The column feature at column \(j\) is the number of contour pixels in that column:

\[
c_j = \sum_{i=1}^{28} C_{ij}, \quad j=1,\ldots,28
\]

The final B2 feature vector is the concatenation:

\[
\mathbf{x}_{B2} =
[r_1,\ldots,r_{28},c_1,\ldots,c_{28}]^T
\in \mathbb{R}^{56}
\]

Thus, the original 784 contour pixels are represented by 56 values, corresponding to a 14-fold reduction in dimensionality. The features preserve the number of boundary pixels along each horizontal and vertical slice, but do not preserve their exact positions within each slice.

## 2.4 Feature Scaling

A StandardScaler is fitted using only the training feature matrix. The fitted transformation is then applied to validation and test features without refitting. This avoids using validation or test statistics during preprocessing. The saved scaler is used to ensure that downstream experiments use a consistent feature transformation.

---

# 3. Supervised Classification

## 3.1 Models and Experimental Settings

Eight classical classifiers and one MLP were evaluated on the 56-dimensional B2 feature vectors. The classical model settings used in the B2 implementation are summarized below.

| Classifier | Main Settings / Training Protocol |
|---|---|
| Logistic Regression | C = 1; max_iter = 1,000; L-BFGS |
| Gaussian Naive Bayes | var_smoothing = 1e-8 |
| Linear Discriminant Analysis | SVD solver |
| Decision Tree | Gini; max_depth = 16; min_samples_split = 10; min_samples_leaf = 4 |
| Random Forest | 150 estimators; max_depth = 22; min_samples_split = 5 |
| HistGradientBoosting | learning_rate = 0.1; max_iter = 150; max_leaf_nodes = 31 |
| k-Nearest Neighbours | k = 5; distance weights; Euclidean; 40,000-sample training subset |
| Support Vector Machine | RBF kernel; C = 5; gamma = scale; 30,000-sample training subset |
| MLP | 56 → 128 → 64 → 10; BatchNorm/ReLU/dropout; AdamW; lr = 0.001; weight decay = 1e-4; 15 epochs; batch size = 256 |

The MLP uses the best checkpoint selected using validation performance. Its training history is recorded for 15 epochs. A CNN was not included in B2 because the B2 input is a 56-dimensional feature vector rather than a two-dimensional image tensor.

## 3.2 Classification Results

Table 3 summarizes the measured validation and test accuracy, macro F1, macro precision and macro recall. ROC-AUC is the macro one-vs-rest score.

| Model | Val Acc (%) | Test Acc (%) | Macro F1 | Macro Precision | Macro Recall | ROC-AUC | Train Time (s) | Inference (ms/1k) | Size (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gradient Boosting | 92.90 | 92.89 | 0.9279 | 0.9282 | 0.9277 | 0.9967 | 55.38 | 37.74 | 5.19 |
| MLP | 90.95 | 91.37 | 0.9122 | 0.9128 | 0.9119 | 0.9952 | 45.80 | 6.48 | 0.07 |
| Random Forest | 89.85 | 90.52 | 0.9037 | 0.9060 | 0.9030 | 0.9939 | 12.27 | 6.80 | 656.34 |
| SVM | 89.33 | 89.48 | 0.8927 | 0.8940 | 0.8924 | 0.9920 | 85.36 | 1,879.96 | 8.90 |
| k-NN | 84.33 | 84.82 | 0.8436 | 0.8494 | 0.8446 | 0.9669 | 0.01 | 63.08 | 8.85 |
| Decision Tree | 80.47 | 80.27 | 0.7989 | 0.8003 | 0.7987 | 0.9202 | 2.96 | 0.19 | 2.00 |
| Logistic Regression | 76.12 | 77.35 | 0.7610 | 0.7683 | 0.7670 | 0.9641 | 2.03 | 0.13 | 0.003 |
| LDA | 72.88 | 74.10 | 0.7258 | 0.7354 | 0.7335 | 0.9582 | 0.31 | 0.13 | 0.008 |
| Naive Bayes | 60.63 | 60.73 | 0.5702 | 0.7033 | 0.6004 | 0.9448 | 0.06 | 1.21 | 0.005 |

Gradient Boosting ranked first by test accuracy and macro F1, achieving 92.89% test accuracy and 0.9279 macro F1. The MLP ranked second by test accuracy at 91.37%, with a much smaller model footprint (approximately 0.07 MB). Random Forest achieved 90.52% test accuracy but required approximately 656.34 MB of model storage.

The SVM achieved 89.48% test accuracy but had the highest inference latency among the tested models, at approximately 1,879.96 ms per 1,000 test samples. This illustrates that high predictive performance must be considered alongside deployment cost.

## 3.3 MLP Learning Behaviour

The MLP was trained for 15 epochs. Training accuracy increased from 72.47% to 86.09%, while validation accuracy increased from 85.92% to 90.95%. Validation accuracy reached its highest recorded value at epoch 15. Training and validation losses decreased over the training period, with final losses of approximately 0.41 and 0.27 respectively.

Validation accuracy remained higher than training accuracy throughout the recorded epochs. This is consistent with regularization effects such as dropout being active during training and disabled during validation. The curves do not show an obvious severe-overfitting pattern within the 15-epoch run.

**Figure 1:** MLP Training and Validation Accuracy and Loss
`results/b2/figures/mlp_learning_curves_b2.png`

## 3.4 ROC Analysis

The five-model ROC comparison includes Gradient Boosting (AUC 0.9967), MLP (0.9952), Random Forest (0.9939), SVM (0.9920) and Logistic Regression (0.9641). All five curves rise steeply toward the top-left region, indicating strong one-vs-rest class ranking performance. The diagonal reference line represents random guessing.

**Figure 2:** ROC Curve Comparison for B2 Classifiers
`results/b2/figures/roc_curves_comparison_b2.png`

The plotted comparison shows the five selected models; it is not a nine-model ROC plot.

## 3.5 Classification Error Analysis

The most frequently confused digit pairs were calculated from the 10,000 test predictions. For each unordered pair \((a,b)\), the bidirectional count is the sum of \(a\) being predicted as \(b\) and \(b\) being predicted as \(a\).

| Classifier | Most Confused Pair | Combined Errors | Directional Detail |
|---|---|---:|---|
| Logistic Regression | 3 ↔ 5 | 291 | 3→5: 51; 5→3: 240 |
| Naive Bayes | 4 ↔ 9 | 816 | 4→9: 815; 9→4: 1 |
| LDA | 3 ↔ 5 | 320 | 3→5: 38; 5→3: 282 |
| Decision Tree | 3 ↔ 5 | 261 | 3→5: 92; 5→3: 169 |
| Random Forest | 3 ↔ 5 | 141 | 3→5: 16; 5→3: 125 |
| Gradient Boosting | 3 ↔ 5 | 85 | 3→5: 23; 5→3: 62 |
| k-NN | 3 ↔ 5 | 240 | 3→5: 74; 5→3: 166 |
| SVM | 3 ↔ 5 | 172 | 3→5: 56; 5→3: 116 |
| MLP | 3 ↔ 5 | 137 | 3→5: 36; 5→3: 101 |

The 3↔5 pair appeared among the top five confusions for all nine classifiers, making it the most consistent error pattern. Other recurring pairs included 4↔9 and 2↔3. Naive Bayes showed a particularly strong directional error, classifying 815 examples of digit 4 as digit 9.

A plausible explanation is that different digits can produce similar counts of contour pixels along rows and columns. Since B2 does not retain the exact within-row or within-column positions of contour pixels, some distinct shapes can map to similar 56-dimensional vectors. This is an interpretation of the representation and observed errors, not a direct causal test.

---

# 4. Unsupervised Clustering

## 4.1 Experimental Setup

The clustering experiment used a reproducible random sample of 3,000 feature vectors selected without replacement from the 162,000 augmented B2 training features, using seed 42. The clustering algorithms were fitted without digit labels. True labels were used only after clustering to calculate external evaluation metrics.

This differs from the supervised evaluation, which reports final classification performance on the separate 10,000-sample test set. Therefore, clustering metrics in this report must not be interpreted as test-set classification scores.

## 4.2 Algorithms and Results

| Algorithm | Clusters | Silhouette | Davies-Bouldin | Calinski-Harabasz | ARI | NMI | Homogeneity | Completeness | V-measure | Fit Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| k-Means | 10 | 0.0746 | 2.3250 | 159.69 | 0.2521 | 0.3211 | 0.3025 | 0.3420 | 0.3211 | 0.376 |
| Agglomerative (Ward) | 10 | 0.0361 | 2.6834 | 130.13 | 0.2285 | 0.3131 | 0.2953 | 0.3332 | 0.3131 | 0.310 |
| Mini-Batch k-Means | 10 | 0.0538 | 2.6229 | 141.64 | 0.1813 | 0.2737 | 0.2632 | 0.2852 | 0.2737 | 0.034 |
| Spectral Clustering | 10 | 0.0338 | 2.1375 | 135.83 | 0.1710 | 0.3109 | 0.2718 | 0.3632 | 0.3109 | 1.392 |
| Gaussian Mixture Model | 10 | 0.0316 | 2.9437 | 112.99 | 0.1615 | 0.2473 | 0.2325 | 0.2642 | 0.2473 | 0.243 |
| DBSCAN | 1* | N/A | N/A | N/A | 0.0463 | 0.0848 | 0.0552 | 0.1834 | 0.0848 | 0.017 |

\*DBSCAN produced one non-noise cluster; 53.03% of samples were labelled as noise. Internal scores are not meaningful for this one-cluster result and were left undefined.

k-Means achieved the highest ARI (0.2521), NMI (0.3211), homogeneity (0.3025) and V-measure (0.3211) among the six baseline algorithms. Spectral Clustering achieved the highest completeness (0.3632), while also obtaining a lower Davies–Bouldin index (2.1375) than the other methods. Mini-Batch k-Means was the fastest method by fit time (approximately 0.034 seconds).

## 4.3 DBSCAN Parameter Investigation

A DBSCAN parameter sweep examined eps values of 5, 6, 7, 8, 9 and 10, with min_samples values of 5, 10 and 15, using the same 3,000-sample training subset. Most settings produced one dominant cluster with either no meaningful secondary cluster or only very small secondary groups. Settings that produced two or three clusters did so with highly imbalanced cluster sizes. The apparent silhouette values for some settings were influenced by these tiny clusters.

The k-distance plot showed a gradual increase rather than a clear elbow. The investigation therefore did not provide convincing evidence for a better DBSCAN setting that recovers meaningful digit groups. The baseline eps = 5.0 and min_samples = 15 configuration was retained for the main comparison.

**Figure 3:** DBSCAN k-Distance Plot
`results/b2/figures/dbscan_k_distance.png`

## 4.4 Clustering Interpretation

The external clustering scores indicate limited alignment between the B2 feature-space groupings and the ten true digit classes. Even the strongest baseline result, k-Means, achieved an ARI of 0.2521 and NMI of 0.3211. These values suggest that the 56 contour projections contain some class-related structure, but do not naturally separate all ten digit categories into distinct clusters.

The relatively low silhouette scores also indicate that the discovered groups are not strongly separated in the standardized B2 feature space. The DBSCAN result suggests that, under the tested distance and density settings, the data do not form multiple well-separated density islands.

---

# 5. Computational Efficiency

The classification experiments show a range of accuracy and computational trade-offs. Gradient Boosting delivered the highest test accuracy (92.89%) but required 55.38 seconds of training. The MLP achieved 91.37% test accuracy with a compact 0.07 MB model size and 6.48 ms per 1,000 samples inference latency. Random Forest achieved 90.52% test accuracy but had a much larger model size of approximately 656.34 MB. SVM was comparatively expensive at 85.36 seconds of training and approximately 1,879.96 ms per 1,000 samples inference latency.

Among the clustering algorithms, Mini-Batch k-Means had the shortest recorded fit time (0.034 seconds), while Spectral Clustering took the longest (1.392 seconds) among the six baseline methods. These times were measured on the 3,000-sample clustering subset and should not be directly compared with classifier training times, which used different data volumes and procedures.

---

# 6. Limitations

- B2 compresses a 28 × 28 contour image into 56 row and column counts. This loses exact two-dimensional spatial relationships and can make different digit shapes appear similar.
- Clustering was performed on a 3,000-sample subset of augmented training features rather than the full training set, due to computational considerations. Clustering results are therefore specific to that sample and configuration.
- DBSCAN did not identify a convincing multi-cluster structure in the tested B2 feature space. Its results were sensitive to eps and min_samples.
- The classical k-NN and SVM models were trained on 40,000- and 30,000-sample subsets respectively, while other classical models used the full augmented training set. Their training times and results should be interpreted with this difference in mind.
- A CNN baseline was not evaluated for B2 because B2 is a vector representation. A fair image-based CNN comparison belongs to the raw-image path rather than this feature-vector experiment.
- Cross-representation conclusions require the corresponding A1, A2, A3, B1 and B3 results under a consistent evaluation protocol.

---

# 7. Conclusions and Future Work

The B2 contour row-and-column representation reduces each 28 × 28 contour image to 56 numerical features while retaining useful boundary-distribution information. In supervised classification, Gradient Boosting achieved the strongest result with 92.89% test accuracy and 0.9279 macro F1. The MLP provided a competitive alternative with 91.37% test accuracy and a compact model footprint.

The most consistent classification error was confusion between digits 3 and 5. This pattern is compatible with the information loss caused by marginal contour projections: B2 records how many contour pixels occur in each row and column, but not their exact arrangement within each slice.

Unsupervised clustering was less successful at recovering the ten semantic digit classes. k-Means achieved the strongest external agreement among the tested baseline methods (ARI 0.2521; NMI 0.3211), while DBSCAN produced one dominant cluster and substantial noise.

Future work should compare all feature spaces using the same partitions and evaluation protocols, investigate richer contour descriptors that preserve local spatial relationships, and explore alternative clustering representations or dimensionality-reduction methods. Any such comparison should retain the project's anti-leakage rules and clearly distinguish training-subset clustering results from held-out test classification results.

---

# Appendix A. B2 Figures

The following figures were generated by the B2 implementation and should be inserted into the final submission at the indicated sections:

1. `results/b2/figures/roc_curves_comparison_b2.png` – Section 3.4, ROC comparison.
2. `results/b2/figures/mlp_learning_curves_b2.png` – Section 3.3, MLP accuracy and loss curves.
3. `results/b2/figures/all_cluster_size_distributions.png` – Section 4.4, cluster size distributions.
4. `results/b2/figures/dbscan_k_distance.png` – Section 4.3, DBSCAN parameter investigation.
5. Clustering ARI comparison and contingency heatmaps from `results/b2/figures/`.

Additional PCA, t-SNE, UMAP and contingency heatmap figures can be included in an extended results appendix if required by the project report guidelines.

---

# Appendix B. Reproducibility Notes

- Shared split: 54,000 training / 6,000 validation / 10,000 test; random seed 42.
- Augmented training size: 162,000 samples.
- B2 feature size: 56.
- Clustering sample: 3,000 training feature vectors sampled without replacement using seed 42.
- Validation and test partitions are not augmented; scaler fitting uses training features only.
- Reported values are taken from the B2 experiment outputs. Unavailable values have not been fabricated.