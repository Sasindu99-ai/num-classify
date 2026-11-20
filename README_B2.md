# EE5130 Mini Project: Feature Set B2 Documentation

## Comparative Study of Supervised Classification and Unsupervised Clustering on MNIST

> **Module:** `src/b2` — Path B (Contour Images), Feature Set B2 (Contour Row + Column Sums)  
> **Course:** EE5130 – Artificial Intelligence and Machine Learning  
> **Dataset:** MNIST Handwritten Digits  
> **Feature Dimensionality:** 56

---

## 1. Assigned Scope Overview

This package implements **Feature Set B2**, which represents each MNIST digit using a binary contour image and compresses that image into a 56-dimensional feature vector.

The B2 pipeline includes:

1. Contour extraction from each 28 × 28 grayscale image.
2. Extraction of 28 row-wise and 28 column-wise contour-pixel counts.
3. Feature scaling using a scaler fitted on training features only.
4. Supervised evaluation of eight classical classifiers and one MLP.
5. Unsupervised comparison of six clustering algorithms.

Unlike A2, which sums grayscale intensity values, B2 counts pixels in a binary contour image. It describes detected boundary distribution rather than total grayscale ink intensity.

## 2. B2 Feature Representation

### 2.1 Contour Extraction

A binary foreground mask is created using an intensity threshold of **100**. A **3 × 3 minimum filter** obtains an eroded foreground mask. Subtracting the eroded mask from the original binary mask leaves the boundary pixels.

Contour pixels are represented by 1 and background pixels by 0.

The same procedure is applied independently to augmented training images and clean validation and test images.

### 2.2 Row and Column Sums

Let \(C\) be the binary contour image, where \(C_{ij}=1\) for a contour pixel and \(C_{ij}=0\) otherwise.

The row-wise features are:

\[
r_i = \sum_{j=1}^{28} C_{ij}, \quad i=1,\ldots,28
\]

The column-wise features are:

\[
c_j = \sum_{i=1}^{28} C_{ij}, \quad j=1,\ldots,28
\]

The final B2 feature vector is:

\[
\mathbf{x}_{B2} =
[r_1,\ldots,r_{28},c_1,\ldots,c_{28}]^T
\in \mathbb{R}^{56}
\]

This reduces 784 contour pixels to 56 features (a 14-fold dimensionality reduction). It retains the number of contour pixels in each row and column, but not their exact positions within each slice.

## 3. Data Preparation and Leakage Prevention

The shared stratified partition uses random seed **42**.

| Partition | Samples | Use |
|---|---:|---|
| Training (original) | 54,000 | Source partition for training |
| Training (after augmentation) | 162,000 | Classifier training; 3,000-sample clustering subset |
| Validation | 6,000 | Model selection and MLP validation monitoring |
| Test | 10,000 | Final supervised evaluation |

The training partition is expanded from 54,000 to 162,000 samples by retaining the originals and adding two augmented variants per original. Validation and test partitions remain unaugmented.

A `StandardScaler` is fitted using only the training feature matrix. The fitted transformation is applied to validation and test features without refitting, preventing validation and test statistics from entering preprocessing.

## 4. Supervised Classification

### 4.1 Models and Settings

Eight classical classifiers and one MLP were evaluated on the 56-dimensional B2 vectors.

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

The MLP uses the best checkpoint selected using validation performance. A CNN was not included because B2 is a 56-dimensional feature vector rather than a two-dimensional image tensor.

### 4.2 Classification Leaderboard

Results on the isolated 10,000-sample test set:

| Model | Val Acc (%) | Test Acc (%) | Macro F1 | Macro Precision | Macro Recall | ROC-AUC | Train Time (s) | Inference (ms/1k) | Size (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gradient Boosting | 92.90 | **92.89** | **0.9279** | 0.9282 | 0.9277 | **0.9967** | 55.38 | 37.74 | 5.19 |
| MLP | 90.95 | 91.37 | 0.9122 | 0.9128 | 0.9119 | 0.9952 | 45.80 | 6.48 | **0.07** |
| Random Forest | 89.85 | 90.52 | 0.9037 | 0.9060 | 0.9030 | 0.9939 | 12.27 | 6.80 | 656.34 |
| SVM | 89.33 | 89.48 | 0.8927 | 0.8940 | 0.8924 | 0.9920 | 85.36 | 1,879.96 | 8.90 |
| k-NN | 84.33 | 84.82 | 0.8436 | 0.8494 | 0.8446 | 0.9669 | 0.01 | 63.08 | 8.85 |
| Decision Tree | 80.
