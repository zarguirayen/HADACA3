# Multi-Omics Cell-Type Deconvolution  
### HADACA3 Data Science Challenge

## Project Overview

This project presents a machine learning solution developed for the HADACA3 multi-omics challenge.

The objective is to estimate cell-type proportions from bulk RNA-seq and DNA methylation data using reference single-cell profiles.

This is a real-world biomedical data science problem involving:
- High-dimensional data
- Multi-modal integration
- Compositional constraints
- Model calibration under competition settings

Final leaderboard score: **0.86**

---

## Problem Statement

Bulk tissue samples contain mixtures of different cell types.  
The goal is to computationally infer the proportion of each cell type using:

- Gene expression (RNA-seq)
- DNA methylation data
- Reference cell-type profiles

This is a constrained regression and compositional learning problem.

---

## Methodology

### 1. Feature Engineering
- Variance-based feature selection
- Independent selection for RNA and methylation
- Dimensionality reduction while preserving biological signal

### 2. Constrained Deconvolution
- Non-Negative Least Squares (NNLS)
- Ensures biologically valid proportions (no negative values)
- Per-sample normalization to satisfy compositional constraints

### 3. Multi-Omics Fusion
- Independent RNA and methylation deconvolution
- Error-based weighting of modalities
- Adaptive fusion per sample

### 4. Compositional Calibration
- Centered Log-Ratio (CLR) transformation
- Ridge regression calibration
- Bias correction in compositional space
- Improved Aitchison-based performance

### 5. Model Optimization
- Hyperparameter tuning
- Stability analysis across datasets
- Competition-driven optimization

---

## Technical Stack

- Python
- NumPy
- Pandas
- SciPy (NNLS)
- Scikit-learn (Ridge Regression)
- HDF5 data handling (h5py)

---

## Performance

| Stage | Approach | Score |
|-------|----------|-------|
| Baseline | Linear Regression | 0.34 |
| NNLS | Constrained Regression | 0.75 |
| Multi-omics Fusion | RNA + Methylation | 0.80 |
| Calibration | Ridge + CLR | **0.86** |

Evaluation metrics included:
- Aitchison distance (compositional metric)
- RMSE
- MAE
- Median dataset aggregation

---

## Skills Demonstrated

This project showcases the following Data Science skills:

- Multi-omics data integration
- High-dimensional regression
- Constrained optimization
- Compositional data analysis
- Model calibration & stacking
- Feature selection strategies
- Performance optimization in competitive settings
- Reproducible ML pipeline design

---
## Results

The model achieved a leaderboard score of **0.86**, placing in the competitive range of submissions.

Key performance strengths:
- Robust multi-omics integration
- Stable generalization across datasets
- Strong compositional accuracy (Aitchison metric)

---

## Future Improvements

- Ensemble calibration models
- Advanced marker selection per cell type
- Batch effect correction
- Bayesian compositional modeling
