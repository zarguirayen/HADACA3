# HADACA3 – Multi-Omics Cell-Type Deconvolution

## Overview

This repository contains my solution to the HADACA3 data challenge.

The objective of the challenge is to estimate cell-type proportions from bulk RNA-seq and DNA methylation data using reference profiles.

This project implements a multi-omics deconvolution pipeline combining constrained regression, feature selection, and supervised calibration.

Final leaderboard score: **0.86**

---

## Methodology

The pipeline includes the following steps:

### 1. Feature Selection
- Variance-based selection of informative genes and CpGs
- RNA and methylation features selected independently

### 2. Deconvolution
- Non-negative least squares (NNLS)
- Enforces biologically valid constraints (non-negative proportions)
- Column-wise normalization to ensure compositional validity

### 3. Multi-Omics Fusion
- Independent RNA and methylation deconvolution
- Error-weighted fusion based on reconstruction error

### 4. Supervised Calibration
- CLR (Centered Log-Ratio) transformation
- Ridge regression calibration
- Improves compositional accuracy

### 5. Final Normalization
- Ensures proportions sum to 1 per sample

---

## Repository Structure
