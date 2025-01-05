# SCImputation: Mitigating Feature Confounding via a Structural Causal Model for Data Imputation

**SCImputation** is a Structural-Causal-based imputation strategy designed to address missing data in both continuous (e.g., microarray) and discrete (e.g., questionnaire) datasets. Unlike conventional local-learning-based methods (e.g., KNNimpute, LLSimpute), SCImputation treats the **target feature** (the one with missing values) as a *confounder* that can bias neighbor selection and imputation. By employing a **back-door adjustment** to correct for this confounding effect, SCImputation achieves more robust and accurate imputation outcomes.

---

## Overview

Most local-learning-based methods assume a uniform neighbor set for every missing value in an instance, overlooking the fact that each *specific feature* may exhibit different correlations with other features. **SCImputation** tackles this via:

1. **Feature-Aware Neighbor Selection**  
   - Each missing entry selects neighbors using a weighted similarity measure, where the weights come from the correlation between the target feature and all other features.  
   - This selection can differ across different missing values in the same instance.

2. **Causal Back-Door Adjustment**  
   - Recognizes that the target feature (\(F\)) might *confound* the estimation of missing values (\(V\)), since \(F\) also dictates the neighbor set (\(N\)).  
   - SCImputation “intervenes” on \(N\) by reverting to the global distribution \(P(F=f)\) rather than using \(P(F=f \mid N)\), thereby eliminating the local bias introduced by correlated features.

3. **Integrated with KNN and LLS**  
   - KNN-based variants: *SC-PKNNimpute*, *SC-MKNNimpute*, *SC-KKNNimpute*, *SC-SKNNimpute*.  
   - LLS-based variants: *SC-PLLSimpute*, *SC-MLLSimpute*, *SC-KLLSimpute*, *SC-SLLSimpute*.

4. **Multiple Correlation Schemes**  
   - Pearson, Mutual Information, Kendall’s, Spearman (choose whichever suits your data best).

5. **Dynamic Neighbor Selection**  
   - Uses a \(\epsilon\)-based threshold to determine how many neighbors to include (CDF-based approach).

---

## Repository Structure

SCImputation/ ├── data/ │ ├── completegds38.mat │ ├── completegds1761.mat │ ├── completegds3835.mat │ ├── missingdata_gds38.mat │ ├── missingdata_gds1761.mat │ ├── missingdata_gds3835.mat │ └── ... (other data files) ├── src/ │ ├── SCImpute_KNN/ # KNN-based SCImputation variants │ ├── SCImpute_LLS/ # LLS-based SCImputation variants │ ├── weighting_schemes/ # code for correlations and weighting │ ├── scimpute_utils/ # utility functions (similarity, etc.) │ └── ... ├── run_demo.m # example script └── README.md


- **`data/`**:  
  Real or sample data, including microarray sets (GDS38, GDS1761, GDS3835). If you have discrete data (NACC, Turkiye Student dataset), place them here too or link externally.  

- **`src/`**:  
  MATLAB implementations for SCImputation methods:
  - KNN-based: `_KNNimpute` scripts.  
  - LLS-based: `_LLSimpute` scripts.  

- **`run_demo.m`**:  
  Demonstrates loading data, running SCImputation, and evaluating results.

---

## Installation

- **MATLAB** (tested on R2021b or newer).  
- [Recommended] *Statistics and Machine Learning Toolbox* for correlation calculations and other functionalities.

---

## Usage

1. **Clone or download** this repository.

2. **Open MATLAB** and navigate to the project directory.

3. **Load a dataset** (example with GDS38):
   ```matlab
   load('data/completegds38.mat');     % e.g., complete_data
   load('data/missingdata_gds38.mat'); % e.g., missing_data
Here, complete_data is the ground truth; missing_data has artificially induced missing entries.

4. **Run SCImputation**. For example, Pearson-based LLS:
imputedData = SC_PLlSimpute_continuous(missing_data);
or Mutual-Information-based KNN:
imputedData = SC_MKNNimpute(missing_data);

5. **Evaluate Performance** (RMSE/MAE for continuous, accuracy for discrete):
rmse = sqrt(mean((complete_data(:) - imputedData(:)).^2));
mae  = mean(abs(complete_data(:) - imputedData(:)));

% For discrete data, e.g., NACC:
% accuracy = sum( imputedData(:) == complete_data(:) ) / numel(complete_data);

## Demo Script

An example run_demo.m might look like:
% Demo for SCImputation on GDS38

clear; clc; close all;

% 1. Load data
load('data/completegds38.mat');       % complete_data
load('data/missingdata_gds38.mat');   % missing_data

% 2. Impute with SCImputation (Pearson-based LLS version)
imputedData = SC_PLlSimpute_continuous(missing_data);

% 3. Evaluate
rmse = sqrt(mean((complete_data(:) - imputedData(:)).^2));
mae  = mean(abs(complete_data(:) - imputedData(:)));
fprintf('RMSE = %.4f\n', rmse);
fprintf('MAE  = %.4f\n', mae);

## Datasets

1. **Microarray Data**:
  GDS38, GDS1761, GDS3835: Each has a complete_*.mat file plus a missingdata_*.mat version with artificially introduced missingness.
2. **Discrete/Questionnaire Data**:
  NACC: Large Alzheimer’s dataset (not provided here due to licensing).
  Turkiye Student: A set of student evaluations (optional).
To replicate the paper’s results, you may need to request or download these discrete datasets from their respective sources and place them in data/.

## Citing SCImputation

