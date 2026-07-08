# Parkinson's Disease Classification from Voice Recordings

## Overview
Binary classification (healthy vs. diseased) using 22 biomedical voice measurement 
features from the UCI Parkinson's dataset. This project compares three modeling 
approaches: classical ML (SVM)

## Dataset
- Source: [UCI Parkinson's Disease Data Set](link)
- 195 voice recordings from 31 subjects (~6 recordings per patient)
- 22 biomedical voice features (jitter, shimmer, HNR, etc.)
- Target: status (0 = healthy, 1 = Parkinson's diagnosed)
- Class distribution: ~75% diseased, ~25% healthy (imbalanced)

## Key Methodology Decisions
- **Group-aware splitting**: Since each patient contributes multiple recordings, 
  standard random splits/CV would leak patient identity between train and test. 
  Used `GroupShuffleSplit`/`GroupKFold` throughout to keep each patient's 
  recordings together.
- **Multicollinearity handling**: Several jitter/shimmer features were highly 
  correlated (r > 0.9). Dropped redundant features, reducing from 22 → 11 features, 
  which improved SVM's mean AUC from 0.68 to 0.74 and reduced fold-to-fold variance.
- **Evaluation metric**: Used ROC-AUC and macro-F1 (not accuracy) due to class 
  imbalance. Reported per-class precision/recall since the healthy class is 
  harder to detect and clinically important not to over-flag.

## Approaches
1. **Classical ML** — SVM (rbf)

## Results (Classical Methods)

### SVM (RBF kernel)

Two configurations were tested on the reduced 11-feature set (after dropping 
highly correlated jitter/shimmer duplicates). Both evaluated using 5-fold 
GroupKFold cross-validation (patient-grouped, pooled out-of-fold predictions).

| Metric | Config 1 (C=10, gamma=0.01) | Config 2 (grid-searched, F1-macro optimized) |
|---|---|---|
| CV ROC-AUC | 0.74 ± 0.09 | — |
| Accuracy | 0.69 | 0.74 |
| Macro F1 | 0.62 | 0.63 |
| Healthy (0) precision | 0.40 | 0.47 |
| Healthy (0) recall | 0.54 | 0.42 |
| Healthy (0) F1 | 0.46 | 0.44 |
| Diseased (1) precision | 0.83 | 0.82 |
| Diseased (1) recall | 0.73 | 0.84 |
| Diseased (1) F1 | 0.78 | 0.83 |

