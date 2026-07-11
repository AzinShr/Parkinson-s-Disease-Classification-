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
2. 2. **Ensemble Methods** - Random Forest & Gradient Boosting

## Results 

| Metric | SVM (best) | Random Forest | Gradient Boosting |
|---|---|---|---|
| Accuracy | 0.74 | 0.75 | 0.76 |
| Macro F1 | 0.63 | 0.62 | **0.65** |
| Healthy (0) precision | 0.47 | 0.48 | **0.53** |
| Healthy (0) recall | 0.42 | 0.33 | 0.40 |
| Healthy (0) F1 | 0.44 | 0.40 | **0.45** |
| Diseased (1) precision | 0.82 | 0.80 | 0.82 |
| Diseased (1) recall | 0.84 | 0.88 | 0.88 |
| Diseased (1) F1 | 0.83 | 0.84 | **0.85** |

**Interpretation:** Gradient Boosting is the strongest model so far by a small 
margin across most metrics. However, ensembles did **not** meaningfully solve 
the healthy-class detection problem seen with SVM — Random Forest actually has 
the worst healthy-class recall of all three models (0.33). This suggests the 
bottleneck is not model choice but data: only 48 healthy recordings (~8 patients) 
are available, split across 5 folds during cross-validation, leaving too little 
minority-class signal for any of these methods to learn robustly.

**Takeaway across classical + ensemble methods:** performance plateaus around 
macro F1 ≈ 0.62–0.65 regardless of algorithm, pointing to a sample-size/class-
imbalance ceiling rather than an algorithmic limitation. This motivates trying 
the neural network approach next, though it's unlikely to fully overcome the 
same underlying data constraint — more useful may be techniques like SMOTE, 
class-weighted loss functions, or gathering additional healthy-patient recordings 
if this were a real clinical project.

