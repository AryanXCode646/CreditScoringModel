# Credit Scoring Model — Evaluation Report

> **Educational Disclaimer**: This model is built for educational ML benchmarking on the 1994 UCI Statlog German Credit dataset. It is **not** validated or suitable for real-world lending or financial underwriting decisions.

- **Selected Model**: `random_forest` (chosen via 5-fold CV `roc_auc`)
- **CV-Tuned Decision Threshold**: `0.46`
- **Train / Test Split**: 800 / 200 (stratified, seed=42)

## 1. Cross-Validation Comparison (Training Set Only)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Brier Score |
|---|---|---|---|---|---|---|
| logistic_regression | 0.6963 | 0.4953 | 0.6542 | 0.5637 | 0.7632 | 0.1984 |
| decision_tree | 0.6512 | 0.4463 | 0.6750 | 0.5373 | 0.6991 | 0.2342 |
| random_forest | 0.7325 | 0.5436 | 0.6750 | 0.6022 | 0.7901 | 0.1804 |

## 2. Held-Out Test Set Comparison (Threshold = 0.50)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Brier Score |
|---|---|---|---|---|---|---|
| logistic_regression | 0.7600 | 0.5698 | 0.8167 | 0.6712 | 0.8150 | 0.1768 |
| decision_tree | 0.6150 | 0.4124 | 0.6667 | 0.5096 | 0.6248 | 0.2820 |
| random_forest | 0.7350 | 0.5455 | 0.7000 | 0.6131 | 0.7910 | 0.1829 |

## 3. Selected Model (random_forest) at Tuned Threshold (0.46)

- **Accuracy**: 0.6900
- **Precision (High Risk)**: 0.4894
- **Recall (High Risk)**: 0.7667
- **F1-Score (High Risk)**: 0.5974
- **ROC-AUC**: 0.7910
- **Brier Score**: 0.1829

### Classification Report

```text
precision    recall  f1-score   support

Good Credit (0)       0.87      0.66      0.75       140
   Bad Risk (1)       0.49      0.77      0.60        60

       accuracy                           0.69       200
      macro avg       0.68      0.71      0.67       200
   weighted avg       0.75      0.69      0.70       200
```

## 4. Leakage, Fairness & Limitations

- **Data Leakage Prevention**: Preprocessing (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`) and threshold selection are performed strictly inside training folds; the 20% test set is never used for fitting or hyperparameter/threshold tuning.
- **Class Imbalance**: The dataset has 70% Good / 30% Bad credit. Balanced class weights are used during training, and threshold tuning balances precision and recall for high-risk detection.
- **Fairness & Ethics**: The raw UCI dataset includes demographic attributes (`age`, `personal_status_sex`, `foreign_worker`). In regulated credit underwriting, protected attributes must be excluded or audited for disparate impact.
