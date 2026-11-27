# Credit Scoring Model

An educational machine learning project that predicts credit risk (Good Credit vs. Bad Credit) using structured financial and demographic data. It features leakage-safe preprocessing, feature engineering, stratified cross-validation, and multi-model evaluation.

> **Educational Disclaimer:** This project is an educational machine learning benchmark trained on the 1994 UCI Statlog German Credit dataset. It is **not** a validated financial underwriting system and must **not** be used for real-world lending decisions.

## Problem Statement & Objectives

Predict an individual's credit risk (`Good Credit (0)` vs. `Bad Credit / High Risk (1)`) from structured financial and demographic history using leakage-safe preprocessing, feature engineering, stratified cross-validation, and multi-model evaluation.

## Dataset

- **Name:** Statlog (German Credit Data)
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
- **License:** CC BY 4.0
- **Size:** 1,000 instances, 20 raw attributes (7 numeric, 13 categorical), 0 missing values
- **Target:** `credit_risk` (`0` = Good Credit / 700 rows, `1` = Bad Credit / 300 rows)

## Technology Stack

- Python 3.12 (`>=3.10`)
- `scikit-learn`, `pandas`, `numpy`, `scipy`
- `matplotlib`, `seaborn`, `joblib`
- `streamlit` (optional interactive UI)
- `pytest`, `ruff`

## Repository Structure

```text
CreditScoringModel/
├── app.py                     # Optional Streamlit prediction interface
├── configs/
│   └── config.json            # Dataset, training, and evaluation configuration
├── data/                      # Dataset directory (gitignored raw files)
├── models/                    # Saved model bundle (gitignored)
├── reports/
│   ├── figures/               # EDA & evaluation plots
│   ├── metrics.json           # Machine-readable evaluation metrics
│   └── model_evaluation.md    # Detailed evaluation report
├── scripts/
│   ├── download_data.py       # Downloads UCI Statlog German Credit dataset
│   └── run_eda.py             # Generates EDA summary & figures
├── src/credit_scoring/
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
└── tests/
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
```

## Usage

```bash
# 1. Download dataset
python scripts/download_data.py

# 2. Run exploratory data analysis
python scripts/run_eda.py

# 3. Train & select best model (5-fold Stratified CV on 80% train split)
python -m credit_scoring.train

# 4. Evaluate on isolated 20% test split and generate figures
python -m credit_scoring.evaluate

# 5. Run sample prediction
python -m credit_scoring.predict --sample

# 6. Launch optional Streamlit UI
streamlit run app.py
```

## Actual Evaluation Results

Models were compared using 5-fold Stratified Cross-Validation (`random_state=42`) and evaluated on an isolated 20% test set (`n=200`).

### 5-Fold Cross-Validation

**Training Split:** `n=800`

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Brier Score |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6963 | 0.4953 | 0.6542 | 0.5637 | 0.7632 | 0.1984 |
| Decision Tree | 0.6512 | 0.4463 | 0.6750 | 0.5373 | 0.6991 | 0.2342 |
| **Random Forest (Selected)** | **0.7325** | **0.5436** | **0.6750** | **0.6022** | **0.7901** | **0.1804** |

### Held-Out Test Set

**Test Set:** `n=200`, model: `random_forest`, CV-tuned threshold: `0.46`

| Metric | Result |
|---|---:|
| Accuracy | 0.6900 |
| Precision (High Risk) | 0.4894 |
| Recall (High Risk) | 0.7667 |
| F1-Score (High Risk) | 0.5974 |
| ROC-AUC | 0.7910 |
| Brier Score | 0.1829 |

See [reports/model_evaluation.md](reports/model_evaluation.md) and `reports/figures/` for the confusion matrix, ROC curves, calibration curve, and top feature importances.

## Testing & Code Quality

```bash
python -m pytest -q
ruff check .
ruff format --check .
```

## Limitations & Future Improvements

- **Historical Context:** Data was collected in 1994 in Germany, with monetary amounts denominated in Deutsche Mark.
- **Fairness:** Raw features include `age`, `personal_status_sex`, and `foreign_worker`. Real-world credit models require fairness auditing and careful assessment of protected attributes.
- **Future Work:** Cost-sensitive optimization aligned with the UCI 5:1 false-negative/false-positive cost matrix and isotonic/Platt probability calibration.

## Author

**Aryan Saxena**
