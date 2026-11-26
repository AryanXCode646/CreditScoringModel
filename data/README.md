# Dataset Documentation — UCI Statlog (German Credit Data)

- **Name**: Statlog (German Credit Data)
- **Source**: UCI Machine Learning Repository (`https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)`)
- **Direct Download URL**: `https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data`
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Instances**: 1,000 rows, 20 attributes (7 numerical, 13 categorical), 0 missing values in raw file
- **Target Variable**:
  - Raw values: `1` = Good credit, `2` = Bad credit (700 Good, 300 Bad)
  - Encoded target (`credit_risk`): `0` = Good / Low Risk, `1` = Bad / High Risk

## Setup

Run the download script from the project root:

```bash
python scripts/download_data.py
```

## Limitations & Fairness Notes

- Collected in 1994 in Germany (historical economic context, Deutsche Mark amounts).
- Contains sensitive/demographic attributes (`age`, `personal_status_sex`, `foreign_worker`) that must not be used in real automated lending decisions without strict regulatory compliance and fairness auditing.
- Small sample size (1,000 records) limits generalization.
