"""Preprocessing pipeline construction for credit scoring."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_scoring.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from credit_scoring.features import ENGINEERED_NUMERIC_FEATURES, CreditFeatureEngineer


def build_preprocessor() -> Pipeline:
    """Build leakage-safe feature engineering and preprocessing pipeline.

    Returns:
        Scikit-learn Pipeline that engineers features, imputes missing values,
        scales numerical features, and one-hot encodes categorical features.
    """
    all_numeric = NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, all_numeric),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("feature_engineering", CreditFeatureEngineer()),
            ("column_transformer", column_transformer),
        ]
    )
