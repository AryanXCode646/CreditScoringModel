"""Domain-driven feature engineering for credit scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ENGINEERED_NUMERIC_FEATURES: list[str] = [
    "monthly_credit_amount",
    "log_credit_amount",
    "installment_burden",
    "credit_per_existing_loan",
]


class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add deterministic financial-history features without fitting on target."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "CreditFeatureEngineer":
        """Stateless transformer; returns self."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute engineered financial features from validated raw features.

        Args:
            X: Input DataFrame containing raw credit features.

        Returns:
            DataFrame with original and engineered columns.
        """
        df = X.copy()
        duration = df["duration_months"].replace(0, np.nan)
        credit_amount = df["credit_amount"].clip(lower=0)
        existing_credits = df["existing_credits"].clip(lower=1)

        df["monthly_credit_amount"] = credit_amount / duration
        df["log_credit_amount"] = np.log1p(credit_amount)
        df["installment_burden"] = df["installment_commitment"] * df["monthly_credit_amount"]
        df["credit_per_existing_loan"] = credit_amount / existing_credits
        return df
