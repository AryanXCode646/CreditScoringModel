"""Dataset loading, schema definition, and download utilities for German Credit data."""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

UCI_GERMAN_CREDIT_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
)

RAW_COLUMNS: list[str] = [
    "checking_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment",
    "installment_commitment",
    "personal_status_sex",
    "other_parties",
    "residence_since",
    "property_magnitude",
    "age",
    "other_payment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "own_telephone",
    "foreign_worker",
    "raw_target",
]

NUMERIC_FEATURES: list[str] = [
    "duration_months",
    "credit_amount",
    "installment_commitment",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents",
]

CATEGORICAL_FEATURES: list[str] = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment",
    "personal_status_sex",
    "other_parties",
    "property_magnitude",
    "other_payment_plans",
    "housing",
    "job",
    "own_telephone",
    "foreign_worker",
]

FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "credit_risk"

CATEGORY_LABELS: dict[str, dict[str, str]] = {
    "checking_status": {
        "A11": "< 0 DM",
        "A12": "0 <= ... < 200 DM",
        "A13": ">= 200 DM / salary assignment",
        "A14": "no checking account",
    },
    "credit_history": {
        "A30": "no credits / all paid back duly",
        "A31": "all credits at this bank paid duly",
        "A32": "existing credits paid duly till now",
        "A33": "delay in paying off in the past",
        "A34": "critical account / other credits existing",
    },
    "purpose": {
        "A40": "car (new)",
        "A41": "car (used)",
        "A42": "furniture/equipment",
        "A43": "radio/television",
        "A44": "domestic appliances",
        "A45": "repairs",
        "A46": "education",
        "A48": "retraining",
        "A49": "business",
        "A410": "others",
    },
    "savings_status": {
        "A61": "< 100 DM",
        "A62": "100 <= ... < 500 DM",
        "A63": "500 <= ... < 1000 DM",
        "A64": ">= 1000 DM",
        "A65": "unknown / no savings account",
    },
    "employment": {
        "A71": "unemployed",
        "A72": "< 1 year",
        "A73": "1 <= ... < 4 years",
        "A74": "4 <= ... < 7 years",
        "A75": ">= 7 years",
    },
    "personal_status_sex": {
        "A91": "male : divorced/separated",
        "A92": "female : divorced/separated/married",
        "A93": "male : single",
        "A94": "male : married/widowed",
        "A95": "female : single",
    },
    "other_parties": {
        "A101": "none",
        "A102": "co-applicant",
        "A103": "guarantor",
    },
    "property_magnitude": {
        "A121": "real estate",
        "A122": "building society savings / life insurance",
        "A123": "car or other",
        "A124": "unknown / no property",
    },
    "other_payment_plans": {
        "A141": "bank",
        "A142": "stores",
        "A143": "none",
    },
    "housing": {
        "A151": "rent",
        "A152": "own",
        "A153": "for free",
    },
    "job": {
        "A171": "unemployed / unskilled - non-resident",
        "A172": "unskilled - resident",
        "A173": "skilled employee / official",
        "A174": "management / self-employed / highly qualified",
    },
    "own_telephone": {
        "A191": "none",
        "A192": "yes, registered",
    },
    "foreign_worker": {
        "A201": "yes",
        "A202": "no",
    },
}


def download_german_credit(
    destination: str | Path = "data/german.data",
    url: str = UCI_GERMAN_CREDIT_URL,
    overwrite: bool = False,
) -> Path:
    """Download the UCI Statlog German Credit dataset.

    Args:
        destination: Local file path where the raw dataset will be saved.
        url: Official UCI dataset URL.
        overwrite: Whether to re-download if the file already exists.

    Returns:
        Path to the downloaded dataset file.

    Raises:
        RuntimeError: If downloading fails.
    """
    dest_path = Path(destination)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and not overwrite:
        return dest_path

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(
            f"Failed to download UCI German Credit dataset from '{url}': {exc}"
        ) from exc

    if len(content) < 1000:
        raise RuntimeError(
            f"Downloaded file from '{url}' appears incomplete ({len(content)} bytes)."
        )

    dest_path.write_bytes(content)
    return dest_path


def load_german_credit(data_path: str | Path = "data/german.data") -> pd.DataFrame:
    """Load the German Credit dataset and encode binary target.

    Target mapping:
        raw_target == 1 (Good Credit) -> credit_risk = 0 (Low Risk)
        raw_target == 2 (Bad Credit)  -> credit_risk = 1 (High Risk)

    Args:
        data_path: Path to the space-delimited `german.data` file.

    Returns:
        DataFrame with named feature columns and `credit_risk` target.

    Raises:
        FileNotFoundError: If the dataset file is missing.
        ValueError: If the file schema or target values are invalid.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found at '{path}'. "
            "Run `python scripts/download_data.py` to fetch the UCI German Credit dataset."
        )

    df = pd.read_csv(path, sep=r"\s+", header=None, names=RAW_COLUMNS)
    if df.empty:
        raise ValueError(f"Dataset file '{path}' is empty.")

    if df.shape[1] != len(RAW_COLUMNS):
        raise ValueError(f"Expected {len(RAW_COLUMNS)} columns in '{path}', found {df.shape[1]}.")

    invalid_targets = set(df["raw_target"].dropna().unique()) - {1, 2}
    if invalid_targets:
        raise ValueError(
            f"Unexpected target values {sorted(invalid_targets)} in '{path}'. Expected {{1, 2}}."
        )

    df[TARGET_COLUMN] = (df["raw_target"] == 2).astype(int)
    df = df.drop(columns=["raw_target"])
    return df


def validate_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and order feature columns for inference or training.

    Args:
        df: Input DataFrame containing credit applicant features.

    Returns:
        Copy of DataFrame containing only validated feature columns in canonical order.

    Raises:
        ValueError: If required columns are missing, DataFrame is empty,
            or numeric columns contain non-positive values where invalid.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Input DataFrame must contain at least one row.")

    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    validated = df[FEATURE_COLUMNS].copy()
    for col in NUMERIC_FEATURES:
        validated[col] = pd.to_numeric(validated[col], errors="coerce")

    if (validated["duration_months"].dropna() <= 0).any():
        raise ValueError("'duration_months' must be strictly positive (> 0).")
    if (validated["credit_amount"].dropna() < 0).any():
        raise ValueError("'credit_amount' cannot be negative.")
    if (validated["age"].dropna() <= 0).any():
        raise ValueError("'age' must be strictly positive (> 0).")

    return validated
