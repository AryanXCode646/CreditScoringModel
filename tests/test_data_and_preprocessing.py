"""Unit tests for dataset loading, validation, preprocessing, and feature engineering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_scoring.data import (
    FEATURE_COLUMNS,
    RAW_COLUMNS,
    TARGET_COLUMN,
    load_german_credit,
    validate_input_dataframe,
)
from credit_scoring.features import ENGINEERED_NUMERIC_FEATURES, CreditFeatureEngineer
from credit_scoring.predict import DEFAULT_SAMPLE_RECORD
from credit_scoring.preprocessing import build_preprocessor


def _make_synthetic_df(n_rows: int = 30, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_rows):
        row = dict(DEFAULT_SAMPLE_RECORD)
        row["duration_months"] = int(rng.integers(6, 48))
        row["credit_amount"] = int(rng.integers(500, 10000))
        row["age"] = int(rng.integers(20, 65))
        row[TARGET_COLUMN] = int(i % 3 == 0)
        rows.append(row)
    return pd.DataFrame(rows)


def test_load_german_credit_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_german_credit(tmp_path / "missing.data")


def test_load_german_credit_parses_valid_file(tmp_path: Path) -> None:
    sample_file = tmp_path / "german.data"
    line_good = " ".join(str(DEFAULT_SAMPLE_RECORD[c]) for c in RAW_COLUMNS[:-1]) + " 1\n"
    line_bad = " ".join(str(DEFAULT_SAMPLE_RECORD[c]) for c in RAW_COLUMNS[:-1]) + " 2\n"
    sample_file.write_text(line_good + line_bad, encoding="utf-8")

    df = load_german_credit(sample_file)
    assert df.shape == (2, len(FEATURE_COLUMNS) + 1)
    assert df[TARGET_COLUMN].tolist() == [0, 1]


def test_validate_input_dataframe_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="Missing required feature columns"):
        validate_input_dataframe(pd.DataFrame([{"age": 30}]))

    bad_duration = dict(DEFAULT_SAMPLE_RECORD, duration_months=0)
    with pytest.raises(ValueError, match="duration_months"):
        validate_input_dataframe(pd.DataFrame([bad_duration]))

    bad_age = dict(DEFAULT_SAMPLE_RECORD, age=-5)
    with pytest.raises(ValueError, match="age"):
        validate_input_dataframe(pd.DataFrame([bad_age]))


def test_feature_engineering_and_missing_value_imputation() -> None:
    df = _make_synthetic_df(n_rows=10)
    X = df[FEATURE_COLUMNS].copy()
    X.loc[0, "credit_amount"] = np.nan
    X.loc[1, "checking_status"] = None

    engineer = CreditFeatureEngineer()
    engineered = engineer.transform(X)
    for col in ENGINEERED_NUMERIC_FEATURES:
        assert col in engineered.columns

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == 10
    assert not np.isnan(transformed).any()
