"""Unit tests for training, evaluation metrics, serialization, and prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_scoring.data import TARGET_COLUMN
from credit_scoring.predict import DEFAULT_SAMPLE_RECORD, load_model_bundle, predict_credit_risk
from credit_scoring.train import (
    compute_classification_metrics,
    save_bundle,
    train_and_select_model,
)


def _make_synthetic_dataset(n_rows: int = 60, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_rows):
        is_bad = int(i % 3 == 0)
        row = dict(DEFAULT_SAMPLE_RECORD)
        row["duration_months"] = int(rng.integers(24, 60) if is_bad else rng.integers(6, 24))
        row["credit_amount"] = int(rng.integers(5000, 15000) if is_bad else rng.integers(500, 4000))
        row["age"] = int(rng.integers(19, 70))
        row["checking_status"] = "A11" if is_bad else "A14"
        row[TARGET_COLUMN] = is_bad
        rows.append(row)
    return pd.DataFrame(rows)


def test_compute_classification_metrics() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.6, 0.9])
    metrics = compute_classification_metrics(y_true, y_prob, threshold=0.5)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_training_reproducibility_serialization_and_prediction(tmp_path: Path) -> None:
    df = _make_synthetic_dataset(n_rows=60, seed=7)
    res1 = train_and_select_model(df, test_size=0.2, cv_folds=3, random_state=42)
    res2 = train_and_select_model(df, test_size=0.2, cv_folds=3, random_state=42)

    assert res1["bundle"]["model_name"] == res2["bundle"]["model_name"]
    assert res1["bundle"]["cv_results"] == res2["bundle"]["cv_results"]

    bundle_path = save_bundle(res1["bundle"], tmp_path / "bundle.joblib")
    loaded = load_model_bundle(bundle_path)

    preds = predict_credit_risk(DEFAULT_SAMPLE_RECORD, loaded)
    assert len(preds) == 1
    assert preds[0]["predicted_class"] in (0, 1)
    assert 0.0 <= preds[0]["probability_high_risk"] <= 1.0
    assert pytest.approx(1.0, abs=1e-3) == (
        preds[0]["probability_high_risk"] + preds[0]["probability_low_risk"]
    )


def test_predict_rejects_invalid_input(tmp_path: Path) -> None:
    df = _make_synthetic_dataset(n_rows=45, seed=11)
    res = train_and_select_model(df, test_size=0.2, cv_folds=3, random_state=42)
    with pytest.raises(ValueError):
        predict_credit_risk({"age": 25}, res["bundle"])
