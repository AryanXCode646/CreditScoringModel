"""Training workflow for the Credit Scoring Model."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from credit_scoring.config import load_config
from credit_scoring.data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_german_credit,
    validate_input_dataframe,
)
from credit_scoring.preprocessing import build_preprocessor


def build_candidate_models(random_state: int = 42) -> dict[str, Any]:
    """Create candidate classification estimators."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def compute_classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute standard binary classification metrics for high-risk class (1)."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }


def find_optimal_threshold(y_true: np.ndarray | pd.Series, y_prob: np.ndarray) -> float:
    """Select the probability threshold maximizing F1-score on out-of-fold predictions."""
    thresholds = np.linspace(0.20, 0.80, 61)
    best_threshold = 0.50
    best_f1 = -1.0
    for thr in thresholds:
        score = f1_score(y_true, (y_prob >= thr).astype(int), zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(thr)
    return round(best_threshold, 2)


def train_and_select_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    cv_folds: int = 5,
    random_state: int = 42,
    selection_metric: str = "roc_auc",
) -> dict[str, Any]:
    """Run stratified split, CV model comparison, selection, and final training.

    Args:
        df: Loaded German Credit DataFrame.
        test_size: Fraction reserved for isolated test evaluation.
        cv_folds: Number of stratified CV folds on the training set.
        random_state: Deterministic random seed.
        selection_metric: CV metric used to select the winning model.

    Returns:
        Dictionary containing the trained model bundle and train/test splits.
    """
    X = validate_input_dataframe(df[FEATURE_COLUMNS])
    y = df[TARGET_COLUMN].astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    candidates = build_candidate_models(random_state=random_state)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    cv_results: dict[str, dict[str, float]] = {}
    oof_thresholds: dict[str, float] = {}
    fitted_pipelines: dict[str, Pipeline] = {}

    for name, estimator in candidates.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", estimator),
            ]
        )
        oof_probs = cross_val_predict(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
        )[:, 1]
        opt_thr = find_optimal_threshold(y_train, oof_probs)
        cv_metrics = compute_classification_metrics(y_train, oof_probs, threshold=0.5)
        cv_metrics["optimal_threshold"] = opt_thr
        cv_results[name] = cv_metrics
        oof_thresholds[name] = opt_thr

        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

    best_model_name = max(cv_results, key=lambda k: cv_results[k][selection_metric])
    best_pipeline = fitted_pipelines[best_model_name]
    best_threshold = oof_thresholds[best_model_name]

    bundle = {
        "model": best_pipeline,
        "model_name": best_model_name,
        "threshold": best_threshold,
        "feature_columns": FEATURE_COLUMNS,
        "target_mapping": {
            0: "Good Credit (Low Risk)",
            1: "Bad Credit (High Risk)",
        },
        "cv_results": cv_results,
        "metadata": {
            "dataset": "UCI Statlog (German Credit Data)",
            "n_total": int(len(df)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "test_size": test_size,
            "cv_folds": cv_folds,
            "random_state": random_state,
            "selection_metric": selection_metric,
            "sklearn_version": sklearn.__version__,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    }

    return {
        "bundle": bundle,
        "fitted_pipelines": fitted_pipelines,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def save_bundle(bundle: dict[str, Any], output_path: str | Path) -> Path:
    """Persist trained model bundle with joblib."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for training the credit scoring model."""
    parser = argparse.ArgumentParser(description="Train Credit Scoring Model.")
    parser.add_argument("--config", default="configs/config.json", help="Path to config JSON.")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
        df = load_german_credit(cfg["dataset"]["local_path"])
        result = train_and_select_model(
            df=df,
            test_size=cfg["dataset"]["test_size"],
            cv_folds=cfg["training"]["cv_folds"],
            random_state=cfg["training"]["random_state"],
            selection_metric=cfg["training"]["selection_metric"],
        )
        model_path = save_bundle(result["bundle"], cfg["training"]["model_output_path"])
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    bundle = result["bundle"]
    print(
        f"Selected best model (by CV {bundle['metadata']['selection_metric']}): "
        f"{bundle['model_name']} (threshold={bundle['threshold']})"
    )
    print("5-Fold CV Results:")
    print(json.dumps(bundle["cv_results"], indent=2))
    print(f"Saved model bundle to: {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
