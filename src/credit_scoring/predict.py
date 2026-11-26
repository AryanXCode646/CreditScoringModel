"""Inference interface for the Credit Scoring Model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from credit_scoring.data import validate_input_dataframe

DEFAULT_SAMPLE_RECORD: dict[str, Any] = {
    "checking_status": "A11",
    "duration_months": 24,
    "credit_history": "A32",
    "purpose": "A40",
    "credit_amount": 4500,
    "savings_status": "A61",
    "employment": "A73",
    "installment_commitment": 3,
    "personal_status_sex": "A93",
    "other_parties": "A101",
    "residence_since": 2,
    "property_magnitude": "A123",
    "age": 32,
    "other_payment_plans": "A143",
    "housing": "A152",
    "existing_credits": 1,
    "job": "A173",
    "num_dependents": 1,
    "own_telephone": "A192",
    "foreign_worker": "A201",
}


def load_model_bundle(
    model_path: str | Path = "models/credit_scoring_bundle.joblib",
) -> dict[str, Any]:
    """Load a saved credit scoring model bundle."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model bundle not found at '{path}'. Run `python -m credit_scoring.train` first."
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(f"Invalid model bundle format in '{path}'.")
    return bundle


def predict_credit_risk(
    records: pd.DataFrame | dict[str, Any] | list[dict[str, Any]],
    bundle: dict[str, Any],
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Predict credit risk class and probabilities for one or more applicants.

    Args:
        records: Single applicant dict, list of dicts, or DataFrame.
        bundle: Loaded model bundle containing fitted pipeline and threshold.
        threshold: Optional decision threshold override.

    Returns:
        List of prediction dictionaries with class, label, and probabilities.
    """
    if isinstance(records, dict):
        df = pd.DataFrame([records])
    elif isinstance(records, list):
        df = pd.DataFrame(records)
    elif isinstance(records, pd.DataFrame):
        df = records.copy()
    else:
        raise ValueError("Input records must be a dict, list of dicts, or DataFrame.")

    validated = validate_input_dataframe(df)
    model = bundle["model"]
    thr = float(bundle.get("threshold", 0.5) if threshold is None else threshold)
    target_mapping = bundle.get(
        "target_mapping",
        {0: "Good Credit (Low Risk)", 1: "Bad Credit (High Risk)"},
    )

    probs = model.predict_proba(validated)
    results: list[dict[str, Any]] = []
    for row_prob in probs:
        prob_good = float(row_prob[0])
        prob_bad = float(row_prob[1])
        pred_class = int(prob_bad >= thr)
        results.append(
            {
                "predicted_class": pred_class,
                "predicted_label": target_mapping[pred_class],
                "probability_high_risk": round(prob_bad, 4),
                "probability_low_risk": round(prob_good, 4),
                "threshold_used": thr,
                "model_name": bundle.get("model_name", "unknown"),
            }
        )
    return results


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for credit scoring inference."""
    parser = argparse.ArgumentParser(description="Predict credit risk for an applicant.")
    parser.add_argument(
        "--model",
        default="models/credit_scoring_bundle.joblib",
        help="Path to saved model bundle.",
    )
    parser.add_argument("--input-json", help="Path to JSON file with applicant record(s).")
    parser.add_argument("--sample", action="store_true", help="Run prediction on built-in sample.")
    args = parser.parse_args(argv)

    try:
        bundle = load_model_bundle(args.model)
        if args.input_json:
            payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
        elif args.sample:
            payload = DEFAULT_SAMPLE_RECORD
        else:
            raise ValueError("Provide either --input-json <path> or --sample.")
        predictions = predict_credit_risk(payload, bundle)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(predictions, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
