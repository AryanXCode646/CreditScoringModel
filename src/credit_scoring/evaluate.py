"""Evaluation, visualization, and reporting for the Credit Scoring Model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split

from credit_scoring.config import load_config
from credit_scoring.data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_german_credit,
    validate_input_dataframe,
)
from credit_scoring.train import (
    build_candidate_models,
    compute_classification_metrics,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from credit_scoring.preprocessing import build_preprocessor  # noqa: E402


def _extract_feature_importance(pipeline: Pipeline, top_n: int = 15) -> pd.DataFrame | None:
    """Extract feature importance or absolute coefficients from a fitted pipeline."""
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    col_transformer = preprocessor.named_steps["column_transformer"]
    feature_names = col_transformer.get_feature_names_out()

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
    else:
        return None

    df_imp = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return df_imp


def _save_plots(
    y_test: np.ndarray,
    test_probs_by_model: dict[str, np.ndarray],
    best_model_name: str,
    best_threshold: float,
    best_pipeline: Pipeline,
    test_metrics_by_model: dict[str, dict[str, float]],
    figures_dir: Path,
) -> list[Path]:
    """Generate and save evaluation plots."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # 1. Model comparison bar chart
    df_comp = pd.DataFrame(test_metrics_by_model).T[
        ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    ]
    ax = df_comp.plot(kind="bar", figsize=(9, 5), rot=0)
    ax.set_title("Test Set Performance Comparison (Threshold = 0.50)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    plt.tight_layout()
    comp_path = figures_dir / "model_comparison.png"
    plt.savefig(comp_path, dpi=150)
    plt.close()
    saved.append(comp_path)

    # 2. Confusion matrix for selected model
    best_probs = test_probs_by_model[best_model_name]
    best_preds = (best_probs >= best_threshold).astype(int)
    cm = confusion_matrix(y_test, best_preds)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    labels = ["Good (0)", "Bad Risk (1)"]
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix — {best_model_name} (thr={best_threshold:.2f})")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=12)
    plt.tight_layout()
    cm_path = figures_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    saved.append(cm_path)

    # 3. ROC curves
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, probs in test_probs_by_model.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc_val = test_metrics_by_model[name]["roc_auc"]
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random Baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves on Held-Out Test Set")
    ax.legend(loc="lower right")
    plt.tight_layout()
    roc_path = figures_dir / "roc_curves.png"
    plt.savefig(roc_path, dpi=150)
    plt.close()
    saved.append(roc_path)

    # 4. Calibration curve for best model
    prob_true, prob_pred = calibration_curve(y_test, best_probs, n_bins=8, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, marker="o", label=best_model_name)
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives (Bad Credit)")
    ax.set_title(f"Probability Calibration — {best_model_name}")
    ax.legend(loc="upper left")
    plt.tight_layout()
    cal_path = figures_dir / "calibration_curve.png"
    plt.savefig(cal_path, dpi=150)
    plt.close()
    saved.append(cal_path)

    # 5. Feature importance
    df_imp = _extract_feature_importance(best_pipeline, top_n=15)
    if df_imp is not None and not df_imp.empty:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        df_plot = df_imp.iloc[::-1]
        ax.barh(df_plot["feature"], df_plot["importance"], color="steelblue")
        ax.set_xlabel("Importance")
        ax.set_title(f"Top 15 Feature Importances — {best_model_name}")
        plt.tight_layout()
        imp_path = figures_dir / "feature_importance.png"
        plt.savefig(imp_path, dpi=150)
        plt.close()
        saved.append(imp_path)

    return saved


def evaluate_bundle(
    bundle: dict[str, Any],
    df: pd.DataFrame,
    figures_dir: str | Path = "reports/figures",
    report_path: str | Path = "reports/model_evaluation.md",
    metrics_json_path: str | Path = "reports/metrics.json",
) -> dict[str, Any]:
    """Evaluate trained bundle and baseline candidates on the isolated test split."""
    meta = bundle["metadata"]
    test_size = float(meta["test_size"])
    random_state = int(meta["random_state"])

    X = validate_input_dataframe(df[FEATURE_COLUMNS])
    y = df[TARGET_COLUMN].astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    candidates = build_candidate_models(random_state=random_state)
    test_metrics_by_model: dict[str, dict[str, float]] = {}
    test_probs_by_model: dict[str, np.ndarray] = {}

    for name, estimator in candidates.items():
        if name == bundle["model_name"]:
            pipe = bundle["model"]
        else:
            pipe = Pipeline(
                steps=[("preprocessor", build_preprocessor()), ("classifier", estimator)]
            )
            pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:, 1]
        test_probs_by_model[name] = probs
        test_metrics_by_model[name] = compute_classification_metrics(y_test, probs, threshold=0.5)

    best_name = bundle["model_name"]
    best_thr = float(bundle["threshold"])
    best_probs = test_probs_by_model[best_name]
    tuned_metrics = compute_classification_metrics(y_test, best_probs, threshold=best_thr)
    best_preds = (best_probs >= best_thr).astype(int)
    cm = confusion_matrix(y_test, best_preds).tolist()
    cls_report = classification_report(
        y_test,
        best_preds,
        target_names=["Good Credit (0)", "Bad Risk (1)"],
        zero_division=0,
    )

    _save_plots(
        y_test=y_test,
        test_probs_by_model=test_probs_by_model,
        best_model_name=best_name,
        best_threshold=best_thr,
        best_pipeline=bundle["model"],
        test_metrics_by_model=test_metrics_by_model,
        figures_dir=Path(figures_dir),
    )

    payload = {
        "selected_model": best_name,
        "tuned_threshold": best_thr,
        "cv_results": bundle["cv_results"],
        "test_metrics_default_threshold": test_metrics_by_model,
        "selected_model_test_metrics_tuned_threshold": tuned_metrics,
        "confusion_matrix_tuned": cm,
        "classification_report_tuned": cls_report,
    }

    metrics_path = Path(metrics_json_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Credit Scoring Model — Evaluation Report",
        "",
        "> **Educational Disclaimer**: This model is built for educational ML benchmarking "
        "on the 1994 UCI Statlog German Credit dataset. It is **not** validated or suitable "
        "for real-world lending or financial underwriting decisions.",
        "",
        f"- **Selected Model**: `{best_name}` (chosen via 5-fold CV `{meta['selection_metric']}`)",
        f"- **CV-Tuned Decision Threshold**: `{best_thr:.2f}`",
        f"- **Train / Test Split**: {meta['n_train']} / {meta['n_test']} "
        f"(stratified, seed={random_state})",
        "",
        "## 1. Cross-Validation Comparison (Training Set Only)",
        "",
        "| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Brier Score |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in bundle["cv_results"].items():
        lines.append(
            f"| {name} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | "
            f"{m['f1_score']:.4f} | {m['roc_auc']:.4f} | {m['brier_score']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 2. Held-Out Test Set Comparison (Threshold = 0.50)",
            "",
            "| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Brier Score |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for name, m in test_metrics_by_model.items():
        lines.append(
            f"| {name} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | "
            f"{m['f1_score']:.4f} | {m['roc_auc']:.4f} | {m['brier_score']:.4f} |"
        )

    lines.extend(
        [
            "",
            f"## 3. Selected Model ({best_name}) at Tuned Threshold ({best_thr:.2f})",
            "",
            f"- **Accuracy**: {tuned_metrics['accuracy']:.4f}",
            f"- **Precision (High Risk)**: {tuned_metrics['precision']:.4f}",
            f"- **Recall (High Risk)**: {tuned_metrics['recall']:.4f}",
            f"- **F1-Score (High Risk)**: {tuned_metrics['f1_score']:.4f}",
            f"- **ROC-AUC**: {tuned_metrics['roc_auc']:.4f}",
            f"- **Brier Score**: {tuned_metrics['brier_score']:.4f}",
            "",
            "### Classification Report",
            "",
            "```text",
            cls_report.strip(),
            "```",
            "",
            "## 4. Leakage, Fairness & Limitations",
            "",
            "- **Data Leakage Prevention**: Preprocessing (`SimpleImputer`, `StandardScaler`, "
            "`OneHotEncoder`) and threshold selection run strictly inside training folds; "
            "the 20% test set is never used for fitting or hyperparameter/threshold tuning.",
            "- **Class Imbalance**: The dataset has 70% Good / 30% Bad credit. Balanced class "
            "weights are used during training, and threshold tuning balances precision and recall "
            "for high-risk detection.",
            "- **Fairness & Ethics**: The raw UCI dataset includes demographic attributes (`age`, "
            "`personal_status_sex`, `foreign_worker`). In regulated credit underwriting, protected "
            "attributes must be excluded or audited for disparate impact.",
        ]
    )
    report_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for evaluating the saved credit scoring model."""
    parser = argparse.ArgumentParser(description="Evaluate Credit Scoring Model.")
    parser.add_argument("--config", default="configs/config.json", help="Path to config JSON.")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
        model_path = Path(cfg["training"]["model_output_path"])
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at '{model_path}'. "
                "Run `python -m credit_scoring.train` first."
            )
        bundle = joblib.load(model_path)
        df = load_german_credit(cfg["dataset"]["local_path"])
        payload = evaluate_bundle(
            bundle=bundle,
            df=df,
            figures_dir=cfg["evaluation"]["figures_dir"],
            report_path=cfg["evaluation"]["report_path"],
            metrics_json_path=cfg["evaluation"]["metrics_json_path"],
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Evaluated {payload['selected_model']} (threshold={payload['tuned_threshold']:.2f})")
    print(json.dumps(payload["selected_model_test_metrics_tuned_threshold"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
