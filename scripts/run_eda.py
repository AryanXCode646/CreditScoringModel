"""Generate exploratory data analysis summary and figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from credit_scoring.config import load_config
from credit_scoring.data import TARGET_COLUMN, load_german_credit


def run_eda(data_path: str | Path, figures_dir: str | Path) -> None:
    """Create EDA summary plots for German Credit dataset."""
    df = load_german_credit(data_path)
    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Target class distribution
    fig, ax = plt.subplots(figsize=(5.5, 4))
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    ax.bar(["Good Credit (0)", "Bad Risk (1)"], counts.values, color=["#2b8a3e", "#c92a2a"])
    ax.set_title("Class Distribution — UCI German Credit")
    ax.set_ylabel("Count")
    for idx, val in enumerate(counts.values):
        ax.text(idx, val + 12, str(val), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "eda_class_distribution.png", dpi=150)
    plt.close()

    # 2. Credit amount and duration by risk class
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    sns.boxplot(data=df, x=TARGET_COLUMN, y="credit_amount", ax=axes[0])
    axes[0].set_xticklabels(["Good (0)", "Bad Risk (1)"])
    axes[0].set_title("Credit Amount by Risk Class")

    sns.boxplot(data=df, x=TARGET_COLUMN, y="duration_months", ax=axes[1])
    axes[1].set_xticklabels(["Good (0)", "Bad Risk (1)"])
    axes[1].set_title("Loan Duration (Months) by Risk Class")
    plt.tight_layout()
    plt.savefig(out_dir / "eda_numeric_distributions.png", dpi=150)
    plt.close()

    print(f"EDA complete: {len(df)} rows, {df.isna().sum().sum()} missing values.")
    print(f"Saved EDA figures to: {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run EDA on German Credit dataset.")
    parser.add_argument("--config", default="configs/config.json", help="Path to config JSON.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_eda(cfg["dataset"]["local_path"], cfg["evaluation"]["figures_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
