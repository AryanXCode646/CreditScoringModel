"""Download the UCI Statlog German Credit dataset."""

from __future__ import annotations

import argparse
import sys

from credit_scoring.config import load_config
from credit_scoring.data import download_german_credit


def main() -> int:
    parser = argparse.ArgumentParser(description="Download UCI German Credit dataset.")
    parser.add_argument("--config", default="configs/config.json", help="Path to config JSON.")
    parser.add_argument("--overwrite", action="store_true", help="Re-download if file exists.")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
        path = download_german_credit(
            destination=cfg["dataset"]["local_path"],
            url=cfg["dataset"]["url"],
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Dataset ready at: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
