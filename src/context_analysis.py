"""Stratify rolling-CV agreement by how much venue history a match has.

The applied question is whether pre-match weather context substitutes for
missing venue history. If the moisture features help most where a ground has
few prior matches, they are not just a proxy for venue identity: they carry the
cold-start signal the project needs.

Inputs are the pooled out-of-fold predictions written by
``evaluate_rolling_cv.py`` (``predictions/*.csv``).

Usage::

    python src\\context_analysis.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

BINS = [(-0.5, 0.5, "0"), (0.5, 2.5, "1-2"), (2.5, 5.5, "3-5"), (5.5, np.inf, "6+")]


def macro_f1(frame: pd.DataFrame) -> float:
    if frame.empty or frame["y_true"].nunique() < 2:
        return float("nan")
    return float(f1_score(frame["y_true"], frame["y_pred"], average="macro"))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=root / "models/rolling_cv_final/predictions",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "models/pitch_proxy_results/context_by_history.csv",
    )
    args = parser.parse_args()

    rows = []
    for format_name in ("T20", "ODI", "Test"):
        frames = {}
        for variant in ("xgb_ctx", "xgb_day"):
            path = args.predictions / f"{variant}@performat_{format_name}.csv"
            if not path.exists():
                continue
            frames[variant] = pd.read_csv(path)
        if len(frames) < 2:
            print(f"{format_name}: missing predictions, skipped")
            continue
        for low, high, label in BINS:
            subset = {
                variant: frame[
                    frame["history_matches"].between(low, high, inclusive="both")
                ]
                for variant, frame in frames.items()
            }
            ctx = subset["xgb_ctx"]
            day = subset["xgb_day"]
            if ctx.empty:
                continue
            rows.append(
                {
                    "match_type": format_name,
                    "history_bin": label,
                    "n": len(ctx),
                    "macro_f1_context": round(macro_f1(ctx), 4),
                    "macro_f1_day_only": round(macro_f1(day), 4),
                    "delta_context_minus_day": round(
                        macro_f1(ctx) - macro_f1(day), 4
                    ),
                    "accuracy_context": round(
                        float((ctx["y_true"] == ctx["y_pred"]).mean()), 4
                    ),
                    "accuracy_day_only": round(
                        float((day["y_true"] == day["y_pred"]).mean()), 4
                    ),
                }
            )

    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
