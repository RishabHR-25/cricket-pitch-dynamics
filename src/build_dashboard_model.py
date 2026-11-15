"""Fit the deployment models the dashboard serves.

The dashboard must not train anything on interaction. This script refits the
paper's headline match model (XGBoost with match-day and prior-window moisture
context, per-format venue windows) on every labelled match and writes one
pipeline per format together with a model card. Performance is *not* estimated
here: the card records the rolling-origin numbers from
``models/rolling_cv_final/cv_summary.csv``, which are the estimates the paper
reports.

Run after ``evaluate_rolling_cv.py``:

    python src\\build_dashboard_model.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import train_pitch_models as T  # noqa: E402

FRAMES = {
    "T20": "t20_w5.csv.gz",
    "ODI": "odi_w5.csv.gz",
    "Test": "test_w10.csv.gz",
}
WINDOWS = {"T20": 5, "ODI": 5, "Test": 10}
EXCLUDED = {"match_id", "date", "match_type", "y"}
CATEGORICAL = ("venue", "venue_id", "city", "gender", "team_type", "toss_decision")


def _json_scalar(value):
    """Convert a pandas/NumPy value to something strict JSON can hold."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def build(root: Path = ROOT) -> None:
    cv = pd.read_csv(root / "models/rolling_cv_final/cv_summary.csv")
    output = root / "models/dashboard"
    output.mkdir(parents=True, exist_ok=True)

    for match_type, filename in FRAMES.items():
        frame = pd.read_csv(root / "models/rolling_cv_final/feature_frames" / filename)
        frame = frame[frame["y"].notna()].copy()

        # A per-format window leaves the other window's history columns entirely
        # missing. Drop them, exactly as evaluate_rolling_cv.py does.
        feature_columns = [
            column
            for column in frame.columns
            if column not in EXCLUDED and not frame[column].isna().all()
        ]
        categorical = [c for c in feature_columns if c in CATEGORICAL]
        numeric = [c for c in feature_columns if c not in categorical]

        pipeline = T._make_pipeline(
            numeric, categorical, device="cpu", model_kind="xgb"
        )
        class_order = list(T.CLASS_ORDER)
        codes = frame["y"].map({label: index for index, label in enumerate(class_order)})
        if codes.isna().any():
            raise ValueError(f"{match_type}: unknown label in feature frame")
        pipeline.fit(frame[feature_columns], codes.to_numpy(dtype="int64"))

        folder = output / match_type.lower()
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, folder / "model.joblib")

        row = cv[
            cv["variant"].eq("xgb:ctx") & cv["match_type"].eq(match_type)
        ]
        if row.empty:
            raise ValueError(f"{match_type}: no xgb:ctx row in cv_summary.csv")
        rolling = row.iloc[0].to_dict()

        card = {
            "match_type": match_type,
            "venue_window": WINDOWS[match_type],
            "class_order": class_order,
            "feature_columns": feature_columns,
            "categorical_columns": categorical,
            "numeric_columns": numeric,
            "training_rows": int(len(frame)),
            "training_start": str(frame["date"].min()),
            "training_end": str(frame["date"].max()),
            "label_counts": {
                str(key): int(value)
                for key, value in frame["y"].value_counts().items()
            },
            "rolling_cv": {
                key: _json_scalar(value) for key, value in rolling.items()
            },
            "provenance": (
                "Refit on all labelled matches for deployment. Performance "
                "estimates are the rolling-origin CV in "
                "models/rolling_cv_final/cv_summary.csv, not in-sample fit."
            ),
        }
        (folder / "model_card.json").write_text(
            json.dumps(card, indent=2), encoding="utf-8"
        )
        print(
            f"{match_type}: {len(frame):,} matches, {len(feature_columns)} features, "
            f"window {WINDOWS[match_type]}, rolling macro-F1 "
            f"{rolling['pooled_macro_f1']:.4f}"
        )

    print(f"\nwritten under {output}")


if __name__ == "__main__":
    build()
