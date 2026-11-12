"""Robustness checks for the ball-by-ball wear estimator.

Three checks that a referee will ask for:

1. **Per-phase estimates.** Pooling phases forces one age slope across the
   powerplay, middle and death. Spinners do not bowl the death in limited-overs
   cricket, so the pooled slope is dominated by overs that contain little spin.
   Re-estimating inside each phase shows where the effect actually lives.
2. **Established bowlers only.** Part-time and incorrectly classified bowlers
   dilute the style contrast. Restricting to bowlers with at least 300 legal
   balls removes most of them, at the cost of sample size.
3. **Alternative age definition.** Ball age is recounted from the start of each
   innings. Using overs instead of ball age, and excluding the first and last
   over of each innings, checks that the result is not an artefact of the
   boundary.

Usage::

    python src\\wear_robustness.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import pitch_dynamics as D

OUTCOMES = ("runs_per_ball", "wicket_rate", "dot_share", "boundary_share")


def phase_estimates(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for match_type, subset in panel.groupby("match_type", sort=False):
        for phase, phase_frame in subset.groupby("phase", observed=True):
            if phase_frame["match_id"].nunique() < 30:
                continue
            if phase_frame.loc[phase_frame["spin"].eq(1), "balls"].sum() < 5000:
                continue
            for outcome in OUTCOMES:
                fitted = D.fit_spec(phase_frame, outcome, "match_fe")
                hit = fitted["terms"][fitted["terms"]["term"].eq("age_x_spin")]
                if hit.empty:
                    continue
                value = hit.iloc[0]
                rows.append(
                    {
                        "match_type": match_type,
                        "phase": phase,
                        "outcome": outcome,
                        "coefficient": round(float(value["coefficient"]), 5),
                        "std_error": round(float(value["std_error"]), 5),
                        "p_value": round(float(value["p_value"]), 5),
                        "cells": int(len(phase_frame)),
                        "matches": int(phase_frame["match_id"].nunique()),
                        "spin_balls": int(
                            phase_frame.loc[phase_frame["spin"].eq(1), "balls"].sum()
                        ),
                        "pace_balls": int(
                            phase_frame.loc[phase_frame["spin"].eq(0), "balls"].sum()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def headline(panel: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for match_type, subset in panel.groupby("match_type", sort=False):
        for outcome in OUTCOMES:
            fitted = D.fit_spec(subset, outcome, "match_fe")
            hit = fitted["terms"][fitted["terms"]["term"].eq("age_x_spin")]
            if hit.empty:
                continue
            value = hit.iloc[0]
            rows.append(
                {
                    "variant": label,
                    "match_type": match_type,
                    "outcome": outcome,
                    "coefficient": round(float(value["coefficient"]), 5),
                    "std_error": round(float(value["std_error"]), 5),
                    "p_value": round(float(value["p_value"]), 5),
                    "cells": int(len(subset)),
                    "matches": int(subset["match_id"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--balls", type=Path, default=root / "data/interim/balls.parquet"
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=root / "models/pitch_dynamics/over_panel.parquet",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "models/pitch_dynamics"
    )
    parser.add_argument("--min-bowler-balls", type=int, default=300)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(args.panel)
    print(f"Loaded panel with {len(panel):,} cells")

    by_phase = phase_estimates(panel)
    by_phase.to_csv(args.output_dir / "wear_by_phase.csv", index=False)
    print("\nPer-phase age x spin:")
    print(by_phase.to_string(index=False))

    # Drop the first and last over of every innings to test boundary effects.
    bounds = panel.groupby(["match_id", "innings"], observed=True)["over"].agg(
        ["min", "max"]
    )
    bounds = bounds.reset_index().rename(columns={"min": "first_over", "max": "last_over"})
    trimmed = panel.merge(bounds, on=["match_id", "innings"], how="left")
    trimmed = trimmed[
        trimmed["over"].gt(trimmed["first_over"])
        & trimmed["over"].lt(trimmed["last_over"])
    ]
    trimmed_results = headline(trimmed, "drop_first_and_last_over")
    print("\nBoundary check:")
    print(trimmed_results.to_string(index=False))

    established = D.load_panel(args.balls, min_bowler_balls=args.min_bowler_balls)
    established_results = headline(established, "established_bowlers")
    established_results.to_csv(
        args.output_dir / "wear_established_bowlers.csv", index=False
    )
    print("\nEstablished bowlers:")
    print(established_results.to_string(index=False))

    combined = pd.concat([trimmed_results, established_results], ignore_index=True)
    combined.to_csv(args.output_dir / "wear_robustness.csv", index=False)


if __name__ == "__main__":
    main()
