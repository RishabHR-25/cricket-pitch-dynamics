"""Identification tests: what the within-innings gradient is not.

The pooled age x spin gradient in ``pitch_dynamics.py`` is precise, stable and
consistent with pitch wear. It is also consistent with at least four other
mechanisms, three of which this module can test directly:

1. **Phase composition.** Spinners bowl a different mix of overs than seamers.
2. **Bowler selection.** Which bowler bowls an over is not random, and it changes
   systematically at the second new ball.
3. **Ball condition.** The ball is replaced at over 80 in Tests and, since
   October 2011, twice per ODI innings.
4. **Batter adaptation.** Batters become more settled as an innings proceeds.

The centrepiece is the second-new-ball event study. The naive comparison looks
like a clean natural experiment: at over 80 the ball is renewed while the pitch
keeps ageing, so a jump in outcomes at that boundary should be a ball-condition
effect. It is not. Across all deliveries the pace dot-ball share rises by about
1.5 percentage points at over 80. Within the same bowler, in the same innings,
it *falls* by about 2 points. The aggregate jump is the arrival of the frontline
seamers, not the arrival of a new ball. This is the same composition problem that
threatens the wear estimate itself, in its most visible form.

The second test uses the October 2011 two-new-ball rule in ODIs. Each ball is
then in play for about 25 overs instead of 50, so ball age accrues at half the
rate per over of play while the pitch wears at the same rate. If the smooth
gradient were ball ageing, it should halve after the rule. It does not.

Usage::

    python src\\identify_wear.py newball
    python src\\identify_wear.py did
    python src\\identify_wear.py all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import pitch_dynamics as P

NEW_BALL_OVER = 80
ODI_RULE_DATE = pd.Timestamp("2011-10-01")

# Two-over bins around the second new ball. The bin ending at over 79 is the
# reference category, so every coefficient is a change relative to the last two
# overs with the old ball.
EVENT_BINS = (
    ("72_73", 72, 74),
    ("74_75", 74, 76),
    ("76_77", 76, 78),
    ("78_79", 78, 80),
    ("80_81", 80, 82),
    ("82_83", 82, 84),
    ("84_85", 84, 86),
    ("86_87", 86, 88),
)
REFERENCE_BIN = "78_79"


def load_deliveries(
    path: Path,
    match_types: tuple[str, ...],
    over_low: int | None = None,
    over_high: int | None = None,
) -> pd.DataFrame:
    """Load legal spin/pace deliveries with delivery-level outcome flags."""
    columns = [
        "match_id",
        "date",
        "match_type",
        "innings",
        "over",
        "legal",
        "bowler_style",
        "bowler_id",
        "runs_total",
        "runs_batter",
        "is_bowler_wicket",
        "wicket_kind",
        "wickets_before",
        "score_before",
    ]
    frame = pd.read_parquet(
        path,
        columns=columns,
        filters=[
            ("legal", "=", True),
            ("bowler_style", "in", ["spin", "pace"]),
            ("match_type", "in", list(match_types)),
        ],
    )
    frame = frame[frame["match_type"].isin(match_types)]
    if over_low is not None:
        frame = frame[frame["over"] >= over_low]
    if over_high is not None:
        frame = frame[frame["over"] < over_high]
    frame = frame.reset_index(drop=True)
    frame["dot"] = frame["runs_total"].eq(0).astype(float)
    frame["boundary"] = frame["runs_batter"].isin([4, 6]).astype(float)
    frame["bowled_lbw"] = frame["wicket_kind"].isin(["bowled", "lbw"]).astype(float)
    frame["spin"] = frame["bowler_style"].eq("spin").astype(float)
    frame["run_rate_before"] = frame["score_before"].astype(float) / (
        frame["over"].clip(lower=1) * P.BALLS_PER_OVER
    )
    return frame


def _bin_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["bin"] = "none"
    for label, low, high in EVENT_BINS:
        frame.loc[frame["over"].between(low, high - 1), "bin"] = label
    frame["unit"] = (
        frame["match_id"].astype(str)
        + "|"
        + frame["innings"].astype(str)
        + "|"
        + frame["bowler_id"].astype(str)
    )
    return frame


def _event_design(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    columns, names = [], []
    for label, _, _ in EVENT_BINS:
        if label == REFERENCE_BIN:
            continue
        columns.append(frame["bin"].eq(label).astype(float).to_numpy())
        names.append(label)
    columns.append(frame["wickets_before"].to_numpy(float))
    names.append("wickets_before")
    return np.column_stack(columns), names


def event_study(
    frame: pd.DataFrame,
    style: str,
    absorb: str,
    outcome: str = "dot",
) -> pd.DataFrame:
    """Event study around over 80, with or without bowler fixed effects.

    ``absorb='match'`` reproduces the naive comparison: it holds the match fixed
    and lets the bowler mix change. ``absorb='unit'`` holds the bowler and the
    innings fixed, and restricts the sample to units that bowl both with the old
    ball and with the new ball.
    """
    subset = frame[frame["bowler_style"].eq(style) & frame["bin"].ne("none")]
    if absorb == "unit":
        bins = subset.groupby("unit")["bin"].agg(lambda values: set(values))
        keep = bins[
            bins.apply(
                lambda values: REFERENCE_BIN in values and "80_81" in values
            )
        ].index
        subset = subset[subset["unit"].isin(keep)]
        codes = subset["unit"].to_numpy()
    else:
        codes = subset["match_id"].to_numpy()
    design, names = _event_design(subset)
    result = P.within_fit(
        subset[outcome].to_numpy(float),
        design,
        names,
        codes,
        np.ones(len(subset)),
        subset["match_id"].to_numpy(),
    )
    result["style"] = style
    result["sample"] = "within_bowler" if absorb == "unit" else "all_deliveries"
    result["outcome"] = outcome
    result["n"] = len(subset)
    result["units"] = int(subset["unit"].nunique())
    result["matches"] = int(subset["match_id"].nunique())
    return result


def run_new_ball(deliveries: Path, output_dir: Path) -> pd.DataFrame:
    """Contrast the naive and within-bowler second-new-ball event studies."""
    frame = load_deliveries(deliveries, ("Test",), over_low=72, over_high=88)
    frame = _bin_frame(frame)
    print(f"Test deliveries in overs 72-87: {len(frame):,}")

    rows = []
    for style in ("pace", "spin"):
        for absorb in ("match", "unit"):
            for outcome in ("dot", "runs_total", "boundary", "is_bowler_wicket"):
                rows.append(event_study(frame, style, absorb, outcome))
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(output_dir / "wear_new_ball_event.csv", index=False)

    print("\nDot-ball event study relative to overs 78-79")
    print(
        table[
            table["outcome"].eq("dot")
            & table["term"].ne("wickets_before")
        ][
            ["style", "sample", "term", "coefficient", "std_error", "p_value"]
        ]
        .round(4)
        .to_string(index=False)
    )
    return table


def run_two_new_ball(panel_path: Path, output_dir: Path) -> pd.DataFrame:
    """Test whether the ODI age x spin gradient responds to the 2011 rule.

    The four format-era strata are disjoint samples, so the difference-in-
    differences standard error is the square root of the sum of the four
    variances. Separate within-format fits are used instead of one pooled
    interaction model because a single linear age term cannot represent the very
    different age profiles of a 20-over and a 50-over innings.
    """
    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["post"] = panel["date"].ge(ODI_RULE_DATE).astype(int)
    panel = panel[panel["match_type"].isin(["ODI", "T20"])].copy()

    rows = []
    for outcome in ("runs_per_ball", "wicket_rate", "dot_share", "boundary_share"):
        slopes: dict[tuple[str, int], tuple[float, float]] = {}
        for match_type in ("ODI", "T20"):
            for post in (0, 1):
                subset = panel[
                    panel["match_type"].eq(match_type) & panel["post"].eq(post)
                ]
                fitted = P.fit_spec(subset, outcome, "match_fe")
                hit = fitted["terms"][
                    fitted["terms"]["term"].eq("age_x_spin")
                ].iloc[0]
                slopes[(match_type, post)] = (
                    float(hit["coefficient"]),
                    float(hit["std_error"]),
                )
                rows.append(
                    {
                        "outcome": outcome,
                        "stratum": f"{match_type}_{'post' if post else 'pre'}",
                        "coef": round(float(hit["coefficient"]), 6),
                        "se": round(float(hit["std_error"]), 6),
                        "p": round(float(hit["p_value"]), 6),
                        "n_matches": int(subset["match_id"].nunique()),
                        "n_cells": int(len(subset)),
                    }
                )
        odi_change = slopes[("ODI", 1)][0] - slopes[("ODI", 0)][0]
        t20_change = slopes[("T20", 1)][0] - slopes[("T20", 0)][0]
        did = odi_change - t20_change
        did_se = float(np.sqrt(sum(value[1] ** 2 for value in slopes.values())))
        rows.append(
            {
                "outcome": outcome,
                "stratum": "difference_in_differences",
                "coef": round(did, 6),
                "se": round(did_se, 6),
                "ci_low": round(did - 1.96 * did_se, 6),
                "ci_high": round(did + 1.96 * did_se, 6),
                "p": round(float(2 * (1 - _norm_cdf(abs(did / did_se)))), 6),
                "odi_change": round(odi_change, 6),
                "t20_change": round(t20_change, 6),
                "predicted_odi_change_if_halved": round(
                    -0.5 * slopes[("ODI", 0)][0], 6
                ),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "wear_two_new_ball.csv", index=False)
    print("\nTwo-new-ball rule: age x spin by format and era")
    print(
        result[result["stratum"].ne("difference_in_differences")][
            ["outcome", "stratum", "coef", "se", "p"]
        ].to_string(index=False)
    )
    print("\nDifference-in-differences (ODI change minus T20 change):")
    print(
        result[result["stratum"].eq("difference_in_differences")][
            [
                "outcome",
                "coef",
                "se",
                "ci_low",
                "ci_high",
                "p",
                "predicted_odi_change_if_halved",
            ]
        ].to_string(index=False)
    )
    return result


def _norm_cdf(value: float) -> float:
    from math import erf, sqrt

    return 0.5 * (1 + erf(value / sqrt(2)))


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
    parser.add_argument(
        "command", choices=("newball", "did", "all"), nargs="?", default="all"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.command in ("newball", "all"):
        run_new_ball(args.balls, args.output_dir)
    if args.command in ("did", "all"):
        run_two_new_ball(args.panel, args.output_dir)
    print(f"\nArtifacts written to {args.output_dir}")


if __name__ == "__main__":
    main()
