"""Estimate within-innings pitch dynamics from ball-by-ball outcomes.

Two linked pieces of evidence:

1. **Wear curves.** For each format and bowler style, the scoring rate, dot-ball
   share and bowler-credited wicket rate are plotted against ball age in
   two-over bins, with Wilson intervals. This is descriptive and assumption-light.
2. **Wear regression.** A weighted linear probability model at the over level
   estimates whether the spin-versus-pace gap changes as the innings ages,
   controlling for phase, wickets lost, current run rate, innings, gender and
   venue fixed effects. Standard errors are clustered by match.

The identifying assumption is not that the pitch is measured; it is that, within
a phase and a venue, a change in the *relative* effectiveness of spin and pace
as the ball ages is a wear signal. Alternative explanations (tactics, batter
risk) are discussed in the report, not hidden.

Usage::

    python src\\analyze_pitch_dynamics.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

FORMATS = ("T20", "ODI", "Test")
AGE_BIN_OVERS = 2


def wilson_interval(successes: float, total: float, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (np.nan, np.nan)
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def wear_curves(balls: pd.DataFrame) -> pd.DataFrame:
    frame = balls.copy()
    frame["age_bin"] = (
        np.floor(frame["ball_age"] / (6 * AGE_BIN_OVERS)) * AGE_BIN_OVERS
    ).astype(int)
    frame["dot"] = frame["runs_total"].eq(0)
    frame["boundary"] = frame["runs_batter"].isin([4, 6])
    grouped = frame.groupby(
        ["match_type", "bowler_style", "age_bin"], observed=True
    ).agg(
        balls=("legal", "size"),
        runs=("runs_total", "sum"),
        wickets=("is_bowler_wicket", "sum"),
        dots=("dot", "sum"),
        boundaries=("boundary", "sum"),
    )
    grouped = grouped.reset_index()
    grouped["runs_per_ball"] = grouped["runs"] / grouped["balls"]
    grouped["wicket_rate"] = grouped["wickets"] / grouped["balls"]
    grouped["dot_share"] = grouped["dots"] / grouped["balls"]
    grouped["boundary_share"] = grouped["boundaries"] / grouped["balls"]
    intervals = grouped.apply(
        lambda row: wilson_interval(row["wickets"], row["balls"]), axis=1
    )
    grouped["wicket_ci_low"] = [interval[0] for interval in intervals]
    grouped["wicket_ci_high"] = [interval[1] for interval in intervals]
    dot_intervals = grouped.apply(
        lambda row: wilson_interval(row["dots"], row["balls"]), axis=1
    )
    grouped["dot_ci_low"] = [interval[0] for interval in dot_intervals]
    grouped["dot_ci_high"] = [interval[1] for interval in dot_intervals]
    return grouped


def over_panel(balls: pd.DataFrame) -> pd.DataFrame:
    """Aggregate legal balls to match x innings x over x bowler-style cells."""
    frame = balls[balls["legal"] & balls["bowler_style"].isin(["spin", "pace"])].copy()
    frame["wickets_before"] = frame["wickets_before"].fillna(0)
    panel = (
        frame.groupby(
            [
                "match_id",
                "match_type",
                "venue_id",
                "gender",
                "innings",
                "over",
                "bowler_style",
            ],
            observed=True,
        )
        .agg(
            balls=("legal", "size"),
            runs=("runs_total", "sum"),
            wickets=("is_bowler_wicket", "sum"),
            wickets_before=("wickets_before", "mean"),
            score_before=("score_before", "mean"),
        )
        .reset_index()
    )
    panel = panel[panel["balls"] >= 3].copy()
    panel["runs_per_ball"] = panel["runs"] / panel["balls"]
    panel["wicket_rate"] = panel["wickets"] / panel["balls"]
    panel["spin"] = panel["bowler_style"].eq("spin").astype(float)
    panel["age"] = panel["over"].astype(float) + 0.5
    panel["run_rate_before"] = panel["score_before"] / (panel["over"].clip(lower=1) * 6)
    return panel


def design_matrix(panel: pd.DataFrame) -> tuple[sparse.csr_matrix, list[str]]:
    frames = []
    names: list[str] = []

    def add(values: np.ndarray, name: str) -> None:
        frames.append(sparse.csr_matrix(values.reshape(-1, 1)))
        names.append(name)

    add(np.ones(len(panel)), "intercept")
    add(panel["age"].to_numpy(dtype=float), "age_over")
    add(panel["spin"].to_numpy(dtype=float), "spin")
    add((panel["age"] * panel["spin"]).to_numpy(dtype=float), "age_x_spin")
    add(panel["wickets_before"].to_numpy(dtype=float), "wickets_before")
    add(panel["run_rate_before"].to_numpy(dtype=float), "run_rate_before")
    add(panel["gender"].eq("male").to_numpy(dtype=float), "male")
    for phase in sorted(panel["_phase"].unique())[1:]:
        add(panel["_phase"].eq(phase).to_numpy(dtype=float), f"phase_{phase}")
    for innings in sorted(panel["innings"].unique())[1:]:
        add(panel["innings"].eq(innings).to_numpy(dtype=float), f"innings_{innings}")
    venue = pd.get_dummies(
        panel["venue_id"], sparse=True, dtype=float, drop_first=True
    )
    venue_names = [f"venue_{column}" for column in venue.columns]
    frames.append(sparse.csr_matrix(venue.to_numpy()))
    names.extend(venue_names)
    return sparse.hstack(frames).tocsr(), names


def cluster_robust_wls(
    X: sparse.csr_matrix,
    y: np.ndarray,
    weights: np.ndarray,
    clusters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted least squares with one-way cluster-robust standard errors."""
    weighted = X.multiply(weights[:, None])
    xtwx = (X.T @ weighted).toarray()
    xtwy = X.T @ (weights * y)
    # Pivoted least squares instead of solve(): a small format slice can have an
    # exactly collinear nuisance column (for example a constant venue dummy).
    beta, *_ = np.linalg.lstsq(xtwx, xtwy, rcond=None)
    residuals = y - X @ beta
    scores = X.multiply(residuals[:, None])
    codes, uniques = pd.factorize(clusters)
    indicator = sparse.csr_matrix(
        (np.ones(len(codes)), (np.arange(len(codes)), codes)),
        shape=(len(codes), len(uniques)),
    )
    cluster_scores = indicator.T @ scores
    meat = cluster_scores.T @ cluster_scores
    bread = np.linalg.pinv(xtwx)
    covariance = bread @ meat @ bread
    n, p = X.shape
    groups = len(uniques)
    covariance *= groups / (groups - 1) * (n - 1) / (n - p)
    variance = np.clip(np.diag(covariance), 0.0, None)
    return beta, np.sqrt(variance)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--balls", type=Path, default=root / "data/interim/balls.parquet"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "models/pitch_dynamics"
    )
    parser.add_argument(
        "--formats", nargs="+", default=list(FORMATS), choices=list(FORMATS)
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    columns = [
        "match_id",
        "date",
        "match_type",
        "venue_id",
        "gender",
        "innings",
        "over",
        "ball_age",
        "legal",
        "bowler_style",
        "runs_batter",
        "runs_total",
        "is_bowler_wicket",
        "wickets_before",
        "score_before",
    ]
    balls = pd.read_parquet(args.balls, columns=columns)
    balls = balls[balls["match_type"].isin(args.formats)]
    print(f"Loaded {len(balls):,} deliveries")
    print(balls["bowler_style"].value_counts().to_string())

    curves = wear_curves(balls)
    curves.to_csv(args.output_dir / "wear_curves.csv", index=False)

    panel = over_panel(balls)
    panel["_phase"] = np.select(
        [
            panel["match_type"].eq("T20") & panel["over"].lt(6),
            panel["match_type"].eq("T20") & panel["over"].lt(15),
            panel["match_type"].eq("T20"),
            panel["match_type"].eq("ODI") & panel["over"].lt(10),
            panel["match_type"].eq("ODI") & panel["over"].lt(40),
            panel["match_type"].eq("ODI"),
            panel["match_type"].eq("Test") & panel["over"].lt(30),
            panel["match_type"].eq("Test") & panel["over"].lt(80),
            panel["match_type"].eq("Test"),
        ],
        [
            "powerplay",
            "middle",
            "death",
            "powerplay",
            "middle",
            "death",
            "new_ball",
            "middle",
            "old_ball",
        ],
        default="other",
    )

    rows = []
    for match_type in args.formats:
        subset = panel[panel["match_type"].eq(match_type)]
        if subset.empty:
            continue
        names = None
        for outcome, label in (
            ("runs_per_ball", "runs_per_ball"),
            ("wicket_rate", "wicket_rate"),
        ):
            result = []
            X, names = design_matrix(subset)
            y = subset[outcome].to_numpy(dtype=float)
            beta, se = cluster_robust_wls(
                X,
                y,
                subset["balls"].to_numpy(dtype=float),
                subset["match_id"].to_numpy(),
            )
            focus = {
                "age_over": "Change per over, pace",
                "spin": "Spin-pace gap at over 0",
                "age_x_spin": "Extra change per over for spin",
            }
            for index, name in enumerate(names):
                if name in focus:
                    from scipy.stats import norm

                    result.append(
                        {
                            "match_type": match_type,
                            "outcome": label,
                            "term": focus[name],
                            "coefficient": round(float(beta[index]), 5),
                            "std_error": round(float(se[index]), 5),
                            "ci_low": round(float(beta[index] - 1.96 * se[index]), 5),
                            "ci_high": round(float(beta[index] + 1.96 * se[index]), 5),
                            "p_value": round(
                                float(2 * (1 - norm.cdf(abs(beta[index] / se[index])))),
                                5,
                            )
                            if se[index] > 0
                            else np.nan,
                        }
                    )
            rows.extend(result)
            print(f"{match_type} {label}: fitted on {len(subset):,} cells")

    regression = pd.DataFrame(rows)
    regression.to_csv(args.output_dir / "wear_regression.csv", index=False)
    print(regression.to_string(index=False))
    print(f"Wrote {args.output_dir / 'wear_curves.csv'}")
    print(f"Wrote {args.output_dir / 'wear_regression.csv'}")


if __name__ == "__main__":
    main()
