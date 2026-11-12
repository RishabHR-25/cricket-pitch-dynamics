"""Out-of-sample validation of the ball-by-ball wear signal.

Two questions decide whether the wear estimate is a real, match-specific pitch
state rather than an artefact of one estimation sample:

1. **Does it persist across innings?** If the coefficient is measuring the
   surface, a match whose first innings shows a strong late spin advantage
   should also show one in its second innings, on the same pitch. The estimator
   is rerun separately inside each innings and the two coefficients are
   correlated across matches.
2. **Does it predict held-out play?** Adding the age x spin term to a model of
   over-level outcomes is compared with the same model without it, under
   expanding-window folds over dates. Improvements are reported with bootstrap
   confidence intervals clustered on matches.

Both tests are deliberately conservative. The persistence test uses no
fixed effects inside the innings, so its per-match coefficients are noisy; a
positive correlation is therefore a lower bound on persistence. The predictive
test adds a single term, so any gain is necessarily small.

Usage::

    python src\\validate_wear.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

FORMATS = ("T20", "ODI", "Test")
OUTCOMES = ("runs_per_ball", "wicket_rate", "dot_share", "boundary_share")


def weighted_slope(frame: pd.DataFrame, outcome: str) -> float | None:
    """Within-innings age x spin interaction from a weighted OLS fit."""
    if frame["spin"].nunique() < 2:
        return None
    if frame.loc[frame["spin"].eq(1), "balls"].sum() < 12:
        return None
    if frame.loc[frame["spin"].eq(0), "balls"].sum() < 12:
        return None
    design = np.column_stack(
        [
            np.ones(len(frame)),
            frame["age"].to_numpy(dtype=float),
            frame["spin"].to_numpy(dtype=float),
            (frame["age"] * frame["spin"]).to_numpy(dtype=float),
            frame["wickets_before"].to_numpy(dtype=float),
            frame["run_rate_before"].to_numpy(dtype=float),
        ]
    )
    weights = frame["balls"].to_numpy(dtype=float)
    target = frame[outcome].to_numpy(dtype=float)
    xtwx = design.T @ (design * weights[:, None])
    xtwy = design.T @ (weights * target)
    beta, *_ = np.linalg.lstsq(xtwx, xtwy, rcond=None)
    return float(beta[3])


def innings_persistence(panel: pd.DataFrame) -> pd.DataFrame:
    """Correlate first-innings and later-innings wear coefficients."""
    rows = []
    for match_type, subset in panel.groupby("match_type", sort=False):
        first = (
            subset[subset["innings"].eq(1)]
            .groupby(["match_id"], observed=True)
            .apply(lambda frame: weighted_slope(frame, "runs_per_ball"))
            .rename("wear_first")
        )
        second = (
            subset[subset["innings"].eq(2)]
            .groupby(["match_id"], observed=True)
            .apply(lambda frame: weighted_slope(frame, "runs_per_ball"))
            .rename("wear_second")
        )
        merged = pd.concat([first, second], axis=1).dropna()
        if len(merged) < 30:
            continue
        pearson = stats.pearsonr(merged["wear_first"], merged["wear_second"])
        spearman = stats.spearmanr(merged["wear_first"], merged["wear_second"])
        slope = stats.linregress(merged["wear_first"], merged["wear_second"])
        rows.append(
            {
                "match_type": match_type,
                "outcome": "runs_per_ball",
                "matches": int(len(merged)),
                "pearson_r": round(float(pearson.statistic), 4),
                "pearson_p": round(float(pearson.pvalue), 6),
                "spearman_r": round(float(spearman.statistic), 4),
                "spearman_p": round(float(spearman.pvalue), 6),
                "regression_slope": round(float(slope.slope), 4),
                "regression_p": round(float(slope.pvalue), 6),
            }
        )

        # Same test for the wicket outcome, where the sign expectation flips.
        first_w = (
            subset[subset["innings"].eq(1)]
            .groupby(["match_id"], observed=True)
            .apply(lambda frame: weighted_slope(frame, "wicket_rate"))
            .rename("wear_first")
        )
        second_w = (
            subset[subset["innings"].eq(2)]
            .groupby(["match_id"], observed=True)
            .apply(lambda frame: weighted_slope(frame, "wicket_rate"))
            .rename("wear_second")
        )
        merged_w = pd.concat([first_w, second_w], axis=1).dropna()
        if len(merged_w) >= 30:
            pearson = stats.pearsonr(merged_w["wear_first"], merged_w["wear_second"])
            rows.append(
                {
                    "match_type": match_type,
                    "outcome": "wicket_rate",
                    "matches": int(len(merged_w)),
                    "pearson_r": round(float(pearson.statistic), 4),
                    "pearson_p": round(float(pearson.pvalue), 6),
                    "spearman_r": round(float(stats.spearmanr(merged_w["wear_first"], merged_w["wear_second"]).statistic), 4),
                    "spearman_p": round(float(stats.spearmanr(merged_w["wear_first"], merged_w["wear_second"]).pvalue), 6),
                    "regression_slope": round(
                        float(stats.linregress(merged_w["wear_first"], merged_w["wear_second"]).slope), 4
                    ),
                    "regression_p": round(
                        float(stats.linregress(merged_w["wear_first"], merged_w["wear_second"]).pvalue), 6
                    ),
                }
            )
    return pd.DataFrame(rows)


def make_folds(dates: np.ndarray, folds: int, min_train_fraction: float) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    unique = np.sort(np.unique(dates))
    start = int(len(unique) * min_train_fraction)
    edges = np.linspace(start, len(unique), folds + 1).astype(int)
    return [
        (unique[edges[index]], unique[edges[index + 1] - 1])
        for index in range(folds)
    ]


NUMERIC_COLUMNS = (
    "age",
    "spin",
    "wickets_before",
    "run_rate_before",
    "batter_balls",
    "innings",
)


def design_matrices(
    train: pd.DataFrame, test: pd.DataFrame, add_wear: bool
):
    """Sparse design matrices; categorical levels come from the training fold."""
    from scipy import sparse
    from sklearn.preprocessing import OneHotEncoder

    encoder = OneHotEncoder(
        handle_unknown="ignore", min_frequency=20, sparse_output=True
    )
    encoder.fit(train[["phase", "venue_id"]])

    def matrix(frame: pd.DataFrame):
        numeric = frame[list(NUMERIC_COLUMNS)].to_numpy(dtype=float)
        male = frame["gender"].eq("male").to_numpy(dtype=float)[:, None]
        columns = [sparse.csr_matrix(numeric), sparse.csr_matrix(male)]
        if add_wear:
            columns.append(
                sparse.csr_matrix(
                    (frame["age"] * frame["spin"]).to_numpy(dtype=float)[:, None]
                )
            )
        columns.append(encoder.transform(frame[["phase", "venue_id"]]))
        return sparse.hstack(columns).tocsr()

    train_matrix = matrix(train)
    test_matrix = matrix(test)
    scale = np.sqrt(
        np.asarray(train_matrix.multiply(train_matrix).mean(axis=0)).ravel()
    )
    scale[scale == 0] = 1.0
    return (
        train_matrix.multiply(1.0 / scale).tocsr(),
        test_matrix.multiply(1.0 / scale).tocsr(),
    )


def ridge_fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    outcome: str,
    add_wear: bool,
) -> np.ndarray:
    from sklearn.linear_model import Ridge

    x_train, x_test = design_matrices(train, test, add_wear)
    weights = train["balls"].to_numpy(dtype=float)
    model = Ridge(alpha=1.0, solver="lsqr")
    model.fit(
        x_train,
        train[outcome].to_numpy(dtype=float),
        sample_weight=weights,
    )
    return model.predict(x_test)


def weighted_mse(
    frame: pd.DataFrame, prediction: np.ndarray
) -> tuple[float, np.ndarray]:
    weights = frame["balls"].to_numpy(dtype=float)
    residual = frame["outcome"].to_numpy(dtype=float) - prediction
    total = weights.sum()
    mse = float(np.sum(weights * residual**2) / total)
    # Per-match contributions for the bootstrap.
    contribution = pd.Series(weights * residual**2).groupby(
        frame["match_id"].to_numpy()
    ).sum()
    weight_by_match = pd.Series(weights).groupby(frame["match_id"].to_numpy()).sum()
    return mse, (contribution / weight_by_match).to_numpy()


def predictive_gain(panel: pd.DataFrame, folds: int, seed: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for match_type, subset in panel.groupby("match_type", sort=False):
        edges = make_folds(
            pd.to_datetime(subset["date"]).to_numpy(), folds, 0.5
        )
        for outcome in OUTCOMES:
            test_frames: list[pd.DataFrame] = []
            baseline_pred: list[np.ndarray] = []
            wear_pred: list[np.ndarray] = []
            for start, end in edges:
                start = pd.Timestamp(start)
                end = pd.Timestamp(end)
                train = subset[pd.to_datetime(subset["date"]) < start]
                test = subset[
                    (pd.to_datetime(subset["date"]) >= start)
                    & (pd.to_datetime(subset["date"]) <= end)
                ]
                if train.empty or test.empty:
                    continue
                test_frames.append(test)
                baseline_pred.append(
                    ridge_fit_predict(train, test, outcome, add_wear=False)
                )
                wear_pred.append(
                    ridge_fit_predict(train, test, outcome, add_wear=True)
                )
            if not test_frames:
                continue
            frame = pd.concat(test_frames, ignore_index=True)
            frame["outcome"] = frame[outcome]
            base = np.concatenate(baseline_pred)
            wear = np.concatenate(wear_pred)
            base_mse, base_by_match = weighted_mse(frame, base)
            wear_mse, wear_by_match = weighted_mse(frame, wear)
            if len(base_by_match) != len(wear_by_match):
                continue
            delta = base_by_match - wear_by_match
            boot = np.array(
                [
                    rng.choice(delta, size=len(delta), replace=True).mean()
                    for _ in range(1000)
                ]
            )
            rows.append(
                {
                    "match_type": match_type,
                    "outcome": outcome,
                    "test_cells": int(len(frame)),
                    "baseline_weighted_mse": round(base_mse, 6),
                    "wear_weighted_mse": round(wear_mse, 6),
                    "improvement": round(base_mse - wear_mse, 6),
                    "relative_improvement": round(
                        (base_mse - wear_mse) / base_mse, 6
                    ),
                    "ci_low": round(float(np.percentile(boot, 2.5)), 6),
                    "ci_high": round(float(np.percentile(boot, 97.5)), 6),
                    "p_improvement_positive": round(
                        float(np.mean(boot > 0)), 4
                    ),
                }
            )
    return pd.DataFrame(rows)


def rolling_coefficients(panel: pd.DataFrame, folds: int) -> pd.DataFrame:
    """Estimate age x spin on each training and test window separately."""
    from pitch_dynamics import fit_spec

    rows = []
    for match_type, subset in panel.groupby("match_type", sort=False):
        edges = make_folds(
            pd.to_datetime(subset["date"]).to_numpy(), folds, 0.5
        )
        for index, (start, end) in enumerate(edges, start=1):
            train = subset[pd.to_datetime(subset["date"]) < pd.Timestamp(start)]
            test = subset[
                (pd.to_datetime(subset["date"]) >= pd.Timestamp(start))
                & (pd.to_datetime(subset["date"]) <= pd.Timestamp(end))
            ]
            if train.empty or test.empty:
                continue
            for outcome in OUTCOMES:
                train_fit = fit_spec(train, outcome, "match_fe")["terms"]
                test_fit = fit_spec(test, outcome, "match_fe")["terms"]
                train_hit = train_fit[train_fit["term"].eq("age_x_spin")]
                test_hit = test_fit[test_fit["term"].eq("age_x_spin")]
                if train_hit.empty or test_hit.empty:
                    continue
                rows.append(
                    {
                        "match_type": match_type,
                        "fold": index,
                        "outcome": outcome,
                        "train_coef": round(float(train_hit.iloc[0]["coefficient"]), 5),
                        "train_p": round(float(train_hit.iloc[0]["p_value"]), 5),
                        "test_coef": round(float(test_hit.iloc[0]["coefficient"]), 5),
                        "test_p": round(float(test_hit.iloc[0]["p_value"]), 5),
                        "same_sign": bool(
                            train_hit.iloc[0]["coefficient"]
                            * test_hit.iloc[0]["coefficient"]
                            > 0
                        ),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel",
        type=Path,
        default=root / "models/pitch_dynamics/over_panel.parquet",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "models/pitch_dynamics"
    )
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(args.panel)
    panel["date"] = pd.to_datetime(panel["date"])
    print(f"Loaded {len(panel):,} over cells from {args.panel}")

    persistence = innings_persistence(panel)
    persistence.to_csv(args.output_dir / "wear_persistence.csv", index=False)
    print("\nCross-innings persistence:")
    print(persistence.to_string(index=False))

    gains = predictive_gain(panel, args.folds, args.seed)
    gains.to_csv(args.output_dir / "wear_predictive_gain.csv", index=False)
    print("\nOut-of-sample predictive gain (baseline minus wear MSE):")
    print(gains.to_string(index=False))

    coefficients = rolling_coefficients(panel, args.folds)
    coefficients.to_csv(
        args.output_dir / "wear_rolling_coefficients.csv", index=False
    )
    print("\nRolling coefficient stability:")
    print(coefficients.to_string(index=False))


if __name__ == "__main__":
    main()
