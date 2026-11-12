"""Estimate within-match pitch wear from ball-by-ball outcomes.

The paper's central claim is that a pitch changes measurably *during* a match,
and that the change can be inferred from how the relative effectiveness of spin
and pace evolves with ball age. This module implements the identification
strategy behind that claim.

Inference problem
-----------------
A naive comparison of spin and pace at different ball ages confounds three
things: the pitch state, the tactical use of each bowling type (spinners rarely
bowl the death), and bowler quality. We separate them as far as open data
allows:

* **Match fixed effects.** All comparisons are made *within a match*, so venue,
  climate, format, team strength and date are absorbed exactly.
* **Phase controls and phase-specific age slopes.** Powerplay, middle and death
  are allowed their own level and their own age trend, so the age x spin term is
  not driven by the known phase composition of each attack.
* **Innings-level fixed effects.** A stricter specification compares cells only
  within the same innings of the same match.
* **Batter-experience and match-state controls.** Mean balls faced by the batter,
  wickets lost and the scoring rate before the over are included because they
  move with innings age.
* **Cluster-robust standard errors** at the match level throughout.

Outcomes
--------
Four complementary outcomes are estimated, because no single one is a clean
pitch measure:

* runs per ball (lower is better for the bowler)
* wicket rate (higher is better)
* dot-ball share (higher is better)
* boundary share (lower is better)

If wear is real, spin's *relative* performance should improve with ball age on
all four: age x spin negative for runs and boundaries, positive for wickets and
dots.

Supporting tests
----------------
* **Cross-innings reuse.** In the same match, is the spin advantage larger in
  innings two and later than in innings one? This is a direct same-surface test
  that does not rely on the age gradient alone.
* **Placebo.** Style labels are permuted within matches and the estimator is
  refitted; the observed coefficient is compared with this null distribution.
* **Convergent validity.** The age x spin effect is re-estimated on venue
  terciles ranked by their historical spin-wicket share. A genuine wear signal
  should be stronger where spin already dominates.
* **Moisture link.** The effect is re-estimated on matches split by pre-match dry
  days and prior rainfall. Drier, older surfaces should wear more.

Usage::

    python src\\pitch_dynamics.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

FORMATS = ("T20", "ODI", "Test")
STYLES = ("spin", "pace")
OUTCOMES = ("runs_per_ball", "wicket_rate", "dot_share", "boundary_share")

# Balls per over is six everywhere in the archive, including The Hundred
# conversions already normalised by Cricsheet to six-ball overs.
BALLS_PER_OVER = 6


def load_panel(path: Path, min_bowler_balls: int = 0) -> pd.DataFrame:
    """Aggregate legal deliveries to match x innings x over x style cells."""
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
        "bowler_id",
        "batter_id",
        "runs_batter",
        "runs_total",
        "is_bowler_wicket",
        "wickets_before",
        "score_before",
    ]
    balls = pd.read_parquet(
        path,
        columns=columns,
        filters=[
            ("legal", "=", True),
            ("bowler_style", "in", list(STYLES)),
        ],
    )
    print(f"Loaded {len(balls):,} legal spin/pace deliveries")

    # Total legal balls already bowled in the match, including deliveries whose
    # bowler style is unknown. This is the physically correct wear clock for
    # Test cricket, where later innings inherit the surface used earlier.
    order = ["match_id", "innings", "ball_age"]
    counts = pd.read_parquet(
        path,
        columns=["match_id", "innings", "ball_age"],
        filters=[("legal", "=", True)],
    ).sort_values(order, kind="stable")
    counts["match_ball"] = counts.groupby("match_id", observed=True).cumcount()
    balls = balls.merge(
        counts[order + ["match_ball"]],
        on=order,
        how="left",
        validate="many_to_one",
    )

    balls = balls[balls["match_type"].isin(FORMATS)]
    if min_bowler_balls > 0:
        volume = balls.groupby("bowler_id", observed=True).size()
        established = set(volume[volume >= min_bowler_balls].index)
        balls = balls[balls["bowler_id"].isin(established)]
        print(
            f"Restricted to {len(established):,} bowlers with at least "
            f"{min_bowler_balls} legal balls ({len(balls):,} deliveries)"
        )

    # Batter experience: legal balls the striker has already faced this innings.
    balls = balls.sort_values(
        ["match_id", "innings", "ball_age"], kind="stable"
    )
    balls["batter_balls"] = (
        balls.groupby(["match_id", "innings", "batter_id"], observed=True)
        .cumcount()
        .astype("float32")
    )

    panel = (
        balls.groupby(
            [
                "match_id",
                "date",
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
            dots=("runs_total", lambda values: int((values == 0).sum())),
            boundaries=("runs_batter", lambda values: int(values.isin([4, 6]).sum())),
            wickets_before=("wickets_before", "mean"),
            run_rate_before=("score_before", "mean"),
            batter_balls=("batter_balls", "mean"),
            match_ball=("match_ball", "mean"),
        )
        .reset_index()
    )
    panel = panel[panel["balls"] >= 3].copy()
    panel["runs_per_ball"] = panel["runs"] / panel["balls"]
    panel["wicket_rate"] = panel["wickets"] / panel["balls"]
    panel["dot_share"] = panel["dots"] / panel["balls"]
    panel["boundary_share"] = panel["boundaries"] / panel["balls"]
    panel["spin"] = panel["bowler_style"].eq("spin").astype(float)
    panel["age"] = panel["over"].astype(float) + 0.5
    panel["match_overs"] = panel["match_ball"] / BALLS_PER_OVER
    panel["later_innings"] = panel["innings"].ge(2).astype(float)
    panel["run_rate_before"] = panel["run_rate_before"] / (
        panel["over"].clip(lower=1) * BALLS_PER_OVER
    )
    panel["phase"] = assign_phase(panel)
    print(f"Aggregated to {len(panel):,} over x style cells")
    return panel


def assign_phase(panel: pd.DataFrame) -> np.ndarray:
    over = panel["over"].to_numpy()
    match_type = panel["match_type"].to_numpy()
    phase = np.full(len(panel), "middle", dtype=object)
    t20 = match_type == "T20"
    odi = match_type == "ODI"
    test = match_type == "Test"
    phase[t20 & (over < 6)] = "powerplay"
    phase[t20 & (over >= 15)] = "death"
    phase[odi & (over < 10)] = "powerplay"
    phase[odi & (over >= 40)] = "death"
    phase[test & (over < 30)] = "new_ball"
    phase[test & (over >= 80)] = "old_ball"
    return phase


def _weighted_group_demean(
    values: np.ndarray,
    codes: np.ndarray,
    weights: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    """Subtract weighted group means from every column of ``values``."""
    if values.ndim == 1:
        values = values[:, None]
        squeeze = True
    else:
        squeeze = False
    totals = np.bincount(codes, weights=weights, minlength=n_groups)
    totals = np.where(totals == 0, 1.0, totals)
    out = np.empty_like(values, dtype=float)
    for column in range(values.shape[1]):
        sums = np.bincount(
            codes, weights=weights * values[:, column], minlength=n_groups
        )
        out[:, column] = values[:, column] - (sums / totals)[codes]
    return out[:, 0] if squeeze else out


def within_fit(
    y: np.ndarray,
    design: np.ndarray,
    column_names: list[str],
    absorb: np.ndarray,
    weights: np.ndarray,
    clusters: np.ndarray,
) -> pd.DataFrame:
    """Weighted least squares with absorbed fixed effects and clustered SEs.

    ``absorb`` is an integer code per row identifying the fixed-effect group.
    Estimation partials the group effects out with the weighted within
    transformation (Frisch-Waugh-Lovell), then runs weighted least squares on the
    demeaned data. Standard errors are one-way cluster-robust.
    """
    codes, uniques = pd.factorize(absorb)
    n_groups = len(uniques)
    n = len(y)

    y_tilde = _weighted_group_demean(y, codes, weights, n_groups)
    x_tilde = _weighted_group_demean(design, codes, weights, n_groups)

    # Drop exactly collinear columns after demeaning (a common nuisance is a
    # phase dummy that is constant within a match, e.g. a Test new-ball phase).
    keep: list[int] = []
    for column in range(x_tilde.shape[1]):
        candidate = x_tilde[:, column]
        norm = np.linalg.norm(candidate)
        if norm <= 1e-12:
            continue
        if keep:
            basis = x_tilde[:, keep]
            projection = basis @ np.linalg.lstsq(basis, candidate, rcond=None)[0]
            residual = candidate - projection
            if np.linalg.norm(residual) / norm < 1e-8:
                continue
        keep.append(column)
    x_tilde = x_tilde[:, keep]
    names = [column_names[index] for index in keep]

    xtwx = x_tilde.T @ (x_tilde * weights[:, None])
    xtwy = x_tilde.T @ (weights * y_tilde)
    beta, *_ = np.linalg.lstsq(xtwx, xtwy, rcond=None)
    residuals = y_tilde - x_tilde @ beta

    scores = x_tilde * (weights * residuals)[:, None]
    cluster_codes, cluster_labels = pd.factorize(clusters)
    indicator = sparse.csr_matrix(
        (np.ones(len(cluster_codes)), (np.arange(len(cluster_codes)), cluster_codes)),
        shape=(len(cluster_codes), len(cluster_labels)),
    )
    cluster_scores = indicator.T @ scores
    meat = cluster_scores.T @ cluster_scores
    bread = np.linalg.pinv(xtwx)
    covariance = bread @ meat @ bread

    n_columns = x_tilde.shape[1] + n_groups
    groups = len(cluster_labels)
    if groups > 1 and n > n_columns:
        covariance *= groups / (groups - 1) * (n - 1) / (n - n_columns)
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))

    from scipy.stats import norm

    rows = []
    for index, name in enumerate(names):
        coefficient = float(beta[index])
        error = float(standard_errors[index])
        z_value = coefficient / error if error > 0 else np.nan
        rows.append(
            {
                "term": name,
                "coefficient": coefficient,
                "std_error": error,
                "ci_low": coefficient - 1.96 * error,
                "ci_high": coefficient + 1.96 * error,
                "p_value": float(2 * (1 - norm.cdf(abs(z_value))))
                if error > 0
                else np.nan,
                "n_cells": n,
                "n_clusters": groups,
            }
        )
    return pd.DataFrame(rows)


def build_design(
    panel: pd.DataFrame,
    spec: str,
    age_column: str = "age",
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Return the design matrix, column names and fixed-effect codes."""
    age = panel[age_column].to_numpy(dtype=float)
    base = [
        np.ones(len(panel)),
        age,
        panel["spin"].to_numpy(dtype=float),
        (age * panel["spin"]).to_numpy(dtype=float),
        panel["wickets_before"].to_numpy(dtype=float),
        panel["run_rate_before"].to_numpy(dtype=float),
        panel["batter_balls"].to_numpy(dtype=float),
        panel["gender"].eq("male").to_numpy(dtype=float),
    ]
    names = [
        "intercept",
        "age",
        "spin",
        "age_x_spin",
        "wickets_before",
        "run_rate_before",
        "batter_balls",
        "male",
    ]
    phases = sorted(panel["phase"].unique())
    if spec in {
        "venue_fe",
        "match_fe",
        "match_innings_fe",
        "phase_slopes",
        "cross_innings",
        "match_use",
    }:
        for phase in phases[1:]:
            base.append(panel["phase"].eq(phase).to_numpy(dtype=float))
            names.append(f"phase_{phase}")
    if spec == "phase_slopes":
        for phase in phases[1:]:
            base.append(
                (panel["phase"].eq(phase) * panel["spin"]).to_numpy(dtype=float)
            )
            names.append(f"spin_x_phase_{phase}")
        for phase in phases[1:]:
            base.append(
                (panel["phase"].eq(phase) * panel["age"]).to_numpy(dtype=float)
            )
            names.append(f"age_x_phase_{phase}")
        for phase in phases[1:]:
            base.append(
                (panel["phase"].eq(phase) * panel["age"] * panel["spin"]).to_numpy(
                    dtype=float
                )
            )
            names.append(f"age_x_spin_x_phase_{phase}")
        base.append((panel["age"] ** 2).to_numpy(dtype=float))
        names.append("age_squared")
        base.append(((panel["age"] ** 2) * panel["spin"]).to_numpy(dtype=float))
        names.append("age_squared_x_spin")
    if spec == "cross_innings":
        base.append(panel["later_innings"].to_numpy(dtype=float))
        names.append("later_innings")
        base.append(
            (panel["later_innings"] * panel["spin"]).to_numpy(dtype=float)
        )
        names.append("later_innings_x_spin")

    if spec == "venue_fe":
        absorb = panel["venue_id"].to_numpy()
    elif spec in {"match_fe", "phase_slopes", "cross_innings", "match_use"}:
        absorb = panel["match_id"].to_numpy()
    elif spec == "match_innings_fe":
        absorb = (
            panel["match_id"].astype(str) + "|" + panel["innings"].astype(str)
        ).to_numpy()
    elif spec == "venue_only":
        absorb = panel["venue_id"].to_numpy()
    else:
        raise ValueError(f"unknown specification {spec!r}")
    return np.column_stack(base), names, absorb


def fit_spec(
    panel: pd.DataFrame,
    outcome: str,
    spec: str,
    age_column: str = "age",
) -> dict[str, Any]:
    design, names, absorb = build_design(panel, spec, age_column)
    result = within_fit(
        panel[outcome].to_numpy(dtype=float),
        design,
        names,
        absorb,
        panel["balls"].to_numpy(dtype=float),
        panel["match_id"].to_numpy(),
    )
    focus = result[result["term"].isin(
        ["age_x_spin", "later_innings_x_spin", "age", "spin"]
    )].copy()
    return {"spec": spec, "outcome": outcome, "terms": focus, "all": result}

def run_battery(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # ``phase_slopes`` interacts phase with age, age squared and spin. It is
    # kept for completeness but is numerically fragile because the quadratic
    # terms are nearly collinear with the absorbed match effects; the stable
    # phase decomposition lives in ``wear_robustness.py``, which refits the
    # headline specification inside each phase separately.
    specs = [
        ("venue_fe", "age"),
        ("match_fe", "age"),
        ("match_innings_fe", "age"),
        ("phase_slopes", "age"),
        ("match_use", "match_overs"),
    ]
    for match_type, subset in panel.groupby("match_type", sort=False):
        for outcome in OUTCOMES:
            for spec, age_column in specs:
                fitted = fit_spec(subset, outcome, spec, age_column)
                focus = fitted["terms"]
                row = {
                    "match_type": match_type,
                    "outcome": outcome,
                    "spec": spec,
                    "age_variable": age_column,
                    "n_cells": int(len(subset)),
                }
                for term in ("age", "spin", "age_x_spin"):
                    hit = focus[focus["term"].eq(term)]
                    if hit.empty:
                        row[f"{term}_coef"] = np.nan
                        continue
                    value = hit.iloc[0]
                    row[f"{term}_coef"] = round(float(value["coefficient"]), 5)
                    row[f"{term}_se"] = round(float(value["std_error"]), 5)
                    row[f"{term}_p"] = round(float(value["p_value"]), 5)
                # Phase-specific age x spin slopes, when the spec estimates them.
                for _, value in fitted["all"].iterrows():
                    term = str(value["term"])
                    if term.startswith("age_x_spin_x_phase_"):
                        row[term] = round(float(value["coefficient"]), 5)
                        row[f"{term}_p"] = round(float(value["p_value"]), 5)
                rows.append(row)
    return pd.DataFrame(rows)


def run_cross_innings(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for match_type, subset in panel.groupby("match_type", sort=False):
        if match_type not in FORMATS:
            continue
        for outcome in OUTCOMES:
            fitted = fit_spec(subset, outcome, "cross_innings")
            focus = fitted["terms"]
            row = {"match_type": match_type, "outcome": outcome, "n_cells": len(subset)}
            for term in ("later_innings", "later_innings_x_spin"):
                hit = focus[focus["term"].eq(term)]
                if hit.empty:
                    continue
                value = hit.iloc[0]
                row[f"{term}_coef"] = round(float(value["coefficient"]), 5)
                row[f"{term}_se"] = round(float(value["std_error"]), 5)
                row[f"{term}_p"] = round(float(value["p_value"]), 5)
            rows.append(row)
    return pd.DataFrame(rows)


def placebo_test(
    panel: pd.DataFrame,
    outcome: str,
    spec: str,
    draws: int,
    seed: int,
    group: tuple[str, ...] = ("match_id", "innings"),
) -> dict[str, Any]:
    """Permute bowling style within innings and refit the age x spin term.

    This asks whether the observed age x spin association is stronger than the
    association produced by randomly allocating styles to the overs actually
    bowled in each innings. Permuting within an innings (rather than across a
    whole match) keeps the innings-level composition of spin and pace fixed, so
    the test isolates the within-innings age gradient. It is a randomisation
    test for the selection channel, not a test of the physical pitch; the report
    says so explicitly.
    """
    observed_fit = fit_spec(panel, outcome, spec)
    observed = observed_fit["terms"]
    observed = observed[observed["term"].eq("age_x_spin")].iloc[0]
    rng = np.random.default_rng(seed)
    null: list[float] = []
    for _ in range(draws):
        shuffled = panel.copy()
        shuffled["spin"] = (
            shuffled.groupby(list(group), observed=True)["spin"]
            .transform(lambda values: rng.permutation(values.to_numpy()))
            .to_numpy()
        )
        fitted = fit_spec(shuffled, outcome, spec)
        hit = fitted["terms"][fitted["terms"]["term"].eq("age_x_spin")]
        if not hit.empty:
            null.append(float(hit.iloc[0]["coefficient"]))
    null_array = np.array(null)
    coefficient = float(observed["coefficient"])
    # Two-sided empirical p-value.
    p_value = float(np.mean(np.abs(null_array) >= abs(coefficient)))
    return {
        "observed": coefficient,
        "null_mean": float(null_array.mean()),
        "null_sd": float(null_array.std()),
        "p_value": p_value,
        "draws": draws,
        "spec": spec,
        "permutation_unit": "|".join(group),
        "null_quantiles": {
            "2.5%": float(np.percentile(null_array, 2.5)),
            "97.5%": float(np.percentile(null_array, 97.5)),
        },
    }


def venue_heterogeneity(panel: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Re-estimate wear on venues ranked by historical spin-wicket share."""
    # Spin share of bowler-credited wickets at each venue, from the cells.
    spin_cells = panel[panel["spin"].eq(1)]
    pace_cells = panel[panel["spin"].eq(0)]
    spin_rate = spin_cells.groupby("venue_id", observed=True)["wickets"].sum() / (
        spin_cells.groupby("venue_id", observed=True)["balls"].sum().clip(lower=1)
    )
    pace_rate = pace_cells.groupby("venue_id", observed=True)["wickets"].sum() / (
        pace_cells.groupby("venue_id", observed=True)["balls"].sum().clip(lower=1)
    )
    share = (spin_rate / (spin_rate + pace_rate)).dropna()
    counts = panel.groupby("venue_id", observed=True)["balls"].sum()
    share = share[counts.reindex(share.index).fillna(0) >= 5000]
    if share.empty:
        return pd.DataFrame()
    rank = pd.qcut(share.rank(method="first"), 3, labels=["low", "mid", "high"])
    mapping = rank.to_dict()
    rows = []
    for label in ("low", "mid", "high"):
        venues = [venue for venue, group in mapping.items() if group == label]
        if not venues:
            continue
        subset = panel[panel["venue_id"].isin(venues)]
        if subset["match_id"].nunique() < 20:
            continue
        fitted = fit_spec(subset, outcome, "match_fe")
        hit = fitted["terms"][fitted["terms"]["term"].eq("age_x_spin")]
        if hit.empty:
            continue
        value = hit.iloc[0]
        rows.append(
            {
                "spin_friendliness": label,
                "diagnostic": "age_x_spin",
                "outcome": outcome,
                "coefficient": round(float(value["coefficient"]), 5),
                "std_error": round(float(value["std_error"]), 5),
                "p_value": round(float(value["p_value"]), 5),
                "n_cells": int(len(subset)),
                "n_venues": len(venues),
                "n_matches": int(subset["match_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def moisture_split(
    panel: pd.DataFrame,
    weather_path: Path,
    outcome: str,
) -> pd.DataFrame:
    """Estimate wear separately for dry and wet pre-match conditions."""
    weather = pd.read_csv(weather_path, low_memory=False)
    keep = ["match_id", "precip_sum_14d", "precip_sum_28d", "dry_days_before", "temp_mean_7d"]
    weather = weather[[column for column in keep if column in weather.columns]]
    weather["match_id"] = weather["match_id"].astype(str)
    frame = panel.copy()
    frame["match_id"] = frame["match_id"].astype(str)
    frame = frame.merge(weather, on="match_id", how="left")
    rows = []
    # Median splits on each moisture descriptor.
    for column in ("dry_days_before", "precip_sum_14d", "precip_sum_28d"):
        if column not in frame.columns or frame[column].isna().all():
            continue
        cut = frame[column].median()
        for label, subset in (
            ("below_median", frame[frame[column] <= cut]),
            ("above_median", frame[frame[column] > cut]),
        ):
            if subset["match_id"].nunique() < 50:
                continue
            fitted = fit_spec(subset, outcome, "match_fe")
            hit = fitted["terms"][fitted["terms"]["term"].eq("age_x_spin")]
            if hit.empty:
                continue
            value = hit.iloc[0]
            rows.append(
                {
                    "descriptor": column,
                    "split": label,
                    "median": round(float(cut), 3),
                    "outcome": outcome,
                    "coefficient": round(float(value["coefficient"]), 5),
                    "std_error": round(float(value["std_error"]), 5),
                    "p_value": round(float(value["p_value"]), 5),
                    "n_cells": int(len(subset)),
                    "n_matches": int(subset["match_id"].nunique()),
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
        "--weather", type=Path, default=root / "data/interim/match_weather.csv"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "models/pitch_dynamics"
    )
    parser.add_argument("--placebo-draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panel = load_panel(args.balls)
    panel.to_parquet(args.output_dir / "over_panel.parquet", index=False)
    panel.groupby(["match_type", "phase"], observed=True).size().to_csv(
        args.output_dir / "panel_phase_counts.csv"
    )

    battery = run_battery(panel)
    battery.to_csv(args.output_dir / "wear_specifications.csv", index=False)
    print("\nSpecification battery (age x spin):")
    print(
        battery[
            ["match_type", "outcome", "spec", "age_x_spin_coef", "age_x_spin_p"]
        ].to_string(index=False)
    )

    cross = run_cross_innings(panel)
    cross.to_csv(args.output_dir / "wear_cross_innings.csv", index=False)
    print("\nCross-innings reuse (later innings x spin):")
    print(cross.to_string(index=False))

    # Placebo: the same estimator on randomly reallocated styles, for every
    # outcome and format, so the null distribution is transparent. 100 draws
    # per cell is enough to read the p-value to about 0.01.
    placebo_rows = []
    for match_type, subset in panel.groupby("match_type", sort=False):
        for outcome in OUTCOMES:
            result = placebo_test(
                subset, outcome, "match_innings_fe", args.placebo_draws, args.seed
            )
            result.update({"match_type": match_type, "outcome": outcome})
            placebo_rows.append(result)
            print(
                f"Placebo {match_type} {outcome}: observed "
                f"{result['observed']:.5f}, null sd {result['null_sd']:.5f}, "
                f"p {result['p_value']:.3f}"
            )
    with (args.output_dir / "wear_placebo.json").open("w", encoding="utf-8") as file:
        json.dump(placebo_rows, file, indent=2)

    venues = []
    for match_type, subset in panel.groupby("match_type", sort=False):
        for outcome in OUTCOMES:
            result = venue_heterogeneity(subset, outcome)
            if not result.empty:
                result["match_type"] = match_type
                venues.append(result)
    if venues:
        venue_table = pd.concat(venues, ignore_index=True)
        venue_table.to_csv(
            args.output_dir / "wear_venue_heterogeneity.csv", index=False
        )
        print("\nVenue heterogeneity:")
        print(venue_table.to_string(index=False))

    moisture = []
    for match_type, subset in panel.groupby("match_type", sort=False):
        for outcome in OUTCOMES:
            result = moisture_split(subset, args.weather, outcome)
            if not result.empty:
                result["match_type"] = match_type
                moisture.append(result)
    if moisture:
        moisture_table = pd.concat(moisture, ignore_index=True)
        moisture_table.to_csv(
            args.output_dir / "wear_moisture_split.csv", index=False
        )
        print("\nMoisture split:")
        print(moisture_table.to_string(index=False))

    with (args.output_dir / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "formats": list(FORMATS),
                "outcomes": list(OUTCOMES),
                "placebo_draws": args.placebo_draws,
                "seed": args.seed,
                "cells": int(len(panel)),
                "matches": int(panel["match_id"].nunique()),
            },
            file,
            indent=2,
        )
    print(f"\nArtifacts written to {args.output_dir}")


if __name__ == "__main__":
    main()
