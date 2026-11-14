"""Decompose the pooled age x spin coefficient into within- and between-bowler parts.

The paper's central claim is that bowler composition explains the discontinuity at
the second new ball but not the smooth age gradient. Section 6.4 supports that
claim by comparing two coefficients: the pooled match-fixed-effects estimate and
the delivery-level estimate that holds the bowler and innings fixed. Two numbers
and a verbal argument are weaker than they should be, because a reader cannot see
*how much* of the pooled coefficient is composition.

This module turns the comparison into an exact accounting identity. For any
coefficient on a regressor ``x``, let

* ``A`` be the pooled design (match fixed effects plus the other controls),
* ``B`` be the finer design (bowler x innings fixed effects plus the same controls),

so ``span(A)`` is contained in ``span(B)``. Write ``x_A`` for the residual of
``x`` on ``A`` and ``x_B`` for the residual of ``x`` on ``B``. Because ``B``
nests ``A``, ``x_B`` is also the residual of ``x_A`` on ``B``. Put
``d = x_A - x_B``, the part of the match-partialled variation that lives
*between* bowler-innings units once the same controls are removed.

Then, exactly,

    x_A'y_A = x_B'y_B + d'y_A          and     x_A'x_A = x_B'x_B + d'd,

so the pooled coefficient splits as

    beta_pooled = w * beta_within + (1 - w) * beta_between

with ``w = x_B'x_B / x_A'x_A``, ``beta_within = x_B'y_B / x_B'x_B`` and
``beta_between = d'y_A / d'd``. The composition term is the arithmetic difference

    C = beta_pooled - beta_within,

which is the omitted-variable bias from leaving the bowler-innings effects out of
the pooled regression. It is reported with a cluster-robust standard error that
accounts for the covariance between the two fits, since both are linear in the
same outcome vector.

The identity is applied to the smooth age x spin gradient in ``run_trend``. At the
second new ball the same identity is not enough, because the within-bowler design
can only be estimated on bowler-innings units that bowl on both sides of the
boundary: it changes the fixed effect *and* the sample. ``run_boundary`` therefore
estimates a ladder that adds one restriction at a time: all deliveries, matches
with and without a continuing bowler, the continuing bowlers at match fixed
effects, and the same rows with bowler x innings fixed effects.

The ladder overturns the reading in the first draft of the paper. The published
comparison ("+1.5 points naive, -2.1 points within bowler, so the sign reverses")
conflates a sample change with a fixed-effects change. In fact the +1.5 point jump
is +4.9 points in matches where the pace attack turns over completely at the
boundary and zero in matches where a seamer continues, and the fixed effect moves
the continuing-bowler estimate by only 0.0006. Selection is the sample, not the
correction.

Usage::

    python src\\decompose_wear.py trend
    python src\\decompose_wear.py boundary
    python src\\decompose_wear.py all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import pitch_dynamics as P
from identify_wear import EVENT_BINS, REFERENCE_BIN, _bin_frame, load_deliveries

TREND_OUTCOMES = ("dot", "runs_total", "boundary", "is_bowler_wicket")
BOUNDARY_OVER_LOW = 72
BOUNDARY_OVER_HIGH = 88
BOUNDARY_BINS = ("80_81", "82_83", "84_85", "86_87")
LADDER_BINS = BOUNDARY_BINS
FULL_BINS = tuple(
    label for label, _, _ in EVENT_BINS if label != REFERENCE_BIN
)


def _basis(
    design: np.ndarray,
    codes: np.ndarray,
    n_groups: int,
    weights: np.ndarray,
) -> np.ndarray:
    """Weighted within transform plus the collinearity drop used by ``within_fit``."""
    tiled = P._weighted_group_demean(design, codes, weights, n_groups)
    keep: list[int] = []
    for column in range(tiled.shape[1]):
        candidate = tiled[:, column]
        norm = np.linalg.norm(candidate)
        if norm <= 1e-12:
            continue
        if keep:
            basis = tiled[:, keep]
            projection = basis @ np.linalg.lstsq(basis, candidate, rcond=None)[0]
            residual = candidate - projection
            if np.linalg.norm(residual) / norm < 1e-8:
                continue
        keep.append(column)
    return np.ascontiguousarray(tiled[:, keep])


def _partial(
    target: np.ndarray,
    basis: np.ndarray,
    codes: np.ndarray,
    n_groups: int,
    weights: np.ndarray,
) -> np.ndarray:
    """Residual of ``target`` on the absorbed effects spanned by ``basis``."""
    demeaned = P._weighted_group_demean(target, codes, weights, n_groups)
    normal = basis.T @ (basis * weights[:, None])
    cross = basis.T @ (weights * demeaned)
    coefficient = np.linalg.lstsq(normal, cross, rcond=None)[0]
    return demeaned - basis @ coefficient


def _cluster_scores(
    residual: np.ndarray,
    regressor: np.ndarray,
    weights: np.ndarray,
    clusters: np.ndarray,
    denominator: float,
) -> np.ndarray:
    """Cluster sums of the influence function for a single-regressor fit."""
    score = weights * regressor * residual / denominator
    codes, uniques = pd.factorize(clusters)
    return np.bincount(codes, weights=score, minlength=len(uniques))


def _fit_stats(regressor: np.ndarray, outcome: np.ndarray, weights: np.ndarray):
    denominator = float(np.sum(weights * regressor**2))
    coefficient = float(np.sum(weights * regressor * outcome) / denominator)
    residual = outcome - coefficient * regressor
    return coefficient, denominator, residual


def decompose(
    outcome: np.ndarray,
    regressor: np.ndarray,
    pooled_basis: np.ndarray,
    pooled_codes: np.ndarray,
    pooled_groups: int,
    within_basis: np.ndarray,
    within_codes: np.ndarray,
    within_groups: int,
    weights: np.ndarray,
    clusters: np.ndarray,
    n_parameters: int,
) -> dict[str, float]:
    """Split one coefficient into within and between components, exactly.

    ``pooled_basis`` and ``within_basis`` are the within-transformed control
    matrices *excluding* the regressor of interest, because the
    Frisch-Waugh-Lovell residual is defined against the other controls.
    """
    x_pooled = _partial(regressor, pooled_basis, pooled_codes, pooled_groups, weights)
    x_within = _partial(regressor, within_basis, within_codes, within_groups, weights)
    y_pooled = _partial(outcome, pooled_basis, pooled_codes, pooled_groups, weights)
    y_within = _partial(outcome, within_basis, within_codes, within_groups, weights)

    beta_pooled, denom_pooled, resid_pooled = _fit_stats(x_pooled, y_pooled, weights)
    beta_within, denom_within, resid_within = _fit_stats(x_within, y_within, weights)

    between = x_pooled - x_within
    denom_between = float(np.sum(weights * between**2))
    beta_between = (
        float(np.sum(weights * between * y_pooled) / denom_between)
        if denom_between > 0
        else np.nan
    )
    weight_within = denom_within / denom_pooled
    weight_between = 1.0 - weight_within

    # Cluster-robust covariance between the two fits, which share the outcome.
    score_pooled = _cluster_scores(resid_pooled, x_pooled, weights, clusters, denom_pooled)
    score_within = _cluster_scores(resid_within, x_within, weights, clusters, denom_within)
    groups = len(score_pooled)
    n = len(outcome)
    correction = 1.0
    if groups > 1 and n > n_parameters:
        correction = groups / (groups - 1) * (n - 1) / (n - n_parameters)
    var_pooled = correction * float(np.sum(score_pooled**2))
    var_within = correction * float(np.sum(score_within**2))
    covariance = correction * float(np.sum(score_pooled * score_within))
    composition = beta_pooled - beta_within
    var_composition = var_pooled + var_within - 2.0 * covariance
    se_composition = float(np.sqrt(max(var_composition, 0.0)))
    z_value = composition / se_composition if se_composition > 0 else np.nan
    from scipy.stats import norm

    return {
        "beta_pooled": beta_pooled,
        "se_pooled": float(np.sqrt(var_pooled)),
        "p_pooled": float(
            2 * (1 - norm.cdf(abs(beta_pooled / np.sqrt(var_pooled))))
        )
        if var_pooled > 0
        else np.nan,
        "beta_within": beta_within,
        "se_within": float(np.sqrt(var_within)),
        "p_within": float(
            2 * (1 - norm.cdf(abs(beta_within / np.sqrt(var_within))))
        )
        if var_within > 0
        else np.nan,
        "beta_between": beta_between,
        "weight_within": weight_within,
        "weight_between": weight_between,
        "composition": composition,
        "se_composition": se_composition,
        "p_composition": float(2 * (1 - norm.cdf(abs(z_value))))
        if np.isfinite(z_value)
        else np.nan,
        "within_component": weight_within * beta_within,
        "between_component": weight_between * beta_between,
        "identity_check": beta_pooled - (
            weight_within * beta_within + weight_between * beta_between
        ),
        "n": n,
        "units": int(len(np.unique(within_codes))),
        "clusters": groups,
    }


def _delivery_frame(deliveries: Path, match_type: str) -> pd.DataFrame:
    """Legal spin/pace deliveries with the controls used in Section 6.4."""
    frame = load_deliveries(deliveries, (match_type,))
    frame = frame.sort_values(
        ["match_id", "innings", "ball_age"], kind="stable"
    ).reset_index(drop=True)
    frame["batter_balls"] = (
        frame.groupby(["match_id", "innings", "batter_id"], observed=True)
        .cumcount()
        .astype(float)
    )
    frame["phase"] = P.assign_phase(frame)
    frame["age"] = frame["over"].astype(float) + 0.5
    frame["male"] = frame["gender"].eq("male").astype(float)
    frame["unit"] = (
        frame["match_id"].astype(str)
        + "|"
        + frame["innings"].astype(str)
        + "|"
        + frame["bowler_id"].astype(str)
    )
    return frame


def _controls(frame: pd.DataFrame) -> list[np.ndarray]:
    columns = [
        frame["wickets_before"].to_numpy(float),
        frame["run_rate_before"].to_numpy(float),
        frame["batter_balls"].to_numpy(float),
        frame["male"].to_numpy(float),
    ]
    for phase in sorted(frame["phase"].unique())[1:]:
        columns.append(frame["phase"].eq(phase).to_numpy(float))
    return columns


def run_trend(deliveries: Path, output_dir: Path) -> pd.DataFrame:
    """Decompose the smooth age x spin gradient, by format and outcome."""
    rows = []
    for match_type in P.FORMATS:
        frame = _delivery_frame(deliveries, match_type)
        age = frame["age"].to_numpy(float)
        spin = frame["spin"].to_numpy(float)
        regressor = age * spin

        other = np.column_stack(
            [np.ones(len(frame)), age, spin, *_controls(frame)]
        )

        pooled = frame["match_id"].to_numpy()
        within = frame["unit"].to_numpy()
        pooled_codes, pooled_uniques = pd.factorize(pooled)
        within_codes, within_uniques = pd.factorize(within)
        weights = np.ones(len(frame))

        pooled_basis = _basis(other, pooled_codes, len(pooled_uniques), weights)
        within_basis = _basis(other, within_codes, len(within_uniques), weights)

        for outcome in TREND_OUTCOMES:
            stats = decompose(
                frame[outcome].to_numpy(float),
                regressor,
                pooled_basis,
                pooled_codes,
                len(pooled_uniques),
                within_basis,
                within_codes,
                len(within_uniques),
                weights,
                pooled,
                n_parameters=len(within_uniques) + 2,
            )
            stats.update(
                {
                    "scenario": "smooth_gradient",
                    "match_type": match_type,
                    "style": "both",
                    "outcome": outcome,
                    "boundary_bin": "",
                    "sample": "full",
                    "controls": "state+experience+phase",
                    "n_matches": int(frame["match_id"].nunique()),
                }
            )
            rows.append(stats)
            print(
                f"{match_type:4s} {outcome:16s} pooled {stats['beta_pooled']:+.6f} "
                f"within {stats['beta_within']:+.6f} between {stats['beta_between']:+.6f} "
                f"w {stats['weight_within']:.3f} C {stats['composition']:+.6f}"
            )
        del frame, other, pooled_basis, within_basis
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "wear_decomposition.csv", index=False)
    return table


def _ladder_step(
    data: pd.DataFrame,
    design: np.ndarray,
    names: list[str],
    target: str,
    absorb: str,
    pooled_basis: np.ndarray | None,
    pooled_codes: np.ndarray,
    pooled_groups: int,
    within_basis: np.ndarray | None,
    within_codes: np.ndarray,
    within_groups: int,
    weights: np.ndarray,
) -> dict[str, float]:
    """One row of the selection ladder, from ``within_fit`` or ``decompose``."""
    regressor = data["bin"].eq(target).to_numpy(float)
    if absorb == "decompose":
        stats = decompose(
            data["dot"].to_numpy(float),
            regressor,
            pooled_basis,
            pooled_codes,
            pooled_groups,
            within_basis,
            within_codes,
            within_groups,
            weights,
            data["match_id"].to_numpy(),
            n_parameters=within_groups + 2,
        )
        return stats
    result = P.within_fit(
        data["dot"].to_numpy(float),
        design,
        names,
        data["unit"].to_numpy() if absorb == "unit" else data["match_id"].to_numpy(),
        weights,
        data["match_id"].to_numpy(),
    )
    hit = result[result["term"].eq(f"bin_{target}")].iloc[0]
    return {
        "beta_pooled": float(hit["coefficient"]),
        "se_pooled": float(hit["std_error"]),
        "p_pooled": float(hit["p_value"]),
        "beta_within": np.nan,
        "se_within": np.nan,
        "beta_between": np.nan,
        "weight_within": np.nan,
        "weight_between": np.nan,
        "composition": np.nan,
        "se_composition": np.nan,
        "p_composition": np.nan,
        "within_component": np.nan,
        "between_component": np.nan,
        "identity_check": 0.0,
        "units": int(data["unit"].nunique()),
    }


def run_boundary(deliveries: Path, output_dir: Path) -> pd.DataFrame:
    """Selection ladder for the second-new-ball jump.

    Five nested comparisons, each adding one restriction, so a reader can see
    exactly which restriction removes the jump:

    1. all deliveries in all matches, match fixed effects;
    2. matches in which no pace/spin bowler bowls on both sides of the boundary;
    3. matches in which at least one does;
    4. those matches, restricted to the bowlers who bowl on both sides;
    5. the same rows with bowler x innings fixed effects (the exact split).

    Steps 1-3 use ``within_fit``; steps 4-5 use the exact decomposition.
    """
    frame = load_deliveries(
        deliveries, ("Test",), over_low=BOUNDARY_OVER_LOW, over_high=BOUNDARY_OVER_HIGH
    )
    frame = _bin_frame(frame)
    frame = frame[frame["bin"].ne("none")].reset_index(drop=True)

    rows = []
    for style in ("pace", "spin"):
        subset = frame[frame["bowler_style"].eq(style)].reset_index(drop=True)

        bins_by_unit = subset.groupby("unit", observed=True)["bin"].agg(
            lambda values: set(values)
        )
        continuing = bins_by_unit[
            bins_by_unit.apply(
                lambda values: REFERENCE_BIN in values and "80_81" in values
            )
        ].index
        continuing_matches = set(subset[subset["unit"].isin(continuing)]["match_id"])
        has_continuing = subset["match_id"].isin(continuing_matches)
        is_continuing = subset["unit"].isin(continuing).to_numpy()

        groups = (
            ("1_all_deliveries", subset),
            ("2_no_continuing_bowler", subset[~has_continuing]),
            ("3_continuing_bowler_matches", subset[has_continuing]),
        )
        for step, data in groups:
            data = data.reset_index(drop=True)
            design = np.column_stack(
                [
                    data["bin"].eq(label).to_numpy(float)
                    for label in FULL_BINS
                ]
                + [data["wickets_before"].to_numpy(float)]
            )
            names = [f"bin_{label}" for label in FULL_BINS] + ["wickets_before"]
            weights = np.ones(len(data))
            for target in LADDER_BINS:
                stats = _ladder_step(
                    data, design, names, target, "match",
                    None, np.zeros(0, int), 0, None, np.zeros(0, int), 0, weights,
                )
                stats.update(
                    {
                        "scenario": "second_new_ball",
                        "match_type": "Test",
                        "style": style,
                        "outcome": "dot",
                        "boundary_bin": target,
                        "step": step,
                        "design": "match_fe",
                        "n": len(data),
                        "clusters": int(data["match_id"].nunique()),
                        "n_matches": int(data["match_id"].nunique()),
                    }
                )
                rows.append(stats)

        # Steps 4 and 5: the sample both designs can share, exact split.
        data = subset[is_continuing].reset_index(drop=True)
        pooled = data["match_id"].to_numpy()
        within = data["unit"].to_numpy()
        pooled_codes, pooled_uniques = pd.factorize(pooled)
        within_codes, within_uniques = pd.factorize(within)
        weights = np.ones(len(data))
        wickets = data["wickets_before"].to_numpy(float)
        for target in LADDER_BINS:
            bin_columns = [
                data["bin"].eq(other).to_numpy(float)
                for other in FULL_BINS
                if other != target
            ]
            design = np.column_stack([*bin_columns, wickets])
            names = [
                f"bin_{other}" for other in FULL_BINS if other != target
            ] + ["wickets_before"]
            pooled_basis = _basis(design, pooled_codes, len(pooled_uniques), weights)
            within_basis = _basis(design, within_codes, len(within_uniques), weights)
            stats = decompose(
                data["dot"].to_numpy(float),
                data["bin"].eq(target).to_numpy(float),
                pooled_basis,
                pooled_codes,
                len(pooled_uniques),
                within_basis,
                within_codes,
                len(within_uniques),
                weights,
                pooled,
                n_parameters=len(within_uniques) + 2,
            )
            for step, design_label in (
                ("4_continuing_bowlers_match_fe", "match_fe"),
                ("5_continuing_bowlers_unit_fe", "unit_fe"),
            ):
                row = dict(stats)
                row.update(
                    {
                        "scenario": "second_new_ball",
                        "match_type": "Test",
                        "style": style,
                        "outcome": "dot",
                        "boundary_bin": target,
                        "step": step,
                        "design": design_label,
                        "n": len(data),
                        "clusters": int(data["match_id"].nunique()),
                        "n_matches": int(data["match_id"].nunique()),
                    }
                )
                rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "wear_boundary_ladder.csv", index=False)
    focus = table[table["boundary_bin"].eq("80_81")]
    print("\nSecond-new-ball ladder, 80-81 relative to 78-79, dot-ball share")
    print(
        focus[
            ["style", "step", "beta_pooled", "se_pooled", "p_pooled", "beta_within", "n"]
        ]
        .round(6)
        .to_string(index=False)
    )
    return table


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
        "command", choices=("trend", "boundary", "all"), nargs="?", default="all"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.command in ("trend", "all"):
        run_trend(args.balls, args.output_dir)
    if args.command in ("boundary", "all"):
        run_boundary(args.balls, args.output_dir)
    print(f"\nArtifacts written to {args.output_dir}")


if __name__ == "__main__":
    main()
