"""Correct the mis-centred within-innings permutation null.

The placebo in ``pitch_dynamics.py`` reallocates spin and pace labels within an
innings and refits the age x spin coefficient. Its null distribution is not
centred on zero: for dot-ball share the null mean is about a fifth of the
observed coefficient in T20 and ODI and three quarters in Tests, and for other
outcomes it can be larger than the observed value in either direction. A
permutation test that compares the observed coefficient with that null is
therefore not testing zero, and its p-value cannot be read as evidence for or
against the gradient.

This module does three things.

1. It re-reads the published permutation summary so the mis-centring is reported
   from the committed artifact rather than recomputed.

2. It reports the ratio of the null mean to the observed coefficient, which is
   the size of the mis-centring in units of the effect being tested.

3. It replaces the permutation null with a null-imposed wild cluster bootstrap.
   The interaction is dropped, the restricted model is fitted, and the restricted
   residuals are multiplied by a Rademacher weight drawn once per match before
   the full specification is refitted. Drawing the weight at the match level
   preserves the clustering that the standard errors use, so the bootstrap spread
   is directly comparable to the match-clustered standard error. The resulting
   null is centred on zero by construction.

The output is ``wear_placebo_bias.csv``. A null-imposed bootstrap is not a test
of the physical pitch; it is the correct randomisation null for this estimator,
and it is the null the paper uses when it says the gradient is precise.

Usage::

    python src\\placebo_bias.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import pitch_dynamics as P

FORMATS = ("T20", "ODI", "Test")
OUTCOMES = ("runs_per_ball", "dot_share", "wicket_rate", "boundary_share")
SPEC = "match_innings_fe"
TERM = "age_x_spin"


def _keep_columns(design: np.ndarray) -> list[int]:
    """Drop exactly collinear columns after the within transformation."""
    keep: list[int] = []
    for column in range(design.shape[1]):
        candidate = design[:, column]
        norm = np.linalg.norm(candidate)
        if norm <= 1e-12:
            continue
        if keep:
            basis = design[:, keep]
            projection = basis @ np.linalg.lstsq(basis, candidate, rcond=None)[0]
            if np.linalg.norm(candidate - projection) / norm < 1e-8:
                continue
        keep.append(column)
    return keep


def _solve(design: np.ndarray, weights: np.ndarray, y: np.ndarray) -> np.ndarray:
    normal = design.T @ (design * weights[:, None])
    rhs = design.T @ (weights * y)
    return np.linalg.solve(normal, rhs)


def _prepare(frame: pd.DataFrame):
    """Weighted within-match x innings design shared by every outcome."""
    design, names, absorb = P.build_design(frame, SPEC)
    weights = frame["balls"].to_numpy(float)
    codes, uniques = pd.factorize(absorb)
    design = P._weighted_group_demean(design, codes, weights, len(uniques))
    keep = _keep_columns(design)
    design = design[:, keep]
    names = [names[index] for index in keep]
    match_codes, _ = pd.factorize(frame["match_id"])
    return names, design, weights, match_codes


def _restricted(names: list[str], design: np.ndarray, weights: np.ndarray, y: np.ndarray):
    """Fit the specification without the interaction and return residuals and fit."""
    drop = names.index(TERM)
    keep = [index for index in range(len(names)) if index != drop]
    reduced = design[:, keep]
    beta = _solve(reduced, weights, y)
    fitted = reduced @ beta
    return y - fitted, fitted


def _bootstrap_null(
    design: np.ndarray,
    weights: np.ndarray,
    match_codes: np.ndarray,
    residuals: np.ndarray,
    fitted: np.ndarray,
    index: int,
    draws: int,
    seed: int,
) -> np.ndarray:
    """Wild cluster bootstrap at the match level, with the interaction imposed at zero."""
    rng = np.random.default_rng(seed)
    n_matches = int(match_codes.max()) + 1
    out = np.empty(draws)
    for draw in range(draws):
        sign = rng.choice([-1.0, 1.0], size=n_matches)
        y_star = fitted + residuals * sign[match_codes]
        out[draw] = _solve(design, weights, y_star)[index]
    return out


def run(
    panel: pd.DataFrame,
    placebo_path: Path,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    published = {
        (row["match_type"], row["outcome"]): row
        for row in json.loads(placebo_path.read_text(encoding="utf-8"))
    }

    rows: list[dict] = []
    for match_type in FORMATS:
        frame = panel[panel["match_type"].eq(match_type)].reset_index(drop=True)
        names, design, weights, match_codes = _prepare(frame)
        index = names.index(TERM)

        for outcome in OUTCOMES:
            y = frame[outcome].to_numpy(float)
            observed = float(_solve(design, weights, y)[index])
            summary = published[(match_type, outcome)]

            residuals, fitted = _restricted(names, design, weights, y)
            null = _bootstrap_null(
                design,
                weights,
                match_codes,
                residuals,
                fitted,
                index,
                draws,
                seed,
            )

            rows.append(
                {
                    "match_type": match_type,
                    "outcome": outcome,
                    "n_cells": len(frame),
                    "observed": observed,
                    "permutation_mean": summary["null_mean"],
                    "permutation_sd": summary["null_sd"],
                    "permutation_p": summary["p_value"],
                    "permutation_ratio": summary["null_mean"] / observed,
                    "bootstrap_mean": float(null.mean()),
                    "bootstrap_sd": float(null.std()),
                    "bootstrap_p": float(np.mean(np.abs(null) >= abs(observed))),
                    "bootstrap_z": float(
                        (observed - null.mean()) / null.std()
                    ),
                    "draws": draws,
                }
            )

        print(f"{match_type}: {len(frame):,} cells, {draws} bootstrap draws")

    table = pd.DataFrame(rows)
    table.to_csv(placebo_path.parent / "wear_placebo_bias.csv", index=False)
    return table


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel",
        type=Path,
        default=root / "models/pitch_dynamics/over_panel.parquet",
    )
    parser.add_argument(
        "--placebo",
        type=Path,
        default=root / "models/pitch_dynamics/wear_placebo.json",
    )
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel)
    table = run(panel, args.placebo, args.draws, args.seed)

    columns = [
        "match_type",
        "outcome",
        "observed",
        "permutation_mean",
        "permutation_ratio",
        "permutation_p",
        "bootstrap_sd",
        "bootstrap_p",
        "bootstrap_z",
    ]
    print()
    print(table[columns].round(6).to_string(index=False))
    print(f"\nArtifact written to {args.placebo.parent / 'wear_placebo_bias.csv'}")


if __name__ == "__main__":
    main()
