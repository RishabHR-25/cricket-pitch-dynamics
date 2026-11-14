"""Minimum detectable effect for every design in the paper.

A null is informative only if the design could have found the effect it was
looking for, and the designs in this paper differ by two orders of magnitude in
resolution. This module reads the standard errors already estimated by the other
modules and reports the smallest effect each design detects with 80% power at the
5% level, which is ``(z_0.975 + z_0.80) * SE``. It estimates nothing itself, so it
is cheap and cannot drift from the tables it summarises.

Usage::

    python src\\power_audit.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import norm

FACTOR = norm.ppf(0.975) + norm.ppf(0.80)


def _row(design: str, estimate: float, error: float, source: str) -> dict:
    return {
        "design": design,
        "estimate": float(estimate),
        "std_error": float(error),
        "mde_80": FACTOR * float(error),
        "source": source,
    }


def run(dynamics_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []

    # Pooled delivery-level within-innings coefficient, dot-ball share.
    pooled = pd.read_csv(dynamics_dir / "wear_decomposition.csv")
    pooled = pooled[pooled["outcome"].eq("dot")]
    for _, value in pooled.iterrows():
        rows.append(
            _row(
                f"within innings, {value['match_type']}",
                value["beta_pooled"],
                value["se_pooled"],
                "wear_decomposition.csv",
            )
        )

    # Within-bowler coefficient, full controls, dot-ball share.
    within = pd.read_csv(dynamics_dir / "wear_within_bowler.csv")
    within = within[
        within["outcome"].eq("dot") & within["spec"].eq("full")
    ]
    for _, value in within.iterrows():
        rows.append(
            _row(
                f"within bowler, {value['match_type']}",
                value["coefficient"],
                value["std_error"],
                "wear_within_bowler.csv",
            )
        )

    # Second-new-ball ladder, pace, first post-boundary bin.
    ladder = pd.read_csv(dynamics_dir / "wear_boundary_ladder.csv")
    ladder = ladder[
        ladder["style"].eq("pace") & ladder["boundary_bin"].eq("80_81")
    ]
    labels = {
        "1_all_deliveries": "second new ball, pace, all deliveries",
        "3_continuing_bowler_matches": (
            "second new ball, pace, matches with a continuing bowler"
        ),
        "4_continuing_bowlers_match_fe": (
            "second new ball, pace, continuing bowlers"
        ),
    }
    for step, label in labels.items():
        value = ladder[ladder["step"].eq(step)].iloc[0]
        rows.append(
            _row(
                label,
                value["beta_pooled"],
                value["se_pooled"],
                "wear_boundary_ladder.csv",
            )
        )

    # Across-innings stress test, all six specifications.
    across = pd.read_csv(dynamics_dir / "wear_across_innings_stress.csv")
    for _, value in across.iterrows():
        rows.append(
            _row(
                f"across innings, {value['spec']} ({value['sample']})",
                value["coefficient"],
                value["std_error"],
                "wear_across_innings_stress.csv",
            )
        )

    table = pd.DataFrame(rows)
    table.to_csv(dynamics_dir / "wear_power_audit.csv", index=False)
    return table


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dynamics-dir",
        type=Path,
        default=root / "models/pitch_dynamics",
    )
    args = parser.parse_args()
    table = run(args.dynamics_dir)
    print(f"MDE factor: {FACTOR:.3f}")
    print(table[["design", "estimate", "std_error", "mde_80"]].round(6).to_string(index=False))
    print(f"\nArtifact written to {args.dynamics_dir}")


if __name__ == "__main__":
    main()
