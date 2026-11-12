"""Measure how stable the proxy labels are to threshold perturbation.

The three behavioural labels (Batting / Pace / Spin) come from hand-set
thresholds. A reviewer will ask how much the downstream conclusions depend on
those constants, so this script recomputes every label with each threshold
scaled by +/-5% and +/-10% and reports agreement and Cohen's kappa against the
shipped labels.

No model is retrained; the audit table already carries every input needed.

Usage::

    python src\\label_sensitivity.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_pitch_models as T  # noqa: E402

SCALES = (0.9, 0.95, 1.05, 1.1)
SPIN_RATIO_MIN = 1.25


def label_with_thresholds(
    match_type: str,
    run_rate: float,
    early_dot_percent: float,
    spin_wickets: float,
    pace_wickets: float,
    run_rate_scale: float,
    dot_scale: float,
    spin_scale: float,
) -> str | None:
    rules = T.LABEL_RULES[match_type]
    if (
        run_rate >= rules["batting_run_rate_min"] * run_rate_scale
        and early_dot_percent <= rules["early_dot_percent_max"] * dot_scale
    ):
        return "Batting"
    if spin_wickets + pace_wickets == 0:
        return None
    if spin_wickets >= 2 and (
        pace_wickets == 0
        or spin_wickets / pace_wickets >= SPIN_RATIO_MIN * spin_scale
    ):
        return "Spin"
    return "Pace"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit", type=Path, default=root / "models/pitch_proxy/proxy_label_audit.csv"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "models/pitch_proxy_results/label_sensitivity.csv",
    )
    args = parser.parse_args()

    audit = pd.read_csv(args.audit)
    audit = audit[audit["role_complete"].eq(1) & audit["y"].notna()].copy()

    variants: list[tuple[str, dict[str, float]]] = []
    for name, prefix in (
        ("run_rate", "run_rate"),
        ("early_dot", "dot"),
        ("spin_ratio", "spin"),
    ):
        for scale in SCALES:
            variants.append((f"{name}_{scale:g}", {f"{prefix}_scale": scale}))
    variants.append(("all_0.9", {"run_rate_scale": 0.9, "dot_scale": 0.9, "spin_scale": 0.9}))
    variants.append(("all_1.1", {"run_rate_scale": 1.1, "dot_scale": 1.1, "spin_scale": 1.1}))

    base = {"run_rate_scale": 1.0, "dot_scale": 1.0, "spin_scale": 1.0}
    rows = []
    for match_type, frame in audit.groupby("match_type", sort=True):
        frame = frame.copy()
        frame["baseline"] = [
            label_with_thresholds(
                match_type,
                row.first_innings_run_rate,
                row.early_dot_percent,
                row.spin_wickets,
                row.pace_wickets,
                **base,
            )
            for row in frame.itertuples()
        ]
        for variant_name, settings in [("baseline", base), *variants]:
            scaled = {**base, **settings}
            labels = [
                label_with_thresholds(
                    match_type,
                    row.first_innings_run_rate,
                    row.early_dot_percent,
                    row.spin_wickets,
                    row.pace_wickets,
                    **scaled,
                )
                for row in frame.itertuples()
            ]
            changed = sum(
                left != right for left, right in zip(frame["baseline"], labels)
            )
            kappa = cohen_kappa_score(frame["baseline"], labels)
            rows.append(
                {
                    "match_type": match_type,
                    "variant": variant_name,
                    "n": len(frame),
                    "changed": changed,
                    "changed_fraction": round(changed / len(frame), 4),
                    "kappa_vs_baseline": round(float(kappa), 4),
                    "batting": sum(label == "Batting" for label in labels),
                    "pace": sum(label == "Pace" for label in labels),
                    "spin": sum(label == "Spin" for label in labels),
                }
            )

    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))

    perturbed = result[result["variant"].ne("baseline")]
    print("\nWorst-case agreement by format (excluding baseline):")
    for match_type, frame in perturbed.groupby("match_type"):
        worst = frame.loc[frame["changed_fraction"].idxmax()]
        print(
            f"  {match_type}: {worst['changed_fraction']:.1%} labels change "
            f"under {worst['variant']} (kappa {worst['kappa_vs_baseline']:.3f})"
        )


if __name__ == "__main__":
    main()
