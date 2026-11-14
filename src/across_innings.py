"""Stress-test the one positive result: the across-innings Test contrast.

Section 8 of the paper reports that spin's dot-ball advantage is about one
percentage point higher in the second and later innings of the same Test match.
That contrast is the only result in the data that looks like a surface effect, so
it carries the constructive half of the paper and is the most likely target for a
referee. The stated confounds are batting side, light, declaration pressure and
follow-on selection. This module tests the three that the data can address.

1. **Follow-on selection.** A follow-on is not random: it happens when one side is
   far ahead, and the side batting again is the one that has just been dismissed
   cheaply. If the contrast is driven by that selection, it should be concentrated
   in follow-on matches. ``run`` reports the contrast separately for matches with
   and without a follow-on, and adds a follow-on indicator and its interaction.

2. **Declaration pressure.** Innings two and three often end in a declaration, so
   the batting side in later innings faces a different objective. The specification
   adds an indicator for an innings that follows a declaration, and its
   interaction with spin.

3. **Batting side.** The cleanest test is not a control but a design. In a Test
   without a follow-on the innings order by team is A, B, A, B, so innings one and
   three are batted by the same team, and so are innings two and four. Comparing
   innings three and four against innings one and two therefore holds the batting
   side fixed by construction while moving two innings further into the match.
   ``paired`` estimates that contrast on non-follow-on matches. If the surface
   story is right it should survive; if the original contrast was a batting-side
   artefact it should not.

   The strictest version indexes each side's own innings rather than the match
   innings number. Every side's second innings is compared with its first, and
   match x team fixed effects absorb the batting side exactly, so the contrast is
   \"the second time this side bats\". A three-innings match contributes only the
   side that batted twice; that is the point, because the estimand is about the
   surface that both innings were played on.

What remains uncontrolled is light, weather within the match, and the match
situation (a third innings is usually a chase or a setup). The module reports what
it can and says so.

Usage::

    python src\\across_innings.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import pitch_dynamics as P

COLUMNS = [
    "match_id",
    "date",
    "match_type",
    "gender",
    "innings",
    "over",
    "ball_age",
    "legal",
    "bowler_style",
    "bowler_id",
    "batter_id",
    "runs_total",
    "runs_batter",
    "is_wicket",
    "is_bowler_wicket",
    "wicket_kind",
    "wickets_before",
    "score_before",
    "batting_team",
]


def load_tests(path: Path) -> pd.DataFrame:
    """Legal spin/pace deliveries from Tests, with innings-level context."""
    frame = pd.read_parquet(
        path,
        columns=COLUMNS,
        filters=[("legal", "=", True), ("match_type", "=", "Test")],
    )
    frame = frame[frame["bowler_style"].isin(["spin", "pace"])].copy()
    frame = frame.sort_values(
        ["match_id", "innings", "ball_age"], kind="stable"
    ).reset_index(drop=True)

    frame["dot"] = frame["runs_total"].eq(0).astype(float)
    frame["spin"] = frame["bowler_style"].eq("spin").astype(float)
    frame["age"] = frame["over"].astype(float) + 0.5
    frame["male"] = frame["gender"].eq("male").astype(float)
    frame["batter_balls"] = (
        frame.groupby(["match_id", "innings", "batter_id"], observed=True)
        .cumcount()
        .astype(float)
    )
    frame["phase"] = P.assign_phase(frame)
    frame["run_rate_before"] = frame["score_before"].astype(float) / (
        frame["over"].clip(lower=1) * P.BALLS_PER_OVER
    )

    context = innings_context(frame)
    frame = frame.merge(
        context, on=["match_id", "innings"], how="left", validate="many_to_one"
    )
    frame["had_follow_on"] = frame.groupby("match_id", observed=True)[
        "follow_on"
    ].transform("max")
    return frame


def innings_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Follow-on and declaration flags from the innings sequence of each match."""
    totals = (
        frame.groupby(["match_id", "innings"], observed=True)
        .agg(
            total=("runs_total", "sum"),
            wickets=("is_wicket", "sum"),
            team=("batting_team", "first"),
        )
        .reset_index()
        .sort_values(["match_id", "innings"], kind="stable")
    )
    per_match = totals.groupby("match_id", observed=True)
    totals["prev_team"] = per_match["team"].shift(1)
    totals["prev_wickets"] = per_match["wickets"].shift(1)
    totals["last_innings"] = per_match["innings"].transform("max")

    # A follow-on is the side that batted second batting again immediately.
    totals["follow_on"] = (
        totals["innings"].ge(3)
        & totals["prev_team"].notna()
        & totals["team"].eq(totals["prev_team"])
    ).astype(int)
    # An innings that ended without all ten wickets, and after which cricket
    # continued, ended in a declaration. Innings one can never be one.
    totals["declared_before"] = (
        totals["innings"].gt(1)
        & totals["prev_wickets"].lt(10)
        & totals["innings"].le(totals["last_innings"])
    ).astype(int)
    totals["repeat_innings"] = totals["innings"].ge(3).astype(int)
    totals["later_innings"] = totals["innings"].ge(2).astype(int)
    # Each side's own innings count: 1 for a side's first innings, 2 for its
    # second. Ordering is by match innings, so a follow-on reappearance counts.
    totals["team_innings"] = (
        totals.sort_values(["match_id", "innings"], kind="stable")
        .groupby(["match_id", "team"], observed=True)
        .cumcount()
        + 1
    )
    totals["second_bat"] = totals["team_innings"].ge(2).astype(int)
    return totals[
        [
            "match_id",
            "innings",
            "follow_on",
            "declared_before",
            "repeat_innings",
            "later_innings",
            "second_bat",
        ]
    ]


def build_design(
    frame: pd.DataFrame, interaction: str, extra: tuple[str, ...] = ()
):
    """Match-state controls plus one innings variable and its spin interaction.

    The design carries no linear innings trend. The panel specification that
    ``wear_cross_innings.csv`` comes from has none either, and a trend would soak
    up the very contrast being tested: within a side's own two innings the
    indicator and the innings number move together.
    """
    age = frame["age"].to_numpy(float)
    spin = frame["spin"].to_numpy(float)
    columns = [
        np.ones(len(frame)),
        age,
        spin,
        age * spin,
        frame["wickets_before"].to_numpy(float),
        frame["run_rate_before"].to_numpy(float),
        frame["batter_balls"].to_numpy(float),
        frame["male"].to_numpy(float),
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
    for phase in sorted(frame["phase"].unique())[1:]:
        columns.append(frame["phase"].eq(phase).to_numpy(float))
        names.append(f"phase_{phase}")

    indicator = frame[interaction].to_numpy(float)
    columns.extend([indicator, indicator * spin])
    names.extend([interaction, f"{interaction}_x_spin"])
    for term in extra:
        value = frame[term].to_numpy(float)
        columns.extend([value, value * spin])
        names.extend([term, f"{term}_x_spin"])
    return np.column_stack(columns), names


def fit_term(
    frame: pd.DataFrame,
    interaction: str,
    term: str,
    extra=(),
    by_team: bool = False,
) -> dict:
    design, names = build_design(frame, interaction, extra)
    if by_team:
        team_code = pd.factorize(frame["batting_team"])[0]
        absorb = frame["match_id"].to_numpy().astype(np.int64) * 1000 + team_code
    else:
        absorb = frame["match_id"].to_numpy()
    result = P.within_fit(
        frame["dot"].to_numpy(float),
        design,
        names,
        absorb,
        np.ones(len(frame)),
        frame["match_id"].to_numpy(),
    )
    hit = result[result["term"].eq(term)].iloc[0]
    return {
        "coefficient": float(hit["coefficient"]),
        "std_error": float(hit["std_error"]),
        "ci_low": float(hit["ci_low"]),
        "ci_high": float(hit["ci_high"]),
        "p_value": float(hit["p_value"]),
        "n": len(frame),
        "n_matches": int(frame["match_id"].nunique()),
        "n_innings": int(frame.groupby(["match_id", "innings"]).ngroups),
    }


def run(deliveries: Path, output_dir: Path) -> pd.DataFrame:
    frame = load_tests(deliveries)
    print(f"{len(frame):,} Test deliveries in {frame['match_id'].nunique():,} matches")

    rows: list[dict] = []

    def record(
        label: str, sample: str, term: str, stats: dict, note: str
    ) -> None:
        rows.append(
            {
                "spec": label,
                "sample": sample,
                "term": term,
                **stats,
                "note": note,
            }
        )
        print(
            f"{label:34s} {sample:14s} {stats['coefficient']:+.5f} "
            f"({stats['std_error']:.5f}) p={stats['p_value']:.4f} "
            f"n={stats['n']:,} matches={stats['n_matches']:,}"
        )

    baseline = fit_term(frame, "later_innings", "later_innings_x_spin")
    record(
        "baseline",
        "all matches",
        "later_innings_x_spin",
        baseline,
        "delivery-level replication of the panel contrast",
    )

    with_controls = fit_term(
        frame,
        "later_innings",
        "later_innings_x_spin",
        extra=("follow_on", "declared_before"),
    )
    record(
        "follow_on + declaration controls",
        "all matches",
        "later_innings_x_spin",
        with_controls,
        "adds follow-on and declaration indicators and their spin interactions",
    )

    no_follow = frame[frame["had_follow_on"].eq(0)]
    record(
        "baseline",
        "no follow-on",
        "later_innings_x_spin",
        fit_term(no_follow, "later_innings", "later_innings_x_spin"),
        "matches without a follow-on",
    )

    follow = frame[frame["had_follow_on"].eq(1)]
    record(
        "baseline",
        "follow-on",
        "later_innings_x_spin",
        fit_term(follow, "later_innings", "later_innings_x_spin"),
        "matches with a follow-on, where the side batting again was just dismissed",
    )

    paired = fit_term(no_follow, "repeat_innings", "repeat_innings_x_spin")
    record(
        "repeated innings, match FE",
        "no follow-on",
        "repeat_innings_x_spin",
        paired,
        "innings 3-4 versus innings 1-2, so each side is compared with itself",
    )

    strict = fit_term(
        no_follow, "second_bat", "second_bat_x_spin", by_team=True
    )
    record(
        "second time batting, match x team FE",
        "no follow-on",
        "second_bat_x_spin",
        strict,
        "each side's second innings against its first, batting side absorbed",
    )

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "wear_across_innings_stress.csv", index=False)
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
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run(args.balls, args.output_dir)


if __name__ == "__main__":
    main()
