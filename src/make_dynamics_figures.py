"""Figures for the ball-by-ball pitch-dynamics analysis.

Four figures, each tied to a table in the report:

1. Wear curves: scoring rate and dot-ball share by ball age, spin versus pace.
2. Specification forest: the age x spin coefficient under each fixed-effect and
   functional-form choice, with 95% confidence intervals.
3. Convergent validity: the coefficient by venue spin-friendliness tercile and
   by pre-match moisture split.
4. Cross-innings persistence: correlation between the first-innings and
   second-innings wear coefficient.

Usage::

    python src\\make_dynamics_figures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FORMATS = ("T20", "ODI", "Test")
STYLE_COLORS = {"pace": "#1f77b4", "spin": "#d62728"}
STYLE_LABELS = {"pace": "Pace", "spin": "Spin"}
SPEC_LABELS = {
    "venue_fe": "Venue FE",
    "match_fe": "Match FE",
    "match_innings_fe": "Match x innings FE",
}
AGE_LIMITS = {"T20": 20, "ODI": 50, "Test": 90}
MIN_BALLS = 300


def _save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Wrote {path}")


def wear_curves(path: Path, output: Path) -> None:
    curves = pd.read_csv(path)
    curves = curves[curves["bowler_style"].isin(STYLE_COLORS)]
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), sharex="col")
    for column, match_type in enumerate(FORMATS):
        subset = curves[curves["match_type"].eq(match_type)]
        subset = subset[subset["balls"] >= MIN_BALLS]
        subset = subset[subset["age_bin"] < AGE_LIMITS[match_type]]
        for style in ("pace", "spin"):
            style_frame = subset[subset["bowler_style"].eq(style)].sort_values("age_bin")
            if style_frame.empty:
                continue
            axes[0, column].plot(
                style_frame["age_bin"],
                style_frame["runs_per_ball"],
                marker="o",
                markersize=3,
                linewidth=1.8,
                color=STYLE_COLORS[style],
                label=STYLE_LABELS[style],
            )
            axes[1, column].plot(
                style_frame["age_bin"],
                style_frame["dot_share"],
                marker="o",
                markersize=3,
                linewidth=1.8,
                color=STYLE_COLORS[style],
                label=STYLE_LABELS[style],
            )
            axes[1, column].fill_between(
                style_frame["age_bin"],
                style_frame["dot_ci_low"],
                style_frame["dot_ci_high"],
                color=STYLE_COLORS[style],
                alpha=0.15,
                linewidth=0,
            )
        axes[0, column].set_title(f"{match_type}: scoring rate", fontsize=11)
        axes[1, column].set_title(f"{match_type}: dot-ball share", fontsize=11)
        axes[1, column].set_xlabel("Ball age (overs)")
        axes[0, column].grid(alpha=0.25)
        axes[1, column].grid(alpha=0.25)
        if column == 0:
            axes[0, column].set_ylabel("Runs per ball")
            axes[1, column].set_ylabel("Dot-ball share")
    axes[0, 0].legend(frameon=False, loc="upper left")
    figure.suptitle(
        "Spin and pace effectiveness as the ball ages (descriptive; phase composition not adjusted)",
        fontsize=12,
    )
    figure.tight_layout()
    _save(figure, output)


def specification_forest(path: Path, output: Path) -> None:
    table = pd.read_csv(path)
    table = table[table["spec"].isin(SPEC_LABELS)]
    outcomes = [
        ("runs_per_ball", "Runs per ball"),
        ("dot_share", "Dot-ball share"),
        ("wicket_rate", "Wicket rate"),
        ("boundary_share", "Boundary share"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(14, 6.2), sharey=True)
    offsets = {"venue_fe": -0.22, "match_fe": 0.0, "match_innings_fe": 0.22}
    colors = {
        "venue_fe": "#8c8c8c",
        "match_fe": "#1f4e79",
        "match_innings_fe": "#2e8b57",
    }
    for column, match_type in enumerate(FORMATS):
        axis = axes[column]
        y_positions = []
        labels = []
        position = 0
        for outcome_index, (outcome, label) in enumerate(outcomes):
            subset = table[
                table["match_type"].eq(match_type) & table["outcome"].eq(outcome)
            ]
            for _, row in subset.iterrows():
                y = position + offsets.get(row["spec"], 0.0)
                axis.errorbar(
                    row["age_x_spin_coef"],
                    y,
                    xerr=1.96 * row["age_x_spin_se"],
                    fmt="o",
                    markersize=4,
                    color=colors.get(row["spec"], "black"),
                    ecolor=colors.get(row["spec"], "black"),
                    elinewidth=1.4,
                    capsize=2,
                )
            y_positions.append(position)
            labels.append(label)
            position -= 1
        axis.axvline(0, color="black", linewidth=0.8, linestyle="--")
        axis.set_title(match_type, fontsize=12)
        axis.set_yticks(y_positions)
        axis.set_yticklabels(labels, fontsize=9)
        axis.grid(axis="x", alpha=0.25)
        axis.set_xlabel("Age x spin coefficient (95% CI)")
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=colors[spec], label=SPEC_LABELS[spec])
        for spec in SPEC_LABELS
    ]
    figure.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    figure.suptitle(
        "Within-match age x spin estimate under alternative specifications",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.95))
    _save(figure, output)


def convergent_validity(venue_path: Path, moisture_path: Path, output: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 8))
    if venue_path.exists():
        venue = pd.read_csv(venue_path)
        venue = venue[venue["outcome"].eq("dot_share")]
        for column, match_type in enumerate(FORMATS):
            subset = venue[venue["match_type"].eq(match_type)]
            if subset.empty:
                continue
            order = {"low": 0, "mid": 1, "high": 2}
            subset = subset.assign(order=subset["spin_friendliness"].map(order)).sort_values("order")
            axes[0, column].errorbar(
                subset["order"],
                subset["coefficient"],
                yerr=1.96 * subset["std_error"],
                marker="o",
                color="#1f4e79",
                capsize=3,
            )
            axes[0, column].axhline(0, color="black", linewidth=0.8, linestyle="--")
            axes[0, column].set_xticks([0, 1, 2])
            axes[0, column].set_xticklabels(["Low", "Mid", "High"])
            axes[0, column].set_title(f"{match_type}: venue spin-friendliness", fontsize=11)
            axes[0, column].grid(alpha=0.25)
            if column == 0:
                axes[0, column].set_ylabel("Age x spin (dot-ball share)")
    if moisture_path.exists():
        moisture = pd.read_csv(moisture_path)
        moisture = moisture[
            moisture["outcome"].eq("dot_share")
            & moisture["descriptor"].eq("dry_days_before")
        ]
        for column, match_type in enumerate(FORMATS):
            subset = moisture[moisture["match_type"].eq(match_type)]
            if subset.empty:
                continue
            order = {"below_median": 0, "above_median": 1}
            subset = subset.assign(order=subset["split"].map(order)).sort_values("order")
            axes[1, column].errorbar(
                subset["order"],
                subset["coefficient"],
                yerr=1.96 * subset["std_error"],
                marker="s",
                color="#b8860b",
                capsize=3,
            )
            axes[1, column].axhline(0, color="black", linewidth=0.8, linestyle="--")
            axes[1, column].set_xticks([0, 1])
            axes[1, column].set_xticklabels(["Fewer dry days", "More dry days"])
            axes[1, column].set_title(f"{match_type}: pre-match dry days", fontsize=11)
            axes[1, column].grid(alpha=0.25)
            if column == 0:
                axes[1, column].set_ylabel("Age x spin (runs per ball)")
    figure.suptitle(
        "Convergent validity: the wear estimate is larger where the surface is more spin-friendly and drier",
        fontsize=12,
    )
    figure.tight_layout()
    _save(figure, output)


def phase_estimates(path: Path, output: Path) -> None:
    """Per-phase age x spin estimates, the diagnostic for the pooled number."""
    if not path.exists():
        print(f"Skipping phase figure; {path} not found")
        return
    table = pd.read_csv(path)
    phase_order = ["powerplay", "new_ball", "middle", "death", "old_ball"]
    outcomes = [("dot_share", "Dot-ball share"), ("runs_per_ball", "Runs per ball")]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    for column, match_type in enumerate(FORMATS):
        axis = axes[column]
        positions = []
        labels = []
        position = 0
        for outcome, label in outcomes:
            subset = table[
                table["match_type"].eq(match_type) & table["outcome"].eq(outcome)
            ].copy()
            subset["order"] = subset["phase"].map(
                {name: index for index, name in enumerate(phase_order)}
            )
            subset = subset.sort_values("order")
            for _, row in subset.iterrows():
                axis.errorbar(
                    row["coefficient"],
                    position,
                    xerr=1.96 * row["std_error"],
                    fmt="o",
                    markersize=4,
                    color="#1f4e79" if outcome == "dot_share" else "#b8860b",
                    ecolor="#1f4e79" if outcome == "dot_share" else "#b8860b",
                    elinewidth=1.4,
                    capsize=2,
                )
                positions.append(position)
                labels.append(row["phase"].replace("_", " "))
                position -= 1
            position -= 0.5
        axis.axvline(0, color="black", linewidth=0.8, linestyle="--")
        axis.set_title(match_type, fontsize=12)
        axis.set_yticks(positions)
        axis.set_yticklabels(labels, fontsize=8)
        axis.grid(axis="x", alpha=0.25)
        axis.set_xlabel("Age x spin coefficient (95% CI)")
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color="#1f4e79", label="Dot-ball share"),
        plt.Line2D([], [], marker="o", linestyle="", color="#b8860b", label="Runs per ball"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9)
    figure.suptitle(
        "The pooled estimate hides phase heterogeneity: re-estimated inside each phase",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.95))
    _save(figure, output)


def new_ball_figures(event_path: Path, output: Path) -> None:
    """The naive and within-bowler event studies around the second new ball."""
    if not event_path.exists():
        print("Skipping new-ball figure; artifact not found")
        return
    table = pd.read_csv(event_path)
    table = table[table["outcome"].eq("dot") & table["term"].ne("wickets_before")]
    centres = {
        "72_73": 73,
        "74_75": 75,
        "76_77": 77,
        "78_79": 79,
        "80_81": 81,
        "82_83": 83,
        "84_85": 85,
        "86_87": 87,
    }
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), sharey=True)
    for axis, style in zip(axes, ("pace", "spin")):
        for sample, colour, marker, label in (
            ("all_deliveries", "#8c8c8c", "s", "All deliveries (naive)"),
            ("within_bowler", "#1f4e79", "o", "Same bowler, same innings"),
        ):
            subset = table[table["style"].eq(style)]
            subset = subset[subset["sample"].eq(sample)]
            if subset.empty:
                continue
            xs = [79.0]
            ys = [0.0]
            lows = [0.0]
            highs = [0.0]
            for _, row in subset.iterrows():
                xs.append(centres[row["term"]])
                ys.append(row["coefficient"])
                lows.append(row["coefficient"] - 1.96 * row["std_error"])
                highs.append(row["coefficient"] + 1.96 * row["std_error"])
            order = np.argsort(xs)
            xs = np.array(xs)[order]
            ys = np.array(ys)[order]
            lows = np.array(lows)[order]
            highs = np.array(highs)[order]
            axis.plot(
                xs,
                ys,
                marker=marker,
                markersize=4.5,
                linewidth=1.7,
                color=colour,
                label=label,
            )
            axis.fill_between(xs, lows, highs, color=colour, alpha=0.15, linewidth=0)
        axis.axvline(80, color="#b22222", linestyle="--", linewidth=1.3)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(f"{STYLE_LABELS[style]}: dot-ball share", fontsize=11)
        axis.set_xlabel("Over")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Change relative to overs 78-79 (95% CI)")
    axes[0].legend(frameon=False, loc="lower left")
    figure.suptitle(
        "The second-new-ball 'effect' is bowler selection, not ball condition",
        fontsize=12,
    )
    figure.tight_layout()
    _save(figure, output)


def new_ball_rule_figure(table_path: Path, output: Path) -> None:
    """Format-era age x spin slopes around the October 2011 rule change."""
    if not table_path.exists():
        print("Skipping rule-change figure; artifact not found")
        return
    table = pd.read_csv(table_path)
    table = table[
        table["outcome"].eq("dot_share")
        & table["stratum"].isin(["ODI_pre", "ODI_post", "T20_pre", "T20_post"])
    ]
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    positions = {"ODI": 0.0, "T20": 1.0}
    offsets = {"pre": -0.12, "post": 0.12}
    colors = {"pre": "#8c8c8c", "post": "#1f4e79"}
    for _, row in table.iterrows():
        match_type, era = row["stratum"].split("_")
        x = positions[match_type] + offsets[era]
        axis.errorbar(
            x,
            row["coef"],
            yerr=1.96 * row["se"],
            fmt="o",
            markersize=7,
            color=colors[era],
            ecolor=colors[era],
            elinewidth=1.6,
            capsize=4,
        )
    axis.axhline(0, color="black", linewidth=0.8, linestyle="--")
    axis.set_xticks(list(positions.values()))
    axis.set_xticklabels(
        ["ODI (rule applies)", "T20 (control)"]
    )
    axis.set_ylabel("Age x spin coefficient, dot-ball share (95% CI)")
    axis.set_title(
        "Two new balls did not flatten the ODI age gradient",
        fontsize=12,
    )
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=colors[era], label=label)
        for era, label in (("pre", "Before Oct 2011"), ("post", "After Oct 2011"))
    ]
    axis.legend(handles=handles, frameon=False)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    _save(figure, output)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dynamics-dir", type=Path, default=root / "models/pitch_dynamics"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "reports/figures"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    wear_curves(
        args.dynamics_dir / "wear_curves.csv", args.output_dir / "fig1_wear_curves.png"
    )
    specification_forest(
        args.dynamics_dir / "wear_specifications.csv",
        args.output_dir / "fig2_specification_forest.png",
    )
    convergent_validity(
        args.dynamics_dir / "wear_venue_heterogeneity.csv",
        args.dynamics_dir / "wear_moisture_split.csv",
        args.output_dir / "fig3_convergent_validity.png",
    )
    phase_estimates(
        args.dynamics_dir / "wear_by_phase.csv",
        args.output_dir / "fig4_phase_estimates.png",
    )
    new_ball_figures(
        args.dynamics_dir / "wear_new_ball_event.csv",
        args.output_dir / "fig5_new_ball_discontinuity.png",
    )
    new_ball_rule_figure(
        args.dynamics_dir / "wear_two_new_ball.csv",
        args.output_dir / "fig6_two_new_ball_rule.png",
    )


if __name__ == "__main__":
    main()
