"""Estimation logic for the dashboard toss advisor.

The dashboard and the backtest import this module so the shipped rule and the
checked rule cannot drift apart. Everything here is descriptive: the estimates
are historical win shares of toss winners by decision, shrunk toward the format
baseline.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

SHRINK_PRIOR = 10.0
MIN_VENUE_N = 25
MIN_EDGE_POINTS = 2.0
MIN_POSTERIOR = 0.95
POSTERIOR_DRAWS = 20_000
POSTERIOR_SEED = 42


def _blank(series: pd.Series) -> pd.Series:
    """True where a text field is missing or empty, including NaN from CSV."""
    return series.isna() | series.astype(str).str.strip().eq("")


def decided_tosses(
    toss: pd.DataFrame,
    match_type: str,
    gender: str,
    team_type: str,
    before: pd.Timestamp,
) -> pd.DataFrame:
    """Matches with a usable toss and a decided result, strictly before cutoff."""
    subset = toss[
        toss["match_type"].eq(match_type)
        & toss["gender"].eq(gender)
        & toss["team_type"].eq(team_type)
        & toss["date"].lt(before)
        & toss["toss_decision"].isin(["bat", "field"])
        & ~_blank(toss["toss_winner"])
        & ~_blank(toss["winner"])
        & toss["result"].ne("no_result")
    ].copy()
    subset["toss_won"] = subset["winner"].eq(subset["toss_winner"])
    return subset


def shrunk(
    wins: float,
    n: float,
    base: float,
    prior: float = SHRINK_PRIOR,
) -> float | np.ndarray:
    """Posterior mean of a Beta(wins + prior*base, losses + prior*(1-base))."""
    wins = np.asarray(wins, dtype=float)
    n = np.asarray(n, dtype=float)
    base = np.asarray(base, dtype=float)
    estimate = np.where(n > 0, (wins + prior * base) / (n + prior), base)
    return float(estimate) if estimate.ndim == 0 else estimate


def beta_interval(
    wins: float,
    n: float,
    base: float,
    prior: float = SHRINK_PRIOR,
) -> tuple[float, float]:
    """Equal-tailed 95% posterior interval for the same Beta posterior."""
    wins = np.asarray(wins, dtype=float)
    n = np.asarray(n, dtype=float)
    base = np.asarray(base, dtype=float)
    alpha = wins + prior * base
    beta = np.maximum(n - wins, 0.0) + prior * (1.0 - base)
    low, high = beta_dist.ppf([0.025, 0.975], alpha, beta)
    if np.ndim(low) == 0:
        return float(low), float(high)
    return low, high


def toss_estimates(
    decided: pd.DataFrame, venue_key: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Venue and format rates for each toss decision, shrunk to the format base."""
    base = (
        decided.groupby("toss_decision", observed=True)["toss_won"]
        .agg(["size", "mean"])
        .rename(columns={"size": "base_n", "mean": "base_rate"})
    )
    venue = (
        decided[decided["venue_key"].eq(venue_key)]
        .groupby("toss_decision", observed=True)["toss_won"]
        .agg(["size", "sum"])
        .rename(columns={"size": "n", "sum": "wins"})
    )
    rows = []
    for decision, label in (("bat", "Bat first"), ("field", "Field first")):
        base_n = int(base["base_n"].get(decision, 0))
        base_rate = float(base["base_rate"].get(decision, 0.5))
        n = int(venue["n"].get(decision, 0))
        wins = int(venue["wins"].get(decision, 0))
        raw = wins / n if n else np.nan
        estimate = shrunk(wins, n, base_rate)
        low, high = beta_interval(wins, n, base_rate)
        rows.append(
            {
                "decision": decision,
                "label": label,
                "n": n,
                "wins": wins,
                "raw": raw,
                "estimate": estimate,
                "low": low,
                "high": high,
                "base_n": base_n,
                "base_rate": base_rate,
            }
        )
    return pd.DataFrame(rows), base.reset_index()


def decision_table(
    decided: pd.DataFrame,
    venue_key: str,
    min_venue_n: int = MIN_VENUE_N,
) -> tuple[pd.DataFrame, pd.DataFrame, bool, int]:
    """Decision rows with the format fallback already applied.

    Returns the rows, the format baseline, whether the venue has enough decided
    matches to be used, and the venue total behind both decisions.
    """
    estimates, base = toss_estimates(decided, venue_key)
    venue_total = int(estimates["n"].sum())
    use_venue = venue_total >= min_venue_n
    if not use_venue:
        for index, row in estimates.iterrows():
            low, high = beta_interval(0, 0, row["base_rate"])
            estimates.at[index, "estimate"] = row["base_rate"]
            estimates.at[index, "low"] = low
            estimates.at[index, "high"] = high
    return estimates, base, use_venue, venue_total


def posterior_probability(
    bat: Mapping,
    field: Mapping,
    draws: int = POSTERIOR_DRAWS,
    seed: int = POSTERIOR_SEED,
) -> float:
    """Posterior probability that batting first beats fielding first."""
    rng = np.random.default_rng(seed)

    def sample(row: Mapping) -> np.ndarray:
        alpha = row["wins"] + SHRINK_PRIOR * row["base_rate"]
        beta = max(row["n"] - row["wins"], 0.0) + SHRINK_PRIOR * (
            1.0 - row["base_rate"]
        )
        return rng.beta(alpha, beta, draws)

    return float((sample(bat) > sample(field)).mean())


def choose(
    bat: Mapping,
    field: Mapping,
    use_venue: bool,
    venue_total: int,
    min_venue_n: int = MIN_VENUE_N,
    min_edge_points: float = MIN_EDGE_POINTS,
    min_posterior: float = MIN_POSTERIOR,
) -> dict:
    """Apply the shipped call rule to the two decision rows.

    ``call`` is ``"bat"``, ``"field"`` or ``"none"``. ``prob`` is the posterior
    probability that the chosen call has the higher rate.
    """
    edge = (float(bat["estimate"]) - float(field["estimate"])) * 100.0
    bat_chance = posterior_probability(bat, field)
    prob = bat_chance if edge > 0 else 1.0 - bat_chance
    enough = (
        use_venue
        and int(bat["n"]) >= min_venue_n
        and int(field["n"]) >= min_venue_n
    )
    if not enough or abs(edge) < min_edge_points or prob < min_posterior:
        call = "none"
    elif edge > 0:
        call = "bat"
    else:
        call = "field"
    return {
        "call": call,
        "edge": edge,
        "prob": prob,
        "enough": enough,
        "venue_total": int(venue_total),
    }


def team_records(decided: pd.DataFrame, teams: Iterable[str]) -> pd.DataFrame:
    """Batting-first and fielding-first records for the selected sides."""
    other = decided["team2"].where(
        decided["team1"].eq(decided["toss_winner"]), decided["team1"]
    )
    first = decided["toss_winner"].where(decided["toss_decision"].eq("bat"), other)
    frame = decided.assign(first_team=first)
    base_bat = float(frame["winner"].eq(frame["first_team"]).mean())

    rows = []
    for team in teams:
        played = frame[frame["team1"].eq(team) | frame["team2"].eq(team)]
        bat = played[played["first_team"].eq(team)]
        field = played[~played["first_team"].eq(team)]
        bat_wins = int(bat["winner"].eq(team).sum())
        field_wins = int(field["winner"].eq(team).sum())
        rows.append(
            {
                "team": team,
                "bat_first_n": len(bat),
                "bat_first_wins": bat_wins,
                "bat_first_rate": bat_wins / len(bat) if len(bat) else np.nan,
                "field_first_n": len(field),
                "field_first_wins": field_wins,
                "field_first_rate": field_wins / len(field) if len(field) else np.nan,
                "bat_first_adj": shrunk(bat_wins, len(bat), base_bat),
                "field_first_adj": shrunk(field_wins, len(field), 1.0 - base_bat),
            }
        )
    return pd.DataFrame(rows)
