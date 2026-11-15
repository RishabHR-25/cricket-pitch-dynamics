"""Build the small, committed artifacts the dashboard reads.

The dashboard must not refit models or scan the full ball-by-ball parquet on
every interaction. This script derives three files from committed inputs:

``data/processed/dashboard_matches.csv``
    One row per match in ``data/interim/balls.parquet`` with the match-level
    statistics used to define the behavioural proxy label and the prior-venue
    history features used by the match model. Both five-match and ten-match
    histories are stored so the app can serve T20/ODI (five) and Test (ten)
    without recomputing.

``data/processed/dashboard_weather_climatology.csv``
    Mean observed weather for each venue x format x calendar month. A future
    fixture has no weather observation, so the app defaults to climatology and
    labels it as such.

``data/processed/dashboard_venues.csv``
    A compact venue directory with format coverage, date range and the latest
    match metadata, used to populate the sidebar quickly.

Run once after ``build_ball_dataset.py`` and ``train_pitch_models.py``:

    python src\\build_dashboard_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import train_pitch_models as T  # noqa: E402

WEATHER_COLUMNS = (
    "temperature_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "relative_humidity_2m_mean",
    "precip_sum_7d",
    "precip_sum_14d",
    "precip_sum_28d",
    "temp_mean_7d",
    "temp_mean_14d",
    "humidity_mean_7d",
    "wind_max_mean_7d",
    "dry_days_before",
)

FORMAT_WINDOWS = {"T20": 5, "ODI": 5, "Test": 10}


def _load_matches(root: Path) -> pd.DataFrame:
    """Load the parsed match table, building it once if it is not cached.

    The parsed table is produced by ``train_pitch_models._parse_matches`` and
    contains the exact match population, labels and toss fields used by the
    match model. Reusing it avoids recomputing labels from the ball table, which
    differs slightly around incomplete first innings.
    """
    cache = root / "data/interim/matches_parsed.parquet"
    if cache.exists():
        frame = pd.read_parquet(cache)
    else:
        roles_path = root / "models/pitch_proxy_results/player_role_join_audit.csv"
        roles_frame = pd.read_csv(roles_path)
        roles = dict(
            zip(
                roles_frame["cricsheet_id"].astype(str),
                roles_frame["style"],
            )
        )
        frame, _ = T._parse_matches(root / "data/raw/cricsheet/all_json.zip", roles)
        frame.to_parquet(cache, index=False)

    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    total = frame["spin_wickets"] + frame["pace_wickets"]
    frame["spin_wicket_share"] = np.where(
        frame["role_complete"].to_numpy() & total.gt(0).to_numpy(),
        frame["spin_wickets"] / total.where(total.gt(0)),
        np.nan,
    )
    frame["venue_key"] = T.venue_rows_key(frame).to_numpy()
    return frame.reset_index(drop=True)


def _history(matches: pd.DataFrame, window: int) -> pd.DataFrame:
    """Prior-venue history with the same window for every format."""
    history = T._add_prior_venue_features(
        matches,
        windows_by_format={fmt: window for fmt in T.FORMATS},
    )
    # The helper names columns by the per-format window it was given; asking for
    # a single window for all formats yields exactly `venue_last{window}_...`.
    return history


def build(root: Path = ROOT) -> None:
    matches = _load_matches(root)
    matches["match_id"] = matches["match_id"].astype(str)

    teams = pd.read_parquet(
        root / "data/interim/balls.parquet",
        columns=["match_id", "innings", "batting_team", "bowling_team"],
    )
    teams = (
        teams[teams["innings"].eq(1)]
        .drop_duplicates("match_id")[
            ["match_id", "batting_team", "bowling_team"]
        ]
        .rename(
            columns={
                "batting_team": "first_batting_team",
                "bowling_team": "first_bowling_team",
            }
        )
    )
    teams["match_id"] = teams["match_id"].astype(str)
    matches = matches.merge(teams, on="match_id", how="left", validate="one_to_one")

    history_five = _history(matches, 5)
    history_ten = _history(matches, 10)
    count_columns = ("venue_matches_in_history", "venue_role_matches_in_history")
    counts = pd.DataFrame(
        {
            column: np.where(
                matches["match_type"].eq("Test"),
                history_ten[column],
                history_five[column],
            )
            for column in count_columns
        },
        index=matches.index,
    )
    features = pd.concat(
        [
            matches,
            history_five.drop(columns=list(count_columns)),
            history_ten.drop(columns=list(count_columns)),
            counts,
        ],
        axis=1,
    )

    weather = pd.read_csv(root / "data/interim/match_weather.csv")
    weather["match_id"] = weather["match_id"].astype(str)
    keep = ["match_id", *WEATHER_COLUMNS]
    features["match_id"] = features["match_id"].astype(str)
    features = features.merge(
        weather[keep], on="match_id", how="left", validate="one_to_one"
    )

    features = features.sort_values(["match_type", "date", "match_id"]).reset_index(
        drop=True
    )
    output = root / "data/processed/dashboard_matches.csv"
    features.to_csv(output, index=False)

    # Month climatology, with venue-overall and format-overall fallbacks kept in
    # separate rows (month = 0 and month = -1) so the app can degrade cleanly.
    features["month"] = features["date"].dt.month
    keys = ["venue_key", "match_type"]
    monthly = (
        features.dropna(subset=["month"])
        .groupby([*keys, "month"], observed=True)[list(WEATHER_COLUMNS)]
        .mean()
        .reset_index()
    )
    venue_mean = (
        features.groupby(keys, observed=True)[list(WEATHER_COLUMNS)]
        .mean()
        .reset_index()
        .assign(month=0)
    )
    format_mean = (
        features.groupby("match_type", observed=True)[list(WEATHER_COLUMNS)]
        .mean()
        .reset_index()
        .assign(venue_key="__all__", month=-1)
    )
    climatology = pd.concat([monthly, venue_mean, format_mean], ignore_index=True)
    climatology.to_csv(
        root / "data/processed/dashboard_weather_climatology.csv", index=False
    )

    venues = (
        features.groupby(["venue_key", "match_type"], observed=True)
        .agg(
            venue=("venue", "last"),
            city=("city", "last"),
            gender=("gender", "last"),
            team_type=("team_type", "last"),
            matches=("match_id", "nunique"),
            first_match=("date", "min"),
            last_match=("date", "max"),
        )
        .reset_index()
    )
    venues.to_csv(root / "data/processed/dashboard_venues.csv", index=False)

    print(f"matches: {len(features):,} rows, {features['match_id'].nunique():,} matches")
    print(features.groupby("match_type")["match_id"].nunique().to_string())
    print("\nlabel counts (role-complete matches):")
    print(features["y"].value_counts(dropna=False).to_string())
    print(f"\nwritten: {output}")
    print(f"written: {root / 'data/processed/dashboard_weather_climatology.csv'}")
    print(f"written: {root / 'data/processed/dashboard_venues.csv'}")


if __name__ == "__main__":
    build()
