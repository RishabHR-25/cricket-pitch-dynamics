"""Build match-level proxy labels and leakage-safe venue features from Cricsheet.

The script consumes Cricsheet's canonical JSON match files (a directory or ZIP)
and a researcher-curated bowler-role CSV. It writes one row per eligible T20
match to separate feature (X) and target (y) files.

Required bowler-role CSV columns:
    bowler,style

where style is ``spin`` or ``pace``. Cricsheet does not provide bowling styles;
the role file is a research input and should be versioned and documented.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import zipfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("cricket_pitch_pipeline")

# Dismissals credited to the bowler. Run outs, retired hurt, and similar
# dismissals are intentionally excluded from spin-versus-pace wicket counts.
BOWLER_CREDITED_DISMISSALS = {
    "bowled",
    "caught",
    "caught and bowled",
    "lbw",
    "stumped",
    "hit wicket",
}

# These are transparent initial T20 proxy thresholds, not physical ground truth.
DEFAULT_BATTING_RUN_RATE = 8.5
DEFAULT_BATTING_POWERPLAY_DOT_PERCENT = 35.0
DEFAULT_SPIN_TO_PACE_RATIO = 1.25
MIN_SPIN_WICKETS_FOR_SPIN_LABEL = 2


@dataclass(frozen=True)
class MatchRecord:
    """Canonical match-level outputs derived from ball-by-ball events."""

    match_id: str
    date: pd.Timestamp
    venue: str
    match_type: str
    toss_decision: str
    toss_winner_batting_first: int
    first_innings_runs: int
    first_innings_run_rate: float
    powerplay_dot_percent: float
    spin_wickets: int
    pace_wickets: int


def _read_match_json(source: Path) -> Iterable[tuple[str, dict[str, Any]]]:
    """Yield ``(match_id, JSON)`` pairs from a JSON file, directory, or ZIP."""
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            names = sorted(
                name
                for name in archive.namelist()
                if name.lower().endswith(".json") and not name.startswith("__MACOSX/")
            )
            if not names:
                raise ValueError(f"No JSON match files found in archive: {source}")
            for name in names:
                with archive.open(name) as match_file:
                    yield Path(name).stem, json.load(match_file)
        return

    if source.is_file() and source.suffix.lower() == ".json":
        with source.open(encoding="utf-8") as match_file:
            yield source.stem, json.load(match_file)
        return

    if source.is_dir():
        files = sorted(source.rglob("*.json"))
        if not files:
            raise ValueError(f"No JSON match files found in directory: {source}")
        for path in files:
            with path.open(encoding="utf-8") as match_file:
                yield path.stem, json.load(match_file)
        return

    raise ValueError(f"Input must be a Cricsheet JSON file, directory, or ZIP: {source}")


def load_bowler_roles(path: Path) -> dict[str, str]:
    """Read and validate the manual bowler-to-style mapping."""
    roles = pd.read_csv(path, dtype=str)
    required = {"bowler", "style"}
    missing = required.difference(roles.columns)
    if missing:
        raise ValueError(f"Bowler-role CSV is missing columns: {sorted(missing)}")

    roles = roles.loc[:, ["bowler", "style"]].dropna()
    roles["bowler"] = roles["bowler"].str.strip()
    roles["style"] = roles["style"].str.strip().str.lower()
    roles = roles[roles["bowler"] != ""]
    invalid = sorted(set(roles["style"]) - {"spin", "pace"})
    if invalid:
        raise ValueError(f"Invalid bowler styles {invalid}; use only 'spin' or 'pace'.")
    if roles["bowler"].duplicated().any():
        duplicate_names = sorted(roles.loc[roles["bowler"].duplicated(), "bowler"].unique())
        raise ValueError(f"Bowler-role CSV contains duplicate names: {duplicate_names[:10]}")
    return dict(zip(roles["bowler"], roles["style"]))


def _is_legal_delivery(extras: dict[str, Any]) -> bool:
    """Wides and no-balls do not count as legal balls."""
    return "wides" not in extras and "noballs" not in extras


def _to_match_record(
    match_id: str,
    data: dict[str, Any],
    bowler_roles: dict[str, str],
    powerplay_overs: int,
) -> MatchRecord | None:
    """Compute one match's label-generating statistics; return None if unusable."""
    info = data.get("info", {})
    if str(info.get("match_type", "")).upper() != "T20":
        return None

    innings_list = data.get("innings", [])
    if not innings_list:
        LOGGER.warning("Skipping %s: no innings data.", match_id)
        return None
    if str(info.get("outcome", {}).get("result", "")).lower() == "no result":
        LOGGER.warning("Skipping %s: match had no result.", match_id)
        return None

    venue = str(info.get("venue", "")).strip()
    if not venue:
        LOGGER.warning("Skipping %s: venue is missing.", match_id)
        return None

    first_innings = innings_list[0]
    first_innings_runs = 0
    first_innings_legal_balls = 0
    powerplay_legal_balls = 0
    powerplay_dot_balls = 0
    spin_wickets = 0
    pace_wickets = 0
    unknown_bowlers: set[str] = set()

    for innings_index, innings in enumerate(innings_list):
        for over in innings.get("overs", []):
            over_number = int(over["over"])
            for delivery in over.get("deliveries", []):
                extras = delivery.get("extras", {})
                runs = delivery.get("runs", {})
                total_runs = int(runs.get("total", 0))
                legal = _is_legal_delivery(extras)

                if innings_index == 0:
                    first_innings_runs += total_runs
                    if legal:
                        first_innings_legal_balls += 1
                        if over_number < powerplay_overs:
                            powerplay_legal_balls += 1
                            if total_runs == 0:
                                powerplay_dot_balls += 1

                for wicket in delivery.get("wickets", []):
                    kind = str(wicket.get("kind", "")).strip().lower()
                    if kind not in BOWLER_CREDITED_DISMISSALS:
                        continue
                    bowler = str(delivery.get("bowler", "")).strip()
                    style = bowler_roles.get(bowler)
                    if style is None:
                        unknown_bowlers.add(bowler)
                    elif style == "spin":
                        spin_wickets += 1
                    else:
                        pace_wickets += 1

    if unknown_bowlers:
        examples = ", ".join(sorted(unknown_bowlers)[:10])
        raise ValueError(
            f"{match_id}: missing spin/pace role for credited-wicket bowler(s): "
            f"{examples}. Add exact Cricsheet bowler names to the role CSV."
        )

    if first_innings_legal_balls == 0 or powerplay_legal_balls == 0:
        LOGGER.warning("Skipping %s: no legal first-innings/powerplay deliveries.", match_id)
        return None

    dates = info.get("dates", [])
    if not dates:
        LOGGER.warning("Skipping %s: match date is missing.", match_id)
        return None
    date = pd.to_datetime(dates[0], errors="coerce")
    if pd.isna(date):
        LOGGER.warning("Skipping %s: invalid match date %r.", match_id, dates[0])
        return None

    teams = info.get("teams", [])
    toss = info.get("toss", {})
    first_team = first_innings.get("team")
    toss_winner_batting_first = int(
        bool(teams) and bool(first_team) and toss.get("winner") == first_team
    )
    run_rate = first_innings_runs / (first_innings_legal_balls / 6.0)
    dot_percent = 100.0 * powerplay_dot_balls / powerplay_legal_balls

    return MatchRecord(
        match_id=match_id,
        date=date,
        venue=venue,
        match_type="T20",
        toss_decision=str(toss.get("decision", "unknown")).strip().lower(),
        toss_winner_batting_first=toss_winner_batting_first,
        first_innings_runs=first_innings_runs,
        first_innings_run_rate=run_rate,
        powerplay_dot_percent=dot_percent,
        spin_wickets=spin_wickets,
        pace_wickets=pace_wickets,
    )


def label_pitch_track(
    first_innings_run_rate: float,
    powerplay_dot_percent: float,
    spin_wickets: int,
    pace_wickets: int,
) -> str:
    """Apply deterministic, ordered T20 rules to generate the proxy target."""
    if not all(
        math.isfinite(value)
        for value in (first_innings_run_rate, powerplay_dot_percent)
    ):
        raise ValueError("Proxy label inputs must be finite numeric values.")
    if spin_wickets < 0 or pace_wickets < 0:
        raise ValueError("Wicket counts cannot be negative.")

    if (
        first_innings_run_rate >= DEFAULT_BATTING_RUN_RATE
        and powerplay_dot_percent <= DEFAULT_BATTING_POWERPLAY_DOT_PERCENT
    ):
        return "Batting"

    if (
        spin_wickets >= MIN_SPIN_WICKETS_FOR_SPIN_LABEL
        and (
            pace_wickets == 0
            or spin_wickets / pace_wickets >= DEFAULT_SPIN_TO_PACE_RATIO
        )
    ):
        return "Spin"

    return "Pace"


def _rolling_venue_history(
    matches: pd.DataFrame, window: int
) -> pd.DataFrame:
    """Return prior-match rolling means without same-day or future leakage.

    Matches on the same venue/date all receive history from strictly earlier
    dates; none can see the other match's completed statistics.
    """
    metric_columns = [
        "first_innings_run_rate",
        "spin_wicket_share",
        "powerplay_dot_percent",
    ]
    history = pd.DataFrame(index=matches.index)

    for venue, venue_rows in matches.groupby("venue", sort=False, dropna=False):
        ordered = venue_rows.sort_values(["date", "match_id"], kind="stable")
        previous_matches: deque[dict[str, float]] = deque(maxlen=window)
        for _, same_day in ordered.groupby(ordered["date"].dt.normalize(), sort=True):
            if previous_matches:
                previous = pd.DataFrame(previous_matches)
                means = previous[metric_columns].mean()
                count = len(previous_matches)
            else:
                means = pd.Series({column: np.nan for column in metric_columns})
                count = 0
            history.loc[same_day.index, metric_columns] = means.to_numpy()
            history.loc[same_day.index, "venue_matches_in_history"] = count
            for row in same_day[metric_columns].to_dict(orient="records"):
                previous_matches.append(row)

    return history.rename(
        columns={
            "first_innings_run_rate": f"venue_last{window}_mean_run_rate",
            "spin_wicket_share": f"venue_last{window}_mean_spin_wicket_share",
            "powerplay_dot_percent": f"venue_last{window}_mean_powerplay_dot_percent",
        }
    )


def build_xy(
    records: list[MatchRecord], history_window: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create model features X, proxy targets y, and an auditable label table."""
    if history_window < 1:
        raise ValueError("history_window must be at least 1.")
    if not records:
        raise ValueError("No eligible T20 matches were found.")

    matches = pd.DataFrame([record.__dict__ for record in records])
    matches = matches.sort_values(["date", "venue", "match_id"], kind="stable")
    matches["spin_wicket_share"] = np.divide(
        matches["spin_wickets"],
        matches["spin_wickets"] + matches["pace_wickets"],
        out=np.zeros(len(matches), dtype=float),
        where=(matches["spin_wickets"] + matches["pace_wickets"]).to_numpy() > 0,
    )
    labels = matches.apply(
        lambda row: label_pitch_track(
            row["first_innings_run_rate"],
            row["powerplay_dot_percent"],
            int(row["spin_wickets"]),
            int(row["pace_wickets"]),
        ),
        axis=1,
    )

    rolling = _rolling_venue_history(matches, history_window)
    X = pd.DataFrame(
        {
            "match_id": matches["match_id"],
            "venue": matches["venue"].replace("", "Unknown"),
            "year": matches["date"].dt.year,
            "month": matches["date"].dt.month,
            "day_of_year": matches["date"].dt.dayofyear,
            "toss_decision": matches["toss_decision"].replace("", "unknown"),
            "toss_winner_batting_first": matches["toss_winner_batting_first"],
        },
        index=matches.index,
    ).join(rolling)
    X["venue_matches_in_history"] = X["venue_matches_in_history"].fillna(0).astype(int)

    # Optional weather observations can be joined later on match_id. The
    # label-generating same-match statistics are deliberately excluded from X.
    y = pd.DataFrame({"match_id": matches["match_id"], "y": labels}, index=matches.index)
    audit = matches[
        [
            "match_id",
            "date",
            "venue",
            "first_innings_runs",
            "first_innings_run_rate",
            "powerplay_dot_percent",
            "spin_wickets",
            "pace_wickets",
        ]
    ].copy()
    audit["y"] = labels
    return X.reset_index(drop=True), y.reset_index(drop=True), audit.reset_index(drop=True)


def build_dataset(
    source: Path,
    bowler_roles_path: Path,
    powerplay_overs: int = 6,
    history_window: int = 5,
    weather_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Cricsheet files, derive match records, and return X/y/audit data."""
    if powerplay_overs < 1:
        raise ValueError("powerplay_overs must be at least 1.")
    bowler_roles = load_bowler_roles(bowler_roles_path)
    records: list[MatchRecord] = []
    skipped_non_t20 = 0
    for match_id, data in _read_match_json(source):
        record = _to_match_record(
            match_id, data, bowler_roles=bowler_roles, powerplay_overs=powerplay_overs
        )
        if record is None:
            if str(data.get("info", {}).get("match_type", "")).upper() != "T20":
                skipped_non_t20 += 1
            continue
        records.append(record)
    if skipped_non_t20:
        LOGGER.info("Ignored %d non-T20 match file(s).", skipped_non_t20)
    X, y, audit = build_xy(records, history_window=history_window)
    if weather_path is not None:
        weather = pd.read_csv(weather_path)
        if "match_id" not in weather.columns:
            raise ValueError("Weather CSV must contain a 'match_id' column.")
        weather["match_id"] = weather["match_id"].astype(str)
        if weather["match_id"].duplicated().any():
            raise ValueError("Weather CSV must contain at most one row per match_id.")
        if weather.shape[1] < 2:
            raise ValueError("Weather CSV must contain at least one weather feature.")
        feature_columns = [column for column in weather.columns if column != "match_id"]
        if not all(pd.api.types.is_numeric_dtype(weather[column]) for column in feature_columns):
            raise ValueError("Weather feature columns must be numeric.")
        overlap = set(feature_columns).intersection(X.columns)
        if overlap:
            raise ValueError(f"Weather column names conflict with X columns: {sorted(overlap)}")
        X = X.merge(weather, on="match_id", how="left", validate="one_to_one")
        missing_weather = int(X[feature_columns].isna().all(axis=1).sum())
        if missing_weather:
            LOGGER.warning(
                "%d match(es) have no weather observations; their weather features remain missing.",
                missing_weather,
            )
    return X, y, audit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create leakage-safe proxy labels and venue features from Cricsheet T20 JSON."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Cricsheet JSON file, directory of JSON files, or downloaded JSON ZIP.",
    )
    parser.add_argument(
        "--bowler-roles",
        type=Path,
        required=True,
        help="CSV containing exact Cricsheet bowler names and spin/pace labels.",
    )
    parser.add_argument(
        "--weather-csv",
        type=Path,
        help="Optional one-row-per-match weather CSV keyed by match_id, with numeric columns.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--powerplay-overs", type=int, default=6)
    parser.add_argument("--history-window", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()
    X, y, audit = build_dataset(
        source=args.input,
        bowler_roles_path=args.bowler_roles,
        powerplay_overs=args.powerplay_overs,
        history_window=args.history_window,
        weather_path=args.weather_csv,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    X.to_csv(args.output_dir / "X.csv", index=False)
    y.to_csv(args.output_dir / "y.csv", index=False)
    audit.to_csv(args.output_dir / "proxy_label_audit.csv", index=False)
    LOGGER.info(
        "Wrote %d match rows to %s; target counts: %s",
        len(X),
        args.output_dir.resolve(),
        y["y"].value_counts().to_dict(),
    )


if __name__ == "__main__":
    main()
