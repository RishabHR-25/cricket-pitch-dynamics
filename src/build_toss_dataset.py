"""Build toss, result and squad artifacts for the dashboard toss advisor.

The ball-by-ball layer in ``data/interim/balls.parquet`` does not carry the toss
or the match result, so this module reads the Cricsheet archive once and writes
three small, committed artifacts:

``data/processed/dashboard_toss_matches.csv``
    One row per T20/ODI/Test match: teams, toss winner and decision, winner and
    result type, venue key, date.

``data/processed/dashboard_teams.csv``
    One row per team x format x gender x team type: decided matches, wins,
    win share, batting-first and fielding-first splits, last appearance.

``data/processed/dashboard_team_players.csv``
    One row per player x team x format x gender x team type: matches, wins,
    win share and last appearance. Used only to populate the XI picker and to
    describe a lineup; it is not an input to the toss estimate.

Run from anywhere:

    python src/build_toss_dataset.py
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import train_pitch_models as T  # noqa: E402


def parse_match(match: dict) -> tuple[dict | None, list[dict]]:
    info = match.get("info", {})
    match_type = str(info.get("match_type", ""))
    dates = info.get("dates", [])
    venue = str(info.get("venue", "")).strip()
    if match_type not in T.FORMATS or not dates or not venue:
        return None, []

    teams = [str(team) for team in info.get("teams", [])]
    if len(teams) != 2:
        return None, []

    toss = info.get("toss") or {}
    toss_winner = str(toss.get("winner", "")).strip()
    toss_decision = str(toss.get("decision", "")).strip().lower()
    outcome = info.get("outcome") or {}
    winner = str(outcome.get("winner", "")).strip()
    by = outcome.get("by") or {}
    result = "decided"
    if outcome.get("result") == "tie":
        result = "tie"
    elif outcome.get("result") in {"no result", "abandoned"}:
        result = "no_result"
    elif not winner:
        result = "unknown"
    elif "runs" in by:
        result = "runs"
    elif "wickets" in by:
        result = "wickets"

    registry = info.get("registry", {}) or {}
    venue_registry = registry.get("venues", {}) or {}
    people = registry.get("people", {}) or {}

    row = {
        "match_id": "",
        "date": str(dates[0]),
        "match_type": match_type,
        "venue": venue,
        "venue_id": str(venue_registry.get(venue, "")),
        "city": str(info.get("city", "")),
        "gender": str(info.get("gender", "unknown")),
        "team_type": str(info.get("team_type", "unknown")),
        "team1": teams[0],
        "team2": teams[1],
        "toss_winner": toss_winner,
        "toss_decision": toss_decision,
        "winner": winner,
        "result": result,
        "method": str(info.get("method", "")),
        "overs": info.get("overs", ""),
    }

    rosters: list[dict] = []
    players = info.get("players") or {}
    for team, names in players.items():
        for name in names:
            rosters.append(
                {
                    "team": str(team),
                    "player": str(name),
                    "player_id": str(people.get(name, "")),
                }
            )
    return row, rosters


def add_venue_key(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["venue_key"] = T.venue_rows_key(frame)
    return frame


def batting_first(frame: pd.DataFrame) -> pd.Series:
    """Team that batted first, whichever way the toss went."""
    other = frame["team2"].where(
        frame["team1"].eq(frame["toss_winner"]), frame["team1"]
    )
    return frame["toss_winner"].where(frame["toss_decision"].eq("bat"), other)


def team_splits(matches: pd.DataFrame) -> pd.DataFrame:
    decided = matches[matches["result"].ne("no_result")].copy()
    decided = decided[decided["winner"].ne("")].copy()
    decided["first"] = batting_first(decided)

    keys = ["match_type", "gender", "team_type", "team"]
    long = pd.concat(
        [
            decided.assign(**{"team": decided["team1"]}),
            decided.assign(**{"team": decided["team2"]}),
        ],
        ignore_index=True,
    )
    long["won"] = long["winner"].eq(long["team"]).astype(int)
    long["batted_first"] = long["first"].eq(long["team"])

    grouped = long.groupby(keys, observed=True)
    table = grouped.agg(
        matches=("won", "size"),
        wins=("won", "sum"),
        last_date=("date", "max"),
    ).reset_index()
    table["win_share"] = table["wins"] / table["matches"]

    bat = long[long["batted_first"]].groupby(keys, observed=True).agg(
        bat_first_matches=("won", "size"),
        bat_first_wins=("won", "sum"),
    )
    field = long[~long["batted_first"]].groupby(keys, observed=True).agg(
        field_first_matches=("won", "size"),
        field_first_wins=("won", "sum"),
    )
    table = table.merge(bat, on=keys, how="left").merge(field, on=keys, how="left")
    for column in ("bat_first_matches", "bat_first_wins", "field_first_matches",
                   "field_first_wins"):
        table[column] = table[column].fillna(0).astype(int)
    return table.sort_values(["match_type", "matches"], ascending=[True, False])


def player_splits(
    matches: pd.DataFrame, rosters: pd.DataFrame
) -> pd.DataFrame:
    decided = matches[matches["result"].ne("no_result")].copy()
    decided = decided[decided["winner"].ne("")]
    merged = rosters.merge(
        decided[
            ["match_id", "match_type", "gender", "team_type", "winner", "date"]
        ],
        on="match_id",
        how="inner",
    )
    merged["won"] = merged["winner"].eq(merged["team"]).astype(int)
    keys = ["match_type", "gender", "team_type", "team", "player", "player_id"]
    table = (
        merged.groupby(keys, observed=True)
        .agg(matches=("won", "size"), wins=("won", "sum"), last_date=("date", "max"))
        .reset_index()
    )
    table["win_share"] = table["wins"] / table["matches"]
    return table.sort_values(
        ["match_type", "team", "matches"], ascending=[True, True, False]
    )


def main() -> None:
    archive = ROOT / "data/raw/cricsheet/all_json.zip"
    output = ROOT / "data/processed"
    output.mkdir(parents=True, exist_ok=True)

    match_rows: list[dict] = []
    roster_rows: list[dict] = []
    with zipfile.ZipFile(archive) as bundle:
        names = [name for name in bundle.namelist() if name.endswith(".json")]
        for index, name in enumerate(names, start=1):
            match = json.loads(bundle.read(name))
            row, rosters = parse_match(match)
            if row is None:
                continue
            match_id = Path(name).stem
            row["match_id"] = match_id
            match_rows.append(row)
            for roster in rosters:
                roster["match_id"] = match_id
                roster_rows.append(roster)
            if index % 5000 == 0:
                print(f"  parsed {index:,} of {len(names):,} scorecards", flush=True)

    matches = add_venue_key(pd.DataFrame(match_rows))
    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    rosters = pd.DataFrame(roster_rows)

    matches = matches.sort_values(["match_type", "date", "match_id"]).reset_index(drop=True)
    teams = team_splits(matches)
    players = player_splits(matches, rosters)

    matches_path = output / "dashboard_toss_matches.csv"
    teams_path = output / "dashboard_teams.csv"
    players_path = output / "dashboard_team_players.csv"
    matches.to_csv(matches_path, index=False)
    teams.to_csv(teams_path, index=False)
    players.to_csv(players_path, index=False)

    print(f"matches: {len(matches):,} -> {matches_path.relative_to(ROOT)}")
    print(f"teams:   {len(teams):,} -> {teams_path.relative_to(ROOT)}")
    print(f"players: {len(players):,} -> {players_path.relative_to(ROOT)}")
    print()
    print(
        matches.groupby("match_type", observed=True)
        .agg(matches=("match_id", "size"), decided=("result", lambda s: s.ne("no_result").sum()))
        .to_string()
    )
    toss = matches[matches["result"].ne("no_result") & matches["winner"].ne("")]
    toss = toss[toss["toss_decision"].isin(["bat", "field"])]
    toss = toss[toss["toss_winner"].ne("")]
    share = (
        toss.assign(toss_won=toss["winner"].eq(toss["toss_winner"]))
        .groupby(["match_type", "toss_decision"], observed=True)["toss_won"]
        .agg(["size", "mean"])
    )
    print()
    print("toss-winner win share by decision")
    print(share.round(3).to_string())


if __name__ == "__main__":
    main()
