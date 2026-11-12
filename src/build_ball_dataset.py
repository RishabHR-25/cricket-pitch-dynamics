"""Build a ball-by-ball dataset for within-innings pitch-dynamics analysis.

Every delivery in the Cricsheet archive becomes one row, with the bowler's
spin/pace style attached from the audited role table. The unit of analysis is a
legal ball; extras are kept so scoring rates are exact, and ``legal`` marks
whether a delivery advances the over.

The point is to separate **ball age** (how much the surface has been used) from
**match phase** (powerplay, middle, death) and from the bowler's style. Pitch
wear is then inferred from the *relative* change in spin and pace effectiveness,
not from any physical measurement.

Output is Parquet because the archive produces roughly five million deliveries.
Rows are written in chunks so memory stays flat.

Usage::

    python src\\build_ball_dataset.py
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_pitch_models as T  # noqa: E402

SCHEMA = pa.schema(
    [
        ("match_id", pa.string()),
        ("date", pa.string()),
        ("match_type", pa.string()),
        ("venue", pa.string()),
        ("venue_id", pa.string()),
        ("city", pa.string()),
        ("gender", pa.string()),
        ("team_type", pa.string()),
        ("innings", pa.int8()),
        ("over", pa.int16()),
        ("ball_in_over", pa.int16()),
        ("ball_age", pa.int32()),
        ("legal", pa.bool_()),
        ("batting_team", pa.string()),
        ("bowling_team", pa.string()),
        ("batter", pa.string()),
        ("batter_id", pa.string()),
        ("bowler", pa.string()),
        ("bowler_id", pa.string()),
        ("bowler_style", pa.string()),
        ("runs_batter", pa.int16()),
        ("runs_extras", pa.int16()),
        ("runs_total", pa.int16()),
        ("is_wicket", pa.bool_()),
        ("is_bowler_wicket", pa.bool_()),
        ("wicket_kind", pa.string()),
        ("player_out", pa.string()),
        ("score_before", pa.int16()),
        ("wickets_before", pa.int8()),
        ("phase", pa.string()),
    ]
)


def phase_for(match_type: str, over: int) -> str:
    """Coarse match phase; kept separate from ball age so the two can be compared."""
    if match_type == "T20":
        if over < 6:
            return "powerplay"
        if over < 15:
            return "middle"
        return "death"
    if match_type == "ODI":
        if over < 10:
            return "powerplay"
        if over < 40:
            return "middle"
        return "death"
    # Test: the new ball is a real tactical phase, not a powerplay.
    if over < 30:
        return "new_ball"
    if over < 80:
        return "middle"
    return "old_ball"


def match_rows(
    match: dict[str, Any],
    match_id: str,
    roles: dict[str, str],
) -> list[dict[str, Any]]:
    info = match.get("info", {})
    match_type = str(info.get("match_type", ""))
    dates = info.get("dates", [])
    venue = str(info.get("venue", "")).strip()
    if match_type not in T.FORMATS or not dates or not venue:
        return []
    registry = info.get("registry", {}).get("people", {})
    venue_registry = info.get("registry", {}).get("venues", {})
    teams = info.get("teams", [])
    rows: list[dict[str, Any]] = []
    for innings_index, innings in enumerate(match.get("innings", [])):
        if innings.get("super_over"):
            continue
        batting_team = str(innings.get("team", ""))
        bowling_team = next((team for team in teams if team != batting_team), "")
        ball_age = 0
        score = 0
        wickets = 0
        for over in innings.get("overs", []):
            over_number = int(over.get("over", 0))
            for ball_index, delivery in enumerate(over.get("deliveries", []), start=1):
                extras = delivery.get("extras", {})
                runs = delivery.get("runs", {})
                legal = "wides" not in extras and "noballs" not in extras
                wicket_list = delivery.get("wickets", []) or []
                wicket_kinds = [
                    str(wicket.get("kind", "")).strip().lower()
                    for wicket in wicket_list
                ]
                bowler_name = str(delivery.get("bowler", ""))
                bowler_id = str(registry.get(bowler_name, ""))
                batter_name = str(delivery.get("batter", ""))
                rows.append(
                    {
                        "match_id": match_id,
                        "date": str(dates[0]),
                        "match_type": match_type,
                        "venue": venue,
                        "venue_id": str(venue_registry.get(venue, "")),
                        "city": str(info.get("city", "")),
                        "gender": str(info.get("gender", "unknown")),
                        "team_type": str(info.get("team_type", "unknown")),
                        "innings": innings_index + 1,
                        "over": over_number,
                        "ball_in_over": ball_index,
                        "ball_age": ball_age,
                        "legal": legal,
                        "batting_team": batting_team,
                        "bowling_team": bowling_team,
                        "batter": batter_name,
                        "batter_id": str(registry.get(batter_name, "")),
                        "bowler": bowler_name,
                        "bowler_id": bowler_id,
                        "bowler_style": roles.get(bowler_id, "unknown"),
                        "runs_batter": int(runs.get("batter", 0)),
                        "runs_extras": int(
                            runs.get("extras", 0)
                        ),
                        "runs_total": int(runs.get("total", 0)),
                        "is_wicket": bool(wicket_list),
                        "is_bowler_wicket": any(
                            kind in T.WICKET_DISMISSALS for kind in wicket_kinds
                        ),
                        "wicket_kind": wicket_kinds[0] if wicket_kinds else "",
                        "player_out": str(
                            wicket_list[0].get("player_out", "") if wicket_list else ""
                        ),
                        "score_before": score,
                        "wickets_before": wickets,
                        "phase": phase_for(match_type, over_number),
                    }
                )
                score += int(runs.get("total", 0))
                if legal:
                    ball_age += 1
                wickets += len(wicket_list)
    return rows


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matches", type=Path, default=root / "data/raw/cricsheet/all_json.zip"
    )
    parser.add_argument("--people", type=Path, default=root / "data/raw/cricsheet/people.csv")
    parser.add_argument("--names", type=Path, default=root / "data/raw/cricsheet/names.csv")
    parser.add_argument(
        "--international-players",
        type=Path,
        default=root / "data/raw/kaggle/international_player_stats.zip",
    )
    parser.add_argument(
        "--worldwide-players",
        type=Path,
        default=root / "data/raw/kaggle/worldwide_player_profiles.zip",
    )
    parser.add_argument(
        "--role-overrides",
        type=Path,
        default=root / "data/interim/bowler_role_overrides.csv",
    )
    parser.add_argument(
        "--output", type=Path, default=root / "data/interim/balls.parquet"
    )
    parser.add_argument(
        "--chunk-matches", type=int, default=500, help="Matches per Parquet row group."
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Stop after N matches (0 = all)."
    )
    args = parser.parse_args()

    roles, _ = T.build_player_roles(
        args.people,
        args.names,
        args.international_players,
        args.worldwide_players,
        args.role_overrides,
    )
    print(f"Loaded {len(roles):,} player roles")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(args.output, SCHEMA, compression="zstd")
    buffer: list[dict[str, Any]] = []
    processed = 0
    balls = 0
    with zipfile.ZipFile(args.matches) as archive:
        names = sorted(
            name
            for name in archive.namelist()
            if name.lower().endswith(".json") and not name.startswith("__MACOSX/")
        )
        for name in names:
            with archive.open(name) as match_file:
                match = __import__("json").load(match_file)
            buffer.extend(match_rows(match, Path(name).stem, roles))
            processed += 1
            if len(buffer) >= 50_000 or processed % args.chunk_matches == 0:
                table = pa.Table.from_pylist(buffer, schema=SCHEMA)
                writer.write_table(table)
                balls += len(buffer)
                buffer = []
                print(f"  {processed:,} matches, {balls:,} balls written")
            if args.limit and processed >= args.limit:
                break
    if buffer:
        table = pa.Table.from_pylist(buffer, schema=SCHEMA)
        writer.write_table(table)
        balls += len(buffer)
    writer.close()
    print(f"Wrote {balls:,} deliveries from {processed:,} matches to {args.output}")


if __name__ == "__main__":
    main()
