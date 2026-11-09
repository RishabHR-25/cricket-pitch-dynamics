"""Build a high-confidence bowling-style override table from public sources.

The training pipeline joins Cricsheet people to Kaggle player profiles by
Cricinfo ID and by unique exact name. That leaves a long tail of players whose
Cricsheet short name (for example ``SO Ngoche``) never equals the profile's full
name (``Shem Ngoche``).

This script resolves a conservative subset of that tail: a player is accepted
only when

1. the Cricsheet name matches a profile by full name or by first-initial plus
   surname in the worldwide profile dataset, **and**
2. the player's Cricsheet team matches the profile's country, **and**
3. every matching profile agrees on the style class (pace or spin).

Anything ambiguous is left to the human review queue
(``data/interim/bowler_role_review_queue.csv``); this script never overrides an
existing mapping and never guesses between conflicting styles.

Output: ``data/interim/bowler_role_overrides.csv`` with provenance columns so
the decision can be audited row by row.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_pitch_models as T  # noqa: E402


def initial_key(name: str) -> tuple[str, str] | None:
    """Return (first initial, surname) for a display name."""
    parts = re.sub(r"[^a-z ]", "", name.lower()).split()
    return (parts[0][0], parts[-1]) if len(parts) >= 2 else None


def player_teams(archive_path: Path) -> dict[str, str]:
    """Map Cricsheet player ID to the team they appear for most often."""
    votes: dict[str, Counter[str]] = defaultdict(Counter)
    with zipfile.ZipFile(archive_path) as archive:
        for name in (n for n in archive.namelist() if n.endswith(".json")):
            match = json.load(archive.open(name))
            registry = match["info"].get("registry", {}).get("people", {})
            teams = match["info"].get("teams", [])
            for inning in match.get("innings", []):
                team = inning.get("team")
                opponents = [t for t in teams if t != team]
                for over in inning.get("overs", []):
                    for delivery in over.get("deliveries", []):
                        batter = registry.get(delivery.get("batter"))
                        if batter:
                            votes[batter][team] += 1
                        bowler = registry.get(delivery.get("bowler"))
                        if bowler and opponents:
                            votes[bowler][opponents[0]] += 1
    return {pid: counts.most_common(1)[0][0] for pid, counts in votes.items() if counts}


def build_overrides(
    archive_path: Path,
    people_path: Path,
    names_path: Path,
    international_zip: Path,
    worldwide_zip: Path,
    queue_path: Path,
) -> pd.DataFrame:
    roles, _ = T.build_player_roles(
        people_path, names_path, international_zip, worldwide_zip
    )

    profiles = pd.read_csv(
        zipfile.ZipFile(worldwide_zip).open("players_data_with_all_info.csv"), dtype=str
    )
    profiles = profiles[profiles["bowlingstyle"].notna()]

    by_full: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    by_init: dict[tuple[str, str], list[tuple[str, str, str, str]]] = defaultdict(list)
    for row in profiles.itertuples():
        style = T._style_class(getattr(row, "bowlingstyle", "") or "")
        if not style:
            continue
        country = getattr(row, "country_name", "") or ""
        entry = (row.fullname, country, style, row.bowlingstyle)
        by_full[T._normalise_name(row.fullname)].append(entry)
        key = initial_key(row.fullname)
        if key:
            by_init[key].append(entry)

    teams = player_teams(archive_path)
    queue = pd.read_csv(queue_path)

    rows: list[dict[str, str]] = []
    for row in queue.itertuples():
        pid = row.cricsheet_id
        if pid in roles:
            continue
        team = teams.get(pid)
        if not team:
            continue

        candidates = list(by_full.get(T._normalise_name(row.bowler), []))
        method = "full name"
        if not candidates:
            key = initial_key(row.bowler)
            candidates = list(by_init.get(key, [])) if key else []
            method = "first initial + surname"
        if not candidates:
            continue

        same_country = [c for c in candidates if c[1].lower() == team.lower()]
        if not same_country:
            continue
        styles = {c[2] for c in same_country}
        if len(styles) != 1:
            continue

        best = same_country[0]
        rows.append(
            {
                "cricsheet_id": pid,
                "bowler": row.bowler,
                "team": team,
                "credited_wickets": int(row.credited_t20i_wickets),
                "style": next(iter(styles)),
                "join_method": method,
                "profile_name": best[0],
                "profile_country": best[1],
                "profile_style": best[3],
                "confidence": "country-unique",
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["credited_wickets", "bowler"], ascending=[False, True]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--matches", type=Path, default=root / "data/raw/cricsheet/all_json.zip")
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
        "--queue", type=Path, default=root / "data/interim/bowler_role_review_queue.csv"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data/interim/bowler_role_overrides.csv",
    )
    args = parser.parse_args()

    overrides = build_overrides(
        args.matches,
        args.people,
        args.names,
        args.international_players,
        args.worldwide_players,
        args.queue,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    overrides.to_csv(args.output, index=False)
    print(
        f"Wrote {len(overrides)} overrides covering "
        f"{int(overrides['credited_wickets'].sum())} credited wickets to {args.output}"
    )
    print(overrides["style"].value_counts().to_string())


if __name__ == "__main__":
    main()
