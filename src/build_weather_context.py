"""Add pre-match weather context to the match-weather table.

The collector stores the full daily ERA5 series for every cached batch, but
``match_weather.csv`` only keeps the match-day value. Pitch behaviour depends
much more on the days *before* the match (rain, drying, heat) than on the match
day itself, so this script re-reads the cache and adds strictly prior windows.

No network calls: it reconstructs the same batches as the collector and loads
the cached responses.

Added columns (all computed from dates strictly before the match date):

* ``precip_sum_7d`` / ``precip_sum_14d`` / ``precip_sum_28d`` — accumulated rain
* ``temp_mean_7d`` / ``temp_mean_14d`` — mean daily temperature
* ``humidity_mean_7d`` — mean relative humidity
* ``wind_max_mean_7d`` — mean daily maximum wind
* ``dry_days_before`` — consecutive rain-free days (< 0.5 mm), capped at 30

Usage::

    python src\\build_weather_context.py
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect_match_weather as C  # noqa: E402

RAIN_THRESHOLD_MM = 0.5
MAX_DRY_DAYS = 30


def load_daily_series(
    archive_path: Path,
    geocode_path: Path,
    cache_dir: Path,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Return location_key -> {iso_date: {variable: value}} from the cache."""
    matches = C._load_matches(archive_path)
    geocodes = C._load_geocode_cache(geocode_path)

    years_by_location: dict[int, set[str]] = defaultdict(set)
    for row in matches.to_dict(orient="records"):
        year = int(row["date"][:4])
        if year >= C.ERA5_START_YEAR:
            years_by_location[year].add(row["location_key"])

    series: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    missing_batches = 0
    for year, location_keys in sorted(years_by_location.items()):
        valid_keys = [
            key for key in sorted(location_keys) if geocodes[key].get("latitude") is not None
        ]
        for index in range(0, len(valid_keys), 25):
            chunk = valid_keys[index : index + 25]
            cache_path = cache_dir / (
                f"{C._weather_cache_key([geocodes[key] for key in chunk], year)}.json.gz"
            )
            if not cache_path.exists():
                missing_batches += 1
                continue
            with gzip.open(cache_path, "rt", encoding="utf-8") as file:
                response = json.load(file)
            by_location = response if isinstance(response, list) else [response]
            for location_key, payload in zip(chunk, by_location):
                daily = payload.get("daily", {})
                for position, day in enumerate(daily.get("time", [])):
                    series[location_key][day] = {
                        variable: (
                            daily.get(variable, [None] * position + [None])[position]
                            if position < len(daily.get(variable, []))
                            else None
                        )
                        for variable in C.DAILY_VARIABLES
                    }
    if missing_batches:
        print(f"WARNING: {missing_batches} cache batches were missing")
    return series


def _window_mean(
    days: list[dict[str, Any]], variable: str
) -> float | None:
    values = [day[variable] for day in days if day.get(variable) is not None]
    return sum(values) / len(values) if values else None


def _window_sum(days: list[dict[str, Any]], variable: str) -> float | None:
    values = [day[variable] for day in days if day.get(variable) is not None]
    return float(sum(values)) if values else None


def build_context(
    weather_path: Path,
    series: dict[str, dict[str, dict[str, Any]]],
    matches: pd.DataFrame,
) -> pd.DataFrame:
    weather = pd.read_csv(weather_path)
    location_by_match = dict(
        zip(matches["match_id"].astype(str), matches["location_key"])
    )

    rows: list[dict[str, Any]] = []
    for record in weather.to_dict(orient="records"):
        match_id = str(record["match_id"])
        location_key = location_by_match.get(match_id)
        day = date.fromisoformat(str(record["date"])[:10])
        history = series.get(location_key, {})

        def collect(days_back: int) -> list[dict[str, Any]]:
            result = []
            for offset in range(days_back, 0, -1):
                key = (day - timedelta(days=offset)).isoformat()
                if key in history:
                    result.append(history[key])
            return result

        week = collect(7)
        fortnight = collect(14)
        month = collect(28)
        dry_days = 0
        for offset in range(1, MAX_DRY_DAYS + 1):
            key = (day - timedelta(days=offset)).isoformat()
            entry = history.get(key)
            if entry is None or entry.get("precipitation_sum") is None:
                break
            if entry["precipitation_sum"] < RAIN_THRESHOLD_MM:
                dry_days += 1
            else:
                break

        rows.append(
            {
                "match_id": record["match_id"],
                "precip_sum_7d": _window_sum(week, "precipitation_sum"),
                "precip_sum_14d": _window_sum(fortnight, "precipitation_sum"),
                "precip_sum_28d": _window_sum(month, "precipitation_sum"),
                "temp_mean_7d": _window_mean(week, "temperature_2m_mean"),
                "temp_mean_14d": _window_mean(fortnight, "temperature_2m_mean"),
                "humidity_mean_7d": _window_mean(week, "relative_humidity_2m_mean"),
                "wind_max_mean_7d": _window_mean(week, "wind_speed_10m_max"),
                "dry_days_before": dry_days,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--matches", type=Path, default=root / "data/raw/cricsheet/all_json.zip")
    parser.add_argument("--weather", type=Path, default=root / "data/interim/match_weather.csv")
    parser.add_argument(
        "--geocode-audit", type=Path, default=root / "data/interim/geocode_audit.csv"
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=root / "data/interim/weather_cache"
    )
    parser.add_argument(
        "--output", type=Path, default=root / "data/interim/match_weather.csv"
    )
    args = parser.parse_args()

    archive_matches = C._load_matches(args.matches)
    series = load_daily_series(args.matches, args.geocode_audit, args.cache_dir)
    context = build_context(args.weather, series, archive_matches)

    weather = pd.read_csv(args.weather)
    merged = weather.merge(context, on="match_id", how="left", validate="one_to_one")
    merged.to_csv(args.output, index=False)

    coverage = merged["precip_sum_7d"].notna().mean()
    print(
        f"Wrote {len(merged)} rows with {len(merged.columns)} columns to {args.output} "
        f"({coverage:.1%} prior-window coverage)"
    )
    print(merged[
        [
            "precip_sum_7d",
            "precip_sum_28d",
            "temp_mean_7d",
            "dry_days_before",
        ]
    ].describe().round(2).to_string())


if __name__ == "__main__":
    main()
