"""Add pre-match weather context to the match-weather table.

The collector stores the full daily ERA5 series for every cached batch, but
``match_weather.csv`` only keeps the match-day value. Pitch behaviour depends
much more on the days *before* the match (rain, drying, heat) than on the match
day itself, so this script re-reads the cache and adds strictly prior windows.

No network calls: it reconstructs the same batches as the collector and loads
the cached responses.

Added columns (all computed from dates strictly before the match date):

* precipitation: 3-, 5-, 7-, 14- and 28-day totals, 7-day maximum daily
  rainfall, 7-day rainy-day count, and a 7-day-versus-28-day trend
* temperature: 3-, 7- and 14-day means, 7-day min/max/range
* humidity: 7-day mean and maximum
* wind: 7-day mean and maximum daily wind
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
    """Return location_key -> {iso_date: {variable: value}} from the cache.

    The cache is keyed by batch (a hash of the venue list for one year), so a
    later venue-list update silently orphans valid batches. Instead of trusting
    the key, index every cached payload by the ERA5 grid coordinate it returns
    and attach each venue to its nearest grid cell. This reuses all cached data
    without a single network call.
    """
    matches = C._load_matches(archive_path)
    geocodes = C._load_geocode_cache(geocode_path)

    payloads: list[tuple[float, float, dict[str, dict[str, Any]]]] = []
    for cache_path in sorted(cache_dir.glob("*.json.gz")):
        with gzip.open(cache_path, "rt", encoding="utf-8") as file:
            response = json.load(file)
        for payload in response if isinstance(response, list) else [response]:
            latitude = payload.get("latitude")
            longitude = payload.get("longitude")
            if latitude is None or longitude is None:
                continue
            daily = payload.get("daily", {})
            days = daily.get("time", [])
            payload_series: dict[str, dict[str, Any]] = {}
            for position, day in enumerate(days):
                payload_series[day] = {
                    variable: (
                        daily[variable][position]
                        if variable in daily and position < len(daily[variable])
                        else None
                    )
                    for variable in C.DAILY_VARIABLES
                }
            payloads.append((float(latitude), float(longitude), payload_series))

    required = {
        row["location_key"]
        for row in matches.to_dict(orient="records")
        if int(row["date"][:4]) >= C.ERA5_START_YEAR
    }

    by_cell: dict[tuple[float, float], dict[str, dict[str, Any]]] = {}

    def series_for_cell(cell: tuple[float, float]) -> dict[str, dict[str, Any]]:
        if cell not in by_cell:
            merged: dict[str, dict[str, Any]] = {}
            for latitude, longitude, payload_series in payloads:
                if abs(latitude - cell[0]) <= 0.01 and abs(longitude - cell[1]) <= 0.01:
                    merged.update(payload_series)
            by_cell[cell] = merged
        return by_cell[cell]

    series: dict[str, dict[str, dict[str, Any]]] = {}
    unmatched = 0
    for location_key in sorted(required):
        entry = geocodes.get(location_key, {})
        latitude = entry.get("latitude")
        longitude = entry.get("longitude")
        if latitude is None or longitude is None:
            unmatched += 1
            continue
        nearest = min(
            payloads,
            key=lambda item: (item[0] - latitude) ** 2 + (item[1] - longitude) ** 2,
            default=None,
        )
        if nearest is None:
            unmatched += 1
            continue
        series[location_key] = series_for_cell((nearest[0], nearest[1]))

    if unmatched:
        print(f"WARNING: {unmatched} venues had no cached weather payload")
    return series


def _window_mean(
    days: list[dict[str, Any]], variable: str
) -> float | None:
    values = [day[variable] for day in days if day.get(variable) is not None]
    return sum(values) / len(values) if values else None


def _window_sum(days: list[dict[str, Any]], variable: str) -> float | None:
    values = [day[variable] for day in days if day.get(variable) is not None]
    return float(sum(values)) if values else None


def _window_extreme(
    days: list[dict[str, Any]], variable: str, mode: str
) -> float | None:
    values = [day[variable] for day in days if day.get(variable) is not None]
    if not values:
        return None
    return float(max(values) if mode == "max" else min(values))


def _rainy_days(days: list[dict[str, Any]], threshold: float) -> int:
    values = [
        day["precipitation_sum"]
        for day in days
        if day.get("precipitation_sum") is not None
    ]
    return sum(value >= threshold for value in values)


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
        three_day = collect(3)
        five_day = collect(5)
        precip_7d = _window_sum(week, "precipitation_sum")
        precip_28d = _window_sum(month, "precipitation_sum")
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

        temp_min = _window_extreme(week, "temperature_2m_mean", "min")
        temp_max = _window_extreme(week, "temperature_2m_mean", "max")

        rows.append(
            {
                "match_id": record["match_id"],
                "precip_sum_3d": _window_sum(three_day, "precipitation_sum"),
                "precip_sum_5d": _window_sum(five_day, "precipitation_sum"),
                "precip_sum_7d": precip_7d,
                "precip_sum_14d": _window_sum(fortnight, "precipitation_sum"),
                "precip_sum_28d": precip_28d,
                "precip_max_7d": _window_extreme(week, "precipitation_sum", "max"),
                "precip_days_7d": _rainy_days(week, RAIN_THRESHOLD_MM),
                "precip_trend_7_28": (
                    precip_7d - precip_28d / 4.0
                    if precip_7d is not None and precip_28d is not None
                    else None
                ),
                "temp_mean_3d": _window_mean(three_day, "temperature_2m_mean"),
                "temp_mean_7d": _window_mean(week, "temperature_2m_mean"),
                "temp_mean_14d": _window_mean(fortnight, "temperature_2m_mean"),
                "temp_min_7d": temp_min,
                "temp_max_7d": temp_max,
                "temp_range_7d": (
                    temp_max - temp_min
                    if temp_max is not None and temp_min is not None
                    else None
                ),
                "humidity_mean_7d": _window_mean(week, "relative_humidity_2m_mean"),
                "humidity_max_7d": _window_extreme(
                    week, "relative_humidity_2m_mean", "max"
                ),
                "wind_mean_7d": _window_mean(week, "wind_speed_10m_max"),
                "wind_max_7d": _window_extreme(week, "wind_speed_10m_max", "max"),
                "wind_max_mean_7d": _window_mean(week, "wind_speed_10m_max"),
                "dry_days_before": dry_days,
            }
        )
    return pd.DataFrame(rows)


CONTEXT_COLUMNS = [
    "precip_sum_3d",
    "precip_sum_5d",
    "precip_sum_7d",
    "precip_sum_14d",
    "precip_sum_28d",
    "precip_max_7d",
    "precip_days_7d",
    "precip_trend_7_28",
    "temp_mean_3d",
    "temp_mean_7d",
    "temp_mean_14d",
    "temp_min_7d",
    "temp_max_7d",
    "temp_range_7d",
    "humidity_mean_7d",
    "humidity_max_7d",
    "wind_mean_7d",
    "wind_max_7d",
    "wind_max_mean_7d",
    "dry_days_before",
]

# Features that survived the extended-vs-core ablation (see PROGRESS.md). The
# shipped table uses this subset because the extra descriptors did not improve
# rolling-CV agreement and slightly hurt one-day internationals.
CORE_CONTEXT_COLUMNS = [
    "precip_sum_7d",
    "precip_sum_14d",
    "precip_sum_28d",
    "temp_mean_7d",
    "temp_mean_14d",
    "humidity_mean_7d",
    "wind_max_mean_7d",
    "dry_days_before",
]


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
    parser.add_argument(
        "--feature-set",
        choices=("core", "extended"),
        default="core",
        help=(
            "core keeps the eight context features that survived ablation; "
            "extended keeps every derived descriptor (ablation only)."
        ),
    )
    args = parser.parse_args()

    archive_matches = C._load_matches(args.matches)
    series = load_daily_series(args.matches, args.geocode_audit, args.cache_dir)
    context = build_context(args.weather, series, archive_matches)

    weather = pd.read_csv(args.weather)
    legacy_context = set(CONTEXT_COLUMNS) | {"wind_max_mean_7d"}
    weather = weather.drop(
        columns=[column for column in weather.columns if column in legacy_context]
    )
    merged = weather.merge(context, on="match_id", how="left", validate="one_to_one")
    if args.feature_set == "core":
        merged = merged[
            [column for column in merged.columns if column not in CONTEXT_COLUMNS]
            + CORE_CONTEXT_COLUMNS
        ]
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
