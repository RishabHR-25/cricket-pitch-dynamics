"""Collect daily, city-level ERA5 weather features for Cricsheet matches.

Open-Meteo's free archive service is intended for non-commercial use. This
collector saves geocoding choices and raw responses so locations and retrievals
can be audited and reproduced.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import re
import time
import threading
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd


LOGGER = logging.getLogger("match_weather")
_REQUEST_LOCK = threading.Lock()
_LAST_REQUEST_TIME = 0.0
MATCH_FORMATS = {"T20", "ODI", "Test"}
ERA5_START_YEAR = 1940
DAILY_VARIABLES = (
    "temperature_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "relative_humidity_2m_mean",
)


def _request_json(url: str, *, retries: int = 3) -> Any:
    global _LAST_REQUEST_TIME
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "SPORTS-EL-academic-research/1.0"},
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        with _REQUEST_LOCK:
            wait = 1.0 - (time.monotonic() - _LAST_REQUEST_TIME)
            if wait > 0:
                time.sleep(wait)
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    _LAST_REQUEST_TIME = time.monotonic()
                    return json.load(response)
            except urllib.error.HTTPError as error:
                _LAST_REQUEST_TIME = time.monotonic()
                last_error = error
                retry_after = error.headers.get("Retry-After")
                if (
                    error.code == 429
                    and retry_after
                    and retry_after.isdigit()
                    and int(retry_after) > 3600
                ):
                    reset = error.headers.get("X-RateLimit-Reset")
                    raise RuntimeError(
                        "Open-Meteo daily API quota is exhausted; the collector cannot "
                        "resume until it resets"
                        + (f" (reset epoch {reset} UTC)" if reset else "")
                        + ". The cache and audit are safe: re-run the same command later."
                    ) from error
                if attempt + 1 < retries:
                    delay = (
                        float(retry_after)
                        if retry_after and retry_after.isdigit()
                        else 60.0 if error.code == 429
                        else 2**attempt
                    )
                    time.sleep(delay)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
                _LAST_REQUEST_TIME = time.monotonic()
                last_error = error
                if attempt + 1 < retries:
                    time.sleep(2**attempt)
    raise RuntimeError(f"Request failed after {retries} attempts: {url}") from last_error


def _normalize_location(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def _load_matches(archive_path: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    with zipfile.ZipFile(archive_path) as archive:
        names = sorted(
            name
            for name in archive.namelist()
            if name.lower().endswith(".json") and not name.startswith("__MACOSX/")
        )
        for name in names:
            with archive.open(name) as match_file:
                match = json.load(match_file)
            info = match.get("info", {})
            match_type = str(info.get("match_type", ""))
            dates = info.get("dates", [])
            if match_type not in MATCH_FORMATS or not dates:
                continue
            match_date = pd.to_datetime(dates[0], errors="coerce")
            if pd.isna(match_date):
                continue
            city = str(info.get("city", "")).strip()
            venue = str(info.get("venue", "")).strip()
            if not venue and not city:
                continue
            rows.append(
                {
                    "match_id": Path(name).stem,
                    "date": match_date.strftime("%Y-%m-%d"),
                    "match_type": match_type,
                    "venue": venue,
                    "city": city,
                    "venue_id": info.get("registry", {}).get("venues", {}).get(venue, ""),
                }
            )
    if not rows:
        raise ValueError(f"No dated T20/ODI/Test matches found in {archive_path}")
    cities_by_venue_id: dict[str, set[str]] = defaultdict(set)
    cities_by_venue_name: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["city"]:
            if row["venue_id"]:
                cities_by_venue_id[row["venue_id"]].add(row["city"])
            cities_by_venue_name[_normalize_location(row["venue"])].add(row["city"])
    for row in rows:
        if not row["city"]:
            candidates = (
                cities_by_venue_id.get(row["venue_id"], set())
                if row["venue_id"]
                else cities_by_venue_name.get(_normalize_location(row["venue"]), set())
            )
            if len(candidates) == 1:
                row["city"] = next(iter(candidates))
            elif "," in row["venue"]:
                row["city"] = row["venue"].rsplit(",", 1)[1].strip()
        row["location_query"] = row["city"] or row["venue"]
        row["location_key"] = _normalize_location(row["location_query"])
    return pd.DataFrame(rows)


def _geocode(query: str, expected_country: str | None = None) -> dict[str, Any]:
    url = "https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode(
        {"name": query, "count": 10, "language": "en", "format": "json"}
    )
    response = _request_json(url)
    results = response.get("results", [])
    if not results:
        return {
            "location_key": _normalize_location(query),
            "query": query,
            "latitude": None,
            "longitude": None,
            "timezone": None,
            "country": None,
            "selected_name": None,
            "candidate_count": 0,
            "candidates_json": "[]",
        }

    exact = [
        item
        for item in results
        if _normalize_location(str(item.get("name", ""))) == _normalize_location(query)
    ]
    candidates = exact or results
    if expected_country:
        country_key = _normalize_location(expected_country)
        matching = [
            item
            for item in candidates
            if _normalize_location(str(item.get("country", ""))) == country_key
        ]
        if not matching:
            # Fall back to matching any candidate in the expected country, even
            # if the place name differs slightly (for example a suburb).
            matching = [
                item
                for item in results
                if _normalize_location(str(item.get("country", ""))) == country_key
            ]
        if matching:
            candidates = matching
    selected = max(
        candidates,
        key=lambda item: int(item.get("population") or 0),
    )

    return {
        "location_key": _normalize_location(query),
        "query": query,
        "latitude": selected.get("latitude"),
        "longitude": selected.get("longitude"),
        "timezone": selected.get("timezone"),
        "country": selected.get("country"),
        "selected_name": selected.get("name"),
        "candidate_count": len(results),
        "candidates_json": json.dumps(results, ensure_ascii=True, separators=(",", ":")),
    }


def _load_geocode_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as file:
        return {
            row["location_key"]: {
                **row,
                "latitude": float(row["latitude"]) if row.get("latitude") else None,
                "longitude": float(row["longitude"]) if row.get("longitude") else None,
                "candidate_count": int(row.get("candidate_count") or 0),
            }
            for row in csv.DictReader(file)
        }


def _write_geocode_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    columns = [
        "location_key",
        "query",
        "latitude",
        "longitude",
        "timezone",
        "country",
        "selected_name",
        "candidate_count",
        "candidates_json",
        "resolution",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            {**cache[key], "location_key": key} for key in sorted(cache)
        )


def _load_location_overrides(path: Path | None) -> dict[str, dict[str, str]]:
    """Load the reviewed venue/city corrections keyed by normalized query.

    The file records, for each location string that Open-Meteo could not
    resolve, the city used for the ERA5 request. This keeps a venue such as
    "Arun Jaitley Stadium" from being geocoded as if it were a town name and
    makes the correction auditable instead of implicit.
    """
    if path is None or not path.exists():
        return {}
    overrides: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            query = (row.get("location_query") or "").strip()
            city = (row.get("resolved_city") or "").strip()
            if not query or not city:
                continue
            overrides[_normalize_location(query)] = {
                "city": city,
                "country": (row.get("country") or "").strip(),
                "latitude": (row.get("latitude") or "").strip(),
                "longitude": (row.get("longitude") or "").strip(),
                "timezone": (row.get("timezone") or "").strip(),
                "note": (row.get("note") or "").strip(),
            }
    return overrides


def _apply_location_overrides(
    geocodes: dict[str, dict[str, Any]],
    overrides: dict[str, dict[str, str]],
    max_workers: int,
    refresh: bool = False,
) -> dict[str, dict[str, Any]]:
    """Re-resolve audited locations using the reviewed venue/city table.

    Only entries whose coordinates are missing or that have not yet been
    resolved by a reviewed override are touched, so repeated runs are
    idempotent and successful geocodes are never silently replaced.
    """
    pending: dict[str, dict[str, str]] = {}
    for key, row in geocodes.items():
        override = overrides.get(key) or overrides.get(
            _normalize_location(row.get("query", ""))
        )
        if override is None:
            continue
        marker = f"reviewed_override:{override['city']}"
        if refresh:
            pending[key] = override
            continue
        if override["latitude"] and override["longitude"]:
            same = (
                row.get("latitude") is not None
                and row.get("longitude") is not None
                and abs(float(row["latitude"]) - float(override["latitude"])) < 1e-6
                and abs(float(row["longitude"]) - float(override["longitude"])) < 1e-6
            )
            if same:
                continue
            # Explicit reviewed coordinates are authoritative.
            pending[key] = override
            continue
        if row.get("resolution") == marker and row.get("latitude") is not None:
            continue
        if row.get("latitude") is not None and not str(row.get("resolution", "")).startswith(
            "reviewed_override:"
        ):
            # An automatic geocode succeeded; leave it alone unless the audit
            # explicitly needs review.
            continue
        pending[key] = override

    if not pending:
        return geocodes

    def resolve(item: tuple[str, dict[str, str]]) -> tuple[str, dict[str, Any]]:
        key, override = item
        marker = f"reviewed_override:{override['city']}"
        if override["latitude"] and override["longitude"]:
            return key, {
                "latitude": float(override["latitude"]),
                "longitude": float(override["longitude"]),
                "timezone": override["timezone"] or None,
                "country": override["country"] or None,
                "selected_name": override["city"],
                "candidate_count": 0,
                "candidates_json": "[]",
            }
        resolved = _geocode(override["city"], override["country"] or None)
        resolved["selected_name"] = resolved.get("selected_name") or override["city"]
        if not resolved.get("country") and override["country"]:
            resolved["country"] = override["country"]
        if not resolved.get("timezone") and override["timezone"]:
            resolved["timezone"] = override["timezone"]
        return key, resolved

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(resolve, item): item[0] for item in pending.items()}
        for future in as_completed(futures):
            key, resolved = future.result()
            marker = f"reviewed_override:{pending[key]['city']}"
            row = dict(geocodes[key])
            row.update(
                {
                    name: value
                    for name, value in resolved.items()
                    if name not in {"location_key", "query"}
                }
            )
            row["resolution"] = marker
            geocodes[key] = row
            if resolved.get("latitude") is None:
                LOGGER.warning("Reviewed override for %r still has no coordinates.", key)
    return geocodes



def _weather_cache_key(locations: list[dict[str, Any]], year: int) -> str:
    start = f"{year}-01-01"
    end = min(f"{year}-12-31", date.today().isoformat())
    coordinate_key = ";".join(
        f"{location['latitude']:.4f},{location['longitude']:.4f}"
        for location in locations
    )
    return hashlib.sha256(
        f"{coordinate_key},{start},{end}".encode("utf-8")
    ).hexdigest()


def _fetch_weather_batch(
    locations: list[dict[str, Any]],
    year: int,
    cache_dir: Path,
) -> list[dict[str, Any]]:
    start = f"{year}-01-01"
    end = min(f"{year}-12-31", date.today().isoformat())
    cache_path = cache_dir / f"{_weather_cache_key(locations, year)}.json.gz"

    if cache_path.exists():
        with gzip.open(cache_path, "rt", encoding="utf-8") as file:
            response = json.load(file)
        return response if isinstance(response, list) else [response]

    params = {
        "latitude": ",".join(str(location["latitude"]) for location in locations),
        "longitude": ",".join(str(location["longitude"]) for location in locations),
        "start_date": start,
        "end_date": end,
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "auto",
        "models": "era5",
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(params)
    result = _request_json(url)
    response = result if isinstance(result, list) else [result]
    if len(response) != len(locations):
        raise ValueError(
            f"ERA5 returned {len(response)} locations for {len(locations)} requested."
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache_path, "wt", encoding="utf-8") as file:
        json.dump(response, file, separators=(",", ":"))
    return response


def collect_weather(
    archive_path: Path,
    output_dir: Path,
    cache_dir: Path,
    max_workers: int = 4,
    overrides_path: Path | None = None,
    geocode_only: bool = False,
    refresh_overrides: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matches = _load_matches(archive_path)
    queries = (
        matches.groupby("location_key", as_index=False)
        .agg(query=("location_query", "first"), start=("date", "min"), end=("date", "max"))
    )
    geocode_path = output_dir / "geocode_audit.csv"
    geocodes = _load_geocode_cache(geocode_path)
    missing_queries = [
        row for row in queries.to_dict(orient="records") if row["location_key"] not in geocodes
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_geocode, row["query"]): row["location_key"]
            for row in missing_queries
        }
        for future in as_completed(futures):
            geocodes[futures[future]] = future.result()

    overrides = _load_location_overrides(overrides_path)
    if overrides:
        geocodes = _apply_location_overrides(
            geocodes, overrides, max_workers, refresh=refresh_overrides
        )
    _write_geocode_cache(geocode_path, geocodes)
    unresolved = {
        key: row
        for key, row in geocodes.items()
        if row.get("latitude") is None and key in set(queries["location_key"])
    }
    if unresolved:
        LOGGER.warning(
            "%d locations used by matches are still unresolved; their weather will be missing: %s",
            len(unresolved),
            ", ".join(sorted(row.get("query", key) for key, row in unresolved.items())),
        )
    if geocode_only:
        LOGGER.info("Geocoding stage complete; skipping weather requests.")
        return pd.DataFrame(), queries


    weather_by_location: dict[str, dict[str, dict[str, Any]]] = {}
    years_by_location: dict[int, set[str]] = defaultdict(set)
    for row in matches.to_dict(orient="records"):
        year = int(row["date"][:4])
        if year >= ERA5_START_YEAR:
            years_by_location[year].add(row["location_key"])
    batches: list[tuple[int, list[str]]] = []
    for year, location_keys in sorted(years_by_location.items()):
        valid_keys = [
            key
            for key in sorted(location_keys)
            if geocodes[key].get("latitude") is not None
        ]
        batches.extend(
            (year, valid_keys[index:index + 25])
            for index in range(0, len(valid_keys), 25)
        )
    cached_batches = sum(
        (cache_dir / f"{_weather_cache_key([geocodes[key] for key in keys], year)}.json.gz").exists()
        for year, keys in batches
    )
    LOGGER.info(
        "Weather batches: %d total, %d cached, %d to fetch.",
        len(batches),
        cached_batches,
        len(batches) - cached_batches,
    )
    completed_batches = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _fetch_weather_batch,
                [geocodes[key] for key in location_keys],
                year,
                cache_dir,
            ): (year, location_keys)
            for year, location_keys in batches
        }
        for future in as_completed(futures):
            year, location_keys = futures[future]
            for location_key, result in zip(location_keys, future.result()):
                daily = result.get("daily", {})
                dates = daily.get("time", [])
                by_day = weather_by_location.setdefault(location_key, {})
                for index, day in enumerate(dates):
                    by_day[day] = {
                        variable: (
                            daily.get(variable, [None] * len(dates))[index]
                            if index < len(daily.get(variable, []))
                            else None
                        )
                        for variable in DAILY_VARIABLES
                    }
            completed_batches += 1
            if completed_batches % 10 == 0 or completed_batches == len(batches):
                LOGGER.info(
                    "Weather batches completed: %d/%d", completed_batches, len(batches)
                )

    rows: list[dict[str, Any]] = []
    for match in matches.to_dict(orient="records"):
        observed = weather_by_location.get(match["location_key"], {}).get(match["date"], {})
        geocode = geocodes[match["location_key"]]
        rows.append(
            {
                "match_id": match["match_id"],
                "date": match["date"],
                "match_type": match["match_type"],
                "city": match["city"],
                "location_query": match["location_query"],
                "weather_country": geocode.get("country"),
                "weather_timezone": geocode.get("timezone"),
                "weather_geocode_name": geocode.get("selected_name"),
                "weather_geocode_resolution": geocode.get("resolution") or "auto",
                "weather_geocode_candidates": geocode.get("candidate_count"),
                **observed,
            }
        )

    weather = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    weather.to_csv(output_dir / "match_weather.csv", index=False)
    missing_geocodes = (
        int(weather["temperature_2m_mean"].isna().sum())
        if "temperature_2m_mean" in weather.columns
        else len(weather)
    )
    LOGGER.info(
        "Wrote %d match-weather rows; %d have no daily temperature observation; "
        "%d of %d locations failed geocoding.",
        len(weather),
        missing_geocodes,
        sum(geocodes[row["location_key"]].get("latitude") is None for row in queries.to_dict(orient="records")),
        len(queries),
    )
    return weather, queries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Cricsheet all-format JSON ZIP.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/interim/weather_cache"))
    parser.add_argument(
        "--overrides",
        type=Path,
        default=Path("data/interim/venue_city_overrides.csv"),
        help="Reviewed venue-to-city corrections applied to unresolved locations.",
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--geocode-only",
        action="store_true",
        help="Resolve and audit locations, then stop before requesting ERA5.",
    )
    parser.add_argument(
        "--refresh-overrides",
        action="store_true",
        help="Re-apply every reviewed override even if the audit is already resolved.",
    )
    args = parser.parse_args()
    if args.max_workers < 1 or args.max_workers > 8:
        parser.error("--max-workers must be between 1 and 8.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    collect_weather(
        args.input,
        args.output_dir,
        args.cache_dir,
        args.max_workers,
        args.overrides,
        args.geocode_only,
        args.refresh_overrides,
    )


if __name__ == "__main__":
    main()
