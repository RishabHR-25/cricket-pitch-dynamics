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


def _geocode(query: str) -> dict[str, Any]:
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
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(cache[key] for key in sorted(cache))


def _fetch_weather_batch(
    locations: list[dict[str, Any]],
    year: int,
    cache_dir: Path,
) -> list[dict[str, Any]]:
    start = f"{year}-01-01"
    end = min(f"{year}-12-31", date.today().isoformat())
    coordinate_key = ";".join(
        f"{location['latitude']:.4f},{location['longitude']:.4f}"
        for location in locations
    )
    cache_key = hashlib.sha256(
        f"{coordinate_key},{start},{end}".encode("utf-8")
    ).hexdigest()
    cache_path = cache_dir / f"{cache_key}.json.gz"
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
    _write_geocode_cache(geocode_path, geocodes)

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
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    if args.max_workers < 1 or args.max_workers > 8:
        parser.error("--max-workers must be between 1 and 8.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    collect_weather(args.input, args.output_dir, args.cache_dir, args.max_workers)


if __name__ == "__main__":
    main()
