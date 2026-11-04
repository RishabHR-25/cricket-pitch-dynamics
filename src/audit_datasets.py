"""Audit the weather table and match inventory before model training.

The goal is to make the input data easy to verify: one command checks that the
weather table has one row per eligible match, that every match in the archive
is represented, that the ERA5 values stay inside plausible physical ranges, and
that the geocoding audit is fully resolved. The JSON report is written next to
the processed data and is safe to attach to a paper appendix.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger("data_audit")

WEATHER_COLUMNS = (
    "temperature_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "relative_humidity_2m_mean",
)
PLAUSIBLE_RANGES = {
    "temperature_2m_mean": (-60.0, 60.0),
    "precipitation_sum": (0.0, 2000.0),
    "wind_speed_10m_max": (0.0, 120.0),
    "relative_humidity_2m_mean": (0.0, 100.0),
}


def _load_archive_match_ids(archive_path: Path) -> set[str]:
    # Imported lazily so the audit can still run without reading the archive.
    from collect_match_weather import _load_matches

    return set(_load_matches(archive_path)["match_id"])


def audit(
    weather_path: Path,
    archive_path: Path | None,
    output_dir: Path,
    min_coverage: float,
) -> dict:
    weather = pd.read_csv(weather_path, dtype={"match_id": str})
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    check("weather_file_exists", weather_path.exists(), str(weather_path))
    check("match_id_unique", not weather["match_id"].duplicated().any(),
          f"{int(weather['match_id'].duplicated().sum())} duplicate match_id values")
    check("match_id_complete", weather["match_id"].notna().all(),
          f"{int(weather['match_id'].isna().sum())} missing match_id values")

    available = [column for column in WEATHER_COLUMNS if column in weather.columns]
    check("weather_columns_present", len(available) == len(WEATHER_COLUMNS),
          f"found {available}")

    if archive_path is not None:
        archive_ids = _load_archive_match_ids(archive_path)
        missing = archive_ids - set(weather["match_id"])
        extra = set(weather["match_id"]) - archive_ids
        check("archive_coverage", not missing,
              f"{len(missing)} eligible matches absent; {len(extra)} unknown ids")

    for column in available:
        numeric = pd.to_numeric(weather[column], errors="coerce")
        low, high = PLAUSIBLE_RANGES[column]
        outside = int(((numeric < low) | (numeric > high)).sum())
        check(f"range_{column}", outside == 0, f"{outside} values outside [{low}, {high}]")

    if available:
        observed = weather[available].notna().all(axis=1)
    else:
        observed = pd.Series(False, index=weather.index)
    coverage = float(observed.mean()) if len(weather) else 0.0
    check("weather_coverage", coverage >= min_coverage,
          f"{coverage:.1%} of rows have all weather variables (threshold {min_coverage:.0%})")

    if "weather_geocode_resolution" in weather.columns:
        resolution = weather["weather_geocode_resolution"].value_counts(dropna=False)
        unresolved = int(resolution.get("unresolved", 0))
    else:
        resolution = pd.Series(dtype="int64")
        unresolved = 0
    check("geocodes_resolved", unresolved == 0,
          f"{unresolved} match rows still use an unresolved location")

    by_format = (
        weather.assign(observed=observed)
        .groupby("match_type", dropna=False)
        .agg(
            matches=("match_id", "size"),
            matches_with_weather=("observed", "sum"),
            start=("date", "min"),
            end=("date", "max"),
        )
        .reset_index()
    )
    by_format["weather_coverage"] = by_format["matches_with_weather"] / by_format["matches"]

    weather["_year"] = pd.to_datetime(weather["date"], errors="coerce").dt.year
    by_year = (
        weather.assign(observed=observed)
        .groupby("_year", dropna=False)
        .agg(matches=("match_id", "size"), matches_with_weather=("observed", "sum"))
        .reset_index()
        .rename(columns={"_year": "year"})
    )
    by_year["weather_coverage"] = by_year["matches_with_weather"] / by_year["matches"]

    output_dir.mkdir(parents=True, exist_ok=True)
    by_format.to_csv(output_dir / "weather_coverage_by_format.csv", index=False)
    by_year.to_csv(output_dir / "weather_coverage_by_year.csv", index=False)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "weather_file": str(weather_path),
        "rows": int(len(weather)),
        "weather_coverage": coverage,
        "geocode_resolution_counts": {str(k): int(v) for k, v in resolution.items()},
        "checks": checks,
        "ok": all(item["ok"] for item in checks),
        "coverage_by_format": by_format.to_dict(orient="records"),
        "coverage_by_year": by_year.to_dict(orient="records"),
    }
    (output_dir / "data_readiness.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather", type=Path, default=Path("data/interim/match_weather.csv"))
    parser.add_argument("--matches", type=Path, default=None,
                        help="Optional Cricsheet archive used to verify full match coverage.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--min-coverage", type=float, default=0.98,
                        help="Minimum fraction of matches with complete weather observations.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = audit(args.weather, args.matches, args.output_dir, args.min_coverage)
    LOGGER.info("Readiness: %s (%d rows, %.1f%% weather coverage)",
                "PASS" if report["ok"] else "FAIL", report["rows"],
                100 * report["weather_coverage"])
    for item in report["checks"]:
        if not item["ok"]:
            LOGGER.warning("FAILED %s: %s", item["check"], item["detail"])
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
