# Data artifacts

This folder holds the inputs and derived tables for the pitch-proxy pipeline.
Everything here is generated or documented; do not hand-edit generated files.
The collector and audit commands are in the repository README.

## Raw inputs (`data/raw/`)

| Path | Description | Source / licence |
| --- | --- | --- |
| `cricsheet/all_json.zip` | Ball-by-ball JSON for all formats (Git LFS) | Cricsheet, ODbL |
| `cricsheet/t20s_json.zip` | T20-only JSON archive | Cricsheet, ODbL |
| `cricsheet/people.csv`, `names.csv`, `venues.csv` | Cricsheet registers | Cricsheet, ODbL |
| `kaggle/international_player_stats.zip` | Bowling styles + Cricinfo IDs | Kaggle, CC0 |
| `kaggle/worldwide_player_profiles.zip` | Broader player-style coverage | Kaggle, CC0 |

## Interim tables (`data/interim/`)

### `match_weather.csv`

One row per T20/ODI/Test match in the archive, produced by
`src/collect_match_weather.py`. Every match gets a row even when ERA5 has no
observation (for example matches before 1940 or an unresolved location).

| Column | Meaning |
| --- | --- |
| `match_id` | Cricsheet JSON filename stem; join key to `train_pitch_models.py` |
| `date` | Match start date (`YYYY-MM-DD`) |
| `match_type` | `T20`, `ODI`, or `Test` |
| `city` | City recorded by Cricsheet (may be blank) |
| `location_query` | The string sent to the geocoder |
| `weather_country`, `weather_timezone` | Geocoder result used for the request |
| `weather_geocode_name` | Resolved place name |
| `weather_geocode_resolution` | `auto`, `reviewed_override:<city>`, or `unresolved` |
| `weather_geocode_candidates` | Number of geocoder candidates returned |
| `temperature_2m_mean` | Daily mean 2 m air temperature (°C) |
| `precipitation_sum` | Daily precipitation total (mm) |
| `wind_speed_10m_max` | Daily maximum 10 m wind speed (km/h) |
| `relative_humidity_2m_mean` | Daily mean relative humidity (%) |

Weather is **daily, city-level ERA5 reanalysis**, not stadium conditions and not
aligned to the match start time. Treat it as ambient context, not a measurement
of the pitch.

### `geocode_audit.csv`

The geocoding cache/audit. One row per distinct location string.

| Column | Meaning |
| --- | --- |
| `location_key` | Normalised location string (join key) |
| `query` | Original location string |
| `latitude`, `longitude` | Selected coordinates, or blank when unresolved |
| `timezone`, `country` | Selected place metadata |
| `selected_name` | Name returned by the geocoder (or the override city) |
| `candidate_count` | Number of candidates returned |
| `candidates_json` | Raw candidate list for manual review |
| `resolution` | `reviewed_override:<city>` for manually reviewed rows |

Rows with no current match are retained as history; they do not affect
`match_weather.csv`.

### `venue_city_overrides.csv`

Reviewed corrections for venue strings that the geocoder cannot resolve (for
example `"Arun Jaitley Stadium"` → Delhi). Columns:
`location_query`, `resolved_city`, `country`, optional `latitude`/`longitude`,
and a `note` recording the review rationale. The collector applies these
overrides before requesting ERA5, and tags the result in
`weather_geocode_resolution`. Edit this file deliberately and re-run the
collector; cached response batches are reused.

### `weather_cache/`

One gzipped JSON file per ERA5 request (a year × up to 25 coordinates). The
cache key is the coordinate list plus the requested date range, so an unchanged
batch never hits the network again. The directory is Git-ignored; a checkpoint
is force-added to the repository so collection can resume across machines.

## Processed tables (`data/processed/`)

Produced by `src/audit_datasets.py`:

| File | Contents |
| --- | --- |
| `data_readiness.json` | Pass/fail checks, row counts, coverage, per-format and per-year summaries |
| `weather_coverage_by_format.csv` | Matches and weather coverage for T20/ODI/Test |
| `weather_coverage_by_year.csv` | Matches and weather coverage by season |

Run the audit before every training run:

```powershell
python src\audit_datasets.py `
  --weather data\interim\match_weather.csv `
  --matches data\raw\cricsheet\all_json.zip
```

The command exits non-zero if a check fails (duplicate or missing match IDs,
out-of-range ERA5 values, matches absent from the weather table, or unresolved
locations).
