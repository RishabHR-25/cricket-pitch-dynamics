# Progress and handoff — 2026-11-09

## TL;DR

- **Weather collection is COMPLETE.** The 2026-11-09 run finished the last 12
  batches: **111/111 batches**, and wrote `data/interim/match_weather.csv` with
  **18,585 rows, 0 missing temperature observations, 0 failed geocodes**.
- **Do not run two collectors at once.** Two concurrent runs on 2026-11-08
  doubled request traffic and repeatedly re-triggered the hourly limit. The
  collector is silent for ~90 s at startup while it parses `all_json.zip`, so a
  quiet log does **not** mean it failed.
- The collector **waits out an hourly 429** using `Retry-After` and resumes
  automatically (capped at 24 waits); only a wait longer than one hour is
  treated as daily exhaustion and fails fast. Patch: `src/collect_match_weather.py`.
- **Geocoding is complete**: 413/413 location strings have coordinates. The 94
  previously unresolved venues were reviewed by hand in
  `data/interim/venue_city_overrides.csv`.
- `match_weather.csv` has **not** been produced yet. No partial file was left
  behind; the collector writes it only after all batches succeed.
- All code and documentation changes below are ready to commit and hand off.
- The **Phase-I report** is written and rendered: `reports/EL_Phase1_Report_Content.md`
  (source), `.docx` (RVCE cover, logo header, footer) and `.pdf`.
- A teammate's **preliminary T20I analysis** was pulled after this work
  (`4d37bed`, `c4e8e26`). It confirms that **bowling-role coverage, not weather,
  is the current binding constraint on model quality** — see the next section.

## Current data inventory

| Item | State |
| --- | --- |
| Matches parsed from `all_json.zip` | 18,585 (T20 14,425 · ODI 3,237 · Test 923) |
| Date range | 2001-12-19 → 2026-11-03 |
| Distinct match locations | 338 |
| Geocode audit | 413 rows, **0 unresolved**, 94 reviewed overrides |
| ERA5 batches required | 111 |
| ERA5 batches completed | **111/111 (complete)** |
| `data/interim/match_weather.csv` | **written**: 18,585 rows, 0 missing weather |

Collection finished on 2026-11-09 19:24 IST: `Wrote 18585 match-weather rows;
0 have no daily temperature observation; 0 of 338 locations failed geocoding.`
A batch's cache key is the coordinate set and year, so the six batches built on
pre-review coordinates are stale; the rest are reused. The cache directory is
`.gitignore`d, so it stays on the local disk only — copy
`data/interim/weather_cache/` if you move machines.

## Pulled teammate results: preliminary T20I analysis

Source: `models/pitch_proxy_t20i_preliminary/` and
`src/run_t20_preliminary_analysis.py` (T20 internationals only, **no weather**,
role-complete matches only).

| Quantity | Value |
| --- | --- |
| T20I matches parsed | 5,849 |
| Matches usable after role join | 869 (14.9%) |
| Credited-wicket coverage | 42.0% |
| Labels | Pace 587 · Spin 213 · Batting 68 |
| Chronological test set | 165 matches (from 2024-02-09) |
| Best test macro-F1 | XGBoost 0.410 (RF 0.405, LR 0.406) |
| Majority-class macro-F1 | 0.245 (but accuracy 0.582 vs XGB 0.497) |
| Time-CV macro-F1 | RF 0.451 · LR 0.400 · XGB 0.395 |
| Calibration (ECE) | RF 0.022 · XGB 0.144 · LR 0.183 |

What this means for the program:

1. **Role coverage is the bottleneck.** Only 15% of T20Is survive the role join;
   `data/processed/role_curation_curve.csv` shows the payoff from curating the
   biggest missing bowlers: top 200 players → 1,318 usable matches, top 400 →
   1,618, top 800 → 2,350. `data/interim/bowler_role_review_queue.csv` is the
   work list. Curating it will help metrics far more than adding weather to the
   current 869 matches.
2. **Do not report accuracy alone.** The majority baseline wins on accuracy
   because Pace dominates; macro-F1 and balanced accuracy are the meaningful
   comparisons, and the current models are only modestly above the venue prior.
3. **`gender` is the top SHAP feature** (0.41). Removing it drops XGB macro-F1
   from 0.410 to 0.347, so a large part of the signal is a gender/label
   association rather than pitch behaviour. Investigate before writing this up;
   consider a gender-stratified model or reporting male/female results
   separately.
4. Weather is limited by Open-Meteo's hourly request window (5,000 calls/hour);
   the collector now waits it out and resumes. When `match_weather.csv` lands,
   re-run this same preliminary comparison with the weather columns to quantify
   whether they add out-of-time signal.

## What changed in this work

`src/collect_match_weather.py`

1. Reviewed `--overrides` table (default
   `data/interim/venue_city_overrides.csv`) applied before ERA5 requests.
   Overrides can carry explicit `latitude`, `longitude`, and `timezone`; explicit
   coordinates are authoritative.
2. New `resolution` column in `geocode_audit.csv`
   (`reviewed_override:<city>`), and new `weather_geocode_name` /
   `weather_geocode_resolution` columns in `match_weather.csv`.
3. `--geocode-only` (resolve and audit without spending weather quota) and
   `--refresh-overrides` (re-apply reviewed rows deliberately).
4. Fixed a bug where merging override results overwrote the audit's
   `location_key`; the writer now always keys rows by the cache key.
5. Shared cache-key helper, per-10-batch progress logging, and rate-limit
   handling: an hourly 429 is waited out via `Retry-After` and retried without
   consuming the retry budget; only a wait longer than one hour is treated as
   daily exhaustion and fails fast.

New/changed data and docs

- `data/interim/venue_city_overrides.csv` — 96 reviewed mappings with a `note`
  explaining each decision.
- `data/interim/geocode_audit.csv` — regenerated, fully resolved.
- `src/audit_datasets.py` — readiness checks (one row per match, match-ID
  uniqueness, archive coverage, plausible ERA5 ranges, geocode resolution) plus
  per-format/per-year coverage CSVs.
- `data/README.md` — schemas, sources, licences, and command reference.
- `README.md` — status section rewritten.

Notable corrections found during review: `Port Soif` → Vale, **Georgia**;
`Alur` → **Indonesia**; `Mulapadu` → wrong district; `St Saviour` missing
country. These are now pinned with exact coordinates.

## Phase-I report

- `reports/EL_Phase1_Report_Content.md` — full report source (~5,900 words)
  following the 12-section EL template, with references and a glossary that
  spells out every abbreviation.
- `reports/EL_Phase1_Report_Content.docx` — rendered report: RVCE cover page
  (title, three members with USNs, group), logo header on every body page,
  "Page N | total" footer, 12 tables and 7 figures.
- The project-timeline slide/figure is deliberately **excluded** from the report.
- Every metric is described as agreement with rule-derived **proxy labels**,
  never as measured pitch physics. The Word/PDF snapshot still states "24 of 111
  batches"; weather is now **complete (111/111, 18,585 rows)**. Refresh those two
  sentences (Sections 5.2 and 7.7) before final submission.

## How to resume

Weather is done. From the repo root:

```powershell
# 1. Check readiness (exits non-zero on failure)
python src\audit_datasets.py `
  --weather data\interim\match_weather.csv `
  --matches data\raw\cricsheet\all_json.zip

# 2. Train the three format models
python src\train_pitch_models.py `
  --matches data\raw\cricsheet\all_json.zip `
  --weather data\interim\match_weather.csv `
  --output-dir models\pitch_proxy `
  --device cpu
```

If weather ever needs a refresh, the collector still works and resumes from the
cache; run only **one** collector at a time:

```powershell
python src\collect_match_weather.py `
  --input data\raw\cricsheet\all_json.zip `
  --output-dir data\interim `
  --cache-dir data\interim\weather_cache
```

Keep `data/interim/weather_cache/` unless you are sure you will never refetch:
the folder is Git-ignored, so the 111 cached batches live only on this disk —
copy the whole folder if you switch machines.

Quota note: a request with 25 locations × 365 days × 4 variables is weighted
far above a single call. One full collection can consume most of the 10,000
daily units, and a single run can also exhaust the 5,000-requests-per-hour
window. The collector now waits out an hourly window and resumes by itself; a
wait longer than one hour is treated as daily exhaustion and fails fast. If a
longer pause is needed, use `--geocode-only` while iterating on locations — it
spends no weather quota.

## Notes for training and metrics

- Labels are behavioural proxies (`Batting` / `Spin` / `Pace`), not measured
  pitch truth. Weather is daily, city-level ERA5 reanalysis, not stadium or
  match-time conditions. Report both caveats with any metrics.
- The split is chronological (latest 20% of dates); do not move weather
  imputation or encoding before the split.
- `metrics.json` already reports accuracy, balanced accuracy, macro/weighted F1,
  log loss, Brier score, confusion matrix, class supports, and a majority
  baseline per format. Compare against that baseline before tuning.
- Check role coverage from the Kaggle joins and per-format class balance before
  chasing scores; an imbalanced proxy target caps what any model can show.
- Weather coverage threshold in the audit is 98% by default; matches with no
  observation stay in the table as missing rather than being dropped silently.

## Known caveats / next steps

1. Only the 94 flagged locations were manually reviewed; the other 319 were
   auto-selected by population among exact-name matches. Spot-check high-volume
   venues (e.g. `Hamilton`, `Victoria`, `St John's`) before publication.
2. Test/ODI label thresholds remain initial operational proxies; run the
   sensitivity analysis described in the README before presenting results.
3. After training, inspect `models/pitch_proxy/*/metrics.json` and the
   `weather_coverage_fraction` field to confirm the weather join actually
   improved out-of-time performance rather than adding noise.
