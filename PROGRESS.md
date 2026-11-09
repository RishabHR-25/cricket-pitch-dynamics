# Progress and handoff — 2026-11-09

## TL;DR

- **Weather collection is COMPLETE.** The 2026-11-09 run finished the last 12
  batches: **111/111 batches**, and wrote `data/interim/match_weather.csv` with
  **18,585 rows, 0 missing temperature observations, 0 failed geocodes**.
- **All three format models are trained and saved** (2026-11-09, CPU, with
  weather context and 527 curated role overrides). Rolling-origin CV macro-F1:
  T20 **0.453**, ODI **0.420**, Test **0.544**; single hold-out: 0.434 / 0.400 /
  0.512 (n=576 / 327 / 128). All above their majority baselines. Artifacts in
  `models/pitch_proxy/` (Git-ignored); reviewable copies in
  `models/pitch_proxy_results/`.
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

## Three-format models trained with weather context (2026-11-09)

Command: `python src\train_pitch_models.py --matches data\raw\cricsheet\all_json.zip
--weather data\interim\match_weather.csv --output-dir models\pitch_proxy --device cpu
--role-overrides data\interim\bowler_role_overrides.csv`

Single chronological hold-out (20% of dates):

| Format | Train / test | Macro-F1 | Balanced acc. | Accuracy | Majority macro-F1 | Per-class F1 (Batting / Pace / Spin) |
| --- | --- | --- | --- | --- | --- | --- |
| T20 | 2,506 / 576 | **0.434** | 0.449 | 0.554 | 0.269 | 0.16 / 0.68 / 0.45 |
| ODI | 1,334 / 327 | **0.400** | 0.413 | 0.465 | 0.248 | 0.19 / 0.57 / 0.44 |
| Test | 529 / 128 | **0.512** | 0.548 | 0.719 | 0.225 | 0.00 / 0.80 / 0.73 |

The hold-out is noisy (bootstrap 95% CI width 0.08–0.11), so the primary
comparison below uses **4-fold rolling-origin cross-validation** over all
labelled matches (train on earlier dates, test on the next contiguous block).

### Pre-match moisture context and model comparison (2026-11-09)

`src/build_weather_context.py` re-reads the ERA5 cache (no new API calls) and
adds strictly prior windows to `match_weather.csv` (18,585 rows, 100% coverage):
7/14/28-day precipitation, 7/14-day mean temperature, 7-day humidity and wind,
and `dry_days_before` (consecutive rain-free days, capped at 30). The previous
day-only table is preserved locally as `data/interim/match_weather_dayonly.csv`.

`src/evaluate_rolling_cv.py` compares variants on identical folds
(`models/rolling_cv/cv_summary.csv`). Pooled macro-F1 over out-of-fold
predictions:

| Variant | T20 | ODI | Test | ECE (T20 / ODI / Test) |
| --- | --- | --- | --- | --- |
| XGBoost, day-only weather (previous default) | 0.421 | 0.405 | 0.525 | 0.066 / 0.140 / 0.106 |
| **XGBoost, moisture context (new default)** | **0.453** | **0.420** | **0.544** | **0.058 / 0.106 / 0.100** |
| Random forest, moisture context | 0.450 | 0.410 | 0.554 | 0.132 / 0.072 / 0.155 |
| XGB + RF probability average, context | 0.458 | 0.418 | 0.541 | 0.079 / 0.069 / 0.069 |
| RF two-stage (Batting vs rest, then Pace vs Spin) | 0.444 | 0.408 | 0.564 | 0.087 / 0.052 / 0.154 |

Decisions:

- **Moisture context is the default.** Under CV it improves XGBoost in every
  format (+0.032 T20, +0.014 ODI, +0.019 Test) and keeps the best single-model
  calibration. The single hold-out disagrees for ODI, which is exactly the
  split noise the CV was built to expose.
- **Keep XGBoost.** RF wins on Test but loses T20/ODI and is much less
  calibrated. The ensemble is near-best everywhere but never best and doubles
  the artifact; the two-stage model helps only Test. Both are recorded in
  `cv_summary.csv` as negative results rather than shipped.
- Weather (day + context) now accounts for 8.9% (T20), 17.7% (ODI) and 22.5%
  (Test) of XGBoost importance. `gender` is still the top feature in T20 and
  ODI; it is absent from the Test top ten.
- Every gain here is within roughly one fold standard deviation (0.02–0.06),
  so the honest claim is "small but consistent improvement", not a step change.

### Role curation addendum (2026-11-09)

`src/build_role_overrides.py` resolves players whose Cricsheet short name does
not match a profile full name. It accepts a player only when the name matches,
the Cricsheet team equals the profile country, and all matching profiles agree
on the style class. It wrote **527 overrides covering 9,768 credited wickets**
(337 pace, 190 spin) to `data/interim/bowler_role_overrides.csv`; 139 conflicting
and 346 domestic-team players stay in the human review queue.

| Quantity | Before overrides | After overrides | Change |
| --- | --- | --- | --- |
| Players with a role | 2,478 | **3,005** | +527 |
| T20 labelled matches | 2,308 | **3,082** | +33.5% |
| ODI labelled matches | 1,199 | **1,661** | +38.5% |
| Test labelled matches | 576 | **657** | +14.1% |
| T20 macro-F1 | 0.431 | **0.437** | +0.006 |
| ODI macro-F1 | 0.415 | **0.417** | +0.002 |
| Test macro-F1 | 0.473 | **0.492** | +0.019 |

(The macro-F1 rows above are the day-only-weather hold-out at the time of the
role work; the moisture-context comparison later in this file supersedes them.)

- Weather coverage is **100%** in every format. With the prior-window features,
  weather's share of XGBoost importance is 8.9% (T20), 17.7% (ODI) and 22.5%
  (Test).
- `gender` is still the top feature in T20 and ODI; it is absent from the Test
  model's top ten.
- The Batting class remains the weak point (F1 0.16 T20, 0.19 ODI, 0.00 Test —
  only 17 Test examples in the hold-out).
- Bug fixed en route: `train_one_format` merged every labeled column with the
  feature frame, so shared context columns were suffixed (`venue_x`/`venue_y`)
  and the feature list no longer matched. It now merges keys + `y` only. The
  scripts had never been run end to end before this.
- Artifacts: `models/pitch_proxy/{t20,odi,test}/{model.joblib,metrics.json,
  feature_importance.csv,test_predictions.csv}` plus `summary.json`; small
  reviewable copies are committed under `models/pitch_proxy_results/`.

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
  never as measured pitch physics. The report now states the finished weather
  position (111/111 batches, 18,585 rows, 100% coverage) instead of the earlier
  "24 of 111" snapshot; Sections 5.2, 7.7, the objectives table and the
  conclusion were refreshed and the `.docx`/PDF rebuilt on 2026-11-09.

## How to resume

Weather is done. From the repo root:

```powershell
# 1. Check readiness (exits non-zero on failure)
python src\audit_datasets.py `
  --weather data\interim\match_weather.csv `
  --matches data\raw\cricsheet\all_json.zip

# 2. Train the three format models (moisture context is automatic)
python src\train_pitch_models.py `
  --matches data\raw\cricsheet\all_json.zip `
  --weather data\interim\match_weather.csv `
  --output-dir models\pitch_proxy `
  --device cpu `
  --role-overrides data\interim\bowler_role_overrides.csv

# 3. Re-run the rolling-origin comparison (~5 min, uses cached frames)
python src\evaluate_rolling_cv.py
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
