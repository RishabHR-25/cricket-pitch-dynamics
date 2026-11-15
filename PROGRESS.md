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
- **Ball-by-ball pitch dynamics (2026-11-12)** are built and audited. The
  within-innings age x spin gradient is precise and stable but is **not**
  identified as pitch wear. The second-new-ball jump at over 80 is a change of
  personnel: naive +1.5 pp pace dot share, +4.9 pp in the matches where no pace
  bowler bowls on both sides of the boundary, and zero where one does. The smooth
  gradient is *not* composition either: an exact decomposition puts the
  composition term at 1% of the coefficient in T20 and negative in ODI and Test.
  The October 2011 two-new-ball rule leaves the ODI gradient unchanged. The last
  candidate, a one-point across-innings Test contrast, is four times larger in
  follow-on matches and falls to +0.0035 (p=0.14) when each side is compared with
  its own first innings.
  Conference draft: `reports/PAPER_DRAFT.md`.

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
  spells out every abbreviation. It now also carries the dynamics identification
  work (§7.10, Tables 7.12–7.14, Figures 12–13), which post-dates the submitted
  build.
- `reports/EL_Phase1_Report_Content.docx` — rendered report as submitted on
  2026-11-09: RVCE cover page (title, three members with USNs, group), logo
  header on every body page, "Page N | total" footer. It contains the Phase-I
  results as they stood at submission and does **not** include §7.10; rebuild it
  from the markdown only if a fresh copy is wanted.
- The project-timeline slide/figure is deliberately **excluded** from the report.
- Every metric is described as agreement with rule-derived **proxy labels**,
  never as measured pitch physics. The report now states the finished weather
  position (111/111 batches, 18,585 rows, 100% coverage) instead of the earlier
  "24 of 111" snapshot; Sections 5.2, 7.7, the objectives table and the
  conclusion were refreshed and the `.docx`/PDF rebuilt on 2026-11-09.

## Ball-by-ball pitch dynamics and identification (2026-11-12)

The match-level model is shipped and untouched. This section is a separate
analysis layer over the raw deliveries, and its purpose is to decide what a
within-match "pitch wear" estimate can actually claim.

### Data and estimator

- `src/build_ball_dataset.py` → `data/interim/balls.parquet`: **6,776,668
  deliveries, 23,241 matches** (T20 / ODI / Test, both genders, 2001-12-19 →
  2026-11-03). Bowling style is attached with the conservative role protocol;
  **22.6% of deliveries have unknown style and are excluded**.
- Analysis sample: **5,116,165 legal spin/pace deliveries → 855,659 over x style
  cells** (three or more legal balls per cell, weighted by ball count).
- `src/pitch_dynamics.py` fits `y = match FE + age + spin + age x spin + controls`
  on the cell panel (`models/pitch_dynamics/over_panel.parquet`), with
  match-clustered standard errors. Venue, match, match x innings and
  match x innings x bowler fixed effects are all reported.
- The estimand is **relative**: how spin's outcomes move against pace's as the
  ball ages. It is never a physical pitch measurement.

### What the gradient does

| Check | Result | Verdict |
| --- | --- | --- |
| Pooled dot-share coefficient | +0.0015 (T20), +0.0015 (ODI), +0.0001 (Test), p<0.001 | precise but ambiguous |
| Within-innings placebo (100 draws) | null mean +0.00027 (T20), +0.00030 (ODI), +0.00009 (Test); T20 runs null = observed, T20 wicket null = 21x observed | **null is mis-centred and unusable**, so permutation p-values are not used |
| Null-imposed wild bootstrap (600 draws) | null centred at 0; sd reproduces the clustered SE (0.00015/0.00009/0.00003); dot z = 9.3, 15.5, 3.9 | gradient is precise and real; the Test p=0.06 is an artefact |
| Phase decomposition | T20 powerplay +0.00752, middle +0.00114, death −0.00623 | **sign reverses** |
| Boundary trim | T20 runs −0.00198 → −0.00010 (p=0.87) | runs channel is an innings-edge artefact |
| Persistence across innings | r = −0.003 / −0.047 / +0.030 | no match-level persistence |
| Out-of-sample gain | 0.002–0.153% of weighted MSE | economically negligible |
| Rolling origin (4 folds) | dot share same sign 12/12; magnitudes stable | not a period effect |
| Within-bowler FE (full controls) | +0.00196 (T20), +0.00223 (ODI), +0.00027 (Test), p<0.001 | **not composition**; never smaller than pooled |
| Exact decomposition, composition term | +0.00002 (T20), −0.00047 (ODI), −0.00012 (Test) | composition is 1% of the coefficient or **negative** |

### What the gradient is not

`src/identify_wear.py` runs the three identification tests.

1. **Second new ball at over 80 (Test).** Event study in two-over bins relative
   to overs 78-79 on 162,467 deliveries, match-clustered SE. Across all
   deliveries pace's dot share rises **+0.0148** (p=0.031) in overs 80-81. The
   rise is not uniform across matches: **+0.0487** (p<0.001) in the 472 matches
   where no pace bowler bowls on both sides of the boundary, **−0.0005**
   (p=0.95) in the 405 where one does, **−0.0204** among those continuing
   bowlers at match FE and **−0.0210** (p=0.044) with bowler x innings FE. The
   pace share of deliveries rises from **40% to 74%** at the boundary. The
   published "within-bowler sign reversal" is therefore mostly a *sample*
   change, not a fixed-effects change; the fixed effect moves the continuing-bowler
   estimate by 0.0006. Outputs: `wear_new_ball_event.csv`,
   `wear_boundary_ladder.csv`.
2. **Two new balls per ODI innings from 2011-10-01.** Each ball ages at about
   half the rate per over of play, so a ball-ageing gradient should roughly
   halve. It does not: ODI dot gradient +0.00129 pre → +0.00138 post, T20 control
   +0.00205 → +0.00146, **DiD +0.00069 (SE 0.00089, p=0.44)** against a predicted
   −0.00064. Strata are disjoint samples, so the DiD SE is the root of the sum
   of the four variances. Output: `wear_two_new_ball.csv`.
3. **Within-bowler age x spin gradient (all formats).** Delivery-level weighted
   least squares with match x innings x bowler fixed effects, match-clustered SE,
   on 5,116,165 deliveries. The spin main effect is absorbed, so the coefficient
   is the spin-minus-pace age-slope difference. With full controls (wickets, run
   rate, batter experience, gender, phase) the dot-share coefficient is
   **+0.00196 (T20), +0.00223 (ODI), +0.00027 (Test)**, all p<0.001, against
   pooled delivery-level estimates of +0.00198, +0.00176 and +0.00014 on the
   same rows. Output: `wear_within_bowler.csv`.
4. **Exact decomposition.** The pooled coefficient is a variance-weighted
   average of the within-bowler and between-bowler slopes, an identity that
   closes to 4e-17. Composition is +0.00002 (T20), −0.00047 (ODI), −0.00012
   (Test), so it never creates the gradient and where it is non-zero it biases a
   wear claim downwards. Output: `wear_decomposition.csv`.
5. **Across-innings stress test (Test, dot share).** The last candidate is the
   later-innings contrast, +0.0122 at match FE on 1,670,736 deliveries in 922
   matches (cell-panel value +0.01073). It is **+0.0390** in the 94 matches with
   a follow-on and **+0.0091** in the 828 without; adding follow-on and
   declaration interactions cuts it 44% to **+0.0068**; comparing each side's
   second innings with its own first under match x team FE gives **+0.0035**
   (SE 0.0024, p=0.14). Two thirds of the contrast is composition and selection.
   The strict interval [−0.001, +0.008] excludes the pooled +0.0107, so the
   pooled magnitude is rejected at 5% but zero is not.
   Output: `wear_across_innings_stress.csv` from `python src\across_innings.py`.
6. **Power audit.** The smallest effect each design detects at 80% power is
   2.80 x SE. Within-innings designs resolve 0.0004–0.0005 per over in T20 and
   ODI (about a fifth of the gradient), so their nulls are not power failures.
   The second-new-ball designs resolve only 1.9–2.9 points, so "no new-ball
   bonus" there means "no bonus above three points"; the selection conclusion
   rests on the 4.9-point between-group difference (3.2 SE), not on that null.
   The across-innings strict design resolves 0.66 points. Output:
   `wear_power_audit.csv` from `python src\power_audit.py`.

Taken together: **the ball-renewal jump is selection, the smooth gradient is not
composition, and neither is identified as surface wear.** The one remaining
candidate, the across-innings Test contrast (dot share +0.01073, p<0.001), loses
two thirds of its size under a within-side design and is not distinguishable from
zero, so no surface-scale signal is confirmed.

### Report and paper

- `reports/EL_Phase1_Report_Content.md` — §4.6 (estimator), §7.8 (placebo with
  real null means), §7.9 (persistence, predictive gain, rolling stability in
  Table 7.12, cross-innings contrast) and the new **§7.10 "Is the gradient ball
  condition or surface change?"** with Table 7.13 (selection ladder), Table 7.14
  (rule DiD), Table 7.15 (within-bowler gradient) and Table 7.16 (across-innings
  stress test).
- `reports/PAPER_DRAFT.md` — conference draft built on the audit result (boundary
  selection, unidentified trend, across-innings contrast not robust): abstract,
  11 sections, 9 tables, reproducibility table, verified ICC rule citations, and
  a short venue-fit note.
- Figures: `reports/figures/fig1`–`fig6` from
  `python src\make_dynamics_figures.py`. `fig5` is the naive-versus-continuing-bowler
  event study; `fig6` is the format-era rule comparison.

### Do not reuse

The earlier decay-profile numbers (+2.1 pp pace dot "spike", ratio 2.07,
"ball condition is first-order") came from a trend-controlled profile that
flips sign under no-trend and linear specifications. They were superseded by the
within-bowler event study and their artifacts were deleted on 2026-11-12. Do not
resurface them.

### Reproduce

```powershell
python src\build_ball_dataset.py
python src\pitch_dynamics.py
python src\placebo_bias.py
python src\wear_robustness.py
python src\validate_wear.py
python src\identify_wear.py all
python src\decompose_wear.py all
python src\across_innings.py
python src\power_audit.py
python src\make_dynamics_figures.py
```

`validate_wear.py` takes tens of minutes on the full panel and streams its table
to `models/pitch_dynamics/validate.log`; redirected Python stdout is buffered,
so an empty-looking log does not mean a stall.

## Dashboard prototype — Pitch Brief (2026-11-15)

`dashboard/app.py` is a read-only Streamlit app that turns the committed
artifacts into a pre-match brief. It does not fit, refit or scan the ball table
on interaction.

- **Inputs.** `data/processed/dashboard_matches.csv` has 18,143 matches (the
  exact parsed population, not the ball-derived table, which differed slightly
  around incomplete first innings). Its labels and every prior-window feature
  match the rolling-CV feature frames with max difference 0.0 and no
  missing-value mismatch. `dashboard_weather_climatology.csv` holds
  venue × format × month means plus venue-overall and format-overall fallbacks.
  `dashboard_venues.csv` is the sidebar directory.
- **Models.** `models/dashboard/{t20,odi,test}/model.joblib` refit the paper's
  headline XGBoost specification on all labelled matches with the per-format
  venue windows (5 / 5 / 10). Each `model_card.json` records 29 features,
  training ranges, label counts and the rolling-origin CV. No in-sample metric
  is reported anywhere.
- **Screens.** Match brief (three proxy-class probabilities, local XGBoost
  contributions, venue snapshot, recent matches, weather inputs); Pitch dynamics
  (phase decomposition, age curves, venue terciles, Test second-new-ball ladder);
  Model card (out-of-fold class report, confusion matrix, calibration, global
  importance); Method & caveats (label rules, estimand, six-check audit,
  limitations).
- **Defaults.** The first venue is the highest-match-count venue with at least 20
  matches and at least 60% role coverage in its last five: Kingsmead for T20,
  R Premadasa for ODI, Mirpur for Test. The dropdown stays ordered by match
  count. Venues with partial role coverage show an explicit imputation note.
- **Weather** is month climatology plus an optional dry or wet build-up scenario,
  not a forecast.
- **Verified.** `streamlit.testing.v1.AppTest` passes for all three formats,
  venue changes, the no-history case, and toss and scenario changes. A local
  server returned `ok` from `/_stcore/health`.
- **Committed inputs.** The final rolling-origin run
  `models/rolling_cv_final` (about 1.3 MB) is force-added, because
  `models/rolling_cv_*/` is otherwise ignored; it holds the CV summary, the
  feature frames behind the deployment refit and the out-of-fold predictions the
  model card reads. `data/interim/matches_parsed.parquet` is the parsed-match
  cache the data build uses when `all_json.zip` is absent.

```powershell
python src\build_dashboard_data.py
python src\build_dashboard_model.py
streamlit run dashboard/app.py
```

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
