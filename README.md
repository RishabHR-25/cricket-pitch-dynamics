# SPORTS-EL

## Cricket pitch proxy-label and feature pipeline

Phase 1 and Phase 2 are implemented in [`src/phase1_phase2.py`](./src/phase1_phase2.py).
The script reads Cricsheet's canonical JSON match files, generates a documented
match-level proxy target (`Batting`, `Spin`, or `Pace`), and produces a
leakage-safe feature table for eligible T20 matches.

### 1. Download ball-by-ball data

Use the [Cricsheet downloads page](https://cricsheet.org/downloads/) and select
a JSON ZIP for a consistent competition/format. For a first experiment, choose
T20 matches (for example, T20 internationals or the Indian Premier League) and
record the competition, gender, date range, download date, and archive filename.
Cricsheet's [JSON format documentation](https://cricsheet.org/format/json/)
describes the source fields.

Cricsheet match data is provided under the
[Open Data Commons Attribution License](https://opendatacommons.org/licenses/by/1.0/).
Read its terms and include attribution and the data version/download date in
the paper and repository. Kaggle can be used to find supplementary datasets,
but prefer Cricsheet as the primary source; before combining a Kaggle dataset,
verify its provenance, license, schema, match coverage, and overlap to avoid
duplicate matches and incompatible definitions.

### 2. Create the required bowler-role lookup

Cricsheet records the bowler's name but does **not** label bowling style as
spin or pace. Prepare a CSV named, for example, `data/bowler_roles.csv`:

```csv
bowler,style
Example Spinner,spin
Example Fast Bowler,pace
```

Add exact names as they appear in Cricsheet JSON. Use a documented, cited
source and a consistent rule for mixed-style bowlers; version this file with
the project and manually verify uncertain entries. The pipeline intentionally
stops if it encounters a credited-wicket taker without a role rather than
silently assigning a style.

### 3. Install and run

From the repository root in PowerShell:

```powershell
python -m pip install pandas numpy
python src\phase1_phase2.py `
  --input data\raw\t20s_json.zip `
  --bowler-roles data\bowler_roles.csv `
  --output-dir data\processed
```

The default proxy thresholds are explicit in the script and should be treated
as starting assumptions for T20, not physical pitch measurements:

1. `Batting` if first-innings run rate is at least 8.5 runs/over and powerplay
   dot-ball percentage is at most 35%.
2. Otherwise, `Spin` if at least two bowler-credited wickets were taken by
   spinners and the spin-to-pace wicket ratio is at least 1.25 (or no
   pace-credited wickets were taken).
3. Otherwise, `Pace`.

Powerplay defaults to six overs. Wides and no-balls are excluded from legal-ball
denominators; the spin/pace comparison counts bowler-credited dismissals only.
These rules and cutoffs need domain review and a sensitivity analysis before
being presented as validated pitch classes.

The output directory contains:

* `X.csv`: venue, year/month/day-of-year, toss decision, whether the toss winner
  batted first, and rolling venue statistics.
* `y.csv`: the match ID and proxy target.
* `proxy_label_audit.csv`: the statistics used to produce each proxy target,
  for review and reproducibility.

The rolling features use at most the previous five matches at that venue.
Matches on the same date do not use each other's outcomes. The first matches
at a venue naturally have missing historical averages; impute these inside
each training fold, not before a time-based split. The same-match statistics
that define `y` are intentionally not included in `X`.

### 4. Add ambient weather

The pipeline can merge a prepared weather CSV using `--weather-csv`. It must
contain one row per match with a `match_id` column and one or more numeric
weather columns; rows without observations remain missing and are reported.
Use venue coordinates plus match dates to retrieve historical observations
from the [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api).
Create and document a venue-to-coordinate table, normalize venue aliases, and
choose weather variables and aggregation windows before modeling. Since
Cricsheet generally does not include match start times, do not claim hourly
match-time alignment unless you obtain and validate start-time data separately.

### 5. Research steps after feature generation

1. Check parsing, match counts, missing dates/venues, role coverage, and proxy
   class balance; manually inspect a sample of the audit rows.
2. Normalize venue aliases (the [Cricsheet Venue Register](https://cricsheet.org/register/)
   can help where an identifier is available) and document the mapping.
3. Add weather features and any independently sourced venue metadata. Keep a
   data manifest with URLs, versions, licenses, and transformation decisions.
4. Freeze the label definition before evaluation; run threshold sensitivity
   checks and explain that the labels are behavioral proxies, not observed
   physical pitch truth.
5. Split train/validation/test data chronologically. Fit imputers and
   categorical encoders on training data only; compare simple baselines with
   the planned multiclass model and report per-class precision, recall, F1,
   confusion matrix, and probability calibration.
6. Only after the prediction stage is validated, build the playing-XI
   optimization stage. It also needs a dated player pool, player roles,
   overseas/nationality status, and explicit team constraints/objective; those
   inputs are not present in ball-by-ball match files.

The supplied abstract PDF currently has no content (0 bytes). Replace it with
the completed abstract before using it to finalize the research objectives,
scope, or paper-specific methodology.

## Separate T20, ODI, and Test models

For the multi-format experiment, use the full
[Cricsheet JSON archive](https://cricsheet.org/downloads/all_json.zip), not just
the T20-international archive. The current local copy is
`data/raw/cricsheet/all_json.zip`. The training code fits a distinct model for
each format, so format-specific score patterns and proxy thresholds are not
mixed.

The full archive is stored with Git LFS because it exceeds GitHub's regular
100 MB file limit. After cloning, install Git LFS and run `git lfs pull` if the
archive is not already present.

Cricsheet does not supply bowling style. The training workflow supplements its
player register with two public Kaggle profile datasets:

* [International Cricket Player Performance Stats](https://www.kaggle.com/datasets/ijajdatanerd/intl-cricketer-stats-odi-t20-tests)
  supplies bowling style and Cricinfo player IDs; the direct ID join is used
  when available.
* [Cricket Players Dataset](https://www.kaggle.com/datasets/dhavalrupapara/cricket-players-worldwide-dataset)
  supplies broader player-style coverage and is used only for a unique exact
  normalized name/alias join when an ID join is unavailable.

Both Kaggle listings currently declare CC0. Their metadata are third-party
player-role labels, not official Cricsheet fields; inspect the source and terms
before publication. The join audit is written to
`models/pitch_proxy/player_role_join_audit.csv`. Matches with an unclassified
credited wicket-taker are excluded from proxy-label training rather than
guessing the player's style. Report this exclusion and role coverage in the
paper.

The team-shareable player registers and role-source ZIPs are in
`data/raw/cricsheet/` and `data/raw/kaggle/`. `data/README.md` documents every
derived artifact, its columns, and its sources.

Geocoding is complete: all 413 location strings in
`data/interim/geocode_audit.csv` have coordinates. The 94 previously unresolved
venues were reviewed by hand and are recorded with a rationale in
`data/interim/venue_city_overrides.csv`; the collector tags them
`reviewed_override:<city>`. Generic place names can geocode to the wrong
country (for example Port Soif → Vale, Georgia; Alur → Indonesia), so keep the
override table authoritative when adding venues.

Weather collection is limited by the **Open-Meteo free-tier request quota**, not
by the pipeline: 111 year/coordinate batches are required. The cache currently
holds 71 files (65 reusable; the rest use pre-review coordinates), leaving 46
batches to fetch, and `data/interim/match_weather.csv` has not been written yet.
The collector caches every successful batch, waits out an hourly 429 using
`Retry-After`, and only fails fast when the wait implies daily exhaustion.
Re-run the same command to resume; no work is repeated. When
`data/interim/match_weather.csv` exists, run the readiness audit before training:

```powershell
python src\audit_datasets.py `
  --weather data\interim\match_weather.csv `
  --matches data\raw\cricsheet\all_json.zip
```

The audit writes `data/processed/data_readiness.json` and coverage CSVs, and
exits non-zero if match coverage, value ranges, or geocoding checks fail. The
weather cache is Git-ignored by default; the checkpoint is force-added so a
teammate can resume on another machine.

### Collect weather and train

Install the project's Python dependencies:

```powershell
python -m pip install pandas numpy scikit-learn xgboost joblib
```

Collect daily city-level historical weather, then train the three models:

```powershell
python src\collect_match_weather.py `
  --input data\raw\cricsheet\all_json.zip `
  --output-dir data\interim `
  --cache-dir data\interim\weather_cache

python src\train_pitch_models.py `
  --matches data\raw\cricsheet\all_json.zip `
  --weather data\interim\match_weather.csv `
  --output-dir models\pitch_proxy `
  --device cuda
```

The weather collector saves a geocoding audit, cached API responses, and
`match_weather.csv`. It uses Open-Meteo's ERA5 daily reanalysis at the match
city (match day, not the stadium or exact start time). The free API is for
non-commercial use; cite Open-Meteo and ERA5, retain the query/cache metadata,
and verify ambiguous city geocodes before a paper submission. ERA5 begins in
1940; older matches remain in the output with missing weather rather than
requesting unavailable observations.

Initial format-specific rules are deliberately explicit and configurable in
`src/train_pitch_models.py`:

* **T20:** `Batting` if first-innings run rate is at least 8.5 and first-six-
  overs dot balls are at most 35%.
* **ODI:** `Batting` if first-innings run rate is at least 5.5 and first-ten-
  overs dot balls are at most 60%.
* **Test:** `Batting` if first-innings run rate is at least 3.8 and first-ten-
  overs dot balls are at most 75%. Tests have no powerplay; this is explicitly
  an opening-ten-overs proxy, not a Test powerplay.
* Otherwise, `Spin` requires at least two spin wickets and a spin-to-pace
  wicket ratio of at least 1.25 (or no pace wickets); otherwise the label is
  `Pace`. All spin/pace counts use bowler-credited dismissals.

The ODI/Test rules are initial operational proxies and need domain validation
and sensitivity analysis. Labels are not physical ground truth. The script
trains only matches with complete bowling-style coverage and enough early
overs; it uses only match-start features, prior same-format venue history (no
same-day/future matches), season/toss/venue metadata, and daily weather. It
keeps the most recent 20% of unique match dates as a chronological test set,
uses class weighting, and compares out-of-time results against a majority
baseline. Metrics measure prediction of the proxy labels, **not** accuracy
against measured pitch behavior.

The training command uses XGBoost's CUDA device when `--device cuda` is set;
use `--device cpu` on machines without a supported CUDA-enabled XGBoost build.

Each format's model folder contains a serialized model, held-out class
probabilities, feature importance, and `metrics.json` with accuracy,
balanced accuracy, macro/weighted F1, log loss, Brier score, confusion matrix,
class supports, and baseline metrics. `proxy_label_audit.csv` and
`summary.json` record the labels and counts used for the run.

## Ball-by-ball pitch dynamics (analysis layer)

The match-level model above is unchanged. A separate analysis layer estimates
whether the *relative* effectiveness of spin and pace changes as the ball ages,
and tests whether that change can be attributed to the surface.

```powershell
python src\build_ball_dataset.py        # data/interim/balls.parquet
python src\pitch_dynamics.py            # over x style panel, specs, placebo
python src\placebo_bias.py              # correct the mis-centred permutation null
python src\wear_robustness.py           # phase, boundary trim, established bowlers
python src\validate_wear.py             # persistence, predictive gain, rolling origin
python src\identify_wear.py all         # second new ball, 2011 rule, within-bowler gradient
python src\decompose_wear.py all        # exact within/between split, selection ladder
python src\across_innings.py            # across-innings stress test
python src\power_audit.py               # smallest detectable effect per design
python src\make_dynamics_figures.py     # reports/figures/fig1-fig6
```

Read the output as a statement about a **relative, population-level gradient**,
never as a physical pitch measurement. The headline identification result is
negative. The naive second-new-ball jump at over 80 (+0.0148 pace dot share) is a
change of personnel: it is +0.0487 in the 472 matches where no pace bowler bowls
on both sides of the boundary and −0.0005 in the 405 where one does, and the pace
share of deliveries rises from 40% to 74% at the boundary. The smooth gradient is
not composition either: an exact decomposition writes the pooled coefficient as
a variance-weighted average of a within-bowler and a between-bowler slope, and the
composition term is +0.00002 in T20 and negative in ODI and Test. The across-innings
Test contrast, the last candidate, is +0.0122 pooled but +0.0035 (p=0.14) when each
side's second innings is compared with its own first. The October
2011 two-new-ball rule leaves the ODI age gradient unchanged. Write-up:
`reports/PAPER_DRAFT.md` and Section 7.10 of
`reports/EL_Phase1_Report_Content.md`. The paper is in standard IEEE conference
format in Markdown (author block, Abstract and Index Terms, Roman-numeral
sections, TABLE I–IX, Fig. 1, Appendix A); it is not yet a compiled PDF, and the
target venue is still to be chosen.

## Dashboard prototype (Pitch Brief)

`dashboard/app.py` is a read-only Streamlit prototype that turns the committed
artifacts into a pre-match brief. It never refits a model or scans
`data/interim/balls.parquet` on interaction.

```powershell
python src\build_dashboard_data.py     # data/processed/dashboard_*.csv
python src\build_dashboard_model.py    # models/dashboard/{t20,odi,test}
python src\build_toss_dataset.py       # toss, result and squad artifacts
streamlit run dashboard/app.py
```

`build_dashboard_data.py` caches the exact parsed match table at
`data/interim/matches_parsed.parquet` (about 50 s from `all_json.zip` if it is
absent). `build_dashboard_model.py` refits the paper's headline match model on
all labelled matches with the per-format venue windows; the model card reports
the rolling-origin CV, not in-sample fit. `build_toss_dataset.py` reads the same
archive for `dashboard_toss_matches.csv`, `dashboard_teams.csv` and
`dashboard_team_players.csv`, which back the Toss advisor tab: the historical
win share of toss winners who bat versus field, shrunk to the format baseline,
plus squad pickers. Probabilities are agreement with the
rule-derived proxy label, never a physical pitch reading. The app ships dark by
default and follows the theme chosen in Streamlit's settings menu. Requires
`streamlit`, `xgboost`, `scikit-learn`, `pandas`, `numpy`, `scipy` and
`matplotlib`.

