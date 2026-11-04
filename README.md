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