# Predicting Cricket Pitch Conditions Using Machine Learning for Data-Driven Match Strategy

**Experiential Learning (EL) — Phase-I Report**

**Theme:** Sports · **Group:** 31

**Team:** Aravind R (1RV25ET007) · Rishab H R (1RV25EC154) · Ujwal S Nair (1RV25EC229)

**Date:** 4 November 2026

## 1. Introduction

### 1.1 Context

Cricket is played on a pitch whose behaviour — how much the ball bounces, seams, turns or comes onto the bat — shapes scoring, wicket-taking and the decision made at the toss. Two grounds can host the same teams on the same day and produce completely different matches because the surface behaves differently. Yet no open, standard dataset records pitch behaviour directly. Public ball-by-ball archives record *what happened* on a pitch, not *what the pitch was like*.

This project asks whether pitch behaviour can be inferred, before a match, from free public data. The input is historical ball-by-ball records, player bowling styles and daily historical weather at the match city. The output is a three-class description of the surface (batting-friendly, pace-friendly or spin-friendly), an explanation of the drivers behind that prediction, and — as the applied goal — a bat-first or field-first recommendation at the toss.

### 1.2 Why the problem matters

Three observations motivate the work.

1. **The toss is worth points.** Sood and Willis (2016), analysing more than 41,000 matches, estimate that winning the toss raises the chance of winning by about 2.8% overall, with larger effects in one-day cricket. A better pre-match read of the surface therefore has a direct competitive value.
2. **Visual assessment is imperfect.** Adams et al. (2005) measured 92 County Championship pitches and compared umpires' ratings with high-speed video. Umpires assessed bounce reliably but not pace; pace had to be modelled from bounce and surface friction. Expert visual inspection is useful but not a measurement.
3. **Venues genuinely differ.** In our own T20I (Twenty20 International) sample, the 13 men's venues with at least 25 matches and at least 60% of wicket-taking roles known range from 7.1 to 9.0 runs per over. That spread is large enough to change selection and toss decisions.

The beneficiaries are teams playing at unfamiliar venues, where local knowledge is thinnest, and analysts who need a reproducible, auditable method rather than anecdote. Because the method uses only free and open data, it is also available to associate and domestic teams that cannot buy commercial scouting services.

## 2. Problem Definition

### 2.1 Problem statement

There is no open, repeatable way to classify how a cricket pitch will behave before a match. As a result, the toss decision and the pace/spin balance of the playing eleven still rest on visual inspection, broadcast commentary and venue experience. Those judgements are hard to check, hard to reproduce and unavailable to teams without a long history at the ground.

Two constraints define the problem. First, no public dataset contains physical pitch measurements linked to match identifiers. Second, weather and pitch behaviour interact (moisture, humidity, wind and temperature affect preparation and deterioration), so any useful method must combine match records with ambient conditions. The scientific difficulty is therefore one of **weak supervision**: the target must be constructed from observed play, and every claim must be honest about the fact that the label is a proxy.

### 2.2 Background and literature review

Research on cricket analytics splits into three streams: physical measurement of pitches, prediction of match outcomes, and prediction of scores or chases. None of the studies reviewed below predicts pitch behaviour itself from open ball-by-ball data.

| Study | Data and method | Key finding | Limitation for our problem |
| --- | --- | --- | --- |
| Adams et al. (2005) | 92 County Championship pitches; bounce, hardness and friction measured; high-speed video on 23 | Umpires rate bounce reliably but not pace; pace can be modelled from bounce and friction | Requires on-site instruments; not scalable to historical matches |
| Morley and Thomas (2005) | English one-day matches; logistic regression | Home teams won about 57%; the toss matters, but team quality and match importance dominate | Models the result, not the pitch |
| Chakraborty et al. (2024) | T20 (Twenty20) winners; logistic regression, support vector machine, random forest, decision tree and XGBoost | Random forest best, 84.06% accuracy | Predicts the winner, not the pitch |
| Shah et al. (2025) | 458 T20Is, 2005–2024; seven classifiers; venue, toss, pitch and rankings | Random forest and decision tree best for high run chases | Pitch is an input, not an output |
| Bhatnagar and Bhatnagar (2025) | Indian Premier League (IPL) 2008–2024; automated machine learning, explainable artificial intelligence, weather and large-language-model reads of pitch reports | Pitch reports and pre-match factors help classify first-innings scores | Relies on subjective pitch reports and is IPL-only |

Three trends emerge. Tree ensembles (random forest and gradient boosting) repeatedly win tabular comparisons; explainable methods are becoming standard; and the newest work tries to combine structured data with unstructured pitch reports. The gap is consistent: existing work treats the pitch as an **input** or predicts the **result**, while the pitch itself remains unmeasured.

### 2.3 Research gap and contribution

This project treats pitch behaviour as the **target**. We construct a documented proxy label from ball-by-ball evidence, build leakage-safe features from earlier matches only, train interpretable multiclass classifiers, and explain each prediction. The contribution is a reproducible open-data pipeline, a venue-level Pitch Behaviour Profile (PBP), a first version of a Pitch Wear Index (PWI), and a planned cold-start method for grounds with little history.

## 3. Objectives

### 3.1 Primary objective

Classify pitch behaviour before a match using free public data, and convert that classification into an interpretable toss recommendation.

### 3.2 Specific objectives and success measures

| # | Objective | Measure of success | Status at Phase-I |
| --- | --- | --- | --- |
| 1 | Complete the data | Weather audit passes at 98% match coverage; bowling styles known for at least 70% of T20I wickets | Partial: weather is complete at 100% coverage with prior-window features; 3,005 players have a known style, but T20 label coverage is 21.8% (53.4% ODI, 71.3% Test) |
| 2 | Build pitch features | PBP and PWI computed for every venue using past matches only | Partial: PBP built for 13 men's T20I venues; PWI is a raw second-innings step, not yet normalised |
| 3 | Train and validate classifiers | Random forest and gradient boosting beat the majority and venue baselines; macro-F1 of at least 0.50 in time-aware cross-validation | Partial: pooled rolling-origin cross-validation macro-F1 0.453 T20, 0.420 ODI, 0.544 Test; all beat their majority baselines, but only Test reaches 0.50 |
| 4 | Explain and handle new grounds | SHapley Additive exPlanations (SHAP) drivers per prediction; cold-start method tested on held-out venues | Partial: gain importance and an earlier SHAP analysis implemented; cold-start matching still planned |
| 5 | Disseminate | IEEE-format conference paper and dashboard prototype | Planned: abstract and report drafted with multi-format results; paper and dashboard pending |

### 3.3 Secondary objectives

- Extend the same pipeline to One Day Internationals (ODIs) and Test matches with format-specific label thresholds.
- Curate the highest-impact missing bowler styles so that more matches become trainable.
- Quantify whether daily weather adds out-of-time predictive value now that the weather table is complete.
- Package the pipeline so that a teammate can reproduce every table and figure from the repository.

## 4. Methodology

### 4.1 Approach and theoretical framework

The design is **quantitative and observational**. We use supervised multiclass learning with **weak supervision**: because no ground-truth pitch class exists, the label is derived by an explicit rule from match events. The rule is frozen before evaluation, documented in code, and stress-tested by changing its thresholds.

The theoretical basis is that pitch behaviour leaves a statistical signature in early-innings scoring and in the **type** of bowler who takes wickets. A surface that grips and turns produces more spin-bowled dismissals; a hard, fast surface produces more pace-bowled dismissals; a flat surface produces high scoring with few early dot balls. This is a behavioural proxy, not a physical measurement, and the report treats it as such throughout.

Tree ensembles are chosen because the features are tabular, mixed (numeric and categorical), non-linear and interaction-heavy. Random forest (Breiman, 2001) averages decorrelated decision trees and is robust to outliers; gradient boosting, specifically XGBoost (Extreme Gradient Boosting; Chen and Guestrin, 2016), fits trees sequentially to residual error and usually gives the best tabular accuracy. Logistic regression provides a linear benchmark. SHAP (Lundberg and Lee, 2017) assigns each feature a contribution to each prediction using Shapley values from cooperative game theory, which makes the model auditable.

The end-to-end flow is:

1. **Ingest** Cricsheet JSON (JavaScript Object Notation) archives and flatten each match to one row.
2. **Enrich** rows with bowling styles from public player registers and with daily weather from the Open-Meteo ERA5 archive.
3. **Label** each match using the format-specific rule.
4. **Engineer** match-start and prior-venue features, with strict time ordering.
5. **Split** chronologically into training and later held-out matches.
6. **Train and compare** majority, venue-prior, logistic-regression, random-forest and XGBoost models.
7. **Evaluate** macro-F1, balanced accuracy, per-class scores, log loss, Brier score, calibration and bootstrap confidence intervals.
8. **Explain** the best model with SHAP and translate drivers into toss guidance.

Figure 1 in Section 11 shows this flow as a flowchart.

### 4.2 Proxy label definition

The label is computed from the **first innings only**, because the pitch's influence on the team batting first is the quantity a pre-match decision needs. For T20 the early window is the six-over powerplay (the fielding-restriction period at the start of an innings). A **dot ball** is a legal delivery from which no run is scored; the dot-ball percentage measures how hard scoring was early. Run rate is runs per over.

| Format | Label = Batting if | Otherwise Label = Spin if | Otherwise |
| --- | --- | --- | --- |
| T20 | First-innings run rate ≥ 8.5 and powerplay dot-ball % ≤ 35% | At least 2 bowler-credited spin wickets and spin-to-pace wicket ratio ≥ 1.25 (or no pace wickets) | Pace |
| ODI | First-innings run rate ≥ 5.5 and first-ten-over dot-ball % ≤ 60% | Same spin rule | Pace |
| Test | First-innings run rate ≥ 3.8 and first-ten-over dot-ball % ≤ 75% | Same spin rule | Pace |

Only **bowler-credited** dismissals count (bowled, caught, caught and bowled, leg before wicket, stumped, hit wicket), so run-outs do not distort the spin/pace comparison. Wides and no-balls are excluded from legal-ball denominators. Tests have no powerplay; the first ten overs are used as an explicit **opening-ten-over proxy**, not as a powerplay.

This rule is an operational assumption, not a law. Section 7.6 reports how much the labels move when the cut-offs change.

### 4.3 Data sources

| Source | Content | Licence and role |
| --- | --- | --- |
| Cricsheet | Ball-by-ball JSON for all formats; venue, city, toss, teams and officials | Open Data Commons Attribution (ODC-By); primary match data |
| Kaggle international player statistics | Bowling style and ESPNcricinfo player identifiers | CC0; primary style source, joined on identifier |
| Kaggle worldwide player profiles | Broader player-style coverage | CC0; fallback join on exact normalised name |
| Open-Meteo historical weather API | Daily ERA5 reanalysis at the match city | Free, non-commercial; daily temperature, precipitation, wind and humidity |

ERA5 is the fifth-generation atmospheric reanalysis of the European Centre for Medium-Range Weather Forecasts (ECMWF): a global, hour-by-hour reconstruction of the atmosphere from observations and a weather model. We use daily aggregates at the match **city**, not the stadium, and on the match **date**, not the start time, because Cricsheet generally does not publish start times. The weather features are therefore ambient context, not pitch measurements.

### 4.4 Feature engineering and leakage control

Features fall into four groups.

- **Match context:** format, gender, team type, venue and venue identifier, city, year, and cyclical month and day-of-year encodings (sine and cosine) so that December and January are adjacent.
- **Toss:** the toss decision (bat or field) and whether the toss winner batted first.
- **Prior venue history:** rolling means over at most the previous five matches at that venue for first-innings run rate, early dot-ball percentage and spin wicket share, plus counts of prior matches and prior matches with complete role information.
- **Weather (when available):** daily mean temperature, precipitation total, maximum wind speed and mean relative humidity.

Leakage is controlled in three ways. First, the same-match statistics that define the label (run rate, dot balls, wickets by style) are never used as features. Second, rolling venue features are computed from strictly earlier dates; matches on the same date do not see each other's outcomes. Third, the train/test split is chronological by unique date, and imputers and encoders are fitted inside each training fold.

### 4.5 Training and evaluation protocol

The most recent 20% of unique match dates form the held-out test period. Class weights compensate for imbalance. Models are compared against two baselines: always predicting the majority class, and predicting from the venue's historical label distribution. Evaluation reports accuracy, balanced accuracy (mean per-class recall, which does not reward ignoring minority classes), macro-F1 (the unweighted mean of per-class F1 scores, where F1 is the harmonic mean of precision and recall), weighted F1, multiclass log loss (a penalty for confident wrong probabilities), the Brier score (mean squared probability error), the confusion matrix, per-class precision/recall/F1, bootstrap confidence intervals, and calibration error. Four-fold expanding-window cross-validation repeats the comparison across earlier periods.

## 5. Project Execution

### 5.1 Planning and design

The team began with a scoping exercise: which decision would a pitch model actually change? The answer was the toss and the composition of the attack, so the target was defined as a surface class rather than a score. The group then audited data availability, discovered that no open pitch measurements exist, and adopted the proxy-label design. The work was divided into data engineering, modelling and domain interpretation, with a shared repository and a written data manifest.

Design decisions taken before implementation:

- Use only free and open data, so results are reproducible and publishable.
- Freeze the label rule before looking at model scores.
- Evaluate on **later** matches only, because a random split would leak future venue knowledge.
- Keep every derived table and cache in the repository so that a run can resume on another machine.
- Treat the label as a proxy in every table, figure and sentence.

### 5.2 Implementation

**Data ingestion.** The full Cricsheet archive contains 18,585 T20, ODI and Test matches after date and format checks. After the early-window completeness check, 14,109 T20, 3,113 ODI and 921 Test matches are eligible for modelling; 3,082, 1,661 and 657 of those can be labelled.

**Player-style enrichment.** Cricsheet records the bowler's name but not the bowling style. Styles are joined from the two Kaggle registers, first on the ESPNcricinfo identifier and then on a conservative exact normalised name. A third, reviewed step resolves 527 players by first initial plus surname, but only when the Cricsheet team matches the profile country and every candidate profile agrees on the style class. When a credited wicket-taker still cannot be classified, the match is excluded rather than guessed. This is deliberate: silent guessing would contaminate the label.

**Geocoding and weather.** Matches are grouped by city (or venue when the city is blank) and geocoded, with a manual review of every location that failed. All 413 location strings now have coordinates; 94 previously unresolved venues were reviewed by hand and recorded with a rationale. The collector batches up to 25 coordinates per year, caches every response, and reuses the cache on restart. Collection is now complete: all 111 year/coordinate batches are cached, and the resulting table holds **18,585 match rows with no missing weather values and no failed geocodes**. The readiness audit passes at 100% weather coverage. A separate script derives prior 7/14/28-day moisture windows from the same cache without new API calls.

**Labels and features.** The labelling pipeline and the training script were run independently and reproduced the same label on all 5,400 labelled matches, which validates the implementation.

**Modelling and explanation.** Separate XGBoost models are trained for T20, ODI and Test with a chronological 80/20 split. Four expanding-window folds provide the rolling-origin cross-validation, and variants are compared on identical folds. Calibration, bootstrap intervals and a role-curation study complete the evaluation.

**Iterations.** The first version used only match context and venue history. Review showed that role coverage, not model choice, limited sample size, so the team added a role-curation curve, a prioritised review queue and an auditable override table. The second iteration added calibration and bootstrap intervals after the first results showed that a single accuracy number hid the class imbalance. The third added match-day weather, then prior-window moisture features and rolling-origin model selection.

## 6. Tools and Techniques Used

### 6.1 Tools

| Tool | Purpose |
| --- | --- |
| Python 3.13 | Main implementation language for ingestion, labelling, modelling and reporting |
| pandas and NumPy | Tabular data processing and numerical arrays |
| scikit-learn | Pipelines, imputation, one-hot encoding, logistic regression, random forest, metrics and cross-validation |
| XGBoost | Gradient-boosted decision trees |
| SHAP | Model explanation with Shapley values |
| Matplotlib | Figures used in the presentation and report |
| Git and Git Large File Storage (LFS) | Version control and storage of the 142 MB archive |
| Open-Meteo API | Geocoding and ERA5 daily weather retrieval |
| Cricsheet and Kaggle registers | Match data and bowling-style labels |
| Microsoft PowerPoint and Word | Presentation and report preparation |

No paid software or paid data was used.

### 6.2 Techniques

- **Weak supervision.** The target is generated by a deterministic rule from observed events, then treated as a noisy label. This is the standard approach when ground truth is unavailable, and it requires explicit sensitivity analysis.
- **Chronological validation.** Training on the past and testing on the future mimics real use and prevents future information from leaking into features. Random cross-validation would overstate performance because the same venue appears on both sides.
- **Class weighting.** Minority classes (especially Batting) receive higher weight during training so that the model does not simply predict the majority class.
- **Bootstrap confidence intervals.** Test matches are resampled with replacement many times to estimate the uncertainty of macro-F1; overlapping intervals are treated as evidence that models are not reliably different.
- **Calibration analysis.** Predicted probabilities are binned and compared with observed frequencies; the Expected Calibration Error (ECE) summarises the gap. Calibration matters because a toss recommendation needs honest probabilities, not just a class.
- **SHAP explanation.** Shapley values from cooperative game theory decompose each prediction into additive feature contributions, giving global importance and per-match drivers.
- **Cold-start matching (planned).** For a venue with no history, find statistically similar venues using profile features and borrow their prior. This is the practical answer to "what about a ground we have never played on?"

## 7. Partial Results

### 7.1 Data audit

| Quantity | T20 | ODI | Test |
| --- | --- | --- | --- |
| Matches with a complete early window | 14,109 | 3,113 | 921 |
| Labelled matches (all wicket-taker styles known) | 3,082 (21.8%) | 1,661 (53.4%) | 657 (71.3%) |
| Pace / Spin / Batting labels | 2,219 / 623 / 240 | 1,129 / 268 / 264 | 428 / 162 / 67 |
| Hold-out test matches | 576 | 327 | 128 |

The role table now covers **3,005 players**: 1,899 from the worldwide profile name
match, 579 from an exact Cricinfo identifier, and 527 from a reviewed override.
The overrides accept a player only when the name matches, the Cricsheet team
equals the profile country, and every candidate profile agrees on the style
class; they add 9,768 credited wickets and are recorded row by row in
`data/interim/bowler_role_overrides.csv`. A further 139 players have conflicting
style evidence and 346 appear only for domestic sides, so they remain in the
human review queue rather than being guessed.

T20 remains the hardest format to label: only about one match in five has every
wicket-taker's style known, against half of ODIs and most Tests. Section 7.6
shows how much the review queue still offers.

### 7.2 Venue behaviour profiles

The first Pitch Behaviour Profile covers the 13 men's T20I venues with at least 25 matches and at least 60% role coverage. Across these grounds, higher-scoring venues have a **lower** share of spin wickets (Pearson correlation r = −0.64). Eden Park averages 9.0 runs per over with 25% of wickets taken by spin; Mirpur averages 7.1 with 42%. This is consistent with the intuition that flat, high-scoring surfaces offer less turn, and it is the first evidence that the proxy captures something real about venue character.

The raw Pitch Wear Index step compares the two innings. Second-innings run rate is a median 0.42 runs per over lower than first-innings run rate. This is **not yet** a wear measure: chasing tactics, target pressure and team strength have not been removed. It is not yet normalised for those factors or for the quality of the batting and bowling units.

### 7.3 Model comparison

Results on the chronological hold-out (the most recent 20% of unique match
dates), using match-day weather and pre-match moisture features:

| Format | Test n | Accuracy | Balanced accuracy | Macro-F1 (95% CI) | Majority macro-F1 | Log loss |
| --- | --- | --- | --- | --- | --- | --- |
| T20 | 576 | 0.554 | 0.449 | 0.434 (0.395–0.474) | 0.269 | 0.915 |
| ODI | 327 | 0.465 | 0.413 | 0.400 (0.348–0.453) | 0.248 | 1.065 |
| Test | 128 | 0.719 | 0.548 | 0.512 (0.459–0.558) | 0.225 | 0.839 |

All three models beat the majority baseline on macro-F1 by a wide margin, and
the Test model also beats it on accuracy. T20 and ODI show the accuracy paradox:
the majority class (Pace) is right more often, yet the model is far better on
the minority classes, which is what macro-F1 rewards. Per-class F1 is
0.16 / 0.68 / 0.45 for T20, 0.19 / 0.57 / 0.44 for ODI and 0.00 / 0.80 / 0.73
for Test (Batting / Pace / Spin). Batting is the persistent failure: it is only
4–16% of labels, and in Test the hold-out contains just 17 examples, so the
model never predicts it. Most remaining error is Pace-versus-Spin confusion,
which is expected because the label depends on **who** took wickets rather than
on a physical measurement.

### 7.4 Rolling-origin cross-validation and calibration

A single hold-out has a 95% interval roughly ±0.04–0.05 wide, so small
differences between variants cannot be read from it. Model selection therefore
uses four expanding-window folds: the first 50% of dates train the first model,
and each subsequent quarter is predicted in turn. Every labelled match is
tested exactly once, always using earlier matches only. Pooled out-of-fold
macro-F1 and Expected Calibration Error (ECE):

| Variant | T20 | ODI | Test | ECE (T20 / ODI / Test) |
| --- | --- | --- | --- | --- |
| XGBoost, match-day weather only (previous default) | 0.421 | 0.405 | 0.525 | 0.066 / 0.140 / 0.106 |
| **XGBoost, moisture context (shipped)** | **0.453** | **0.420** | **0.544** | **0.058 / 0.106 / 0.100** |
| Random forest, moisture context | 0.450 | 0.410 | 0.554 | 0.132 / 0.072 / 0.155 |
| Average of XGBoost and random-forest probabilities | 0.458 | 0.418 | 0.541 | 0.079 / 0.069 / 0.069 |
| Two-stage random forest (Batting vs rest, then Pace vs Spin) | 0.444 | 0.408 | 0.564 | 0.087 / 0.052 / 0.154 |

Adding the prior-window moisture features improves XGBoost in every format
(+0.032 T20, +0.014 ODI, +0.019 Test) and keeps the best calibration of the
single models; it is therefore the default. Random forest wins on Test but loses
on T20 and ODI and is markedly less calibrated. The probability average is
close to the best everywhere and best calibrated, but it never wins and doubles
the artifact, so it is recorded rather than shipped. The two-stage model helps
only Test. Every difference is within roughly one fold standard deviation
(0.02–0.06), so the defensible claim is a small but consistent improvement, not
a step change. The single hold-out in Section 7.3 is noisier and disagrees on
ODI, which is precisely why the cross-validation is reported alongside it.

### 7.5 Explainability

The shipped models are explained with XGBoost gain importance. The strongest
signals are match context rather than weather: `gender` contributes 9.4% of
importance in T20 and 13.0% in ODI, venue identity is the next largest block,
and the prior five-match spin-wicket share is the most useful continuous venue
feature. Weather, including the new moisture windows, accounts for 8.9% of
importance in T20, 17.7% in ODI and 22.5% in Test; in Test the leading weather
terms are 7-day and 14-day mean temperature. The earlier T20I SHAP analysis
reached the same conclusion about gender, so the effect is not an artefact of
one explanation method.

That finding is important and uncomfortable. Gender is not a property of a
pitch; it is a proxy for a different distribution of matches, venues and player
pools. The label rule also behaves differently by category: women's matches are
1.8% Batting, 50.7% Pace and 47.4% Spin, while men's matches are 10.6% Batting,
75.3% Pace and 14.1% Spin. A single set of thresholds is therefore not
transferable without review. Until this is resolved, the model should be read as
a **descriptive** tool and, if used at all, with gender-stratified reporting.

### 7.6 Role coverage and label sensitivity

The role-curation curve shows the return on manual review:

| Top missing players curated | Wicket coverage | Role-complete T20Is |
| --- | --- | --- |
| 0 | 42.0% | 919 |
| 50 | 48.8% | 1,021 |
| 100 | 53.4% | 1,110 |
| 200 | 60.1% | 1,318 |
| 400 | 69.5% | 1,618 |
| 800 | 81.4% | 2,350 |
| 3,291 (all) | 100% | 5,849 |

Curating the top 200 bowlers raises the usable sample by about 50%; curating 400 nearly doubles it. This is the single largest lever available for model quality and it costs no external data.

The first pass of that review is complete. A conservative rule — name match
plus Cricsheet team equals profile country plus agreement on style class —
resolved **527 players covering 9,768 credited wickets** without a single guess;
conflicting and domestic-side players were left in the queue. That raised
labelled matches from 2,308 to 3,082 in T20 (+33.5%), 1,199 to 1,661 in ODI
(+38.5%) and 576 to 657 in Test (+14.1%). Every added label carries its join
method in `player_role_join_audit.csv`, so the decision can be audited row by
row.

Threshold sensitivity changes 0–9.2% of labels. Tightening the early dot-ball limit to 30% changes 4.5% of labels; changing the spin-to-pace ratio to 1.0 changes 9.2%. The labels are therefore reasonably stable to small threshold moves, but not to large ones. The cut-offs must be reviewed by format and by category before publication.

### 7.7 Weather features

Weather collection is **complete**: 413 locations are resolved, all 111
year/coordinate batches are cached, and `match_weather.csv` contains 18,585 rows
with 0 missing temperature observations and 0 failed geocodes. The readiness
audit passes at 100% coverage. The free Open-Meteo tier throttled the process
repeatedly — an hourly request window during collection and the daily quota at
the end — so the collector was extended to wait out short limits and resume from
its cache.

The table now also carries strictly prior windows derived from the same cached
ERA5 series, with no additional API calls: 7-, 14- and 28-day precipitation,
7- and 14-day mean temperature, 7-day humidity and wind, and the number of
consecutive dry days before the match. Pitches dry over days, so these features
are physically better motivated than the match-day value alone. In
rolling-origin cross-validation they improve the model in all three formats
(Section 7.4), and weather's share of importance grows with match length to
22.5% in Test.

## 8. Results and Discussion

### 8.1 Phase-I results

The project has produced a working, reproducible pipeline; a documented proxy
label; a complete geocoding audit; venue behaviour profiles; and a validated
three-format model comparison. On the chronological hold-out the macro-F1 is
0.434 for T20, 0.400 for ODI and 0.512 for Test, against majority baselines of
0.269, 0.248 and 0.225. Rolling-origin cross-validation gives the more reliable
estimates of 0.453, 0.420 and 0.544. The models are therefore learning beyond
"Pace is most common", but the margin is modest. The accuracy paradox in T20 and
ODI (the majority class is more accurate but far worse on macro-F1) shows that
headline accuracy is the wrong summary for this imbalanced target. The Batting
class, gender confounding and T20 role coverage remain the main weaknesses.

### 8.2 Were the objectives met?

| Objective | Verdict |
| --- | --- |
| Complete the data | **Largely met.** Geocoding is complete and audited; weather is complete at 100% match coverage with prior-window features; 3,005 players have a known style after 527 audited overrides. T20 label coverage is still 21.8% because every wicket-taker must be known. |
| Build pitch features | **Partly met.** PBP exists for 13 men's venues; PWI is a raw step and is not yet normalised for team strength. |
| Train and validate classifiers | **Partly met.** All three formats beat their majority baselines on macro-F1; the 0.50 target is met only by Test in cross-validation (0.544). T20 and ODI reach 0.453 and 0.420. |
| Explain and handle new grounds | **Partly met.** Gain importance and an earlier SHAP analysis are interpreted; cold-start matching is designed but not validated. |
| Disseminate | **Partly met.** The abstract and this report are drafted with multi-format results; the paper and dashboard remain. |

### 8.3 Unexpected findings

1. **Gender dominates the explanation.** A variable that should not describe a pitch is the strongest predictor. This points to confounding between match category and label distribution, and it must be addressed before any applied claim.
2. **Accuracy and macro-F1 disagree.** The majority baseline wins on accuracy but loses badly on macro-F1. Reporting accuracy alone would have made the models look worse than they are; reporting only macro-F1 would hide the weak Batting class.
3. **Role coverage, not weather, is the bottleneck.** T20 label coverage is 21.8% because every wicket-taker must be known, against 53.4% in ODI and 71.3% in Test. The 527 audited overrides added a third more T20 matches; finishing the review queue is still worth more than further model tuning.
4. **Expert labels are noisy by nature.** The Adams et al. result (umpires reliable on bounce, unreliable on pace) is a useful reminder that even human assessment is imperfect, which supports the decision to report proxy agreement rather than claim physical truth.

### 8.4 Limitations and threats to validity

- **Proxy labels.** The target is a behavioural rule, not a measurement. A match labelled Spin may be a pace-friendly pitch on which spinners happened to take wickets.
- **Small, imbalanced sample.** 5,400 labelled matches across three formats, but only 571 Batting examples (67 in Test); confidence intervals are wide.
- **Selection bias.** Requiring complete role information favours recent matches, men's internationals and better-documented players.
- **Weather is ambient and city-level.** It is not stadium conditions and is not aligned to the start time.
- **Single-split risk.** Cross-validation and held-out rankings differ, so no model is reliably best yet.
- **Geographic ambiguity.** Generic place names can geocode to the wrong country; 94 such cases were reviewed by hand, but the remaining automatic geocodes deserve spot checks.

## 9. Prototype (Software)

### 9.1 Prototype description

The prototype is a command-line software system with five modules:

| Module | Function |
| --- | --- |
| `src/collect_match_weather.py` | Parses the Cricsheet archive, geocodes locations, fetches and caches ERA5 daily weather, writes the geocode audit and `match_weather.csv` |
| `src/build_weather_context.py` | Derives prior 7/14/28-day moisture features from the cached ERA5 series without new API calls |
| `src/phase1_phase2.py` | Generates proxy labels and the leakage-safe T20 feature table |
| `src/train_pitch_models.py` | Builds features for T20/ODI/Test, joins weather, trains and evaluates the models, and writes metrics |
| `src/build_role_overrides.py` | Resolves bowler styles with a name-plus-country-plus-agreeing-style rule; writes the audited override table |
| `src/evaluate_rolling_cv.py` | Compares model and weather variants on identical expanding-window folds |
| `src/run_t20_preliminary_analysis.py` | Runs the earlier T20I comparison, sensitivity analysis, role-curation curve, calibration and SHAP |
| `src/audit_datasets.py` | Checks match coverage, value ranges, duplicates and geocoding; writes a readiness report |

Outputs are plain CSV (comma-separated values) and JSON files, so they can be inspected without the code. The weather cache is content-addressed: a batch's filename is a hash of its coordinates and date range, which makes restarts exact and prevents silent recomputation.

### 9.2 Development process

The system was built in vertical slices. The first slice read one archive, produced one label per match and wrote a feature table. The second added the second format and the weather collector. The third added model comparison and explanation. Each slice ended with a reproducibility check: rerun from the archive and compare row counts and label agreement. Challenges included a 142 MB archive (moved to Git LFS), ambiguous venue names (solved with a reviewed override table), missing start times (solved by using daily city weather), and API rate limits (solved with response caching and a fail-fast message).

### 9.3 Testing and validation

- **Label agreement:** the labelling pipeline and the training script reproduced the same label on all 5,400 labelled matches (100%).
- **Audit:** the readiness audit checks one row per match, unique identifiers, archive coverage, plausible value ranges and unresolved locations, and exits non-zero on failure.
- **Out-of-time testing:** every model is evaluated on later matches only; rolling features use earlier dates only.
- **Robustness:** four-fold expanding-window cross-validation, bootstrap confidence intervals, threshold sensitivity and calibration analysis.
- **Reproducibility:** every response is cached, every override is documented, and the repository records the exact commands.

## 10. Conclusion

### 10.1 Summary

The project set out to make pitch behaviour measurable with open data. It has delivered the infrastructure to do so: a complete match inventory, a documented proxy label, a reviewed geocoding layer, leakage-safe features, a complete weather table with prior-window moisture, and a three-format model comparison with explanation and calibration. The answer is cautious but positive: in rolling-origin cross-validation the models reach macro-F1 0.453 (T20), 0.420 (ODI) and 0.544 (Test), against majority baselines of 0.269, 0.248 and 0.225, and venue profiles show a plausible relationship between scoring and spin (r = −0.64). The models are not yet ready to advise a captain. Accuracy is still below the majority baseline in T20 and ODI, the Batting class is weak, and gender is a confounder. The remaining work is clear: finish the role-review queue, normalise the Pitch Wear Index, resolve the gender confounder, and validate cold-start behaviour before claiming applied value.

### 10.2 Personal reflections

**Aravind R.** Working on this project changed how I think about "data-driven" sport. I began assuming that more data automatically means better answers; I finished understanding that the hard part is defining the target honestly. Building the label rule forced me to read the laws of the game closely and to defend every threshold. I also learned the value of reproducibility: caching weather responses and versioning the archives felt tedious at the time, but it is the reason the team could stop and resume without losing work. The biggest lesson was to report uncertainty — confidence intervals and calibration error — rather than a single flattering number.

**Rishab H R.** My main contribution was the data engineering and the weather pipeline. I learned how much scientific risk sits in the unglamorous steps: geocoding a venue to the wrong country, silently guessing a bowler's style, or letting a future match leak into a rolling average would each invalidate the results. Writing the review table for 94 ambiguous locations taught me that automation needs a human checkpoint. I also learned to design for failure: the API quota ran out mid-run, but because every response was cached, no work was lost. This project improved both my Python engineering and my judgement about when a result is trustworthy.

**Ujwal S Nair.** I focused on the modelling and interpretation. Seeing XGBoost beat logistic regression on macro-F1 but lose on accuracy was a useful lesson in choosing metrics that match the decision. The SHAP analysis was the most surprising part: the strongest feature was gender, which is not a pitch property at all. That taught me to interrogate explanations rather than treat them as confirmation. I also learned how much domain reasoning matters in feature design — a rolling venue average is only meaningful if it respects the order of matches. The project has made me more careful about the difference between predicting a proxy and explaining a physical process.

## 11. Visuals

The figures below summarise the problem, the method, the data coverage and the planned decision output. They are reproduced from the Phase-I presentation.

![Project flowchart](presentation_images/flowchart.png)

**Figure 1 — Project flowchart.** End-to-end flow from data collection to toss recommendation.

![Problem framing](presentation_images/problem.png)

**Figure 2 — Problem framing.** Why pitch behaviour is unmeasured and why the toss decision matters.

![Label rules](presentation_images/label_rules.png)

**Figure 3 — Label rules.** The batting, spin and pace proxy rules used for each format.

![Pitch types](presentation_images/pitch_types.png)

**Figure 4 — Pitch types.** The three behavioural classes used by the model.

![Venue profiles](presentation_images/venues.png)

**Figure 5 — Venue profiles.** Run rate, boundary share, wickets per twenty overs and spin wicket share by venue.

![Data coverage](presentation_images/coverage.png)

**Figure 6 — Data coverage.** Match counts, label coverage and the role-coverage bottleneck.

![Proposed output](presentation_images/output_card.png)

**Figure 7 — Proposed output.** The toss recommendation card with drivers and confidence.

## 12. Outcome of the Work

**Primary target: conference publication.** An IEEE-format paper is planned. The abstract and this report are drafted; the full paper needs the final role-curation pass, the gender-confounder analysis and the dashboard evaluation.

**Stretch target: journal publication.** An extended paper is feasible if the ODI and Test models and the normalised Pitch Wear Index hold up on later matches.

**Software and data.** The repository is a reusable open-data pipeline with documented schemas, cached weather responses and a readiness audit. It is the team's main tangible artifact at this stage.

**Patent and product.** No patent or commercial product is claimed. The toss-recommendation layer is a planned applied output, not a delivered product.

## References

[1] W. A. Adams, S. W. Baker, D. M. James, and R. J. Young, "Measuring and modelling the bounce and pace of County Championship cricket pitches," *International Turfgrass Society Research Journal*, vol. 10, pp. 1021–1026, 2005.

[2] M. Bhatnagar and M. Bhatnagar, "Analyzing key factors influencing IPL cricket scores using explainability and multimodal data," *Journal of Quantitative Analysis in Sports*, vol. 21, no. 3, pp. 253–267, 2025, doi: 10.1515/jqas-2025-0006.

[3] L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001, doi: 10.1023/A:1010933404324.

[4] S. Chakraborty et al., "Cricket data analytics: Forecasting T20 match winners through machine learning," *International Journal of Knowledge-Based and Intelligent Engineering Systems*, vol. 28, no. 1, pp. 73–92, 2024, doi: 10.3233/KES-230060.

[5] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2016, pp. 785–794, doi: 10.1145/2939672.2939785.

[6] Cricsheet, "Downloads." [Online]. Available: https://cricsheet.org/downloads/

[7] H. Hersbach et al., "The ERA5 global reanalysis," *Quarterly Journal of the Royal Meteorological Society*, vol. 146, no. 730, pp. 1999–2049, 2020, doi: 10.1002/qj.3803.

[8] D. M. James, M. J. Carré, and S. J. Haake, "The playing performance of county cricket pitches," *Sports Engineering*, vol. 7, no. 1, pp. 1–14, 2004.

[9] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems*, vol. 30, 2017, pp. 4765–4774.

[10] Open-Meteo, "Historical weather API." [Online]. Available: https://open-meteo.com/en/docs/historical-weather-api

[11] B. Morley and D. Thomas, "An investigation of home advantage and other factors affecting outcomes in English one-day cricket matches," *Journal of Sports Sciences*, vol. 23, no. 3, pp. 261–268, 2005.

[12] S. A. A. Shah, Q. Zaman, D. Wasim, J. Allohibi, A. A. Alharbi, and M. Shabbir, "Optimal model for predicting highest runs chase outcomes in T-20 international cricket using modern classification algorithms," *Alexandria Engineering Journal*, vol. 114, pp. 588–598, 2025.

[13] G. Sood and D. Willis, "Fairly random: The impact of winning the toss on the probability of winning," arXiv:1605.08753, 2016.

## Appendix A — Glossary of Abbreviations

| Abbreviation | Full form | Meaning in this report |
| --- | --- | --- |
| API | Application Programming Interface | Programmatic interface used to query Open-Meteo |
| AutoML | Automated Machine Learning | Automatic model selection and tuning (reviewed literature) |
| CI | Confidence Interval | Range that likely contains the true metric |
| CPU | Central Processing Unit | Standard processor used for training when no GPU is available |
| CSV | Comma-Separated Values | Plain-text tabular file format |
| CV | Cross-Validation | Repeated train/test splits for robust evaluation |
| ECE | Expected Calibration Error | Average gap between predicted confidence and observed accuracy |
| EL | Experiential Learning | The course under which this project is undertaken |
| ERA5 | ECMWF Reanalysis version 5 | Global historical weather reconstruction |
| ECMWF | European Centre for Medium-Range Weather Forecasts | Producer of ERA5 |
| F1 | F1-score | Harmonic mean of precision and recall |
| GPU | Graphics Processing Unit | Accelerator for model training |
| IPL | Indian Premier League | Domestic T20 league reviewed in the literature |
| JSON | JavaScript Object Notation | Structured text format used by Cricsheet |
| LFS | Large File Storage | Git extension for large binary files |
| LLM | Large Language Model | Text model used in one reviewed study |
| LR | Logistic Regression | Linear classification baseline |
| ML | Machine Learning | Learning patterns from data |
| ODI | One Day International | 50-over international format |
| ODC-By | Open Data Commons Attribution License | Licence of the Cricsheet data |
| PBP | Pitch Behaviour Profile | Venue-level summary of scoring and wicket style |
| PWI | Pitch Wear Index | Planned measure of within-match surface change |
| RF | Random Forest | Ensemble of decision trees |
| SHAP | SHapley Additive exPlanations | Game-theoretic feature-attribution method |
| SVM | Support Vector Machine | Margin-based classifier (reviewed literature) |
| T20 | Twenty20 | 20-over format |
| T20I | Twenty20 International | 20-over international match |
