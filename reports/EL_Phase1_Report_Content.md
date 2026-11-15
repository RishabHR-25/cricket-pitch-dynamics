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

There is no open, repeatable way to measure how a cricket pitch behaves, either before a match or while it is being played. As a result, the toss decision and the pace/spin balance of the playing eleven still rest on visual inspection, broadcast commentary and venue experience, and no method describes how the surface changes as the innings progresses. Those judgements are hard to check, hard to reproduce and unavailable to teams without a long history at the ground.

Two constraints define the problem. First, no public dataset contains physical pitch measurements linked to match identifiers. Second, weather and pitch behaviour interact (moisture, humidity, wind and temperature affect preparation and deterioration), so any useful method must combine match records with ambient conditions. The scientific difficulty is therefore one of **weak supervision**: both the match-level target and the within-match wear signal must be constructed from observed play, and every claim must be honest about the fact that neither is a physical measurement.

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

This project treats pitch behaviour as the **target**, at two time scales. At the match scale we construct a documented proxy label from ball-by-ball evidence, build leakage-safe features from earlier matches only, train interpretable multiclass classifiers, and explain each prediction. At the **ball-by-ball scale** we estimate how the relative effectiveness of spin and pace changes as a ball ages, and then test whether that change can be attributed to the surface. To our knowledge, no prior study predicts an interpretable pitch-behaviour class from open ball-by-ball data alone, and none tests whether a within-match wear estimate from such data is identified; published work either measures pitches physically or uses the pitch as an input to a result model.

The contribution is fivefold:

1. **Ball-by-ball estimation and identification of within-match pitch dynamics.** A within-match fixed-effects model isolates the age x spin interaction from venue, date, phase, match state and batter experience, on 5.1 million legal deliveries. The estimator is then subjected to a randomisation placebo, per-phase estimates, a boundary check, persistence and predictive tests, two ball-renewal natural experiments, and a delivery-level within-bowler test. The result is a precise, stable gradient that is **not identified as surface wear**: selection explains the ball-renewal jump, but the smooth gradient survives bowler fixed effects and still fails attribution. The identification analysis is the paper's primary contribution; Section 7.10 is its main results section.
2. **Convergent and predictive validation of the inferred state.** A genuine pitch state must behave like one: it should persist from the first innings to the second on the same surface, should be stronger at venues where spin already dominates, should increase after dry pre-match weather, and should improve held-out prediction. All four tests are run and reported.
3. **A reproducible proxy-label benchmark.** A frozen, format-specific rule converts first-innings scoring behaviour and bowler-credited wicket types into Batting, Pace and Spin classes, so the task can be rebuilt by anyone from the public Cricsheet archive. Section 7.6 quantifies label stability under threshold perturbation rather than asserting it.
4. **An audited bowler-style curation protocol.** A conservative three-source rule (exact name, team-country agreement, style-class agreement) adds 527 reviewed overrides covering 9,768 credited wickets without guessing, raising the labelable T20 sample by 33.5%. The same role table supplies the spin/pace contrast for every ball in the dynamics dataset.
5. **Leakage-safe weather context with honest evaluation.** Prior-window moisture features are computed only from dates before the match, every result is reported under rolling-origin cross-validation with fold spread and bootstrap confidence intervals, and the optimal venue-history window is estimated per format rather than assumed.

Together these make the pipeline an auditable baseline for pitch-behaviour prediction, a venue-level Pitch Behaviour Profile (PBP), a Pitch Wear Index (PWI) defined as the innings-level age x spin coefficient, and a validated cold-start direction for grounds with little history.

## 3. Objectives

### 3.1 Primary objective

Estimate how a cricket pitch changes during a match from open ball-by-ball data, and classify pitch behaviour before a match in a way that can be converted into an interpretable toss recommendation.

### 3.2 Specific objectives and success measures

| # | Objective | Measure of success | Status at Phase-I |
| --- | --- | --- | --- |
| 1 | Complete the data | Weather audit passes at 98% match coverage; bowling styles known for at least 70% of T20I wickets | Partial: weather is complete at 100% coverage with prior-window features; 3,005 players have a known style, but T20 label coverage is 21.8% (53.4% ODI, 71.3% Test) |
| 2 | Build pitch features | PBP and PWI computed for every venue using past matches only | Partial: PBP built for 13 men's T20I venues; PWI is now defined as the innings-level age x spin coefficient and estimated for every match with sufficient spin and pace cells |
| 3 | Estimate within-match wear | A pre-specified age x spin coefficient that is stable across fixed-effect specifications and survives a within-innings randomisation test | Partly met: the coefficient is stable across specifications and rolling windows. The standard randomisation null is mis-centred and unusable, but a null-imposed wild cluster bootstrap restores a zero-centred test under which the gradient is significant; the identification tests in Section 7.10 then reject a wear interpretation. The across-innings Test contrast remains the only candidate, but two thirds of it is composition and it is not distinguishable from zero. |
| 4 | Validate the inferred state | Cross-innings persistence, venue spin-friendliness gradient, moisture gradient and out-of-sample predictive gain | Partly met: the two gradients are detected; the across-innings Test contrast falls by two thirds under a within-side design and is not significant; the per-match slope does not persist and the predictive gain is small (reported with intervals) |
| 5 | Train and validate classifiers | Random forest and gradient boosting beat the majority and venue baselines; macro-F1 of at least 0.50 in time-aware cross-validation | Partial: pooled rolling-origin cross-validation macro-F1 0.453 T20, 0.422 ODI, 0.568 Test; all beat their majority baselines, and Test clears 0.50 |
| 6 | Explain and handle new grounds | SHapley Additive exPlanations (SHAP) drivers per prediction; cold-start method tested on held-out venues | Partial: gain importance and an earlier SHAP analysis implemented; cold-start matching still planned |
| 7 | Disseminate | IEEE-format conference paper and dashboard prototype | Partial: the conference paper draft is written (`reports/PAPER_DRAFT.md`) and the report contains the full identification analysis; the dashboard prototype is pending |

### 3.3 Secondary objectives

- Extend the same pipeline to One Day Internationals (ODIs) and Test matches with format-specific label thresholds.
- Curate the highest-impact missing bowler styles so that more matches become trainable.
- Quantify whether daily weather adds out-of-time predictive value now that the weather table is complete.
- Test whether the inferred wear state links to pre-match moisture and to venue spin-friendliness.
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
9. **Estimate dynamics** on the separate ball-by-ball layer: within-match age x spin models, placebo tests, cross-innings persistence, convergent validity and out-of-sample prediction.

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
- **Prior venue history:** rolling means over prior matches at that venue for first-innings run rate, early dot-ball percentage and spin wicket share, plus counts of prior matches and prior matches with complete role information. The history length is a tuned hyper-parameter, selected per format by rolling-origin cross-validation (Section 7.4).
- **Weather (when available):** match-day mean temperature, precipitation total, maximum wind speed and mean relative humidity, plus the core prior-window descriptors that survived ablation: 7-, 14- and 28-day precipitation totals, 7- and 14-day mean temperature, 7-day mean humidity, mean daily maximum wind, and the number of consecutive dry days before the match.

Leakage is controlled in three ways. First, the same-match statistics that define the label (run rate, dot balls, wickets by style) are never used as features. Second, rolling venue features are computed from strictly earlier dates; matches on the same date do not see each other's outcomes. Third, the train/test split is chronological by unique date, and imputers and encoders are fitted inside each training fold.

### 4.5 Training and evaluation protocol

The most recent 20% of unique match dates form the held-out test period. Class weights compensate for imbalance. Models are compared against two baselines: always predicting the majority class, and predicting from the venue's historical label distribution. Evaluation reports accuracy, balanced accuracy (mean per-class recall, which does not reward ignoring minority classes), macro-F1 (the unweighted mean of per-class F1 scores, where F1 is the harmonic mean of precision and recall), weighted F1, multiclass log loss (a penalty for confident wrong probabilities), the Brier score (mean squared probability error), the confusion matrix, per-class precision/recall/F1, bootstrap confidence intervals, and calibration error. Four-fold expanding-window cross-validation repeats the comparison across earlier periods. Intervals combine the spread across folds with a 1,000-resample percentile bootstrap over matches, and training is single-threaded with a fixed seed so that every reported number can be reproduced exactly.

### 4.6 Ball-by-ball dynamics estimator

The match-level model answers "what kind of pitch was this?". The dynamics analysis answers a different question: "how did the surface change while the match was being played?". Because no open dataset records the physical pitch, the estimand is a **relative** one: as the ball ages, does spin become more effective than pace, after holding match, phase and match state constant?

The unit is an **over x bowling-style cell**: the legal balls a spin bowler bowled in one over, and the legal balls a pace bowler bowled in the same over of the same innings. Aggregating to overs removes delivery-level noise and gives every cell a natural weight (balls bowled). Four outcomes are used because no single one measures a pitch: runs per ball, dot-ball share, wicket rate and boundary share. If wear is real, spin's relative performance should improve with ball age on all four (lower runs and boundaries, higher dots and wickets).

The estimating equation is a weighted linear probability model with absorbed fixed effects:

> outcome = alpha(match) + beta1 * age + beta2 * spin + beta3 * (age x spin) + controls + error

The coefficient of interest is **beta3**, the extra change per over for spin relative to pace. Absorbing match fixed effects means venue, date, format, climate, team strength and pitch identity are held constant exactly; only within-match variation identifies beta3. Controls are match phase, wickets already lost, the scoring rate before the over, the number of balls the striker has already faced, and gender. Standard errors are clustered by match.

Three specifications are reported. The **match fixed-effects** specification is the headline. The **match x innings** specification is stricter and uses only within-innings variation. The **match-use** specification replaces innings age with total legal balls bowled in the match, which is the physically correct wear clock for Test cricket because later innings inherit the surface used earlier.

The identifying assumption is stated plainly: within a phase, and conditional on match state, a change in the relative effectiveness of spin and pace as the ball ages is evidence of surface change. That assumption is too strong on its own, because the ball also changes: a new ball swings and seams, and that advantage decays within a few overs. The pooled coefficient therefore mixes **ball condition** with **pitch condition**, and neither a phase decomposition nor a smooth time trend can separate them; holding the bowler fixed removes only one of the remaining alternatives. Tactics, batter risk and bowler selection are further alternative explanations, so the analysis includes (i) a randomisation placebo that reallocates bowling styles within innings, together with a null-imposed wild cluster bootstrap that corrects the placebo's mis-centring, (ii) per-phase estimates, (iii) a cross-innings persistence test on the same pitch, (iv) convergent-validity tests against venue spin-friendliness and pre-match moisture, (v) an out-of-sample predictive check, (vi) two ball-renewal natural experiments, the second new ball in Tests and the 2011 two-new-ball rule in One Day Internationals, and (vii) a delivery-level within-bowler specification, and (viii) a restriction ladder for the across-innings contrast, all analysed in Section 7.10. None of these measures the pitch physically; together they test whether the inferred state behaves like a pitch state.

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

An early descriptive version of the Pitch Wear Index compared second-innings with first-innings run rate: the second innings is a median 0.42 runs per over lower. That comparison is **not** a wear measure, because chasing tactics, target pressure and team strength have not been removed, and it is retained here only as the motivation for the estimator in Section 4.6. The PWI used from Section 7.8 onwards is the within-match age x spin coefficient, which absorbs those confounders by construction.

### 7.3 Model comparison

Results on the chronological hold-out (the most recent 20% of unique match
dates), using match-day weather and pre-match moisture features:

| Format | Test n | Accuracy | Balanced accuracy | Macro-F1 (95% CI) | Majority macro-F1 | Log loss |
| --- | --- | --- | --- | --- | --- | --- |
| T20 | 576 | 0.559 | 0.461 | 0.443 (0.400–0.486) | 0.269 | 0.912 |
| ODI | 327 | 0.474 | 0.428 | 0.414 (0.361–0.464) | 0.248 | 1.058 |
| Test | 128 | 0.742 | 0.582 | 0.561 (0.487–0.646) | 0.225 | 0.796 |

All three models beat the majority baseline on macro-F1 by a wide margin, and
the Test model also beats it on accuracy. T20 and ODI show the accuracy paradox:
the majority class (Pace) is right more often, yet the model is far better on
the minority classes, which is what macro-F1 rewards. Per-class F1 is
0.17 / 0.69 / 0.47 for T20, 0.22 / 0.58 / 0.44 for ODI and 0.11 / 0.82 / 0.76
for Test (Batting / Pace / Spin). Batting is the persistent failure: it is only
4–16% of labels, and in Test the hold-out contains just 17 examples, so the
model rarely predicts it. Most remaining error is Pace-versus-Spin confusion,
which is expected because the label depends on **who** took wickets rather than
on a physical measurement.

### 7.4 Rolling-origin cross-validation and calibration

A single hold-out has a 95% interval roughly ±0.04–0.05 wide, so small
differences between variants cannot be read from it. Model selection therefore
uses four expanding-window folds: the first 50% of dates train the first model,
and each subsequent quarter is predicted in turn. Every labelled match is
tested exactly once, always using earlier matches only. The venue-history window
was then tuned per format on the same folds (five prior matches for T20 and ODI,
ten for Test). Pooled out-of-fold macro-F1 with a 1,000-resample bootstrap 95%
interval, and Expected Calibration Error (ECE):

| Variant | T20 | ODI | Test | ECE (T20 / ODI / Test) |
| --- | --- | --- | --- | --- |
| XGBoost, match-day weather only | 0.424 (0.399–0.449) | 0.424 (0.390–0.457) | 0.531 (0.491–0.576) | 0.069 / 0.130 / 0.081 |
| **XGBoost, moisture context (shipped)** | **0.453 (0.429–0.480)** | **0.422 (0.386–0.456)** | **0.568 (0.516–0.625)** | **0.053 / 0.110 / 0.090** |
| Random forest, moisture context | 0.447 (0.422–0.473) | 0.419 (0.384–0.454) | 0.554 (0.508–0.610) | 0.129 / 0.080 / 0.148 |

The prior-window moisture features add **+0.029 macro-F1 in T20 and +0.037 in
Test**, with overlapping intervals but a consistent direction across the two
feature sets tested. In ODI they are indistinguishable from match-day weather
(−0.002, almost identical intervals), so no weather gain is claimed for that
format. Random forest is close on T20 but loses on Test and is far less
calibrated (ECE 0.129 versus 0.053 in T20), which is why it is recorded rather
than shipped. The earlier grid also tested a probability average of XGBoost and
random forest and a two-stage random forest; neither was best in more than one
format, and both are documented in `PROGRESS.md` rather than shipped.

The format-specific window is the tuning change that survives scrutiny: Test
improves from 0.544 at a five-match window to 0.568 at ten (+0.024), and the
direction held in both the core and expanded weather feature sets. T20 and ODI
were best or tied at five; longer windows lost 0.02–0.03. The single hold-out in
Section 7.3 is noisier and disagrees on ODI, which is precisely why the
cross-validation is reported alongside it.

To test whether weather context substitutes for missing venue history, the
pooled out-of-fold predictions were stratified by the number of prior matches at
the ground (`context_by_history.csv`). The context gain is largest in the 3–5
match band (Test +0.14, T20 +0.03), but the zero-history strata contain only
17–22 matches, so a cold-start claim is **not** supported yet; it is flagged as
the next experiment rather than a result.

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

To test that stability directly, every label was recomputed with each cut-off
scaled by ±5% and ±10% (`src/label_sensitivity.py`, output
`label_sensitivity.csv`). Agreement is measured against the shipped labels:

| Perturbation | ODI | T20 | Test |
| --- | --- | --- | --- |
| Worst single cut-off at ±5% | 6.2% changed (κ = 0.87) | 2.7% (κ = 0.94) | 4.7% (κ = 0.90) |
| All cut-offs at ±10% | 13.0% changed (κ = 0.72) | 7.7% (κ = 0.82) | 10.7% (κ = 0.78) |

The spin rule is the most stable component (at most 3.2% change even at ±10%),
and the early dot-ball limit is the least stable. Labels are therefore robust to
plausible small moves but not to simultaneous large ones, which is the honest
qualification a reader needs. Cut-offs should be reviewed by format and by
category before publication.

### 7.7 Weather features

Weather collection is **complete**: 413 locations are resolved, all 111
year/coordinate batches are cached, and `match_weather.csv` contains 18,585 rows
with 0 missing temperature observations and 0 failed geocodes. The readiness
audit passes at 100% coverage. The free Open-Meteo tier throttled the process
repeatedly — an hourly request window during collection and the daily quota at
the end — so the collector was extended to wait out short limits and resume from
its cache.

The table also carries strictly prior windows derived from the same cached ERA5
series, with no additional API calls: 7-, 14- and 28-day precipitation, 7- and
14-day mean temperature, 7-day humidity and wind, and the number of consecutive
dry days before the match. Pitches dry over days, so these features are
physically better motivated than the match-day value alone.

An extended 19-descriptor set (adding 3- and 5-day rain totals, rain-day counts,
rain trend, temperature range and wind/humidity extremes) was built and ablated.
It gave no overall gain and slightly hurt ODI, so the eight core descriptors
above are shipped; the negative result is recorded rather than hidden. In
rolling-origin cross-validation the core context adds +0.029 macro-F1 in T20 and
+0.037 in Test and is neutral in ODI (Section 7.4). Weather's share of gain
importance grows with match length, reaching about a fifth in Test.

Because batch cache keys hash the venue list, a later venue-list update
orphaned seven 2026 batches. The loader now indexes every cached payload by its
ERA5 grid coordinate instead of trusting the key, which recovered the 2026 data
and restored 100% prior-window coverage without a new API call.

### 7.8 Within-match pitch dynamics

The ball-by-ball layer contains **6,776,668 deliveries from 23,241 matches**.
After restricting to legal deliveries with a known bowling style, **5,116,165
deliveries** remain, which aggregate to **855,659 over x style cells**. Of all
deliveries, 22.6% have an unknown bowler style and are excluded; this is the
price of the conservative role protocol and it is reported rather than hidden.

Table 7.7 gives the age x spin coefficient under the three fixed-effect
specifications. The sign convention is that a negative coefficient on runs per
ball means spin's scoring advantage grows as the ball ages, which is what wear
predicts.

**Table 7.7: Age x spin coefficient by outcome and specification**

| Format | Outcome | Venue FE | Match FE | Match x innings FE |
| --- | --- | --- | --- | --- |
| T20 | Runs per ball | −0.00214 (p<0.001) | −0.00198 (p<0.001) | −0.00182 (p<0.001) |
| T20 | Dot-ball share | +0.00176 (p<0.001) | +0.00151 (p<0.001) | +0.00138 (p<0.001) |
| T20 | Wicket rate | +0.00010 (p=0.110) | +0.00009 (p=0.187) | +0.00001 (p=0.870) |
| T20 | Boundary share | −0.00010 (p=0.349) | −0.00012 (p=0.285) | −0.00011 (p=0.332) |
| ODI | Runs per ball | −0.00160 (p<0.001) | −0.00131 (p<0.001) | −0.00108 (p<0.001) |
| ODI | Dot-ball share | +0.00157 (p<0.001) | +0.00146 (p<0.001) | +0.00140 (p<0.001) |
| ODI | Wicket rate | −0.00020 (p<0.001) | −0.00018 (p<0.001) | −0.00019 (p<0.001) |
| ODI | Boundary share | −0.00002 (p=0.666) | +0.00003 (p=0.511) | +0.00008 (p=0.107) |
| Test | Runs per ball | −0.00010 (p=0.110) | −0.00012 (p=0.051) | −0.00009 (p=0.163) |
| Test | Dot-ball share | +0.00012 (p<0.001) | +0.00013 (p<0.001) | +0.00012 (p<0.001) |
| Test | Wicket rate | −0.00000 (p=0.709) | −0.00000 (p=0.421) | −0.00001 (p=0.049) |
| Test | Boundary share | −0.00002 (p=0.112) | −0.00003 (p=0.035) | −0.00002 (p=0.135) |

Read as a set, the table says something more specific than "spin improves with
age". Two channels are stable and two are not. The **dot-ball share** is positive
in every format under every specification, and it is the only channel that never
changes sign. The **runs-per-ball** coefficient is negative and significant in
T20 and ODI but only marginal in Test. The **wicket rate** is null in T20 and
Test and *negative* in ODI, which is the wrong sign for wear. The **boundary
share** is null except for a weak negative in Test. A wicket mixes catches,
run-outs, tail-enders and bowler skill, so it is a poor indicator of turn; the
dot ball is the cleaner behavioural marker.

Table 7.8 asks whether the two live channels survive three checks: dropping the
first and last over of every innings, restricting to established bowlers with at
least 300 legal balls, and estimating only in the middle overs, where both
styles genuinely bowl.

**Table 7.8: Robustness of the match fixed-effects estimate**

| Format | Outcome | Pooled | Drop first/last over | Established bowlers | Middle overs only |
| --- | --- | --- | --- | --- | --- |
| T20 | Dot-ball share | +0.00151 | +0.00142 | +0.00162 | +0.00114 |
| T20 | Runs per ball | −0.00198 | −0.00010 (n.s.) | −0.00216 | +0.00408 |
| ODI | Dot-ball share | +0.00146 | +0.00144 | +0.00147 | +0.00109 |
| ODI | Runs per ball | −0.00131 | −0.00101 | −0.00135 | +0.00064 (p=0.056) |
| Test | Dot-ball share | +0.00013 | +0.00016 | +0.00013 | +0.00027 |
| Test | Runs per ball | −0.00012 | −0.00021 | −0.00012 | −0.00028 (n.s.) |

All dot-ball cells are significant at p<0.01. The dot-ball result is robust: it
survives every check in every format, and the middle-overs estimate — the one that
contains the least role confound — is positive in all three formats. The **runs**
result is not. In T20 it collapses to −0.0001 (p=0.87) once the first and last
over of each innings are removed, and inside the middle overs it turns reliably
positive (+0.0041, p=0.002). The pooled T20 runs coefficient is therefore carried
by the innings boundary, which is exactly where pace bowls and spin rarely does.
In ODI the runs coefficient is more stable, but it too reverses inside the middle
overs.

That reversal is the key diagnostic, and Table 7.9 shows why. Re-estimating
inside each phase separates three regimes.

**Table 7.9: Age x spin coefficient inside each phase**

| Format | Phase | Dot-ball share | Runs per ball |
| --- | --- | --- | --- |
| T20 | Powerplay | +0.00752 (p<0.001) | −0.02589 (p<0.001) |
| T20 | Middle | +0.00114 (p=0.010) | +0.00408 (p=0.002) |
| T20 | Death | −0.00623 (p<0.001) | +0.01396 (p=0.010) |
| ODI | Powerplay | +0.00240 (p=0.024) | −0.01144 (p<0.001) |
| ODI | Middle | +0.00109 (p<0.001) | +0.00064 (p=0.056) |
| ODI | Death | +0.00112 (p=0.294) | −0.00410 (p=0.195) |
| Test | New ball | −0.00101 (p<0.001) | +0.00112 (p=0.011) |
| Test | Middle | +0.00027 (p=0.004) | −0.00028 (p=0.173) |
| Test | Old ball | −0.00003 (p=0.761) | +0.00041 (p=0.085) |

The powerplay carries the largest wear-consistent effects: in T20 and ODI, spin's
dot share rises, its run concession falls, its boundary share falls and its
wicket rate rises as the powerplay progresses. But a six- or ten-over window is
not enough time for a pitch to change by that much, and a wear process cannot
reverse sign within it. The obvious explanation is **ball condition, not pitch
condition**: the new ball swings and seams, that advantage decays within the
first few overs, and spin gains relative to pace as it does. Section 7.10 tests
that explanation at the second-new-ball boundary and with bowler fixed effects.
At the boundary, the aggregate new-ball jump is the return of the frontline
seamers, and it disappears in matches where a seamer continues. Across the
innings, however, the gradient does not shrink when the bowler is held fixed, and
an exact decomposition puts the composition term at 1% of the coefficient or
less, so composition is not a general explanation either. The powerplay gradient
is therefore best read as a
**phase-specific pattern whose mechanism is not identified**: it is too large and
too short-lived for wear, and the ball-renewal test does not support a ball-age
reading.

In the middle overs the pattern is different and more cautious. Spin's dot share
rises relative to pace in all three formats (+0.0011 in T20 and ODI, +0.0003 in
Test), but so does its boundary share in T20 and ODI, and its wicket rate does
not rise. The pattern is not "spin becomes more economical"; it is "spin becomes
more attacking", with more dot balls and more boundaries and fewer singles. That
is consistent with a surface offering more turn and bounce — beat the bat or
disappear — but it is equally consistent with a field-setting change, because
attacking fields produce exactly this signature. The data cannot separate the two,
and the report does not claim that they can.

At the death in T20 the sign reverses entirely, but the cell contains only
115,922 spin balls against 313,905 pace balls, so this is a small-sample regime
and the least informative of the three.

Test cricket is different again. There is no powerplay, the new-ball phase runs
to 30 overs, and the dot-ball coefficient is positive in the middle overs
(+0.00027, p=0.004) but negative in the new-ball phase. The pooled Test dot
coefficient (+0.00013) is a mixture of the two. The pooled Test runs coefficient
is not significant under either null, and Section 7.9 discusses what that means
for the Test claim.

**Placebo.** Reallocating spin and pace labels randomly within each innings and
refitting the same match x innings specification does **not** produce a
zero-centred null, and the shift is not small. For dot-ball share the null mean is
+0.00027 in T20 and +0.00030 in ODI, about a fifth of the observed +0.00138 and
+0.00140, and +0.00009 in Test, three quarters of the observed +0.00012. For other
outcomes it moves in both directions: the T20 runs null mean equals the observed
coefficient (−0.00182 in both), and the T20 wicket null mean is twenty-one times
the observed. A null that can reproduce an effect exactly, or invert it, is not a
null against which a p-value can be read, so the permutation p-values are reported
but not used as evidence.

The repair is to impose the null instead of permuting the regressor. Dropping the
interaction, fitting the restricted model, and resampling its residuals with a
Rademacher weight drawn once per match gives a null centred at zero by
construction. Its standard deviation reproduces the match-clustered standard
error (0.00015 against 0.00016 in T20, 0.00009 against 0.00009 in ODI, and
0.00003 against 0.00003 in Test), and the dot-ball gradient is 9.3, 15.5 and 3.9
bootstrap standard deviations from zero. The Test permutation p = 0.06 is
therefore an artefact of the mis-centred null, not evidence of a weak effect. The
gradient is precise and real; what fails, in Sections 7.9 and 7.10, is
attribution. Output:
`wear_placebo_bias.csv` from `python src\placebo_bias.py`.

### 7.9 Validation of the wear signal

A coefficient can be precise and still be meaningless. Four tests ask whether the
inferred state behaves like a property of the pitch rather than an artefact of the
estimation sample.

**Cross-innings persistence.** If the coefficient measures a surface, the same
match should show a similar coefficient in both innings. Estimating the
coefficient separately inside each innings and correlating the two across matches
gives the answer in Table 7.10.

**Table 7.10: Correlation of the innings-level coefficient across the two innings
of the same match**

| Format | Outcome | Matches | Pearson r | p |
| --- | --- | --- | --- | --- |
| T20 | Runs per ball | 7,801 | −0.003 | 0.805 |
| T20 | Wicket rate | 7,801 | +0.002 | 0.882 |
| ODI | Runs per ball | 2,689 | −0.047 | 0.014 |
| ODI | Wicket rate | 2,689 | +0.026 | 0.173 |
| Test | Runs per ball | 763 | +0.030 | 0.402 |
| Test | Wicket rate | 763 | +0.000 | 0.994 |

The correlations are at or near zero, and the one nominally significant value
(ODI, r = −0.05) is negative and tiny. **The per-match coefficient does not
persist.** This is a genuine limit, not a technicality: it means the estimator is
not yet a measurement of any individual pitch.

The most likely reason is classical measurement error. A single innings supplies
20 to 50 overs, often with only a handful of spin overs in the death, so the
innings-level slope is estimated with a very large standard error; attenuation
then drives the cross-innings correlation towards zero whatever the underlying
state. The population coefficient is precise only because it pools 333,000 T20
cells. The correct reading is therefore that the **Pitch Wear Index is a
league-wide and venue-level statistic**, and that a match-level index needs many
more spin overs per innings than a T20 provides.

**Out-of-sample predictive gain.** A wear term should help predict held-out play.
Table 7.11 compares a ridge model of over-level outcomes with and without the
age x spin term, using four expanding-window folds over dates.

**Table 7.11: Weighted mean squared error with and without the age x spin term**

| Format | Outcome | Baseline MSE | With wear | Relative gain | 95% CI |
| --- | --- | --- | --- | --- | --- |
| T20 | Runs per ball | 0.44720 | 0.44719 | +0.002% | −0.019% to +0.003% |
| T20 | Dot-ball share | 0.04351 | 0.04349 | +0.038% | −0.014% to +0.041% |
| ODI | Runs per ball | 0.28697 | 0.28685 | +0.041% | +0.023% to +0.076% |
| ODI | Dot-ball share | 0.04475 | 0.04468 | +0.153% | +0.089% to +0.253% |
| Test | Runs per ball | 0.22946 | 0.22945 | +0.004% | −0.001% to +0.007% |
| Test | Dot-ball share | 0.03766 | 0.03765 | +0.022% | +0.011% to +0.032% |

The gains are positive and, for ODI, statistically distinguishable from zero, but
they are economically negligible: a fraction of a thousandth of the residual
variance. Adding one interaction to a model that already contains age, spin,
venue and phase cannot rescue a signal this small. The honest conclusion is that
the coefficient is a **structural estimate of how the relative effectiveness of
spin and pace changes**, not a predictive feature to be shipped in the match-level
model. It belongs in the analysis layer, not the feature store.

**Rolling coefficient stability.** Refitting the match fixed-effects model on four
expanding windows and then on the following window tests whether the sign is a
property of the period or of the sport.

**Table 7.12: Rolling-origin stability of the match fixed-effects coefficient**

| Format | Outcome | Training windows | Held-out windows | Same sign |
| --- | --- | --- | --- | --- |
| T20 | Dot-ball share | +0.00150 to +0.00155 | +0.00134 to +0.00167 | 4/4 |
| T20 | Runs per ball | −0.00257 to −0.00184 | −0.00471 to −0.00087 | 4/4 |
| T20 | Wicket rate | +0.00005 to +0.00010 | −0.00007 to +0.00020 | 3/4 |
| T20 | Boundary share | −0.00033 to −0.00013 | −0.00074 to +0.00025 | 2/4 |
| ODI | Dot-ball share | +0.00144 to +0.00154 | +0.00102 to +0.00182 | 4/4 |
| ODI | Runs per ball | −0.00132 to −0.00071 | −0.00231 to −0.00105 | 4/4 |
| ODI | Wicket rate | −0.00021 to −0.00018 | −0.00033 to −0.00011 | 4/4 |
| ODI | Boundary share | +0.00005 to +0.00020 | −0.00011 to +0.00003 | 1/4 |
| Test | Dot-ball share | +0.00006 to +0.00010 | +0.00012 to +0.00045 | 4/4 |
| Test | Runs per ball | −0.00011 to +0.00002 | −0.00046 to −0.00002 | 3/4 |
| Test | Wicket rate | −0.00001 to +0.00001 | −0.00004 to −0.00001 | 2/4 |
| Test | Boundary share | −0.00003 to −0.00000 | −0.00009 to −0.00000 | 4/4 |

Two things are being tested here and they give different answers. **Magnitude**
is stable: the T20 dot-ball coefficient sits within a ten-thousandth of 0.0015 in
every training window, and the held-out estimates stay the same order of
magnitude. **Sign** is stable for dot-ball share in all three formats and for
runs in T20 and ODI, but not for wicket rate or boundary share, and the Test
coefficients are an order of magnitude smaller than the limited-overs ones. A
coefficient that is this stable is measuring something real about how the game is
played; the question, taken up in Section 7.10, is whether that something is the
pitch.



**Cross-innings level contrast.** A different persistence test compares later
innings with the first innings of the same match, at the level of the spin-pace
gap rather than the slope. In T20, spin's relative wicket rate is higher in later
innings (+0.00194, p=0.002) and its dot share is higher at p=0.095. In Test, the
same contrast is positive for dot share (+0.01073, p<0.001) and wicket rate
(+0.00113, p=0.007). In ODI the sign reverses for runs (+0.02664, p<0.001) and
boundaries (+0.00550, p<0.001). The limited-overs reversal is the dew and
chasing confound discussed in Section 8.3; it is reported rather than dropped,
because a validator that only reports agreeing tests is not a validator. The Test
contrast is the one candidate that looks like a surface effect, so Section 7.10
subjects it to the same restriction ladder used at the boundary, and two thirds of
it does not survive.

**What survives.** Two things. First, the **population-level** relative ball-age
profile is real, precise and stable: in T20 and ODI, spin's dot-ball share rises
relative to pace as the innings progresses, and this is significant under the
corrected null, the boundary trim, the established-bowler restriction and the
phase decomposition.
Second, the **mid-innings** component is the one most likely to be surface
related, and it is consistent in sign across all three formats. What does not
survive is the stronger claim that the pooled coefficient is a uniform physical
wear rate, or that the index measures an individual pitch. The paper that comes
out of this work has to be built on the first two statements.

### 7.10 Is the gradient ball condition or surface change?

The tests above establish that the gradient is not a uniform wear rate, but they
do not say what it is. Two alternatives remain: **ball condition**, because the
ball softens and scuffs as an innings proceeds, and **bowler selection**, because
which bowler bowls an over changes with age. Three tests separate them.

**Test 1: the second new ball is a selection trap.** From over 80 the fielding
side may take a new ball, and in practice does. At that boundary the ball is
renewed while the surface continues to age smoothly, which looks like a clean
natural experiment. It is not. Table 7.13 reports an event study relative to
overs 78-79, in two-over bins, on 162,467 Test deliveries in overs 72-87. The
first row holds the match fixed and lets the bowler mix change. The last two hold
the bowler and innings fixed, so they compare the same bowler before and after
the boundary. Standard errors are clustered by match.

The published comparison ordinarily stops at the first and last rows, reports a
sign reversal, and calls the reversal a fixed-effects result. That reading is
wrong, because the within-bowler specification can only be estimated on
bowler-innings units that bowl on both sides of the boundary: it changes the
fixed effect *and* the sample at the same time. The middle rows separate the two
changes by adding one restriction at a time.

**Table 7.13: The second-new-ball jump is a change of personnel (dot-ball share,
overs 80-81 relative to 78-79)**

| Comparison | Matches (pace / spin) | Pace | Spin |
| --- | --- | --- | --- |
| Match fixed effects, all deliveries | 877 / 860 | +0.0148 (p=0.031) | −0.0100 (p=0.129) |
| Match fixed effects, matches with no continuing bowler | 472 / 347 | +0.0487 (p<0.001) | +0.0251 (p=0.522) |
| Match fixed effects, matches with a continuing bowler | 405 / 513 | −0.0005 (p=0.952) | −0.0125 (p=0.069) |
| Match fixed effects, continuing bowlers only | 405 / 513 | −0.0204 (p=0.051) | −0.0306 (p<0.001) |
| Bowler x innings fixed effects, continuing bowlers | 405 / 513 | −0.0210 (p=0.044) | −0.0344 (p<0.001) |

Across all deliveries, pace's dot-ball share rises by 1.5 percentage points in
the first two overs with the new ball, which is exactly what the ball-condition
story predicts. The rise is not present in the average match. In the 405 matches
where at least one pace bowler bowls on both sides of over 80 the naive estimate
is already zero, and restricting further to those bowlers makes it slightly
negative. The +1.5 point aggregate is carried by the other 472 matches, where the
pace attack turns over completely at the boundary, so a ball effect and a change
of bowler cannot be separated by construction. Within matches where bowlers
continue, adding bowler-innings fixed effects moves the estimate by 0.0006, about
3% of the gap. Composition is not a correction the fixed effects apply; it is the
sample.

That null is a bound rather than a zero. With a standard error of 0.008 the
design rules out a pace jump larger than about 2.3 percentage points in the
continuing-bowler matches, roughly half the +4.9 point jump seen where the attack
turns over, but it cannot rule out the 1.5 point aggregate on its own. The
selection conclusion rests on the difference between the two groups, which is 4.9
points at 3.2 standard errors, and on the exact decomposition of the smooth trend
in Section 7.10.

The pattern continues beyond the first two overs. On continuing bowlers the pace
coefficient is −0.0293 at overs 82-83, −0.0305 at 84-85 and −0.0449 at 86-87,
against +0.0035, +0.0029 and −0.0045 in the naive specification. The overall pace
share of deliveries rises from 40% before over 80 to 74% after it, so the naive
comparison attributes to the ball what belongs to the attack. Whatever the
continuing-bowler decline reflects — batters settling, spell fatigue, or a harder
ball that finds the edge rather than the middle — it is not a new-ball bonus, and
the boundary cannot be used to identify ball condition. This is the composition
problem caught in its most visible form, but it is a statement about the
boundary, not about the smooth trend.

**Test 2: the two-new-ball rule in ODIs.** From 1 October 2011 each ODI innings
uses two new balls, one from each end, in alternate overs [14]. Each ball is therefore
in play for at most about 25 overs instead of 50, so the ball ages at half the
rate per over of play. The pitch, by contrast, wears at the same rate per over.
If the smooth age x spin gradient were driven by ball ageing, it should roughly
halve after the rule. Twenty20, which has no mid-innings ball change, is used as
a control for common period effects. Table 7.14 gives the answer.

**Table 7.14: Age x spin coefficient for dot-ball share before and after the
two-new-ball rule**

| Stratum | Coefficient (SE) |
| --- | --- |
| ODI before October 2011 | +0.00129 (0.00020) |
| ODI after October 2011 | +0.00138 (0.00010) |
| T20 before October 2011 | +0.00205 (0.00085) |
| T20 after October 2011 | +0.00146 (0.00016) |
| Difference-in-differences (ODI change minus T20 change) | +0.00069 (0.00089), p=0.44 |
| Predicted ODI change if the gradient were ball ageing | −0.00064 |

The ODI gradient did not fall. It rose by 0.0001, against a prediction of
−0.0006, and the difference-in-differences is positive. The confidence interval
[−0.0011, +0.0024] spans zero and is wide enough to include the ball-ageing
prediction, so the test is evidence against that explanation rather than a
rejection of it.

**Test 3: does the smooth gradient survive bowler fixed effects?** The second new
ball is a discontinuity, and a discontinuity can be dominated by who bowls it.
The smooth trend need not be. A delivery-level model with match x innings x bowler
fixed effects removes the between-bowler mix entirely. The spin main effect is
absorbed, so the age x spin coefficient is the difference between the spin and
pace age slopes with each bowler-innings level held fixed. Table 7.15 reports
dot-ball share under four nested specifications, with standard errors clustered by
match. There are 1,983,526 T20, 1,461,903 ODI and 1,670,736 Test deliveries.

**Table 7.15: Within-bowler age x spin coefficient for dot-ball share**

| Format | No controls | Match state | + Batter and gender | + Phase |
| --- | --- | --- | --- | --- |
| T20 | +0.00197 (0.00019) | +0.00218 (0.00019) | +0.00218 (0.00018) | +0.00196 (0.00019) |
| ODI | +0.00260 (0.00011) | +0.00250 (0.00010) | +0.00227 (0.00010) | +0.00223 (0.00011) |
| Test | +0.00034 (0.00004) | +0.00035 (0.00004) | +0.00025 (0.00004) | +0.00027 (0.00004) |

The gradient does not collapse, and it does not grow either: it is unchanged in
T20 and larger in ODI and Test. In the full specification it is +0.00196 in T20,
+0.00223 in ODI and +0.00027 in Test, against +0.00198, +0.00176 and +0.00014 for
the pooled match fixed-effects estimate on exactly the same deliveries. All
coefficients are significant at p<0.001.

A comparison of two coefficients is still an inference, so the same coefficient
was decomposed exactly. The pooled coefficient is a variance-weighted average of
a within-bowler and a between-bowler slope, with weights given by the share of
the residual variation in the age x spin interaction that survives within
bowler-innings units. The composition term, which is the pooled coefficient minus
the within-bowler coefficient, is +0.00002 in T20, −0.00047 in ODI and −0.00012
in Test. It is therefore 1% of the coefficient in T20 and *negative* in the other
two formats: where composition matters at all it makes the gradient look smaller
than it is, which is the opposite of what the composition hypothesis predicts.
Bowler selection does **not generate** the pooled estimate. The runs and boundary
channels move in the same direction; the wicket-rate channel is small and mixed.

This result removes one alternative explanation and it also limits the paper's
thesis. The boundary is selection, but the smooth trend is not. Two caveats keep
the result from becoming a wear claim. First, the comparison conditions on
bowlers who bowl on both sides of an age contrast, so it selects long spells and
stable roles and is not an experiment. Second, bowler fixed effects remove the
between-bowler mix but not spell stage, field changes or batter adaptation, all of
which move with age within a bowler. What survives is a robust description of how
the relative effectiveness of spin and pace changes with ball age. What does not
survive is the attribution of that description to the surface: the phase
reversal, the absent persistence, the negligible predictive gain and the rule
test all still stand.

Taken together, the first three tests separate the two alternatives. The ball is
neither the source of the aggregate jump at renewal nor a supported explanation
for the smooth gradient. The bowler explains the jump but not the trend. The
within-innings coefficient is therefore neither a uniform wear rate nor a
ball-age rate nor a bowler-composition artefact.

**Test 4: does the across-innings contrast survive?** The cross-innings contrast
of Section 7.9 is the last candidate, because it compares two innings of the same
match and the surface has had days rather than overs to change. On pooled data it
is +0.01073 in Test dot-ball share. The same
restriction ladder used at the boundary applies here, and it removes most of the
contrast. Table 7.16 reports it on 1,670,736 Test deliveries in 922 matches, with
one restriction added at a time and standard errors clustered by match.

**Table 7.16: Stress-testing the across-innings Test contrast (dot-ball share)**

| Design | Matches | Coefficient (SE) | p |
| --- | --- | --- | --- |
| Match fixed effects, all matches | 922 | +0.0122 (0.0028) | <0.001 |
| Add follow-on and declaration interactions | 922 | +0.0068 (0.0029) | 0.020 |
| Match fixed effects, matches without a follow-on | 828 | +0.0091 (0.0029) | 0.002 |
| Match fixed effects, matches with a follow-on | 94 | +0.0390 (0.0085) | <0.001 |
| Repeated innings (3-4 versus 1-2), no follow-on | 828 | +0.0036 (0.0024) | 0.135 |
| Each side's second innings versus its first, match x team fixed effects | 828 | +0.0035 (0.0024) | 0.137 |

The first row is a delivery-level replication of the cell-panel estimate in
Section 7.9. It is 0.0015 larger because the panel weights each over x style cell
by its deliveries and enters cell means of the controls, whereas the replication
treats each delivery as a unit; both estimate the same population coefficient, and
the delivery-level version is used here because the ladder requires identical rows
on both sides of each restriction. Three restrictions then move the number. The contrast is 4.3 times
larger in the 94 matches that produced a follow-on, where one side was far enough
ahead to enforce it and the side batting again had just been dismissed cheaply.
Adding a follow-on indicator and a declaration indicator, each interacted with
spin, cuts it by 44% to +0.0068, because a third of second-innings deliveries and
half of fourth-innings deliveries come in an innings that began after a
declaration. The strictest design avoids comparing different batting sides at
all: in a Test without a follow-on the innings alternate between the sides, so
each side's second innings can be compared with its own first, with match x team
fixed effects absorbing the batting side exactly. That returns +0.0035
(SE 0.0024, p=0.14), about a third of the headline, with a 95% interval of
[−0.001, +0.008].

The strict design is not an experiment. A third innings is often a declaration
setup and a fourth is a chase, so the batting side's objective differs between
the two observations, and the match-state controls do not capture strategic
intent. We therefore do not read +0.0035 as the surface effect. The correct
reading is that the pooled +0.0107 is mostly composition and selection. The strict
design cannot distinguish its own estimate from zero, but its 95% interval,
[−0.001, +0.008], does not contain +0.0107, so it rejects the pooled magnitude at
the 5% level while leaving open whether a smaller effect of up to about 0.008 is
present. The only
wear-consistent evidence in the data is therefore this contrast at a third of its
pooled size, and it is not statistically distinguishable from zero; there is no
confirmed surface-scale signal. Output:
`wear_across_innings_stress.csv` from `python src\across_innings.py`.

## 8. Results and Discussion

### 8.1 Phase-I results

The project has produced a working, reproducible pipeline; a documented proxy
label; a complete geocoding audit; venue behaviour profiles; a validated
three-format model comparison; and a ball-by-ball estimator of within-match
pitch wear. On the chronological hold-out the match-level macro-F1 is
0.443 for T20, 0.414 for ODI and 0.561 for Test, against majority baselines of
0.269, 0.248 and 0.225. Rolling-origin cross-validation gives the more reliable
estimates of 0.453, 0.422 and 0.568. The models are therefore learning beyond
"Pace is most common", but the margin is modest. The accuracy paradox in T20 and
ODI (the majority class is more accurate but far worse on macro-F1) shows that
headline accuracy is the wrong summary for this imbalanced target. The Batting
class, gender confounding and T20 role coverage remain the main weaknesses.

The dynamics analysis adds a second, independent line of evidence. Across
5.1 million legal spin and pace deliveries aggregated to 855,000 over-level
cells, the within-match estimator shows that the relative effectiveness of spin
and pace changes systematically as the ball ages, and the change is precisely
estimated and stable across four rolling windows. It is not, however, identified
as surface wear. The phase decomposition reverses sign; the T20 runs channel
disappears under a boundary trim; the permutation null is mis-centred, so its
p-values are unusable, but a null-imposed bootstrap shows the dot-ball gradient is
significant; the per-match coefficient has no cross-innings persistence and
negligible predictive value. The identification tests in Section 7.10 then remove the two leading
alternatives in opposite directions: the second-new-ball jump is entirely bowler
selection, while the smooth gradient is *not* bowler composition, because holding
the bowler fixed makes it larger. Halving the rate at which an ODI ball ages also
leaves the gradient unchanged. The paper that follows is therefore built on two
identification results — the boundary is selection and the trend is not
composition but is also not identified as surface wear — and on the demonstration
that even the across-innings Test difference, the last candidate, loses two thirds
of its size under a within-side design. It is not built on a
claim that the within-innings coefficient measures an individual pitch.

### 8.2 Were the objectives met?

| Objective | Verdict |
| --- | --- |
| Complete the data | **Largely met.** Geocoding is complete and audited; weather is complete at 100% match coverage with prior-window features; 3,005 players have a known style after 527 audited overrides. T20 label coverage is still 21.8% because every wicket-taker must be known. |
| Build pitch features | **Partly met.** PBP exists for 13 men's venues; the PWI is now defined as the innings-level age x spin coefficient and estimated for every match with enough spin and pace cells, but it is not yet normalised for team strength or bowler quality. |
| Estimate within-match wear | **Partly met.** The age x spin coefficient is estimated precisely and is stable across fixed-effect specifications, outcomes and rolling windows. Section 7.10 shows it is not identified as wear: the apparent second-new-ball effect is concentrated in matches where the pace attack turns over at the boundary and is bounded below about 2.3 points where a seamer continues, and halving the rate at which an ODI ball ages leaves the ODI gradient unchanged. An exact decomposition finds a negligible composition term for the smooth trend, so the trend is not a composition artifact either; it is simply not attributable to the surface. The across-innings Test contrast is the only remaining candidate, and two thirds of it does not survive a within-side design, so no surface-scale signal is confirmed. |
| Validate the inferred state | **Partly met.** Both convergent-validity gradients (venue spin-friendliness and pre-match dryness) and the population-level cross-innings contrast are detected; the per-match slope does not persist across innings, and the out-of-sample predictive gain is positive but small, as expected for a single added term. |
| Train and validate classifiers | **Partly met.** All three formats beat their majority baselines on macro-F1; the 0.50 target is met by Test in cross-validation (0.568). T20 and ODI reach 0.453 and 0.422. |
| Explain and handle new grounds | **Partly met.** Gain importance and an earlier SHAP analysis are interpreted; cold-start matching is designed but not validated. |
| Disseminate | **Partly met.** The conference paper draft is written (`reports/PAPER_DRAFT.md`) and this report contains the full identification analysis; the dashboard prototype remains. |

### 8.3 Unexpected findings

1. **Gender dominates the explanation.** A variable that should not describe a pitch is the strongest predictor. This points to confounding between match category and label distribution, and it must be addressed before any applied claim.
2. **Accuracy and macro-F1 disagree.** The majority baseline wins on accuracy but loses badly on macro-F1. Reporting accuracy alone would have made the models look worse than they are; reporting only macro-F1 would hide the weak Batting class.
3. **Role coverage, not weather, is the bottleneck.** T20 label coverage is 21.8% because every wicket-taker must be known, against 53.4% in ODI and 71.3% in Test. The 527 audited overrides added a third more T20 matches; finishing the review queue is still worth more than further model tuning.
4. **Expert labels are noisy by nature.** The Adams et al. result (umpires reliable on bounce, unreliable on pace) is a useful reminder that even human assessment is imperfect, which supports the decision to report proxy agreement rather than claim physical truth.
5. **The pooled dynamics coefficient is a mixture, and the mix is measurable.** A single age slope across the whole innings hides three regimes: the powerplay, where pace's new-ball advantage decays; the middle, where spin's dot share drifts up in every format; and the death, where T20 reverses. The second-new-ball event study makes the mixture visible: the aggregate jump at over 80 is the return of the frontline seamers and is absent in matches where a seamer continues. But the smooth trend is not the same object. An exact decomposition writes the pooled coefficient as a variance-weighted average of a within-bowler and a between-bowler slope, and the composition term is 1% of the coefficient in T20 and negative in ODI and Test, so bowler composition does not create the gradient. The honest object to publish is the pair of results — selection at the boundary, an unidentified trend across the innings — not a single pooled number.
6. **The Test result is a genuine null, not a failure of power, and the last candidate does not survive either.** With 278,746 Test cells the standard errors are tiny, so the absence of a pooled runs effect is not a sample-size problem. The dot-ball channel is positive and significant under the corrected null, but the new-ball phase points the other way, and the second-new-ball jump is confined to matches where the attack turns over at the boundary. The one contrast that is positive on several channels is the later-innings Test gap, which is consistent with a slow surface process at the innings scale. It is also four times larger in matches where a follow-on happened, falls by 44% once follow-on and declaration interactions are added, and returns a third of its size when each side is compared with its own first innings (p=0.14). The contrast is the right candidate at the right scale, but this data set cannot confirm it, and the honest claim is an upper bound rather than an estimate.
7. **A second-innings ODI effect points the wrong way.** Later innings show a *relative* improvement for spin in runs and boundaries (positive coefficients), which contradicts wear. The most plausible explanation is dew and the chasing team's risk profile, both of which are properties of the second innings rather than the surface. This is reported as a confound, not as evidence against wear.
8. **The per-match Pitch Wear Index is not yet reliable.** The population coefficient is precise and reproducible, but the same coefficient estimated inside a single innings does not correlate between the first and second innings of the same match (r ≈ 0.00 in T20, 0.03 in Test, −0.05 in ODI). Classical measurement-error attenuation explains the direction: one innings of 20 to 50 overs cannot pin down a slope that needs hundreds of thousands of cells to separate from noise. The PWI is therefore a league-wide and venue-level measure for now, not a per-match instrument, and the report says so instead of presenting the correlation as support.

### 8.4 Limitations and threats to validity

- **Proxy labels.** The target is a behavioural rule, not a measurement. A match labelled Spin may be a pace-friendly pitch on which spinners happened to take wickets.
- **The dynamics estimand is relative.** It identifies how spin's effectiveness changes relative to pace; it does not measure the pitch, the ball, or the bowler's skill. "Wear" is shorthand for that relative change.
- **Style is not always known.** 22.6% of deliveries have an unknown bowler style and are excluded, and a part-time spinner classified as pace would attenuate the contrast. The established-bowler check bounds this.
- **Selection is not fully exogenous.** Captains choose when to bowl spin, and that choice responds to the match situation. Phase controls, match-state controls, the within-innings placebo and the delivery-level within-bowler specification address this only partially; the last conditions on bowlers who bowl across an age contrast, so it selects long spells and stable roles.
- **Wicket rate is contaminated.** A wicket credited to a bowler can come from a catch, a run-out-adjacent dismissal or a tail-ender's error, so it mixes pitch behaviour with execution and opposition quality.
- **Test cricket has no clean wear clock.** Innings age, ball condition, follow-ons and declarations move together, which is why the match-use specification is reported alongside the innings-age specification rather than instead of it.
- **Innings order is not random.** In the strict across-innings design each side's second innings is compared with its first, which holds the batting side fixed but not the match situation: a third innings is often a declaration setup and a fourth is a chase. The match-state controls absorb the prevailing run rate and wickets lost but not the strategic intent behind them. The bias is unsigned, which is why the strict estimate is reported as an unresolved bound rather than as a surface effect.
- **Small, imbalanced sample at the match level.** 5,400 labelled matches across three formats, but only 571 Batting examples (67 in Test); confidence intervals are wide.
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

The project set out to make pitch behaviour measurable with open data. It has delivered the infrastructure to do so: a complete match inventory, a documented proxy label, a reviewed geocoding layer, leakage-safe features, a complete weather table with prior-window moisture, a three-format model comparison with explanation and calibration, and a ball-by-ball estimator of within-match dynamics together with the tests needed to identify it. The answer is cautious. At the match level, rolling-origin cross-validation reaches macro-F1 0.453 (T20), 0.422 (ODI) and 0.568 (Test) against majority baselines of 0.269, 0.248 and 0.225, and venue profiles show a plausible relationship between scoring and spin (r = −0.64). At the ball-by-ball level the estimate is precise and stable: spin's dot-ball share rises relative to pace as the ball ages in T20 and ODI, by about 0.0015 per over, and the result survives a boundary trim, an established-bowler restriction and four rolling windows. It is not, however, identified as surface wear. The standard permutation null is mis-centred to the point of being unusable, but a null-imposed wild cluster bootstrap restores a zero-centred test under which the gradient is significant in all three formats; the effect then reverses sign by phase, the per-match slope does not persist across innings, and the identification tests point in two directions: the apparent new-ball effect is bowler selection, while holding the bowler fixed does not shrink the smooth gradient and an exact decomposition puts the composition term at 1% of it or less, so the trend is not composition either. Halving the rate at which an ODI ball ages leaves the gradient unchanged. The coefficient is therefore a robust description of relative style effectiveness, not a reading of the surface; the across-innings Test contrast is the last candidate, and two thirds of it does not survive a within-side design. The models are not yet ready to advise a captain. The Batting class is weak, gender is a confounder at the match level, and the dynamics estimand is relative rather than physical. The remaining work is clear: extend the role review, design an instrument that separates attack composition from surface change at the innings scale, resolve the gender confounder, validate cold-start behaviour, and build the dashboard that turns the coefficients into a pre-match decision aid.

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

The next six figures are generated directly from the ball-by-ball analysis and are the visual evidence for Sections 7.8 to 7.10.

![Wear curves](reports/figures/fig1_wear_curves.png)

**Figure 8 — Wear curves.** Scoring rate and dot-ball share by ball age for spin and pace, in each format. This is descriptive and does not adjust for phase composition; the adjusted estimates are in Table 7.7.

![Specification forest](reports/figures/fig2_specification_forest.png)

**Figure 9 — Specification forest.** The age x spin coefficient under each fixed-effect specification, with 95% confidence intervals. Agreement across specifications, not the size of any single estimate, is the point.

![Convergent validity](reports/figures/fig3_convergent_validity.png)

**Figure 10 — Convergent validity.** The age x spin estimate by venue spin-friendliness tercile and by pre-match dry days.

![Per-phase estimates](reports/figures/fig4_phase_estimates.png)

**Figure 11 — Per-phase estimates.** The same coefficient re-estimated inside each phase. The pooled effect is not uniform across phases, which is why the report reads the phase decomposition as a diagnostic rather than a footnote.

![Second-new-ball event study](reports/figures/fig5_new_ball_discontinuity.png)

**Figure 12 — Second-new-ball event study.** Dot-ball share in two-over bins relative to overs 78-79. Grey: all deliveries, with match fixed effects. Blue: bowlers who bowl on both sides of the boundary, within the same bowler and innings. The naive series bumps at the boundary; the continuing-bowler series declines through it. The bump is the return of the frontline seamers, not the new ball, and the blue series changes the sample as well as the fixed effect, which is why Table 7.13 reports each change separately.

![Two-new-ball rule](reports/figures/fig6_two_new_ball_rule.png)

**Figure 13 — Two-new-ball rule.** Age x spin coefficient for dot-ball share in ODI and in the T20 control, before and after October 2011. The ODI gradient does not fall when each ball is in play for half as many overs, which is what a ball-ageing explanation requires.

## 12. Outcome of the Work

**Primary target: conference publication.** The conference draft is written (`reports/PAPER_DRAFT.md`) and is built on identification rather than on a positive predictive claim. The within-innings wear proxy is precise and stable but is not identified as surface change: the second-new-ball jump is a selection trap, the smooth gradient survives bowler fixed effects and so is not composition either, and the phase, persistence, prediction and rule tests still fail. The report presents the boundary and the trend as two separate results rather than as one wear estimate.

**Stretch target: journal publication.** An extended paper is feasible around the
identification audit itself: the boundary ladder, the exact decomposition, and the
stress test that removes two thirds of the across-innings contrast. A confirmatory
study of surface change at the innings scale would additionally need a design that
separates surface change from attack composition, which the within-innings
coefficient is not and the report says so.

**Software and data.** The repository is a reusable open-data pipeline with documented schemas, cached weather responses, a readiness audit and a reproducible identification module.

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

[14] International Cricket Council, "David Richardson answers your questions," 4 October 2011. [Online]. Available: https://www.icc-cricket.com/news/david-richardson-answers-your-questions

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
| FE | Fixed Effects | Match or innings indicators absorbed in the dynamics regression |
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
| PWI | Pitch Wear Index | Innings-level age x spin coefficient estimated per match |
| RF | Random Forest | Ensemble of decision trees |
| SE | Standard Error | Precision of an estimated coefficient |
| SHAP | SHapley Additive exPlanations | Game-theoretic feature-attribution method |
| SVM | Support Vector Machine | Margin-based classifier (reviewed literature) |
| T20 | Twenty20 | 20-over format |
| T20I | Twenty20 International | 20-over international match |
