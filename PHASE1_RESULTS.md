# Phase 1 preliminary results (4 November 2026)

Reproduce from the repo root (needs `pandas numpy scikit-learn xgboost joblib`):

    python src\run_t20_preliminary_analysis.py

## Scope limits

- **T20 internationals only.** `data/raw/cricsheet/all_json.zip` in this copy is a 134-byte Git LFS pointer, so ODI and Test could not run. Run `git lfs pull` in a clone.
- **No weather.** `match_weather.csv` does not exist. The 24 cached ERA5 batches cover 2001-2011 only, and the collector could not reach Open-Meteo from this environment. Re-run it after the quota reset (05:30 IST).
- **Role-complete matches only.** Metrics measure agreement with proxy labels, not pitch physics.

## README steps run

| Step | Result | Output |
| --- | --- | --- |
| 2. Bowler-role lookup | 1,069 unambiguous names; 42% of credited T20I wickets | `data/bowler_roles.csv` |
| 2. Manual review queue | 3,291 unresolved bowlers, ranked by wickets. Candidates are unverified hints and are not used | `data/interim/bowler_role_review_queue.csv` |
| 3. `phase1_phase2.py` (unmodified) | 843 matches: Pace 573, Spin 203, Batting 67 | `data/processed/X.csv`, `y.csv`, `proxy_label_audit.csv` |
| 5.1 Checks | 5,849 matches, 5,713 parsed, 868 labelled (15%); no missing dates or venues | `audit_sample_for_manual_review.csv` (25 rows still need reading by a person) |
| 5.2 Venue aliases | 317 venue strings map to 301 Cricsheet venue IDs | `venue_behaviour_profiles.csv` |
| 5.4 Threshold sensitivity | Up to 9.2% of labels change; spin:pace ratio is the most sensitive cut-off | `label_threshold_sensitivity.csv` |
| 5.5 Chronological models | See below | `models/pitch_proxy_t20i_preliminary/` |

The label logic in `phase1_phase2.py` and `train_pitch_models.py` agreed on all 842 matches compared.

## Held-out results (train to 2024-02-02, test from 2024-02-09, n = 165)

| Model | Accuracy | Balanced accuracy | Macro-F1 (95% CI) | Log loss |
| --- | --- | --- | --- | --- |
| Majority class | 0.582 | 0.333 | 0.245 (0.223-0.265) | 0.913 |
| Venue prior | 0.594 | 0.359 | 0.318 (0.265-0.367) | 0.958 |
| Logistic regression | 0.455 | 0.463 | 0.406 (0.333-0.474) | 1.126 |
| Random forest | 0.497 | 0.409 | 0.405 (0.341-0.474) | 0.925 |
| XGBoost | 0.497 | 0.427 | 0.410 (0.336-0.491) | 1.015 |

Trained models beat the baselines on macro-F1 but not on accuracy. Without the men's/women's feature, XGBoost macro-F1 falls to 0.347. Five-fold time-aware CV mean macro-F1: Random forest 0.45, Logistic regression 0.40, XGBoost 0.40.

## Findings to act on

1. **Bowler styles are the bottleneck.** Curating the top 400 unresolved bowlers lifts role-complete matches from 919 to 1,618.
2. **One threshold set does not fit both categories.** Batting labels: men 11%, women 2%. Spin labels: men 14%, women 47%.
3. **The abstract and the code differ.** The abstract promises four classes, Random Forest + Gradient Boosting, a Pitch Wear Index, cold-start matching and SHAP. The code has three classes and XGBoost. The Pitch Wear Index here is only a raw run-rate difference (`raw_second_minus_first_rr`), not normalised for team strength.
4. **README is out of date on one point.** It says the abstract PDF is empty; it is complete.
