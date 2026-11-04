"""Train separate, chronological XGBoost pitch-proxy models by match format.

The target is a deterministic behavioral proxy, not observed physical pitch
ground truth. Weather is city-level ERA5 daily reanalysis, not stadium
measurements. All reported metrics therefore measure agreement with the proxy.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


LOGGER = logging.getLogger("pitch_model_training")
FORMATS = ("T20", "ODI", "Test")
CLASS_ORDER = ("Batting", "Pace", "Spin")
WICKET_DISMISSALS = {
    "bowled",
    "caught",
    "caught and bowled",
    "lbw",
    "stumped",
    "hit wicket",
}
LABEL_RULES = {
    "T20": {
        "early_overs": 6,
        "batting_run_rate_min": 8.5,
        "early_dot_percent_max": 35.0,
    },
    "ODI": {
        "early_overs": 10,
        "batting_run_rate_min": 5.5,
        "early_dot_percent_max": 60.0,
    },
    "Test": {
        "early_overs": 10,
        "batting_run_rate_min": 3.8,
        "early_dot_percent_max": 75.0,
    },
}


def _normalise_name(value: str) -> str:
    ascii_name = (
        unicodedata.normalize("NFKD", str(value))
        .encode("ascii", "ignore")
        .decode("ascii")
        .casefold()
    )
    return re.sub(r"[^a-z0-9]+", " ", ascii_name).strip()


def _style_class(value: str) -> str | None:
    style = _normalise_name(value)
    if not style or style in {"na", "n a", "none", "unknown", "not available"}:
        return None
    spin_terms = (
        "spin",
        "break",
        "orthodox",
        "chinaman",
        "googly",
        "legbreak",
        "offbreak",
        "wrist",
        "finger",
    )
    pace_terms = ("fast", "medium", "seam", "swing", "pace")
    is_spin = any(term in style for term in spin_terms)
    is_pace = any(term in style for term in pace_terms)
    if is_spin == is_pace:
        return None
    return "spin" if is_spin else "pace"


def _load_style_sources(
    international_zip: Path, worldwide_zip: Path
) -> tuple[dict[str, dict[str, str]], dict[str, list[tuple[str, str]]]]:
    """Build exact Cricinfo-ID and conservative exact-name style lookups."""
    cricinfo_rows: dict[str, set[str]] = defaultdict(set)
    with zipfile.ZipFile(international_zip) as archive:
        with archive.open("all_players.csv") as data_file:
            for row in csv.DictReader(io.TextIOWrapper(data_file, encoding="utf-8-sig")):
                style = _style_class(row.get("bowling_style", ""))
                if style:
                    cricinfo_rows[str(row["id"]).strip()].add(style)
    cricinfo_styles = {
        identifier: next(iter(styles))
        for identifier, styles in cricinfo_rows.items()
        if len(styles) == 1
    }

    name_rows: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with zipfile.ZipFile(worldwide_zip) as archive:
        with archive.open("players_data_with_all_info.csv") as data_file:
            for row in csv.DictReader(io.TextIOWrapper(data_file, encoding="utf-8-sig")):
                style = _style_class(row.get("bowlingstyle", ""))
                if not style:
                    continue
                full_name = row.get("fullname") or (
                    f"{row.get('firstname', '')} {row.get('lastname', '')}"
                )
                key = _normalise_name(full_name)
                if key:
                    name_rows[key].append((str(row.get("id", "")), style))
    return (
        {"cricinfo": cricinfo_styles},
        {name: rows for name, rows in name_rows.items()},
    )


def build_player_roles(
    people_path: Path,
    names_path: Path,
    international_zip: Path,
    worldwide_zip: Path,
) -> tuple[dict[str, str], pd.DataFrame]:
    """Join public role sources to Cricsheet identifiers and retain provenance."""
    id_lookup, name_lookup = _load_style_sources(international_zip, worldwide_zip)
    people: dict[str, dict[str, str]] = {}
    aliases: dict[str, set[str]] = defaultdict(set)
    with people_path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            identifier = row["identifier"]
            people[identifier] = row
            for column in ("name", "unique_name"):
                if row.get(column):
                    aliases[identifier].add(row[column])
    with names_path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if row.get("identifier") in people and row.get("name"):
                aliases[row["identifier"]].add(row["name"])

    roles: dict[str, str] = {}
    audit: list[dict[str, str]] = []
    for identifier, person in people.items():
        cricinfo_ids = {
            value.strip()
            for column, value in person.items()
            if column.startswith("key_cricinfo") and value and value.strip()
        }
        direct_styles = {
            id_lookup["cricinfo"][cricinfo_id]
            for cricinfo_id in cricinfo_ids
            if cricinfo_id in id_lookup["cricinfo"]
        }
        if len(direct_styles) == 1:
            roles[identifier] = next(iter(direct_styles))
            audit.append(
                {
                    "cricsheet_id": identifier,
                    "name": person.get("name", ""),
                    "style": roles[identifier],
                    "source": "Kaggle international player stats",
                    "join_method": "exact Cricinfo ID",
                }
            )
            continue
        if direct_styles:
            continue

        candidates: dict[str, set[str]] = defaultdict(set)
        for alias in aliases[identifier]:
            for source_id, style in name_lookup.get(_normalise_name(alias), []):
                candidates[source_id].add(style)
        if len(candidates) != 1:
            continue
        candidate_styles = next(iter(candidates.values()))
        if len(candidate_styles) != 1:
            continue
        roles[identifier] = next(iter(candidate_styles))
        audit.append(
            {
                "cricsheet_id": identifier,
                "name": person.get("name", ""),
                "style": roles[identifier],
                "source": "Kaggle worldwide player profiles",
                "join_method": "unique exact normalized name/alias",
            }
        )
    return roles, pd.DataFrame(audit)


def _label_match(
    match_type: str,
    run_rate: float,
    early_dot_percent: float,
    spin_wickets: int,
    pace_wickets: int,
) -> str | None:
    rules = LABEL_RULES[match_type]
    if (
        run_rate >= rules["batting_run_rate_min"]
        and early_dot_percent <= rules["early_dot_percent_max"]
    ):
        return "Batting"
    if spin_wickets + pace_wickets == 0:
        return None
    if spin_wickets >= 2 and (
        pace_wickets == 0 or spin_wickets / pace_wickets >= 1.25
    ):
        return "Spin"
    return "Pace"


def _parse_matches(
    archive_path: Path, roles: dict[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_path) as archive:
        names = sorted(
            name
            for name in archive.namelist()
            if name.lower().endswith(".json") and not name.startswith("__MACOSX/")
        )
        for name in names:
            with archive.open(name) as match_file:
                match = json.load(match_file)
            info = match.get("info", {})
            match_type = str(info.get("match_type", ""))
            if match_type not in FORMATS:
                continue
            if str(info.get("outcome", {}).get("result", "")).lower() == "no result":
                continue
            dates = info.get("dates", [])
            if not dates:
                continue
            match_date = pd.to_datetime(dates[0], errors="coerce")
            if pd.isna(match_date):
                continue
            venue = str(info.get("venue", "")).strip()
            if not venue:
                continue

            innings_list = match.get("innings", [])
            if not innings_list:
                continue
            first_innings = innings_list[0]
            balls_per_over = int(info.get("balls_per_over", 6))
            early_overs = LABEL_RULES[match_type]["early_overs"]
            first_runs = 0
            first_legal_balls = 0
            early_legal_balls = 0
            early_dot_balls = 0
            spin_wickets = 0
            pace_wickets = 0
            unknown_wicket_style = False

            registry = info.get("registry", {}).get("people", {})
            for innings_index, innings in enumerate(innings_list):
                for over in innings.get("overs", []):
                    over_number = int(over.get("over", 0))
                    for delivery in over.get("deliveries", []):
                        extras = delivery.get("extras", {})
                        total_runs = int(delivery.get("runs", {}).get("total", 0))
                        legal = "wides" not in extras and "noballs" not in extras
                        if innings_index == 0:
                            first_runs += total_runs
                            if legal:
                                first_legal_balls += 1
                                if over_number < early_overs:
                                    early_legal_balls += 1
                                    if total_runs == 0:
                                        early_dot_balls += 1
                        for wicket in delivery.get("wickets", []):
                            if str(wicket.get("kind", "")).strip().lower() not in WICKET_DISMISSALS:
                                continue
                            bowler_id = registry.get(delivery.get("bowler", ""))
                            style = roles.get(bowler_id)
                            if style == "spin":
                                spin_wickets += 1
                            elif style == "pace":
                                pace_wickets += 1
                            else:
                                unknown_wicket_style = True

            if (
                first_legal_balls == 0
                or early_legal_balls < early_overs * balls_per_over
            ):
                continue
            run_rate = first_runs / (first_legal_balls / balls_per_over)
            early_dot_percent = 100.0 * early_dot_balls / early_legal_balls
            style_complete = not unknown_wicket_style
            label = (
                _label_match(
                    match_type,
                    run_rate,
                    early_dot_percent,
                    spin_wickets,
                    pace_wickets,
                )
                if style_complete
                else None
            )
            teams = info.get("teams", [])
            toss = info.get("toss", {})
            first_batting_team = first_innings.get("team")
            toss_winner_batted_first = int(
                bool(first_batting_team) and toss.get("winner") == first_batting_team
            )
            city = str(info.get("city", "")).strip()
            venue_registry = info.get("registry", {}).get("venues", {})
            venue_id = venue_registry.get(venue, "")
            rows.append(
                {
                    "match_id": Path(name).stem,
                    "date": match_date,
                    "match_type": match_type,
                    "venue": venue,
                    "venue_id": venue_id,
                    "city": city,
                    "gender": str(info.get("gender", "unknown")),
                    "team_type": str(info.get("team_type", "unknown")),
                    "toss_decision": str(toss.get("decision", "unknown")).lower(),
                    "toss_winner_batted_first": toss_winner_batted_first,
                    "first_innings_runs": first_runs,
                    "first_innings_run_rate": run_rate,
                    "early_dot_percent": early_dot_percent,
                    "early_overs": early_overs,
                    "early_legal_balls": early_legal_balls,
                    "spin_wickets": spin_wickets,
                    "pace_wickets": pace_wickets,
                    "role_complete": style_complete,
                    "y": label,
                }
            )
    if not rows:
        raise ValueError("No eligible T20, ODI, or Test matches were parsed.")
    matches = pd.DataFrame(rows).sort_values(["date", "match_id"], kind="stable")
    return matches, matches[matches["y"].notna()].copy()


def _add_prior_venue_features(matches: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Create format-specific venue history using strictly earlier match dates."""
    result = pd.DataFrame(index=matches.index)
    metrics = [
        "first_innings_run_rate",
        "early_dot_percent",
    ]
    for match_type, format_rows in matches.groupby("match_type", sort=False):
        for venue_key, venue_rows in format_rows.groupby(
            venue_rows_key(format_rows), sort=False, dropna=False
        ):
            ordered = venue_rows.sort_values(["date", "match_id"], kind="stable")
            prior: deque[dict[str, float]] = deque(maxlen=window)
            for _, same_day in ordered.groupby(ordered["date"].dt.normalize(), sort=True):
                history = pd.DataFrame(prior)
                for column in metrics:
                    result.loc[same_day.index, f"venue_last{window}_mean_{column}"] = (
                        history[column].mean() if not history.empty else np.nan
                    )
                role_history = history.loc[
                    history["role_complete"].eq(1), "spin_wicket_share"
                ] if not history.empty else pd.Series(dtype=float)
                result.loc[
                    same_day.index, f"venue_last{window}_mean_spin_wicket_share"
                ] = role_history.mean() if not role_history.empty else np.nan
                result.loc[same_day.index, "venue_matches_in_history"] = len(prior)
                result.loc[same_day.index, "venue_role_matches_in_history"] = int(
                    history["role_complete"].sum() if not history.empty else 0
                )
                for _, row in same_day.iterrows():
                    prior.append(
                        {
                            "first_innings_run_rate": float(row["first_innings_run_rate"]),
                            "early_dot_percent": float(row["early_dot_percent"]),
                            "spin_wicket_share": float(row["spin_wicket_share"])
                            if pd.notna(row["spin_wicket_share"])
                            else np.nan,
                            "role_complete": int(row["role_complete"]),
                        }
                    )
    return result


def venue_rows_key(rows: pd.DataFrame) -> pd.Series:
    """Use a stable Cricsheet venue ID where supplied, otherwise normalized name."""
    fallback = rows["venue"].map(_normalise_name)
    registered = rows["venue_id"].fillna("").astype(str).str.strip()
    return registered.where(registered.ne(""), fallback)


def build_features(matches: pd.DataFrame, weather_path: Path) -> pd.DataFrame:
    matches = matches.copy()
    total_wickets = matches["spin_wickets"] + matches["pace_wickets"]
    matches["spin_wicket_share"] = np.divide(
        matches["spin_wickets"],
        total_wickets,
        out=np.full(len(matches), np.nan, dtype=float),
        where=(total_wickets.to_numpy() > 0) & matches["role_complete"].to_numpy(),
    )
    history = _add_prior_venue_features(matches)
    features = matches[
        [
            "match_id",
            "date",
            "match_type",
            "venue",
            "venue_id",
            "city",
            "gender",
            "team_type",
            "toss_decision",
            "toss_winner_batted_first",
        ]
    ].copy()
    features["year"] = features["date"].dt.year
    features["month_sin"] = np.sin(2 * np.pi * features["date"].dt.month / 12)
    features["month_cos"] = np.cos(2 * np.pi * features["date"].dt.month / 12)
    features["day_of_year_sin"] = np.sin(
        2 * np.pi * features["date"].dt.dayofyear / 365.25
    )
    features["day_of_year_cos"] = np.cos(
        2 * np.pi * features["date"].dt.dayofyear / 365.25
    )
    features = features.join(history)
    weather = pd.read_csv(weather_path)
    if "match_id" not in weather.columns:
        raise ValueError(f"{weather_path} must contain match_id.")
    weather["match_id"] = weather["match_id"].astype(str)
    if weather["match_id"].duplicated().any():
        raise ValueError(f"{weather_path} contains duplicate match_id values.")
    weather_columns = [
        column
        for column in (
            "temperature_2m_mean",
            "precipitation_sum",
            "wind_speed_10m_max",
            "relative_humidity_2m_mean",
        )
        if column in weather.columns
    ]
    if not weather_columns:
        raise ValueError(f"{weather_path} has no supported numeric weather features.")
    features["match_id"] = features["match_id"].astype(str)
    return features.merge(
        weather[["match_id", *weather_columns]],
        on="match_id",
        how="left",
        validate="one_to_one",
    )


def _chronological_split(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    dates = np.array(sorted(data["date"].dt.normalize().unique()))
    if len(dates) < 7:
        raise ValueError("At least seven unique match dates are required for a time split.")
    train_cutoff = pd.Timestamp(dates[int((len(dates) - 1) * 0.8)])
    normalized_dates = data["date"].dt.normalize()
    train = data[normalized_dates <= train_cutoff]
    test = data[normalized_dates > train_cutoff]
    return train, test, {
        "strategy": "chronological_by_unique_date",
        "train_fraction_target": 0.8,
        "train_end": train_cutoff.strftime("%Y-%m-%d"),
        "test_start": (
            test["date"].min().strftime("%Y-%m-%d") if not test.empty else ""
        ),
    }


def _make_pipeline(
    numeric_columns: list[str], categorical_columns: list[str], device: str = "cpu"
) -> Pipeline:
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True))
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="Unknown"),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=2,
                                sparse_output=True,
                            ),
                        ),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(CLASS_ORDER),
        n_estimators=350,
        max_depth=4,
        learning_rate=0.04,
        min_child_weight=4,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        eval_metric="mlogloss",
        tree_method="hist",
        device=device,
        n_jobs=4,
        random_state=42,
    )
    return Pipeline([("preprocessing", preprocessing), ("model", model)])


def _probability_matrix(
    estimator: Any, X: pd.DataFrame
) -> np.ndarray:
    raw = estimator.predict_proba(X)
    estimator_classes = list(estimator.classes_)
    if estimator_classes and all(
        isinstance(label, (int, np.integer)) for label in estimator_classes
    ):
        class_to_index = {
            CLASS_ORDER[int(label)]: index
            for index, label in enumerate(estimator_classes)
            if 0 <= int(label) < len(CLASS_ORDER)
        }
    else:
        class_to_index = {
            str(label): index for index, label in enumerate(estimator_classes)
        }
    probabilities = np.zeros((len(X), len(CLASS_ORDER)), dtype=float)
    for target_index, label in enumerate(CLASS_ORDER):
        if label in class_to_index:
            probabilities[:, target_index] = raw[:, class_to_index[label]]
    row_sums = probabilities.sum(axis=1)
    nonzero = row_sums > 0
    probabilities[nonzero] /= row_sums[nonzero, None]
    probabilities[~nonzero] = 1.0 / len(CLASS_ORDER)
    return probabilities


def _score_predictions(
    truth: pd.Series, predicted: np.ndarray, probabilities: np.ndarray
) -> dict[str, Any]:
    one_hot = np.column_stack([(truth.to_numpy() == label) for label in CLASS_ORDER])
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=CLASS_ORDER, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, predicted, labels=CLASS_ORDER, average="weighted", zero_division=0)),
        "multiclass_log_loss": float(log_loss(truth, probabilities, labels=CLASS_ORDER)),
        "multiclass_brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "classification_report": classification_report(
            truth,
            predicted,
            labels=CLASS_ORDER,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            truth, predicted, labels=CLASS_ORDER
        ).tolist(),
    }


def train_one_format(
    match_type: str,
    labeled: pd.DataFrame,
    features: pd.DataFrame,
    output_dir: Path,
    device: str = "cpu",
) -> dict[str, Any]:
    merged = labeled.merge(features, on=["match_id", "date", "match_type"], validate="one_to_one")
    merged = merged.sort_values(["date", "match_id"], kind="stable")
    train, test, split = _chronological_split(merged)
    if train.empty or test.empty:
        raise ValueError(f"{match_type}: chronological split produced an empty partition.")

    excluded = {"match_id", "date", "match_type", "y"}
    feature_columns = [column for column in features.columns if column not in excluded]
    categorical = [
        column
        for column in (
            "venue",
            "venue_id",
            "city",
            "gender",
            "team_type",
            "toss_decision",
        )
        if column in feature_columns
    ]
    numeric = [column for column in feature_columns if column not in categorical]

    train_labels = train["y"].astype(str)
    test_labels = test["y"].astype(str)
    missing_train_classes = set(CLASS_ORDER).difference(train_labels.unique())
    if missing_train_classes:
        raise ValueError(
            f"{match_type}: chronological training period is missing proxy class(es) "
            f"{sorted(missing_train_classes)}; do not fit a three-class model on this split."
        )
    pipeline = _make_pipeline(numeric, categorical, device)
    sample_weights = compute_sample_weight("balanced", train_labels)
    train_codes = train_labels.map(
        {label: index for index, label in enumerate(CLASS_ORDER)}
    ).to_numpy(dtype=np.int64)
    pipeline.fit(
        train[feature_columns],
        train_codes,
        model__sample_weight=sample_weights,
    )

    model_dir = output_dir / match_type.lower()
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_dir / "model.joblib")

    model_predicted_codes = pipeline.predict(test[feature_columns]).astype(int)
    model_predicted = np.asarray(
        [CLASS_ORDER[index] for index in model_predicted_codes], dtype=object
    )
    model_probabilities = _probability_matrix(pipeline, test[feature_columns])
    model_metrics = _score_predictions(test_labels, model_predicted, model_probabilities)

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(train[feature_columns], train_labels)
    baseline_predicted = baseline.predict(test[feature_columns])
    baseline_probabilities = _probability_matrix(baseline, test[feature_columns])
    baseline_metrics = _score_predictions(
        test_labels, baseline_predicted, baseline_probabilities
    )

    transformed_names = pipeline.named_steps["preprocessing"].get_feature_names_out()
    importance = pipeline.named_steps["model"].feature_importances_
    pd.DataFrame(
        {"feature": transformed_names, "importance": importance}
    ).sort_values("importance", ascending=False).to_csv(
        model_dir / "feature_importance.csv", index=False
    )

    prediction_frame = pd.DataFrame(
        {
            "match_id": test["match_id"].to_numpy(),
            "date": test["date"].dt.strftime("%Y-%m-%d").to_numpy(),
            "y_true": test_labels.to_numpy(),
            "y_pred": model_predicted,
        }
    )
    for index, label in enumerate(CLASS_ORDER):
        prediction_frame[f"probability_{label.lower()}"] = model_probabilities[:, index]
    prediction_frame.to_csv(model_dir / "test_predictions.csv", index=False)
    results = {
        "match_type": match_type,
        "training_device": device,
        "target_semantics": "deterministic behavioral proxy; not physical pitch ground truth",
        "label_rules": LABEL_RULES[match_type],
        "class_order": list(CLASS_ORDER),
        "sample_counts": {
            "eligible_total": int(len(merged)),
            "train": int(len(train)),
            "test": int(len(test)),
            "train_classes": train_labels.value_counts().to_dict(),
            "test_classes": test_labels.value_counts().to_dict(),
        },
        "split": split,
        "model": model_metrics,
        "majority_baseline": baseline_metrics,
        "interpretation": (
            "Test metrics are chronological out-of-time agreement with proxy labels. "
            "Do not interpret them as accuracy against measured pitch physics."
        ),
    }
    with (model_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    return results


def run_training(
    archive_path: Path,
    people_path: Path,
    names_path: Path,
    international_zip: Path,
    worldwide_zip: Path,
    weather_path: Path,
    output_dir: Path,
    device: str = "cpu",
) -> dict[str, Any]:
    roles, role_audit = build_player_roles(
        people_path, names_path, international_zip, worldwide_zip
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    role_audit.to_csv(output_dir / "player_role_join_audit.csv", index=False)
    all_matches, _ = _parse_matches(archive_path, roles)
    all_matches["spin_wicket_share"] = np.divide(
        all_matches["spin_wickets"],
        all_matches["spin_wickets"] + all_matches["pace_wickets"],
        out=np.full(len(all_matches), np.nan, dtype=float),
        where=all_matches["role_complete"].to_numpy()
        & ((all_matches["spin_wickets"] + all_matches["pace_wickets"]).to_numpy() > 0),
    )
    features = build_features(all_matches, weather_path)
    all_matches[
        [
            "match_id",
            "date",
            "match_type",
            "venue",
            "city",
            "first_innings_run_rate",
            "early_dot_percent",
            "spin_wickets",
            "pace_wickets",
            "role_complete",
            "y",
        ]
    ].to_csv(output_dir / "proxy_label_audit.csv", index=False)
    summary: dict[str, Any] = {
        "training_device": device,
        "split_strategy": "chronological_by_unique_date_80_20",
        "source_match_counts": {
            str(key): int(value)
            for key, value in all_matches["match_type"].value_counts().items()
        },
        "role_mapping_count": len(roles),
        "role_mapping_sources": {
            str(key): int(value)
            for key, value in role_audit["source"].value_counts().items()
        },
        "format_coverage": {},
        "models": {},
    }
    weather_columns = [
        column
        for column in (
            "temperature_2m_mean",
            "precipitation_sum",
            "wind_speed_10m_max",
            "relative_humidity_2m_mean",
        )
        if column in features.columns
    ]
    for match_type in FORMATS:
        format_matches = all_matches.loc[all_matches["match_type"].eq(match_type)]
        eligible = all_matches.loc[
            all_matches["match_type"].eq(match_type) & all_matches["y"].notna()
        ].copy()
        if eligible.empty:
            raise ValueError(f"No label-complete matches available for {match_type}.")
        format_features = features.loc[features["match_type"].eq(match_type)]
        weather_present = (
            format_features[weather_columns].notna().any(axis=1)
            if weather_columns
            else pd.Series(False, index=format_features.index)
        )
        summary["format_coverage"][match_type] = {
            "parsed_matches_with_complete_early_window": int(len(format_matches)),
            "role_complete_matches": int(format_matches["role_complete"].sum()),
            "proxy_labeled_matches": int(len(eligible)),
            "proxy_class_counts": {
                str(key): int(value)
                for key, value in eligible["y"].value_counts().items()
            },
            "weather_coverage_fraction": float(weather_present.mean())
            if len(weather_present)
            else 0.0,
            "weather_columns": weather_columns,
        }
        summary["models"][match_type] = train_one_format(
            match_type, eligible, features, output_dir, device
        )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=Path, default=Path("data/raw/cricsheet/all_json.zip"))
    parser.add_argument("--people", type=Path, default=Path("data/raw/cricsheet/people.csv"))
    parser.add_argument("--names", type=Path, default=Path("data/raw/cricsheet/names.csv"))
    parser.add_argument(
        "--international-players",
        type=Path,
        default=Path("data/raw/kaggle/international_player_stats.zip"),
    )
    parser.add_argument(
        "--worldwide-players",
        type=Path,
        default=Path("data/raw/kaggle/worldwide_player_profiles.zip"),
    )
    parser.add_argument("--weather", type=Path, default=Path("data/interim/match_weather.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/pitch_proxy"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = run_training(
        args.matches,
        args.people,
        args.names,
        args.international_players,
        args.worldwide_players,
        args.weather,
        args.output_dir,
        args.device,
    )
    for match_type, metrics in summary["models"].items():
        LOGGER.info(
            "%s: test n=%d, macro-F1=%.3f, balanced accuracy=%.3f, majority macro-F1=%.3f",
            match_type,
            metrics["sample_counts"]["test"],
            metrics["model"]["macro_f1"],
            metrics["model"]["balanced_accuracy"],
            metrics["majority_baseline"]["macro_f1"],
        )


if __name__ == "__main__":
    main()
