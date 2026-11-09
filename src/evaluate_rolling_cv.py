"""Compare pitch-proxy model variants with rolling-origin cross-validation.

A single 20% hold-out has a 95% confidence interval roughly +/-0.04 macro-F1, so
small gains from a single split cannot be trusted. This script evaluates the
same variants on expanding-window folds: each fold trains on every match before
a cutoff and tests on the next contiguous block of dates. That mirrors how the
model would be used (predict the future from the past) and reuses every match
for testing exactly once.

Variants compared (same folds for every variant):

* ``xgb`` / ``rf`` -- gradient boosting versus random forest
* weather feature set ``day`` (match-day only) versus ``ctx`` (match-day plus
  prior 7/14/28-day moisture windows)

Outputs:

* ``models/rolling_cv/cv_results.json`` -- per-fold and pooled metrics
* ``models/rolling_cv/cv_summary.csv`` -- one row per variant and format
* ``models/rolling_cv/feature_frames/{format}.csv.gz`` -- cached feature frames

Usage::

    python src\\evaluate_rolling_cv.py
    python src\\evaluate_rolling_cv.py --variants xgb:ctx xgb:day
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_pitch_models as T  # noqa: E402

DAY_WEATHER = (
    "temperature_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "relative_humidity_2m_mean",
    "precipitation_hours",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
)

CONTEXT_WEATHER = DAY_WEATHER + (
    "precip_sum_7d",
    "precip_sum_14d",
    "precip_sum_28d",
    "temp_mean_7d",
    "temp_mean_14d",
    "humidity_mean_7d",
    "wind_max_mean_7d",
    "dry_days_before",
)


def load_datasets(
    archive_path: Path,
    people_path: Path,
    names_path: Path,
    international_zip: Path,
    worldwide_zip: Path,
    weather_path: Path,
    overrides_path: Path | None,
    cache_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Build one labelled feature frame per format, optionally from cache."""
    datasets: dict[str, pd.DataFrame] = {}
    for match_type in T.FORMATS:
        cache_path = cache_dir / f"{match_type.lower()}.csv.gz"
        if cache_path.exists():
            frame = pd.read_csv(cache_path, parse_dates=["date"])
            datasets[match_type] = frame
            print(f"{match_type}: loaded {len(frame)} rows from {cache_path}")
            continue

        if "all_matches" not in datasets:  # type: ignore[comparison-overlap]
            roles, _ = T.build_player_roles(
                people_path, names_path, international_zip, worldwide_zip, overrides_path
            )
            all_matches, _ = T._parse_matches(archive_path, roles)
            all_matches["spin_wicket_share"] = np.divide(
                all_matches["spin_wickets"],
                all_matches["spin_wickets"] + all_matches["pace_wickets"],
                out=np.full(len(all_matches), np.nan, dtype=float),
                where=all_matches["role_complete"].to_numpy()
                & (
                    (all_matches["spin_wickets"] + all_matches["pace_wickets"]).to_numpy()
                    > 0
                ),
            )
            features = T.build_features(all_matches, weather_path)
            datasets["all_matches"] = all_matches  # type: ignore[assignment]
            datasets["features"] = features  # type: ignore[assignment]

        all_matches = datasets["all_matches"]  # type: ignore[assignment]
        features = datasets["features"]  # type: ignore[assignment]
        labeled = all_matches[
            all_matches["y"].notna() & all_matches["match_type"].eq(match_type)
        ][["match_id", "date", "match_type", "y"]]
        frame = labeled.merge(
            features, on=["match_id", "date", "match_type"], validate="one_to_one"
        ).sort_values(["date", "match_id"], kind="stable")
        cache_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cache_path, index=False)
        datasets[match_type] = frame
        print(f"{match_type}: built {len(frame)} rows and cached to {cache_path}")

    datasets.pop("all_matches", None)
    datasets.pop("features", None)
    return datasets


def make_folds(
    frame: pd.DataFrame, n_folds: int, min_train_fraction: float
) -> list[tuple[int, int]]:
    """Return (train_end_index, test_end_index) over sorted unique dates."""
    dates = np.array(sorted(frame["date"].dt.normalize().unique()))
    start = int(len(dates) * min_train_fraction)
    block = max(1, (len(dates) - start) // n_folds)
    folds = []
    for index in range(n_folds):
        test_end = start + block * (index + 1)
        if index == n_folds - 1:
            test_end = len(dates)
        if test_end <= start + block * index:
            break
        folds.append((start + block * index, test_end))
    return folds


def ece(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    predicted = np.asarray(T.CLASS_ORDER)[probabilities.argmax(axis=1)]
    correct = predicted == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(total)


def _fit_binary(
    train: pd.DataFrame,
    feature_columns: list[str],
    numeric: list[str],
    categorical: list[str],
    target: pd.Series,
    base_kind: str,
    device: str,
) -> tuple[Any, dict[str, int]] | None:
    """Fit a binary pipeline and return it with its class-to-code map."""
    labels = target.astype(str)
    if labels.nunique() < 2:
        return None
    classes = sorted(labels.unique())
    codes = labels.map({label: index for index, label in enumerate(classes)})
    pipeline = T._make_pipeline(numeric, categorical, device, base_kind)
    weights = compute_sample_weight("balanced", labels)
    pipeline.fit(
        train[feature_columns], codes.to_numpy(dtype=np.int64),
        model__sample_weight=weights,
    )
    return pipeline, {label: index for index, label in enumerate(classes)}


def _predict_two_stage(
    stage_one: tuple[Any, dict[str, int]],
    stage_two: tuple[Any, dict[str, int]],
    test: pd.DataFrame,
    feature_columns: list[str],
) -> np.ndarray:
    """Combine P(Batting) with P(Spin | not Batting) into three-class probabilities."""
    first, first_index = stage_one
    second, second_index = stage_two
    paddings = first.predict_proba(test[feature_columns])
    p_batting = paddings[:, first_index["Batting"]]
    rest = second.predict_proba(test[feature_columns])
    p_spin_given_rest = rest[:, second_index["Spin"]]
    p_pace = (1.0 - p_batting) * (1.0 - p_spin_given_rest)
    p_spin = (1.0 - p_batting) * p_spin_given_rest
    return np.column_stack([p_batting, p_pace, p_spin])


def evaluate_variant(
    frame: pd.DataFrame,
    weather_mode: str,
    model_kind: str,
    folds: list[tuple[int, int]],
    device: str,
) -> dict[str, Any]:
    dates = np.array(sorted(frame["date"].dt.normalize().unique()))
    excluded = {"match_id", "date", "match_type", "y"}
    feature_columns = [column for column in frame.columns if column not in excluded]
    if weather_mode == "day":
        allowed = set(DAY_WEATHER)
        feature_columns = [
            column
            for column in feature_columns
            if column not in set(CONTEXT_WEATHER) - allowed
        ]
    elif weather_mode != "ctx":
        raise ValueError(f"unknown weather mode {weather_mode!r}")
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

    fold_results = []
    pooled_true: list[np.ndarray] = []
    pooled_pred: list[np.ndarray] = []
    pooled_probabilities: list[np.ndarray] = []
    for fold_index, (train_end, test_end) in enumerate(folds, start=1):
        train_dates = dates[:train_end]
        test_dates = dates[train_end:test_end]
        normalized = frame["date"].dt.normalize()
        train = frame[normalized.isin(train_dates)]
        test = frame[normalized.isin(test_dates)]
        if train.empty or test.empty:
            continue
        train_labels = train["y"].astype(str)
        test_labels = test["y"].astype(str)
        if not set(T.CLASS_ORDER).issubset(train_labels.unique()):
            print(f"    fold {fold_index}: skipped, missing a class in train")
            continue

        if model_kind == "avg":
            codes = train_labels.map(
                {label: index for index, label in enumerate(T.CLASS_ORDER)}
            ).to_numpy(dtype=np.int64)
            member_probabilities = []
            for member in ("xgb", "rf"):
                member_pipeline = T._make_pipeline(
                    numeric, categorical, device, member
                )
                member_weights = compute_sample_weight("balanced", train_labels)
                member_pipeline.fit(
                    train[feature_columns], codes, model__sample_weight=member_weights
                )
                member_probabilities.append(
                    T._probability_matrix(member_pipeline, test[feature_columns])
                )
            probabilities = np.mean(member_probabilities, axis=0)
        elif model_kind == "rf2":
            base_kind = "rf"
            stage_one = _fit_binary(
                train,
                feature_columns,
                numeric,
                categorical,
                (train["y"].astype(str) == "Batting").map(
                    {True: "Batting", False: "rest"}
                ),
                base_kind,
                device,
            )
            rest_mask = train["y"].astype(str) != "Batting"
            stage_two = _fit_binary(
                train.loc[rest_mask],
                feature_columns,
                numeric,
                categorical,
                train.loc[rest_mask, "y"].astype(str),
                base_kind,
                device,
            )
            if stage_one is None or stage_two is None:
                print(f"    fold {fold_index}: skipped, two-stage class missing")
                continue
            probabilities = _predict_two_stage(
                stage_one, stage_two, test, feature_columns
            )
        else:
            pipeline = T._make_pipeline(numeric, categorical, device, model_kind)
            weights = compute_sample_weight("balanced", train_labels)
            codes = train_labels.map(
                {label: index for index, label in enumerate(T.CLASS_ORDER)}
            ).to_numpy(dtype=np.int64)
            pipeline.fit(
                train[feature_columns], codes, model__sample_weight=weights
            )
            probabilities = T._probability_matrix(pipeline, test[feature_columns])
        predicted = np.asarray(T.CLASS_ORDER, dtype=object)[probabilities.argmax(axis=1)]
        truth = test_labels.to_numpy()

        fold_results.append(
            {
                "fold": fold_index,
                "train_n": int(len(train)),
                "test_n": int(len(test)),
                "test_start": str(test["date"].min().date()),
                "test_end": str(test["date"].max().date()),
                "macro_f1": float(f1_score(truth, predicted, average="macro")),
                "accuracy": float((truth == predicted).mean()),
            }
        )
        pooled_true.append(truth)
        pooled_pred.append(predicted)
        pooled_probabilities.append(probabilities)
        print(
            f"    fold {fold_index}: train {len(train)}, test {len(test)}, "
            f"macro-F1 {fold_results[-1]['macro_f1']:.3f}"
        )

    if not fold_results:
        return {"error": "no usable folds"}
    truth = np.concatenate(pooled_true)
    predicted = np.concatenate(pooled_pred)
    probabilities = np.vstack(pooled_probabilities)
    return {
        "weather_mode": weather_mode,
        "model_family": model_kind,
        "n_raw_features": len(feature_columns),
        "folds": fold_results,
        "mean_fold_macro_f1": float(np.mean([f["macro_f1"] for f in fold_results])),
        "std_fold_macro_f1": float(np.std([f["macro_f1"] for f in fold_results])),
        "pooled_macro_f1": float(f1_score(truth, predicted, average="macro")),
        "pooled_accuracy": float((truth == predicted).mean()),
        "pooled_ece": ece(truth, probabilities),
        "pooled_n": int(len(truth)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--matches", type=Path, default=root / "data/raw/cricsheet/all_json.zip")
    parser.add_argument("--people", type=Path, default=root / "data/raw/cricsheet/people.csv")
    parser.add_argument("--names", type=Path, default=root / "data/raw/cricsheet/names.csv")
    parser.add_argument(
        "--international-players",
        type=Path,
        default=root / "data/raw/kaggle/international_player_stats.zip",
    )
    parser.add_argument(
        "--worldwide-players",
        type=Path,
        default=root / "data/raw/kaggle/worldwide_player_profiles.zip",
    )
    parser.add_argument("--weather", type=Path, default=root / "data/interim/match_weather.csv")
    parser.add_argument(
        "--role-overrides",
        type=Path,
        default=root / "data/interim/bowler_role_overrides.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "models/rolling_cv")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--min-train-fraction", type=float, default=0.5)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["xgb:ctx", "xgb:day", "rf:ctx", "rf:day", "avg:ctx", "avg:day"],
        help="Variant names of the form model:weather_mode.",
    )
    parser.add_argument("--formats", nargs="+", default=list(T.FORMATS))
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Ignore cached feature frames and rebuild them from the archive.",
    )
    args = parser.parse_args()

    cache_dir = args.output_dir / "feature_frames"
    if args.rebuild:
        for path in cache_dir.glob("*.csv.gz"):
            path.unlink()
    datasets = load_datasets(
        args.matches,
        args.people,
        args.names,
        args.international_players,
        args.worldwide_players,
        args.weather,
        args.role_overrides,
        cache_dir,
    )

    results: dict[str, Any] = {
        "folds": args.folds,
        "min_train_fraction": args.min_train_fraction,
        "variants": {},
    }
    rows = []
    for spec in args.variants:
        model_kind, weather_mode = spec.split(":")
        for match_type in args.formats:
            frame = datasets[match_type]
            folds = make_folds(frame, args.folds, args.min_train_fraction)
            print(f"{spec} | {match_type}: {len(frame)} rows, {len(folds)} folds")
            outcome = evaluate_variant(
                frame, weather_mode, model_kind, folds, args.device
            )
            outcome["match_type"] = match_type
            results["variants"].setdefault(spec, {})[match_type] = outcome
            if "error" not in outcome:
                rows.append(
                    {
                        "variant": spec,
                        "match_type": match_type,
                        "folds": len(outcome["folds"]),
                        "pooled_n": outcome["pooled_n"],
                        "mean_fold_macro_f1": round(outcome["mean_fold_macro_f1"], 4),
                        "std_fold_macro_f1": round(outcome["std_fold_macro_f1"], 4),
                        "pooled_macro_f1": round(outcome["pooled_macro_f1"], 4),
                        "pooled_accuracy": round(outcome["pooled_accuracy"], 4),
                        "pooled_ece": round(outcome["pooled_ece"], 4),
                        "n_raw_features": outcome["n_raw_features"],
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "cv_results.json"
    summary_path = args.output_dir / "cv_summary.csv"
    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as file:
            previous = json.load(file)
        merged_variants = previous.get("variants", {})
        for spec, per_format in results["variants"].items():
            merged_variants.setdefault(spec, {}).update(per_format)
        results["variants"] = merged_variants
    with results_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    summary = pd.DataFrame(rows)
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        summary = pd.concat([existing, summary], ignore_index=True)
        summary = summary.drop_duplicates(
            subset=["variant", "match_type"], keep="last"
        )
    summary.to_csv(summary_path, index=False)
    print()
    print(summary.sort_values(["match_type", "pooled_macro_f1"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
