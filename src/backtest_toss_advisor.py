"""Backtest the dashboard toss advisor on previous matches.

Question tested
---------------
If the advisor had been shown only matches strictly before a fixture, would its
bat/field call have picked the decision with the higher toss-winner win share?

Design
------
* Expanding window. For every fixture, the venue and format rates use only
  decided matches with a known toss dated strictly earlier. Nothing from the
  fixture itself or from later matches enters the estimate.
* The shipped rule is imported from ``src/toss_advisor.py``; the backtest does
  not re-implement the shrinkage, the fallback or the call threshold.
* Evaluation starts in 2010 with a warm-up of at least 100 prior decided
  matches in the format/gender/team-type group, so the early thin record is not
  graded.

Three checks are reported
-------------------------
1. **Venue-season sign accuracy.** Predictions are frozen at the first fixture
   of each venue-season; the realized season is the bat-minus-field difference
   in toss-winner win share among that season's decided matches at the venue
   (at least 20 matches behind each decision). This is a genuine out-of-sample
   directional test: the prediction never sees the season it is graded on.
2. **Calibration of the rate forecasts.** For each fixture the forecast for the
   decision actually taken is scored against whether the toss winner won
   (Brier score, Brier skill against the format baseline, calibration slope).
3. **Followed-versus-against.** Among clear calls, the toss-winner win share
   when the side followed the call versus when it went the other way. This is
   confounded by self-selection, so it is reported as a check, not as proof.

A venue-shuffled placebo and one-at-a-time sensitivity to the prior strength,
minimum venue sample, minimum edge and posterior threshold are also run.

Run from anywhere:

    python src/backtest_toss_advisor.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import toss_advisor as TA  # noqa: E402

TOSS_PATH = ROOT / "data/processed/dashboard_toss_matches.csv"
TABLE_DIR = ROOT / "reports/tables"
PROCESSED_DIR = ROOT / "data/processed"

EVAL_START = pd.Timestamp("2010-01-01")
WARMUP_FORMAT_N = 100
BLOCK_SIZE = 30
BLOCK_MIN_N = 5
WINDOW_HORIZON = 20
WINDOW_MIN_N = 5
POSTERIOR_POINTS = 512
BOOTSTRAP_DRAWS = 1000
SEED = 42

GROUP_KEYS = ["match_type", "gender", "team_type"]
VENUE_KEYS = GROUP_KEYS + ["venue_key"]
COUNT_COLUMNS = ["n_bat", "w_bat", "n_field", "w_field"]


# --------------------------------------------------------------------------- #
# Data and prior-only counts
# --------------------------------------------------------------------------- #
def load_matches() -> pd.DataFrame:
    frame = pd.read_csv(
        TOSS_PATH,
        parse_dates=["date"],
        dtype={"match_id": str, "venue_key": str},
    )
    blank_winner = frame["winner"].isna() | frame["winner"].astype(str).str.strip().eq("")
    blank_toss = frame["toss_winner"].isna() | frame["toss_winner"].astype(str).str.strip().eq("")
    frame = frame[
        frame["toss_decision"].isin(["bat", "field"])
        & ~blank_toss
        & ~blank_winner
        & frame["result"].ne("no_result")
    ].copy()
    frame["toss_won"] = frame["winner"].eq(frame["toss_winner"]).astype(int)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values(
        ["match_type", "gender", "team_type", "date", "match_id"]
    )
    return frame.reset_index(drop=True)


def _daily_totals(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    working = frame.copy()
    working["n_bat"] = working["toss_decision"].eq("bat").astype(int)
    working["w_bat"] = working["n_bat"] * working["toss_won"]
    working["n_field"] = working["toss_decision"].eq("field").astype(int)
    working["w_field"] = working["n_field"] * working["toss_won"]
    daily = (
        working.groupby(keys + ["date"], observed=True)[COUNT_COLUMNS]
        .sum()
        .reset_index()
        .sort_values(keys + ["date"])
    )
    # Cumulative totals through each date, then shifted one date so a fixture is
    # never scored on a match from its own day.
    cumulative = daily.groupby(keys, observed=True)[COUNT_COLUMNS].cumsum()
    grouper = [daily[key] for key in keys]
    prior = cumulative.groupby(grouper, sort=False).shift(1).fillna(0)
    return pd.concat([daily[keys + ["date"]], prior], axis=1)


def add_prior_counts(frame: pd.DataFrame) -> pd.DataFrame:
    format_prior = _daily_totals(frame, GROUP_KEYS).rename(
        columns={column: f"f_{column}" for column in COUNT_COLUMNS}
    )
    venue_prior = _daily_totals(frame, VENUE_KEYS).rename(
        columns={column: f"v_{column}" for column in COUNT_COLUMNS}
    )
    out = frame.merge(format_prior, on=GROUP_KEYS + ["date"], how="left")
    out = out.merge(venue_prior, on=VENUE_KEYS + ["date"], how="left")
    count_columns = [f"{p}_{c}" for p in ("f", "v") for c in COUNT_COLUMNS]
    out[count_columns] = out[count_columns].fillna(0)
    return out


# --------------------------------------------------------------------------- #
# Shipped estimates, vectorised
# --------------------------------------------------------------------------- #
def add_estimates(
    frame: pd.DataFrame,
    prior: float = TA.SHRINK_PRIOR,
    min_venue_n: int = TA.MIN_VENUE_N,
    min_edge_points: float = TA.MIN_EDGE_POINTS,
    min_posterior: float = TA.MIN_POSTERIOR,
    posterior_points: int = POSTERIOR_POINTS,
) -> pd.DataFrame:
    frame = frame.copy()
    nf_bat = frame["f_n_bat"].to_numpy(float)
    nf_field = frame["f_n_field"].to_numpy(float)
    base_bat = np.divide(
        frame["f_w_bat"].to_numpy(float),
        nf_bat,
        out=np.full(len(frame), 0.5),
        where=nf_bat > 0,
    )
    base_field = np.divide(
        frame["f_w_field"].to_numpy(float),
        nf_field,
        out=np.full(len(frame), 0.5),
        where=nf_field > 0,
    )
    est_bat = TA.shrunk(frame["v_w_bat"].to_numpy(float), frame["v_n_bat"].to_numpy(float), base_bat, prior)
    est_field = TA.shrunk(frame["v_w_field"].to_numpy(float), frame["v_n_field"].to_numpy(float), base_field, prior)
    est_bat = np.asarray(est_bat, dtype=float)
    est_field = np.asarray(est_field, dtype=float)

    frame["base_bat"] = base_bat
    frame["base_field"] = base_field
    frame["est_bat"] = est_bat
    frame["est_field"] = est_field
    frame["edge_points"] = (est_bat - est_field) * 100.0

    a_bat = frame["v_w_bat"].to_numpy(float) + prior * base_bat
    b_bat = frame["v_n_bat"].to_numpy(float) - frame["v_w_bat"].to_numpy(float) + prior * (1.0 - base_bat)
    a_field = frame["v_w_field"].to_numpy(float) + prior * base_field
    b_field = frame["v_n_field"].to_numpy(float) - frame["v_w_field"].to_numpy(float) + prior * (1.0 - base_field)
    a_bat = np.maximum(a_bat, 1e-9)
    b_bat = np.maximum(b_bat, 1e-9)
    a_field = np.maximum(a_field, 1e-9)
    b_field = np.maximum(b_field, 1e-9)

    frame["venue_enough"] = (
        frame["v_n_bat"].ge(min_venue_n) & frame["v_n_field"].ge(min_venue_n)
    )
    frame["format_enough"] = (
        frame["f_n_bat"].ge(WARMUP_FORMAT_N) & frame["f_n_field"].ge(WARMUP_FORMAT_N)
    )

    prob = np.full(len(frame), 0.5)
    eligible = np.flatnonzero(
        frame["venue_enough"].to_numpy()
        & frame["format_enough"].to_numpy()
        & frame["edge_points"].abs().ge(min_edge_points).to_numpy()
    )
    if len(eligible):
        prob[eligible] = posterior_bat_better(
            a_bat[eligible],
            b_bat[eligible],
            a_field[eligible],
            b_field[eligible],
            points=posterior_points,
        )
    frame["prob_bat"] = prob
    frame["prob_call"] = np.where(
        frame["edge_points"].gt(0), frame["prob_bat"], 1.0 - frame["prob_bat"]
    )

    clear = (
        frame["venue_enough"]
        & frame["format_enough"]
        & frame["edge_points"].abs().ge(min_edge_points)
        & frame["prob_call"].ge(min_posterior)
    )
    frame["call"] = np.where(
        clear, np.where(frame["edge_points"].gt(0), "bat", "field"), "none"
    )
    frame["source"] = np.where(frame["venue_enough"], "venue", "format")
    return frame


def posterior_bat_better(
    a1: np.ndarray,
    b1: np.ndarray,
    a2: np.ndarray,
    b2: np.ndarray,
    points: int = POSTERIOR_POINTS,
) -> np.ndarray:
    """P(X > Y) for independent Beta(a1, b1) and Beta(a2, b2).

    Uses E[F_Y(X)] with X drawn on a fixed quantile grid of its own Beta, which
    is a midpoint rule in probability space. The 512-point grid is accurate to
    about 0.002 and is vectorised, so the sensitivity sweep is tractable. The
    dashboard keeps its seeded Monte Carlo version.
    """
    a1 = np.atleast_1d(np.asarray(a1, dtype=float))[:, None]
    b1 = np.atleast_1d(np.asarray(b1, dtype=float))[:, None]
    a2 = np.atleast_1d(np.asarray(a2, dtype=float))[:, None]
    b2 = np.atleast_1d(np.asarray(b2, dtype=float))[:, None]
    u = (np.arange(points, dtype=float) + 0.5) / points
    x = beta_dist.ppf(u[None, :], a1, b1)
    return beta_dist.cdf(x, a2, b2).mean(axis=1)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def block_table(
    predictions: pd.DataFrame,
    block_size: int = BLOCK_SIZE,
    min_n: int = BLOCK_MIN_N,
) -> pd.DataFrame:
    """One row per consecutive within-venue block of fixtures.

    The prediction is the one issued for the first fixture of the block, using
    only matches before that fixture. The realized outcome is the bat-minus-field
    difference in toss-winner win share over the block itself, so no fixture in
    the block contributes to its own forecast.
    """
    graded = predictions[predictions["format_enough"]].copy()
    graded = graded[graded["date"].ge(EVAL_START)]
    rows = []
    columns = VENUE_KEYS + [
        "venue", "block_start", "block_end", "n_matches", "n_bat", "n_field",
        "predicted_edge_points", "predicted_call", "predicted_prob",
        "realized_bat_share", "realized_field_share", "realized_edge_points",
    ]
    for key, group in graded.groupby(VENUE_KEYS, observed=True):
        group = group.sort_values(["date", "match_id"]).reset_index(drop=True)
        for start in range(0, len(group) - block_size + 1, block_size):
            block = group.iloc[start : start + block_size]
            first = block.iloc[0]
            if not bool(first["venue_enough"]):
                continue  # the advisor had no venue record to speak from
            n_bat = int(block["toss_decision"].eq("bat").sum())
            n_field = int(block["toss_decision"].eq("field").sum())
            if n_bat < min_n or n_field < min_n:
                continue
            bat_won = float(
                block.loc[block["toss_decision"].eq("bat"), "toss_won"].mean()
            )
            field_won = float(
                block.loc[block["toss_decision"].eq("field"), "toss_won"].mean()
            )
            rows.append(
                {
                    "match_type": key[0],
                    "gender": key[1],
                    "team_type": key[2],
                    "venue_key": key[3],
                    "venue": group["venue"].iloc[0],
                    "block_start": first["date"],
                    "block_end": block["date"].max(),
                    "n_matches": len(block),
                    "n_bat": n_bat,
                    "n_field": n_field,
                    "predicted_edge_points": float(first["edge_points"]),
                    "predicted_call": first["call"],
                    "predicted_prob": float(first["prob_call"]),
                    "realized_bat_share": bat_won,
                    "realized_field_share": field_won,
                    "realized_edge_points": (bat_won - field_won) * 100.0,
                }
            )
    blocks = pd.DataFrame(rows, columns=columns)
    if blocks.empty:
        return blocks
    blocks["agrees"] = (
        np.sign(blocks["predicted_edge_points"]) == np.sign(blocks["realized_edge_points"])
    ).astype(int)
    blocks["realized_abs_edge"] = blocks["realized_edge_points"].abs()
    return blocks


def call_windows(
    predictions: pd.DataFrame,
    horizon: int = WINDOW_HORIZON,
    min_n: int = WINDOW_MIN_N,
    non_overlapping: bool = False,
) -> pd.DataFrame:
    """Grade each clear call against the venue's next matches.

    The window starts after the call fixture, so the outcome is strictly future
    information. With ``non_overlapping`` the scan accepts a call only when its
    window does not overlap the previous accepted window at the same venue,
    which removes most of the dependence between graded calls.
    """
    graded = predictions[predictions["format_enough"]].copy()
    graded = graded[graded["date"].ge(EVAL_START)]
    rows = []
    for key, group in graded.groupby(VENUE_KEYS, observed=True):
        group = group.sort_values(["date", "match_id"]).reset_index(drop=True)
        last_end = -1
        for position in np.flatnonzero(group["call"].ne("none").to_numpy()):
            if non_overlapping and position <= last_end:
                continue
            call = group.iloc[position]
            future = group.iloc[position + 1 : position + 1 + horizon]
            future = future[future["date"].gt(call["date"])]
            if future.empty:
                continue
            n_bat = int(future["toss_decision"].eq("bat").sum())
            n_field = int(future["toss_decision"].eq("field").sum())
            if n_bat < min_n or n_field < min_n:
                continue
            bat_won = float(
                future.loc[future["toss_decision"].eq("bat"), "toss_won"].mean()
            )
            field_won = float(
                future.loc[future["toss_decision"].eq("field"), "toss_won"].mean()
            )
            rows.append(
                {
                    "match_type": key[0],
                    "gender": key[1],
                    "team_type": key[2],
                    "venue_key": key[3],
                    "venue": call["venue"],
                    "call_date": call["date"],
                    "call": call["call"],
                    "predicted_edge_points": float(call["edge_points"]),
                    "predicted_prob": float(call["prob_call"]),
                    "baseline_edge_points": float(call["base_bat"] - call["base_field"]) * 100.0,
                    "n_matches": len(future),
                    "n_bat": n_bat,
                    "n_field": n_field,
                    "realized_bat_share": bat_won,
                    "realized_field_share": field_won,
                    "realized_edge_points": (bat_won - field_won) * 100.0,
                }
            )
            last_end = position + horizon
    windows = pd.DataFrame(rows)
    if windows.empty:
        return windows
    windows["predicted_call"] = windows["call"]
    windows["agrees"] = (
        np.sign(windows["predicted_edge_points"])
        == np.sign(windows["realized_edge_points"])
    ).astype(int)
    windows["baseline_agrees"] = (
        np.sign(windows["baseline_edge_points"])
        == np.sign(windows["realized_edge_points"])
    ).astype(int)
    windows["realized_abs_edge"] = windows["realized_edge_points"].abs()
    return windows


def sign_accuracy(blocks: pd.DataFrame, clear_only: bool) -> dict:
    subset = blocks[blocks["predicted_call"].ne("none")] if clear_only else blocks
    n = len(subset)
    if n == 0:
        return {
            "blocks": 0,
            "hits": 0,
            "accuracy": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "match_weighted_accuracy": float("nan"),
            "venue_balanced_accuracy": float("nan"),
            "venues": 0,
            "max_windows_per_venue": 0,
            "mean_realized_abs_edge": float("nan"),
        }
    hits = int(subset["agrees"].sum())
    weights = subset["n_matches"].to_numpy(float)
    weighted = float(np.average(subset["agrees"].to_numpy(float), weights=weights))
    by_venue = subset.groupby("venue_key").agg(
        predicted=("predicted_edge_points", "mean"),
        realized=("realized_edge_points", "mean"),
        windows=("agrees", "size"),
    )
    venue_balanced = float(
        (np.sign(by_venue["predicted"]) == np.sign(by_venue["realized"])).mean()
    )

    # The same venue contributes many blocks, so resample venues, not blocks.
    rng = np.random.default_rng(SEED)
    venues = subset["venue_key"].to_numpy()
    unique, inverse = np.unique(venues, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        picked = rng.integers(0, len(unique), size=len(unique))
        index = np.concatenate([members[i] for i in picked])
        draws.append(float(subset["agrees"].to_numpy(float)[index].mean()))
    return {
        "blocks": n,
        "hits": hits,
        "accuracy": hits / n,
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "match_weighted_accuracy": weighted,
        "venue_balanced_accuracy": venue_balanced,
        "venues": int(by_venue.shape[0]),
        "max_windows_per_venue": int(by_venue["windows"].max()),
        "mean_realized_abs_edge": float(subset["realized_abs_edge"].mean()),
    }


def calibration(predictions: pd.DataFrame) -> dict:
    frame = predictions[predictions["format_enough"]].copy()
    frame = frame[frame["toss_decision"].isin(["bat", "field"])]
    p = np.where(
        frame["toss_decision"].eq("bat"), frame["est_bat"], frame["est_field"]
    )
    baseline = np.where(
        frame["toss_decision"].eq("bat"), frame["base_bat"], frame["base_field"]
    )
    y = frame["toss_won"].to_numpy(float)
    p = np.clip(np.asarray(p, dtype=float), 0.01, 0.99)
    baseline = np.clip(np.asarray(baseline, dtype=float), 0.01, 0.99)
    brier = float(np.mean((p - y) ** 2))
    brier_base = float(np.mean((baseline - y) ** 2))
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    model.fit(logit, y)
    slope = float(model.coef_[0, 0])
    intercept = float(model.intercept_[0])

    rng = np.random.default_rng(SEED)
    venues = frame["venue_key"].to_numpy()
    unique, inverse = np.unique(venues, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    skill_draws = []
    slope_draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        picked = rng.integers(0, len(unique), size=len(unique))
        index = np.concatenate([members[i] for i in picked])
        y_boot = y[index]
        p_boot = p[index]
        base_boot = baseline[index]
        b_boot = float(np.mean((p_boot - y_boot) ** 2))
        bb_boot = float(np.mean((base_boot - y_boot) ** 2))
        if bb_boot > 0:
            skill_draws.append(1 - b_boot / bb_boot)
        logit_boot = np.log(p_boot / (1 - p_boot)).reshape(-1, 1)
        if len(np.unique(y_boot)) == 2:
            fit = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
            fit.fit(logit_boot, y_boot)
            slope_draws.append(float(fit.coef_[0, 0]))

    def interval(values):
        if not values:
            return float("nan"), float("nan")
        return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))

    skill = 1 - brier / brier_base if brier_base > 0 else float("nan")
    skill_low, skill_high = interval(skill_draws)
    slope_low, slope_high = interval(slope_draws)
    return {
        "matches": len(frame),
        "venues": int(frame["venue_key"].nunique()),
        "brier": brier,
        "brier_format_baseline": brier_base,
        "brier_skill": skill,
        "brier_skill_ci_low": skill_low,
        "brier_skill_ci_high": skill_high,
        "calibration_slope": slope,
        "calibration_slope_ci_low": slope_low,
        "calibration_slope_ci_high": slope_high,
        "calibration_intercept": intercept,
        "observed_win_share": float(y.mean()),
        "mean_forecast": float(p.mean()),
    }


def followed_vs_against(predictions: pd.DataFrame) -> dict:
    frame = predictions[predictions["call"].ne("none")].copy()
    frame["followed"] = frame["call"].eq(frame["toss_decision"])
    followed = frame[frame["followed"]]
    against = frame[~frame["followed"]]
    diff = float(followed["toss_won"].mean() - against["toss_won"].mean())
    f_values = followed["toss_won"].to_numpy(float)
    a_values = against["toss_won"].to_numpy(float)
    rng = np.random.default_rng(SEED)
    f_draws = rng.choice(f_values, size=(BOOTSTRAP_DRAWS, len(f_values)), replace=True).mean(axis=1)
    a_draws = rng.choice(a_values, size=(BOOTSTRAP_DRAWS, len(a_values)), replace=True).mean(axis=1)
    draws = f_draws - a_draws
    return {
        "clear_calls": len(frame),
        "followed_matches": len(followed),
        "followed_win_share": float(followed["toss_won"].mean()),
        "against_matches": len(against),
        "against_win_share": float(against["toss_won"].mean()),
        "difference_points": diff * 100.0,
        "difference_ci_low": float(np.percentile(draws, 2.5)) * 100.0,
        "difference_ci_high": float(np.percentile(draws, 97.5)) * 100.0,
    }


def sensitivity(base: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grid = []
    for prior in (5.0, 10.0, 25.0):
        grid.append({"parameter": "prior pseudo-matches", "value": prior,
                     "prior": prior, "min_venue_n": TA.MIN_VENUE_N,
                     "min_edge_points": TA.MIN_EDGE_POINTS,
                     "min_posterior": TA.MIN_POSTERIOR})
    for n in (10, 25, 50):
        grid.append({"parameter": "minimum venue n", "value": n,
                     "prior": TA.SHRINK_PRIOR, "min_venue_n": n,
                     "min_edge_points": TA.MIN_EDGE_POINTS,
                     "min_posterior": TA.MIN_POSTERIOR})
    for edge in (0.0, 2.0, 5.0):
        grid.append({"parameter": "minimum edge (points)", "value": edge,
                     "prior": TA.SHRINK_PRIOR, "min_venue_n": TA.MIN_VENUE_N,
                     "min_edge_points": edge, "min_posterior": TA.MIN_POSTERIOR})
    for posterior in (0.80, 0.95, 0.99):
        grid.append({"parameter": "minimum posterior", "value": posterior,
                     "prior": TA.SHRINK_PRIOR, "min_venue_n": TA.MIN_VENUE_N,
                     "min_edge_points": TA.MIN_EDGE_POINTS, "min_posterior": posterior})

    evaluable = base[base["format_enough"]].copy()
    for setting in grid:
        predictions = add_estimates(
            evaluable,
            prior=setting["prior"],
            min_venue_n=int(setting["min_venue_n"]),
            min_edge_points=setting["min_edge_points"],
            min_posterior=setting["min_posterior"],
            posterior_points=128,
        )
        windows = call_windows(predictions)
        metrics = sign_accuracy(windows, clear_only=True)
        clear = int(predictions["call"].ne("none").sum())
        rows.append(
            {
                **setting,
                "clear_calls": clear,
                "clear_call_share": clear / len(predictions) if len(predictions) else np.nan,
                "windows": metrics["blocks"],
                "venues": metrics["venues"],
                "sign_accuracy": metrics["accuracy"],
                "accuracy_ci_low": metrics["ci_low"],
                "accuracy_ci_high": metrics["ci_high"],
                "venue_balanced_accuracy": metrics["venue_balanced_accuracy"],
            }
        )
    return pd.DataFrame(rows)


def cluster_accuracy(frame: pd.DataFrame, column: str) -> dict:
    """Accuracy of a 0/1 column with a venue-cluster bootstrap interval."""
    if frame.empty:
        return {"n": 0, "accuracy": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "venues": 0}
    values = frame[column].to_numpy(float)
    venues = frame["venue_key"].to_numpy()
    unique, inverse = np.unique(venues, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        picked = rng.integers(0, len(unique), size=len(unique))
        index = np.concatenate([members[i] for i in picked])
        draws.append(float(values[index].mean()))
    return {
        "n": len(frame),
        "accuracy": float(values.mean()),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "venues": int(len(unique)),
    }


def direction_placebo(windows: pd.DataFrame, seed: int = SEED) -> dict:
    """Random bat/field calls on the same windows; expected accuracy 0.5."""
    rng = np.random.default_rng(seed)
    realized = np.sign(windows["realized_edge_points"].to_numpy(float))
    random_sign = rng.choice([-1.0, 1.0], size=len(windows))
    frame = windows.copy()
    frame["random_agrees"] = (random_sign == realized).astype(int)
    return cluster_accuracy(frame, "random_agrees")


def baseline_benchmark(windows: pd.DataFrame) -> dict:
    """Format baseline on the same windows, without the venue adjustment."""
    return cluster_accuracy(windows, "baseline_agrees")


def examples(predictions: pd.DataFrame, per_type: int = 20) -> pd.DataFrame:
    """One recent clear call per venue, so the table spans venues and outcomes."""
    frame = predictions[predictions["call"].ne("none")].copy()
    frame["followed"] = frame["call"].eq(frame["toss_decision"])
    frame["toss_winner_won"] = frame["toss_won"].eq(1)
    frame["call_outcome"] = np.select(
        [
            frame["followed"] & frame["toss_winner_won"],
            frame["followed"] & ~frame["toss_winner_won"],
        ],
        ["followed, won", "followed, lost"],
        default="not followed",
    )
    frame = frame.sort_values("date", ascending=False)
    frame = frame.groupby(
        ["match_type", "venue_key"], observed=True, group_keys=False
    ).head(1)
    frame = frame.groupby("match_type", observed=True, group_keys=False).head(per_type)
    columns = [
        "date", "match_type", "gender", "team_type", "venue", "team1", "team2",
        "toss_winner", "toss_decision", "call", "edge_points", "prob_call",
        "winner", "toss_winner_won", "call_outcome",
    ]
    return frame[columns].sort_values("date", ascending=False)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    matches = load_matches()
    counts = add_prior_counts(matches)
    predictions = add_estimates(counts)
    evaluable = predictions[
        predictions["format_enough"] & predictions["date"].ge(EVAL_START)
    ].copy()
    blocks = block_table(evaluable)
    windows = call_windows(evaluable)
    windows_independent = call_windows(evaluable, non_overlapping=True)

    summary_rows = [
        {
            "check": "clear-call forward-window sign accuracy (overlapping)",
            **sign_accuracy(windows, clear_only=True),
        },
        {
            "check": "clear-call forward-window sign accuracy (non-overlapping)",
            **sign_accuracy(windows_independent, clear_only=True),
        },
        {
            "check": "within-venue block sign accuracy (all venue predictions)",
            **sign_accuracy(blocks, clear_only=False),
        },
        {"check": "rate calibration", **calibration(evaluable)},
        {"check": "followed versus against", **followed_vs_against(evaluable)},
        {"check": "format baseline on the same windows", **baseline_benchmark(windows)},
        {"check": "random-direction placebo", **direction_placebo(windows)},
    ]

    summary = pd.DataFrame(summary_rows)
    sens = sensitivity(predictions[predictions["date"].ge(EVAL_START)].copy())
    example_table = examples(evaluable)
    per_match = evaluable[
        [
            "match_id", "date", "match_type", "gender", "team_type", "venue",
            "venue_key", "team1", "team2", "toss_winner", "toss_decision", "winner",
            "toss_won", "call", "edge_points", "prob_call", "venue_enough",
            "source",
        ]
    ]

    summary.to_csv(TABLE_DIR / "toss_backtest_summary.csv", index=False)
    blocks.to_csv(TABLE_DIR / "toss_backtest_venue_blocks.csv", index=False)
    windows.to_csv(TABLE_DIR / "toss_backtest_call_windows.csv", index=False)
    sens.to_csv(TABLE_DIR / "toss_backtest_sensitivity.csv", index=False)
    example_table.to_csv(TABLE_DIR / "toss_backtest_examples.csv", index=False)
    per_match.to_csv(PROCESSED_DIR / "toss_backtest_predictions.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)
    print(f"decided matches with a usable toss: {len(matches):,}")
    print(f"evaluable matches (>= {EVAL_START.date()}, format warm-up met): {len(evaluable):,}")
    print(f"venue blocks with >={BLOCK_MIN_N} per decision: {len(blocks):,}")
    print(f"clear-call windows: {len(windows):,} ({len(windows_independent):,} non-overlapping)")
    print()
    print(summary.to_string(index=False))
    print()
    print("sensitivity")
    print(sens[["parameter", "value", "clear_calls", "windows", "venues",
                "sign_accuracy", "accuracy_ci_low", "accuracy_ci_high",
                "venue_balanced_accuracy"]].to_string(index=False))
    print()
    print("recent clear calls")
    print(example_table.head(12).to_string(index=False))
    print()
    print(f"tables -> {TABLE_DIR.relative_to(ROOT)}")
    print(f"per-match -> {(PROCESSED_DIR / 'toss_backtest_predictions.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
