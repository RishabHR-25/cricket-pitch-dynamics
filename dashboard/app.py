"""Pitch Brief -- a pre-match decision aid built from the pitch-dynamics study.

Run from the repository root:

    streamlit run dashboard/app.py

The app reads only committed artifacts. It does not train, refit, or scan the
ball-by-ball parquet on interaction. The match model is a deployment refit of
the paper's headline specification (see ``src/build_dashboard_model.py``); its
performance numbers are the rolling-origin estimates in
``models/rolling_cv_final/cv_summary.csv``.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import toss_advisor as TA  # noqa: E402

MATCHES_PATH = ROOT / "data/processed/dashboard_matches.csv"
CLIMATOLOGY_PATH = ROOT / "data/processed/dashboard_weather_climatology.csv"
VENUES_PATH = ROOT / "data/processed/dashboard_venues.csv"
DYNAMICS = ROOT / "models/pitch_dynamics"
TOSS_PATH = ROOT / "data/processed/dashboard_toss_matches.csv"
TEAMS_PATH = ROOT / "data/processed/dashboard_teams.csv"
TEAM_PLAYERS_PATH = ROOT / "data/processed/dashboard_team_players.csv"
MODEL_ROOT = ROOT / "models/dashboard"
CV_PATH = ROOT / "models/rolling_cv_final/cv_summary.csv"
PREDICTIONS = ROOT / "models/rolling_cv_final/predictions"

FORMATS = ("T20", "ODI", "Test")
CLASS_ORDER = ("Batting", "Pace", "Spin")
CLASS_COLORS = {"Batting": "#059669", "Pace": "#2563eb", "Spin": "#d97706"}
CLASS_SHORT = {"Batting": "Batting", "Pace": "Pace", "Spin": "Spin"}
WINDOW = {"T20": 5, "ODI": 5, "Test": 10}
PHASE_ORDER = {
    "T20": ["powerplay", "middle", "death"],
    "ODI": ["powerplay", "middle", "death"],
    "Test": ["new_ball", "middle", "old_ball"],
}
PHASE_LABEL = {
    "powerplay": "Powerplay",
    "middle": "Middle",
    "death": "Death",
    "new_ball": "New ball",
    "old_ball": "Old ball",
}
WEATHER_COLUMNS = (
    "temperature_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "relative_humidity_2m_mean",
    "precip_sum_7d",
    "precip_sum_14d",
    "precip_sum_28d",
    "temp_mean_7d",
    "temp_mean_14d",
    "humidity_mean_7d",
    "wind_max_mean_7d",
    "dry_days_before",
)
PRETTY_FEATURE = {
    "venue": "Venue",
    "venue_id": "Venue ID",
    "city": "City",
    "gender": "Gender",
    "team_type": "Team type",
    "toss_decision": "Toss decision",
    "toss_winner_batted_first": "Toss winner batted first",
    "year": "Year",
    "month_sin": "Season (sin)",
    "month_cos": "Season (cos)",
    "day_of_year_sin": "Day of year (sin)",
    "day_of_year_cos": "Day of year (cos)",
    "venue_matches_in_history": "Matches at venue",
    "venue_role_matches_in_history": "Matches with full role data",
    "venue_last5_mean_first_innings_run_rate": "Venue run rate (last 5)",
    "venue_last5_mean_early_dot_percent": "Venue early dot % (last 5)",
    "venue_last5_mean_spin_wicket_share": "Venue spin wicket share (last 5)",
    "venue_last10_mean_first_innings_run_rate": "Venue run rate (last 10)",
    "venue_last10_mean_early_dot_percent": "Venue early dot % (last 10)",
    "venue_last10_mean_spin_wicket_share": "Venue spin wicket share (last 10)",
    "temperature_2m_mean": "Temperature",
    "precipitation_sum": "Match-day precipitation",
    "wind_speed_10m_max": "Maximum wind",
    "relative_humidity_2m_mean": "Relative humidity",
    "precip_sum_7d": "Rain, previous 7 days",
    "precip_sum_14d": "Rain, previous 14 days",
    "precip_sum_28d": "Rain, previous 28 days",
    "temp_mean_7d": "Temperature, previous 7 days",
    "temp_mean_14d": "Temperature, previous 14 days",
    "humidity_mean_7d": "Humidity, previous 7 days",
    "wind_max_mean_7d": "Wind, previous 7 days",
    "dry_days_before": "Dry days before match",
}

# Theme-aware palette. The app ships dark by default (see .streamlit/config.toml)
# but follows the theme picked in Streamlit's settings menu, so both palettes are
# defined here and applied to the custom HTML and to every matplotlib figure.
THEME_TYPE = (st.context.theme or {}).get("type") or "dark"
IS_DARK = THEME_TYPE != "light"
PALETTE = {
    "panel": "#111b2d" if IS_DARK else "#ffffff",
    "border": "#22314a" if IS_DARK else "#e2e8f0",
    "text": "#e8eef8" if IS_DARK else "#0f172a",
    "muted": "#9fb0c9" if IS_DARK else "#475569",
    "axis": "#cbd5e1" if IS_DARK else "#0f172a",
    "accent": "#f59e0b" if IS_DARK else "#0f172a",
    "neutral": "#64748b" if IS_DARK else "#94a3b8",
    "track": "#22314a" if IS_DARK else "#e2e8f0",
    "shadow": "rgba(0,0,0,.35)" if IS_DARK else "rgba(15,23,42,.08)",
    "verdict_bg": "#1e1a10" if IS_DARK else "#fffbeb",
    "verdict_text": "#fcd34d" if IS_DARK else "#92400e",
}

plt.rcParams.update(
    {
        "figure.facecolor": PALETTE["panel"],
        "axes.facecolor": PALETTE["panel"],
        "savefig.facecolor": PALETTE["panel"],
        "axes.edgecolor": PALETTE["border"],
        "axes.labelcolor": PALETTE["muted"],
        "text.color": PALETTE["text"],
        "xtick.color": PALETTE["muted"],
        "ytick.color": PALETTE["muted"],
        "grid.color": PALETTE["border"],
        "legend.facecolor": PALETTE["panel"],
        "legend.edgecolor": PALETTE["border"],
    }
)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_matches() -> pd.DataFrame:
    return pd.read_csv(
        MATCHES_PATH,
        parse_dates=["date"],
        dtype={"match_id": str, "venue_key": str, "venue_id": str},
    )


@st.cache_data(show_spinner=False)
def load_climatology() -> pd.DataFrame:
    return pd.read_csv(CLIMATOLOGY_PATH, dtype={"venue_key": str})


@st.cache_data(show_spinner=False)
def load_venues() -> pd.DataFrame:
    return pd.read_csv(
        VENUES_PATH, parse_dates=["first_match", "last_match"], dtype={"venue_key": str}
    )


@st.cache_data(show_spinner=False)
def load_cv() -> pd.DataFrame:
    return pd.read_csv(CV_PATH)


@st.cache_data(show_spinner=False)
def load_predictions(match_type: str) -> pd.DataFrame:
    return pd.read_csv(
        PREDICTIONS / f"xgb_ctx@performat_{match_type}.csv",
        parse_dates=["date"],
        dtype={"match_id": str},
    )


@st.cache_data(show_spinner=False)
def load_dynamics() -> dict[str, pd.DataFrame]:
    files = {
        "phase": "wear_by_phase.csv",
        "curves": "wear_curves.csv",
        "regression": "wear_regression.csv",
        "heterogeneity": "wear_venue_heterogeneity.csv",
        "decomposition": "wear_decomposition.csv",
        "ladder": "wear_boundary_ladder.csv",
        "placebo": "wear_placebo_bias.csv",
        "robustness": "wear_robustness.csv",
        "persistence": "wear_persistence.csv",
        "gain": "wear_predictive_gain.csv",
        "across": "wear_across_innings_stress.csv",
        "power": "wear_power_audit.csv",
        "specifications": "wear_specifications.csv",
    }
    return {name: pd.read_csv(DYNAMICS / filename) for name, filename in files.items()}


@st.cache_data(show_spinner=False)
def load_toss_matches() -> pd.DataFrame:
    return pd.read_csv(
        TOSS_PATH,
        parse_dates=["date"],
        dtype={"match_id": str, "venue_key": str, "venue_id": str},
    )


@st.cache_data(show_spinner=False)
def load_teams() -> pd.DataFrame:
    return pd.read_csv(
        TEAMS_PATH, parse_dates=["last_date"], dtype={"team": str}
    )


@st.cache_data(show_spinner=False)
def load_team_players() -> pd.DataFrame:
    return pd.read_csv(
        TEAM_PLAYERS_PATH,
        parse_dates=["last_date"],
        dtype={"team": str, "player": str, "player_id": str},
    )


@st.cache_resource(show_spinner=False)
def load_model(match_type: str):
    folder = MODEL_ROOT / match_type.lower()
    model = joblib.load(folder / "model.joblib")
    card = json.loads((folder / "model_card.json").read_text(encoding="utf-8"))
    return model, card


# --------------------------------------------------------------------------- #
# Feature construction
# --------------------------------------------------------------------------- #
def pretty_feature(name: str) -> str:
    return PRETTY_FEATURE.get(name, name.replace("_", " ").title())


def raw_feature(transformed: str) -> str:
    if transformed.startswith("numeric__"):
        return transformed[len("numeric__") :]
    body = transformed[len("categorical__") :] if transformed.startswith("categorical__") else transformed
    for raw in ("venue_id", "venue", "city", "gender", "team_type", "toss_decision"):
        if body == raw or body.startswith(raw + "_"):
            return raw
    return body


def compute_history(
    matches: pd.DataFrame,
    match_type: str,
    venue_key: str,
    cutoff: pd.Timestamp,
    window: int,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    prior = matches[
        matches["match_type"].eq(match_type)
        & matches["venue_key"].eq(venue_key)
        & matches["date"].lt(cutoff)
    ].sort_values(["date", "match_id"])
    recent = prior.tail(window)
    role = recent[recent["role_complete"]]
    spin = role["spin_wicket_share"].dropna()
    prefix = f"venue_last{window}_mean"
    values = {
        f"{prefix}_first_innings_run_rate": float(recent["first_innings_run_rate"].mean())
        if len(recent)
        else np.nan,
        f"{prefix}_early_dot_percent": float(recent["early_dot_percent"].mean())
        if len(recent)
        else np.nan,
        f"{prefix}_spin_wicket_share": float(spin.mean()) if len(spin) else np.nan,
        "venue_matches_in_history": float(len(recent)),
        "venue_role_matches_in_history": float(len(role)),
    }
    return values, prior, recent


def weather_defaults(
    climatology: pd.DataFrame,
    venue_key: str,
    match_type: str,
    month: int,
) -> dict[str, float]:
    candidates = [(venue_key, month), (venue_key, 0), ("__all__", -1)]
    for key, value in candidates:
        row = climatology[
            climatology["venue_key"].eq(key)
            & climatology["match_type"].eq(match_type)
            & climatology["month"].eq(value)
        ]
        if not row.empty:
            return {column: float(row.iloc[0][column]) for column in WEATHER_COLUMNS}
    return {column: np.nan for column in WEATHER_COLUMNS}


def apply_scenario(weather: dict[str, float], scenario: str) -> dict[str, float]:
    values = dict(weather)
    if scenario == "Dry build-up":
        values["precipitation_sum"] = 0.0
        values["precip_sum_7d"] = 0.0
        values["precip_sum_14d"] = 0.0
        values["precip_sum_28d"] = 0.0
        values["dry_days_before"] = 14.0
    elif scenario == "Wet build-up":
        values["precipitation_sum"] = max(values.get("precipitation_sum", 0.0), 2.0)
        for column in ("precip_sum_7d", "precip_sum_14d", "precip_sum_28d"):
            values[column] = values.get(column, 0.0) * 2.5 + 1.0
        values["dry_days_before"] = 0.0
    return values


def build_feature_row(
    card: dict,
    history: dict[str, float],
    weather: dict[str, float],
    match_date: date,
    toss_decision: str,
    gender: str,
    team_type: str,
    venue_name: str,
    venue_id: str,
    city: str,
) -> pd.DataFrame:
    stamp = pd.Timestamp(match_date)
    month = stamp.month
    day_of_year = stamp.day_of_year
    record: dict[str, object] = {
        "venue": venue_name,
        "venue_id": venue_id if pd.notna(venue_id) else np.nan,
        "city": city if pd.notna(city) else np.nan,
        "gender": gender,
        "team_type": team_type,
        "toss_decision": toss_decision.lower(),
        "toss_winner_batted_first": int(toss_decision.lower() == "bat"),
        "year": int(stamp.year),
        "month_sin": float(np.sin(2 * np.pi * month / 12)),
        "month_cos": float(np.cos(2 * np.pi * month / 12)),
        "day_of_year_sin": float(np.sin(2 * np.pi * day_of_year / 365.25)),
        "day_of_year_cos": float(np.cos(2 * np.pi * day_of_year / 365.25)),
    }
    record.update(history)
    record.update(weather)
    frame = pd.DataFrame([record])
    for column in card["feature_columns"]:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[card["feature_columns"]]


def predict_probabilities(model, card: dict, row: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(row)[0]
    classes = list(model.classes_)
    wanted = list(range(len(card["class_order"])))
    if classes != wanted:
        raw = np.array([raw[classes.index(index)] for index in wanted])
    return raw


def local_contributions(
    model, card: dict, row: pd.DataFrame, predicted_index: int
) -> pd.DataFrame:
    preprocessing = model.named_steps["preprocessing"]
    estimator = model.named_steps["model"]
    transformed = preprocessing.transform(row)
    names = list(preprocessing.get_feature_names_out())
    contributions = estimator.get_booster().predict(
        xgb.DMatrix(transformed, feature_names=names), pred_contribs=True
    )
    contributions = np.asarray(contributions)
    if contributions.ndim == 3:
        contributions = contributions[0, predicted_index, :]
    elif contributions.ndim == 2:
        contributions = contributions[0]
    series = pd.Series(contributions[:-1], index=names)
    grouped = series.groupby(series.index.map(raw_feature)).sum()
    grouped = grouped[grouped.abs().gt(0)]
    table = grouped.rename("contribution").reset_index()
    table.columns = ["feature", "contribution"]
    table["label"] = table["feature"].map(pretty_feature)
    return table.reindex(
        table["contribution"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Toss advisor
# --------------------------------------------------------------------------- #
# The estimation logic and the shipped call rule live in src/toss_advisor.py so
# that this app and src/backtest_toss_advisor.py cannot drift apart. Only the
# rendering helper stays here.


def toss_decision_card(
    label: str, value: float, low: float, high: float, active: bool, color: str
) -> None:
    active_class = "active" if active else ""
    st.markdown(
        f"""
        <div class="prob-card {active_class}">
          <div class="prob-label">{label}{' · lean' if active else ''}</div>
          <div class="prob-value" style="color:{color};">{value:.1%}</div>
          <div class="prob-bar"><span style="width:{value * 100:.1f}%;background:{color};"></span></div>
          <div class="small-muted" style="margin-top:8px;">95% interval {low:.1%} to {high:.1%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Presentation helpers
# --------------------------------------------------------------------------- #
def clean_axes(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def show_figure(fig) -> None:
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def probability_card(label: str, value: float, active: bool) -> None:
    color = CLASS_COLORS[label]
    active_class = "active" if active else ""
    badge = " · expected" if active else ""
    st.markdown(
        f"""
        <div class="prob-card {active_class}">
          <div class="prob-label">{label}{badge}</div>
          <div class="prob-value" style="color:{color};">{value:.0%}</div>
          <div class="prob-bar"><span style="width:{value * 100:.1f}%;background:{color};"></span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div class="section-title">{title}</div>'
        + (f'<div class="section-sub">{subtitle}</div>' if subtitle else ""),
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Pitch Brief",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; max-width: 1280px; }
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 58%, #334155 100%);
        color: #f8fafc; padding: 26px 30px; border-radius: 18px; margin-bottom: 18px;
    }
    .hero-kicker { font-size: 11px; letter-spacing: .18em; text-transform: uppercase;
                   color: #94a3b8; font-weight: 700; }
    .hero-title { font-size: 38px; font-weight: 800; line-height: 1.05; margin: 6px 0 8px 0; }
    .hero-sub { font-size: 15px; color: #cbd5e1; max-width: 850px; }
    .badge { display: inline-block; padding: 4px 10px; border-radius: 999px;
             font-size: 11px; font-weight: 600; margin-right: 6px; margin-top: 13px; }
    .badge-warn { background: #7c2d12; color: #fed7aa; }
    .badge-neutral { background: #1e293b; color: #cbd5e1; border: 1px solid #334155; }
    .prob-card { border: 1px solid __BORDER__; border-radius: 14px; padding: 14px 16px;
                 background: __PANEL__; height: 100%; }
    .prob-card.active { border-color: __ACCENT__; box-shadow: 0 4px 18px __SHADOW__; }
    .prob-label { font-size: 12px; letter-spacing: .08em; text-transform: uppercase;
                  color: __MUTED__; font-weight: 700; }
    .prob-value { font-size: 32px; font-weight: 800; line-height: 1.1; margin-top: 2px; }
    .prob-bar { height: 8px; border-radius: 999px; background: __TRACK__;
                margin-top: 9px; overflow: hidden; }
    .prob-bar span { display: block; height: 100%; border-radius: 999px; }
    .section-title { font-size: 20px; font-weight: 800; color: __TEXT__;
                     margin: 4px 0 2px 0; }
    .section-sub { color: __MUTED__; font-size: 13px; margin-bottom: 8px; }
    .verdict { border-left: 4px solid #d97706; background: __VERDICT_BG__;
               padding: 14px 16px; border-radius: 0 12px 12px 0; margin: 10px 0 14px 0; }
    .verdict b { color: __VERDICT_TEXT__; }
    .small-muted { color: __MUTED__; font-size: 12px; }
    .toss-call { border: 1px solid __BORDER__; border-radius: 14px; padding: 18px 20px;
                 background: __PANEL__; }
    .toss-call .kicker { font-size: 11px; letter-spacing: .16em; text-transform: uppercase;
                         color: __MUTED__; font-weight: 700; }
    .toss-call .choice { font-size: 30px; font-weight: 800; line-height: 1.1; margin: 4px 0; }
    .toss-call .detail { font-size: 13px; color: __MUTED__; }
    .player-chip { display: inline-block; padding: 3px 9px; border-radius: 999px;
                   border: 1px solid __BORDER__; background: __PANEL__; color: __TEXT__;
                   font-size: 12px; margin: 0 5px 5px 0; }
    </style>
    """.replace("__BORDER__", PALETTE["border"])
    .replace("__PANEL__", PALETTE["panel"])
    .replace("__ACCENT__", PALETTE["accent"])
    .replace("__SHADOW__", PALETTE["shadow"])
    .replace("__MUTED__", PALETTE["muted"])
    .replace("__TRACK__", PALETTE["track"])
    .replace("__TEXT__", PALETTE["text"])
    .replace("__VERDICT_BG__", PALETTE["verdict_bg"])
    .replace("__VERDICT_TEXT__", PALETTE["verdict_text"]),
    unsafe_allow_html=True,
)

matches = load_matches()
climatology = load_climatology()
venue_directory = load_venues()
cv = load_cv()
dynamics = load_dynamics()
toss_matches = load_toss_matches()
team_players = load_team_players()

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Pre-match brief · prototype</div>
      <div class="hero-title">Pitch Brief</div>
      <div class="hero-sub">
        What the ball-by-ball record says about spin, pace and the surface —
        and, just as importantly, what it cannot say.
      </div>
      <span class="badge badge-warn">Proxy label, not a pitch measurement</span>
      <span class="badge badge-neutral">6.78M deliveries</span>
      <span class="badge badge-neutral">Rolling-origin validation</span>
      <span class="badge badge-neutral">Read-only committed artifacts</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- Sidebar ------------------------------------- #
st.sidebar.markdown("### Build a brief")
match_type = st.sidebar.selectbox("Format", FORMATS, index=0)

format_venues = venue_directory[venue_directory["match_type"].eq(match_type)].copy()
format_venues = format_venues.sort_values("matches", ascending=False)
if format_venues.empty:
    st.error("No venues found for this format.")
    st.stop()

# Pick a first screen that shows the model's inputs rather than a venue whose
# last five matches all lack bowler-role data. The dropdown itself stays ordered
# by match count so it remains easy to scan.
recent_role = (
    matches.sort_values(["match_type", "venue_key", "date", "match_id"])
    .groupby(["match_type", "venue_key"], observed=True)
    .tail(5)
    .groupby(["match_type", "venue_key"], observed=True)["role_complete"]
    .mean()
)
format_venues["recent_role"] = [
    float(recent_role.get((match_type, key), np.nan))
    for key in format_venues["venue_key"]
]
eligible_default = format_venues[
    format_venues["matches"].ge(20) & format_venues["recent_role"].ge(0.6)
]
default_venue_key = (
    eligible_default.iloc[0]["venue_key"]
    if not eligible_default.empty
    else format_venues.iloc[0]["venue_key"]
)

venue_options = list(format_venues["venue_key"])
venue_labels = {
    row.venue_key: (
        f"{row.venue} — {row.city}" if isinstance(row.city, str) and row.city else row.venue
    )
    for row in format_venues.itertuples()
}
venue_key = st.sidebar.selectbox(
    "Venue",
    venue_options,
    format_func=lambda key: venue_labels.get(key, key),
    index=venue_options.index(default_venue_key),
    key=f"venue_{match_type}",
    help="Search by venue name. The list is ordered by match count.",
)

venue_rows = matches[
    matches["match_type"].eq(match_type) & matches["venue_key"].eq(venue_key)
].sort_values(["date", "match_id"])
latest = venue_rows.iloc[-1]

default_date = max(date.today(), (latest["date"].date() + timedelta(days=1)))
match_date = st.sidebar.date_input(
    "Match date",
    value=default_date,
    min_value=date(2001, 1, 1),
    max_value=date(2030, 12, 31),
    key=f"date_{venue_key}",
    help="Used for season features and to select the prior-history cutoff.",
)

# Everything the brief shows about this venue is restricted to matches strictly
# before the fixture date, so an earlier date cannot leak later results.
cutoff = pd.Timestamp(match_date)
past = venue_rows[venue_rows["date"].lt(cutoff)]

toss_counts = past["toss_decision"].value_counts(normalize=True)
default_toss = toss_counts.idxmax() if len(toss_counts) else "field"
toss_choice = st.sidebar.radio(
    "Toss decision",
    ["Field", "Bat"],
    index=0 if str(default_toss).lower() == "field" else 1,
    horizontal=True,
    key=f"toss_{venue_key}",
    help="At this venue, the historically more common choice is preselected. The toss is not known pre-match.",
)
if len(toss_counts):
    st.sidebar.caption(
        f"At this venue, {toss_counts.get('field', 0):.0%} of toss winners chose to field "
        f"(n={len(past):,} before this date)."
    )

with st.sidebar.expander("Advanced inputs"):
    genders = ["male", "female"]
    teams = ["international", "club"]
    venue_genders = set(past["gender"].dropna()) or set(venue_rows["gender"].dropna())
    venue_teams = set(past["team_type"].dropna()) or set(venue_rows["team_type"].dropna())
    default_gender = "male" if "male" in venue_genders else "female"
    default_team = "international" if "international" in venue_teams else "club"
    gender = st.selectbox(
        "Gender",
        genders,
        index=genders.index(default_gender),
        key=f"gender_{venue_key}",
    )
    team_type = st.selectbox(
        "Team type",
        teams,
        index=teams.index(default_team),
        key=f"team_{venue_key}",
    )
    scenario = st.selectbox(
        "Build-up moisture",
        ["Climatology", "Dry build-up", "Wet build-up"],
        index=0,
        key=f"scenario_{venue_key}",
        help="A scenario for the prior 7/14/28-day rain features, not a forecast.",
    )

st.sidebar.caption(
    "The model predicts agreement with a rule-derived behavioural label. "
    "It does not measure the pitch."
)
st.sidebar.info(
    "Team and player pickers are in the **Toss & line-ups** tab, second from the left."
)

# ----------------------------- Model --------------------------------------- #
model, card = load_model(match_type)
window = int(card["venue_window"])
history_values, prior, recent = compute_history(
    matches, match_type, venue_key, pd.Timestamp(match_date), window
)
if prior.empty:
    st.warning(
        "This venue has no earlier matches in the selected format. History features "
        "are missing and the model falls back to its imputation."
    )

weather_base = weather_defaults(
    climatology, venue_key, match_type, pd.Timestamp(match_date).month
)
weather = apply_scenario(weather_base, scenario)
row = build_feature_row(
    card,
    history_values,
    weather,
    match_date,
    toss_choice,
    gender,
    team_type,
    latest["venue"],
    latest["venue_id"],
    latest["city"],
)
probabilities = predict_probabilities(model, card, row)
predicted_index = int(np.argmax(probabilities))
predicted_class = card["class_order"][predicted_index]
contributions = local_contributions(model, card, row, predicted_index)

venue_profile = venue_directory[
    venue_directory["match_type"].eq(match_type)
    & venue_directory["venue_key"].eq(venue_key)
].iloc[0]
format_mean_run_rate = float(
    recent["first_innings_run_rate"].mean()
) if len(recent) else np.nan
format_average = matches[matches["match_type"].eq(match_type)]["first_innings_run_rate"].mean()

# ----------------------------- Tabs ---------------------------------------- #
tab_brief, tab_toss, tab_dynamics, tab_model, tab_method = st.tabs(
    ["Match brief", "Toss & line-ups", "Pitch dynamics", "Model card", "Method & caveats"]
)

# ============================= Match brief ================================= #
with tab_brief:
    section(
        f"{latest['venue']} · {match_type} · {match_date:%d %b %Y}",
        f"{gender.title()} · {team_type.title()} · toss: {toss_choice.lower()}",
    )

    cols = st.columns(3)
    for column, label, value in zip(cols, card["class_order"], probabilities):
        with column:
            probability_card(label, float(value), label == predicted_class)

    st.caption(
        f"Expected proxy class: **{predicted_class}**. "
        f"Rolling-origin macro-F1 for this format is {card['rolling_cv']['pooled_macro_f1']:.3f} "
        f"(majority-class share {card['label_counts'].get('Pace', 0) / card['training_rows']:.0%}). "
        "Probabilities are model agreement with the rule-derived label, not a physical pitch reading."
    )

    left, right = st.columns([1.15, 1])

    with left:
        section("Top drivers for this brief", "Log-odds contributions to the expected class")
        top = contributions.head(8).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7.2, 3.9))
        colors = [
            CLASS_COLORS[predicted_class] if value > 0 else PALETTE["neutral"]
            for value in top["contribution"]
        ]
        ax.barh(top["label"], top["contribution"], color=colors, height=0.62)
        ax.axvline(0, color=PALETTE["axis"], linewidth=1)
        ax.set_xlabel(f"Contribution toward {predicted_class} (log-odds)")
        clean_axes(ax)
        ax.grid(axis="x", alpha=0.25)
        ax.grid(axis="y", visible=False)
        fig.tight_layout()
        show_figure(fig)
        st.caption(
            "Positive bars push toward the expected class; grey bars push away. "
            "These are model associations, not causal effects."
        )

    with right:
        section("Venue snapshot", "Prior matches used by the model")
        m1, m2 = st.columns(2)
        m1.metric("Matches on record", f"{int(venue_profile['matches']):,}")
        m2.metric("In history", f"{len(recent)} of last {window}")
        m3, m4 = st.columns(2)
        m3.metric(
            f"Run rate (last {window})",
            f"{format_mean_run_rate:.2f}" if pd.notna(format_mean_run_rate) else "n/a",
            help=f"Format average: {format_average:.2f}" if pd.notna(format_average) else None,
        )
        m4.metric(
            f"Early dot % (last {window})",
            f"{recent['early_dot_percent'].mean():.1f}%" if len(recent) else "n/a",
            help="Share of legal balls in the early overs that concede no run.",
        )
        m5, m6 = st.columns(2)
        spin_value = recent["spin_wicket_share"].dropna()
        m5.metric(
            f"Spin share (last {window})",
            f"{spin_value.mean():.2f}" if len(spin_value) else "n/a",
            help="Spin share of bowler-credited wickets where both styles are known.",
        )
        role_share = (
            recent["role_complete"].mean() if len(recent) else np.nan
        )
        m6.metric(
            "Role coverage",
            f"{role_share:.0%}" if pd.notna(role_share) else "n/a",
            help="Share of prior matches where every bowler-credited wicket had a known style.",
        )
        st.caption(
            f"Last match at this venue before {match_date:%d %b %Y}: "
            + (f"{past['date'].max():%d %b %Y}." if len(past) else "none on record.")
        )
        if len(spin_value) == 0:
            st.info(
                f"None of the last {len(recent)} matches at this venue has complete "
                "bowler-role data, so the spin-history input is imputed. The "
                "probabilities are less anchored here."
            )
        elif pd.notna(role_share) and role_share < 0.5:
            st.info(
                f"Only {role_share:.0%} of the last {len(recent)} matches at this venue "
                "have complete bowler-role data, so the spin-history input is partly "
                "imputed. The probabilities are less anchored here."
            )

    section("Recent matches at this venue", "What the proxy label looked like")
    table = past.tail(8).copy()
    table = table[
        [
            "date",
            "first_batting_team",
            "first_bowling_team",
            "first_innings_runs",
            "first_innings_run_rate",
            "early_dot_percent",
            "spin_wicket_share",
            "toss_decision",
            "y",
        ]
    ].rename(
        columns={
            "date": "Date",
            "first_batting_team": "Batting first",
            "first_bowling_team": "Bowling first",
            "first_innings_runs": "First-innings runs",
            "first_innings_run_rate": "Run rate",
            "early_dot_percent": "Early dot %",
            "spin_wicket_share": "Spin wicket share",
            "toss_decision": "Toss",
            "y": "Proxy label",
        }
    )
    table["Date"] = pd.to_datetime(table["Date"]).dt.strftime("%d %b %Y")
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "Run rate": st.column_config.NumberColumn(format="%.2f"),
            "Early dot %": st.column_config.NumberColumn(format="%.1f"),
            "Spin wicket share": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    with st.expander("Weather inputs used"):
        st.caption(
            f"Month climatology for {match_date:%B} at this venue, with the selected "
            f"build-up scenario applied. This is not a forecast."
        )
        weather_table = pd.DataFrame(
            {
                "Input": [
                    "Mean temperature (°C)",
                    "Match-day precipitation (mm)",
                    "Relative humidity (%)",
                    "Maximum wind (km/h)",
                    "Rain, previous 7 days (mm)",
                    "Rain, previous 14 days (mm)",
                    "Rain, previous 28 days (mm)",
                    "Dry days before match",
                ],
                "Value": [
                    weather.get("temperature_2m_mean", np.nan),
                    weather.get("precipitation_sum", np.nan),
                    weather.get("relative_humidity_2m_mean", np.nan),
                    weather.get("wind_speed_10m_max", np.nan),
                    weather.get("precip_sum_7d", np.nan),
                    weather.get("precip_sum_14d", np.nan),
                    weather.get("precip_sum_28d", np.nan),
                    weather.get("dry_days_before", np.nan),
                ],
            }
        )
        st.dataframe(
            weather_table,
            hide_index=True,
            width="stretch",
            column_config={"Value": st.column_config.NumberColumn(format="%.1f")},
        )

    # Audit verdict for this format
    dot_row = dynamics["decomposition"][
        dynamics["decomposition"]["match_type"].eq(match_type)
        & dynamics["decomposition"]["outcome"].eq("dot")
        & dynamics["decomposition"]["scenario"].eq("smooth_gradient")
    ].iloc[0]
    dot_bias = dynamics["placebo"][
        dynamics["placebo"]["match_type"].eq(match_type)
        & dynamics["placebo"]["outcome"].eq("dot_share")
    ].iloc[0]
    composition_share = dot_row["composition"] / dot_row["beta_pooled"]
    st.markdown(
        f"""
        <div class="verdict">
          <b>The audit in one line.</b>
          The dot-share gradient is precise — {dot_bias['observed']:+.5f} per over
          in the cell panel under match × innings fixed effects, {dot_bias['bootstrap_z']:.1f}
          bootstrap standard deviations from zero (bootstrap sd
          {dot_bias['bootstrap_sd']:.5f}) — but it is <b>not identified as surface
          wear</b>: runs per ball reverses sign by phase in every format, dot share
          reverses in T20, the T20 runs channel disappears when innings edges are
          trimmed, and the second-new-ball jump is bowler selection. In the
          delivery-level decomposition — a different sample with more controls —
          composition accounts for {composition_share:+.0%} of its own coefficient,
          so bowler mix is not the explanation either.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================= Toss advisor =============================== #
with tab_toss:
    section(
        f"Toss & line-ups · {latest['venue']} · {match_type}",
        f"{gender.title()} · {team_type.title()} · matches before {match_date:%d %b %Y}",
    )
    st.caption(
        "If you win the toss, should you bat or field? The estimate is the share of "
        "toss winners who went on to win after making each choice at this venue. "
        "It is an association in the historical record, not a causal effect of the toss, "
        "and the backtest in Method & caveats does not support it as a prediction."
    )

    toss_cutoff = pd.Timestamp(match_date)
    decided = TA.decided_tosses(
        toss_matches, match_type, gender, team_type, toss_cutoff
    )

    if decided.empty:
        st.warning(
            "No decided matches with a usable toss are on record for this format, "
            "gender and team type before the selected date."
        )
    else:
        team_counts = pd.concat(
            [decided["team1"], decided["team2"]], ignore_index=True
        ).value_counts()
        team_options = list(team_counts.index)

        venue_decided = decided[decided["venue_key"].eq(venue_key)]
        if len(venue_decided):
            recent_pair = venue_decided.sort_values("date").iloc[-1]
            default_pair = [recent_pair["team1"], recent_pair["team2"]]
        else:
            default_pair = team_options[:2]
        default_pair = [team for team in default_pair if team in team_options]
        while len(default_pair) < 2 and len(team_options) >= 2:
            candidate = team_options[len(default_pair)]
            if candidate not in default_pair:
                default_pair.append(candidate)

        st.markdown("#### 1 · Pick the two sides")
        left_team, right_team = st.columns(2)
        with left_team:
            team_a = st.selectbox(
                "Team A",
                team_options,
                index=team_options.index(default_pair[0]),
                key=f"toss_team_a_{match_type}_{venue_key}",
            )
        with right_team:
            other_options = [team for team in team_options if team != team_a]
            default_b = default_pair[1] if default_pair[1] != team_a else other_options[0]
            team_b = st.selectbox(
                "Team B",
                other_options,
                index=other_options.index(default_b),
                key=f"toss_team_b_{match_type}_{venue_key}",
            )

        def squad_frame(team: str) -> pd.DataFrame:
            squad = team_players[
                team_players["match_type"].eq(match_type)
                & team_players["gender"].eq(gender)
                & team_players["team_type"].eq(team_type)
                & team_players["team"].eq(team)
            ].copy()
            return squad.sort_values(
                ["last_date", "matches"], ascending=[False, False]
            )

        st.markdown("#### 2 · Line-ups")
        st.caption(
            "Names come from the Cricsheet scorecards for this format. The picker "
            "defaults to the eleven most recent players; the records shown are "
            "career records in this format and are descriptive only."
        )
        lineup_columns = st.columns(2)
        selections: dict[str, list[str]] = {}
        for column, team in zip(lineup_columns, (team_a, team_b)):
            squad = squad_frame(team)
            with column:
                st.markdown(f"**{team}**")
                if squad.empty:
                    st.info("No squad records for this side in the archive.")
                    selections[team] = []
                    continue
                choices = list(squad["player"])
                default_xi = choices[:11]
                chosen = st.multiselect(
                    f"{team} XI",
                    choices,
                    default=default_xi,
                    key=f"toss_xi_{match_type}_{venue_key}_{team}",
                    label_visibility="collapsed",
                    placeholder="Type to search for a player",
                )
                selections[team] = chosen
                picked = squad[squad["player"].isin(chosen)]
                if len(picked):
                    matches_played = int(picked["matches"].sum())
                    wins_played = int(picked["wins"].sum())
                    st.caption(
                        f"{len(picked)} selected · {matches_played:,} format appearances "
                        f"· {wins_played / matches_played:.1%} win share"
                    )

        st.markdown("#### 3 · The toss call")
        estimates, base_est, use_venue, venue_total = TA.decision_table(
            decided, venue_key
        )
        bat = estimates[estimates["decision"].eq("bat")].iloc[0]
        field = estimates[estimates["decision"].eq("field")].iloc[0]
        verdict = TA.choose(bat, field, use_venue, venue_total)
        edge = verdict["edge"]
        prob = verdict["prob"]
        enough = verdict["enough"]
        if not enough:
            call = "No clear edge"
            call_color = PALETTE["muted"]
            call_detail = (
                "This venue does not have enough decided matches behind both "
                f"choices (n={int(bat['n'])} bat, {int(field['n'])} field). The "
                "figures below fall back to the format baseline; treat the toss "
                "as close to neutral here."
            )
        elif verdict["call"] == "none":
            call = "No clear edge"
            call_color = PALETTE["muted"]
            call_detail = (
                f"The adjusted difference is {edge:+.1f} points and the posterior "
                f"probability that one choice is better is only {prob:.0%}. "
                "Conditions, team make-up and the XI matter more than the toss on "
                "this record."
            )
        elif verdict["call"] == "bat":
            call = "Bat first"
            call_color = "#10b981"
            call_detail = (
                f"Toss winners who batted first won {bat['estimate']:.1%} of the time "
                f"here, against {field['estimate']:.1%} for those who fielded: "
                f"a {edge:+.1f}-point edge over {int(venue_total):,} decided matches, "
                f"with a {prob:.0%} posterior probability that batting is the better "
                "choice."
            )
        else:
            call = "Field first"
            call_color = "#3b82f6"
            call_detail = (
                f"Toss winners who fielded first won {field['estimate']:.1%} of the time "
                f"here, against {bat['estimate']:.1%} for those who batted: "
                f"a {abs(edge):.1f}-point edge over {int(venue_total):,} decided matches, "
                f"with a {prob:.0%} posterior probability that fielding is the better "
                "choice."
            )

        source = (
            "venue and format history" if enough else "format baseline"
        )
        st.markdown(
            f"""
            <div class="toss-call">
              <div class="kicker">Historical lean if you win the toss · {source}</div>
              <div class="choice" style="color:{call_color};">{call}</div>
              <div class="detail">{call_detail}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.warning(
            "Backtest verdict: this venue lean is not validated. On 15,653 prior-only "
            "fixtures from 2010, the rule made 260 clear calls (1.7%). Across 89 forward "
            "windows at 8 venues, the direction held 81% of the time by matches but 50% "
            "when venues are weighted equally; the format baseline scored 52%, and the "
            "venue-adjusted win rates were worse than the format baseline as probability "
            "forecasts (Brier skill -1.6%). Treat the panel as descriptive venue history, "
            "not a recommendation."
        )

        card_columns = st.columns(2)
        with card_columns[0]:
            toss_decision_card(
                "Bat first",
                float(bat["estimate"]),
                float(bat["low"]),
                float(bat["high"]),
                call == "Bat first",
                "#10b981",
            )
        with card_columns[1]:
            toss_decision_card(
                "Field first",
                float(field["estimate"]),
                float(field["low"]),
                float(field["high"]),
                call == "Field first",
                "#3b82f6",
            )

        chart_left, chart_right = st.columns([1, 1.05])
        with chart_left:
            fig, ax = plt.subplots(figsize=(5.6, 2.3))
            positions = [1, 0]
            values = [float(bat["estimate"]), float(field["estimate"])]
            errors = np.array(
                [
                    [values[0] - float(bat["low"]), values[1] - float(field["low"])],
                    [float(bat["high"]) - values[0], float(field["high"]) - values[1]],
                ]
            )
            ax.barh(
                positions,
                values,
                color=["#10b981", "#3b82f6"],
                height=0.5,
            )
            ax.errorbar(
                values,
                positions,
                xerr=errors,
                fmt="none",
                ecolor=PALETTE["axis"],
                capsize=4,
                linewidth=1.2,
            )
            ax.axvline(0.5, color=PALETTE["muted"], linestyle="--", linewidth=1)
            ax.set_yticks(positions, ["Bat first", "Field first"])
            span_low = min(float(bat["low"]), float(field["low"])) - 0.02
            span_high = max(float(bat["high"]), float(field["high"])) + 0.02
            ax.set_xlim(max(0.0, span_low), min(1.0, span_high))
            ax.set_xlabel("Share of toss winners who won")
            clean_axes(ax)
            ax.grid(axis="x", alpha=0.25)
            ax.grid(axis="y", visible=False)
            fig.tight_layout()
            show_figure(fig)
            st.caption(
                "Bars are shrunk toward the format baseline and the whiskers are "
                "95% posterior intervals, so small venue samples move toward 50%."
            )

        with chart_right:
            st.markdown("**Venue history by toss decision**")
            display = estimates.copy()
            display["Decision"] = display["label"]
            display["Matches"] = display["n"].astype(int)
            display["Wins"] = display["wins"].astype(int)
            display["Raw"] = display["raw"] * 100
            display["Adjusted"] = display["estimate"] * 100
            display["Interval"] = display.apply(
                lambda row: f"{row['low']:.1%} to {row['high']:.1%}", axis=1
            )
            display["Format baseline"] = display["base_rate"] * 100
            st.dataframe(
                display[
                    ["Decision", "Matches", "Wins", "Raw", "Adjusted", "Interval",
                     "Format baseline"]
                ],
                hide_index=True,
                width="stretch",
                column_config={
                    "Raw": st.column_config.NumberColumn(format="%.1f%%"),
                    "Adjusted": st.column_config.NumberColumn(format="%.1f%%"),
                    "Format baseline": st.column_config.NumberColumn(format="%.1f%%"),
                },
            )
            st.caption(
                f"Format baseline: {match_type} toss winners who bat first win "
                f"{base_est.set_index('toss_decision')['base_rate'].get('bat', 0.5):.0%} "
                "of decided matches with a usable toss."
            )

        st.markdown("#### 4 · What the line-ups add")
        st.caption(
            "Batting-first and fielding-first records for the two selected sides in "
            "this format, before the selected date. They are context: the toss call "
            "above is estimated from venue and format history, not from a player model."
        )
        records = TA.team_records(decided, [team_a, team_b])
        records["Bat first"] = records.apply(
            lambda row: f"{int(row['bat_first_wins'])}-{int(row['bat_first_n'] - row['bat_first_wins'])}",
            axis=1,
        )
        records["Field first"] = records.apply(
            lambda row: f"{int(row['field_first_wins'])}-{int(row['field_first_n'] - row['field_first_wins'])}",
            axis=1,
        )
        records["Adjusted edge (bat − field)"] = (
            records["bat_first_adj"] - records["field_first_adj"]
        ) * 100
        records["bat_first_rate"] = records["bat_first_rate"] * 100
        records["field_first_rate"] = records["field_first_rate"] * 100
        st.dataframe(
            records[
                [
                    "team",
                    "Bat first",
                    "bat_first_rate",
                    "Field first",
                    "field_first_rate",
                    "Adjusted edge (bat − field)",
                ]
            ].rename(
                columns={
                    "team": "Team",
                    "bat_first_rate": "Bat-first win %",
                    "field_first_rate": "Field-first win %",
                }
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "Bat-first win %": st.column_config.NumberColumn(format="%.1f%%"),
                "Field-first win %": st.column_config.NumberColumn(format="%.1f%%"),
                "Adjusted edge (bat − field)": st.column_config.NumberColumn(format="%+.1f"),
            },
        )

        st.info(
            "The toss decision is chosen under conditions the analyst does not see: "
            "weather, pitch inspection, team composition and era. A higher historical "
            "win share after batting first is not proof that batting first causes wins, "
            "and this table does not adjust for team strength. No-result and tied "
            "matches are excluded. ODI history mixes the pre- and post-2011 ball rules."
        )

# ============================= Dynamics ================================== #
with tab_dynamics:
    section(
        "The object the paper estimates",
        "Age × spin coefficient: how spin's outcomes move against pace's as the ball ages",
    )
    st.caption(
        "A positive dot-share coefficient means spin's dot share rises relative to pace "
        "as the ball gets older. The estimand is relative and is never a physical "
        "measurement of the pitch. Legal spin and pace deliveries only."
    )

    format_phase = dynamics["phase"][dynamics["phase"]["match_type"].eq(match_type)].copy()
    format_curves = dynamics["curves"][dynamics["curves"]["match_type"].eq(match_type)].copy()
    format_het = dynamics["heterogeneity"][
        dynamics["heterogeneity"]["match_type"].eq(match_type)
        & dynamics["heterogeneity"]["outcome"].eq("dot_share")
    ].copy()

    k1, k2, k3 = st.columns(3)
    k1.metric(
        "Pooled dot gradient",
        f"{dot_bias['observed']:+.5f}",
        help="Age × spin coefficient, match × innings fixed effects, clustered by match.",
    )
    k2.metric(
        "Phase range",
        f"{format_phase[format_phase['outcome'].eq('dot_share')]['coefficient'].min():+.5f}"
        f" to {format_phase[format_phase['outcome'].eq('dot_share')]['coefficient'].max():+.5f}",
        help="Smallest and largest phase-specific dot-share coefficient.",
    )
    k3.metric(
        "Composition share",
        f"{composition_share:+.0%}",
        help="Delivery-level decomposition: composition term divided by its pooled coefficient, on a different sample from the cell panel.",
    )

    section("Phase decomposition", "The same coefficient, estimated inside each phase")
    order = PHASE_ORDER[match_type]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.7))
    for ax, outcome, title in (
        (axes[0], "dot_share", "Dot-ball share"),
        (axes[1], "runs_per_ball", "Runs per ball"),
    ):
        subset = (
            format_phase[format_phase["outcome"].eq(outcome)]
            .set_index("phase")
            .reindex(order)
        )
        x = np.arange(len(subset))
        values = subset["coefficient"].to_numpy()
        errors = 1.96 * subset["std_error"].to_numpy()
        colors = ["#d97706" if value >= 0 else "#2563eb" for value in values]
        ax.bar(x, values, color=colors, width=0.62, alpha=0.9)
        ax.errorbar(
            x, values, yerr=errors, fmt="none", ecolor="#334155", capsize=4, lw=1
        )
        ax.axhline(0, color=PALETTE["axis"], linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([PHASE_LABEL[phase] for phase in order])
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.set_ylabel("Coefficient")
        clean_axes(ax)
    fig.tight_layout()
    show_figure(fig)
    st.caption(
        "Amber is positive, blue is negative. Runs per ball changes sign across phases "
        "in every format, and the T20 dot-share coefficient reverses too; a surface that "
        "wears in one direction cannot produce that pattern by itself."
    )

    section("Ball-age curves", "Descriptive curves behind the regression")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
    for ax, outcome, title, ci_low, ci_high in (
        (axes[0], "dot_share", "Dot-ball share", "dot_ci_low", "dot_ci_high"),
        (axes[1], "runs_per_ball", "Runs per ball", None, None),
    ):
        for style, color in (("spin", "#d97706"), ("pace", "#2563eb")):
            subset = format_curves[format_curves["bowler_style"].eq(style)].sort_values(
                "age_bin"
            )
            ax.plot(
                subset["age_bin"],
                subset[outcome],
                color=color,
                linewidth=2,
                label=style.title(),
            )
            if ci_low and ci_high:
                ax.fill_between(
                    subset["age_bin"],
                    subset[ci_low],
                    subset[ci_high],
                    color=color,
                    alpha=0.16,
                    linewidth=0,
                )
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("Ball age (overs)")
        ax.set_ylabel(outcome.replace("_", " ").title())
        ax.legend(frameon=False, fontsize=9)
        clean_axes(ax)
    fig.tight_layout()
    show_figure(fig)
    st.caption(
        "Bands are 95% binomial intervals on the dot share. These curves mix venues, "
        "phases and bowler quality; the regression panel is the identified object."
    )

    left, right = st.columns(2)
    with left:
        section("Venue terciles", "Age × spin coefficient by historical spin friendliness")
        subset = (
            format_het.set_index("spin_friendliness")
            .reindex(["low", "mid", "high"])
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(5.6, 3.4))
        x = np.arange(len(subset))
        values = subset["coefficient"].to_numpy()
        errors = 1.96 * subset["std_error"].to_numpy()
        ax.bar(
            x,
            values,
            color=[PALETTE["neutral"], PALETTE["muted"], "#d97706"],
            width=0.62,
        )
        ax.errorbar(
            x, values, yerr=errors, fmt="none", ecolor="#334155", capsize=4, lw=1
        )
        ax.axhline(0, color=PALETTE["axis"], linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(["Low", "Mid", "High"])
        ax.set_xlabel("Historical spin-wicket share tercile")
        ax.set_ylabel("Coefficient")
        clean_axes(ax)
        fig.tight_layout()
        show_figure(fig)
        st.caption(
            "Terciles are defined within the format from venue spin-wicket share "
            "(at least 5,000 balls per venue). They are not a property of the selected venue."
        )

    with right:
        if match_type == "Test":
            section("Second new ball", "Why the over-80 jump is not wear")
            ladder = dynamics["ladder"]
            labels = {
                "1_all_deliveries": "All deliveries",
                "2_no_continuing_bowler": "Attack turns over",
                "3_continuing_bowler_matches": "Same bowler continues",
            }
            subset = (
                ladder[
                    ladder["style"].eq("pace")
                    & ladder["boundary_bin"].eq("80_81")
                    & ladder["design"].eq("match_fe")
                    & ladder["step"].isin(list(labels))
                ]
                .set_index("step")
                .reindex(list(labels))
                .reset_index()
            )
            subset["label"] = subset["step"].map(labels)
            fig, ax = plt.subplots(figsize=(5.6, 3.4))
            x = np.arange(len(subset))
            values = subset["beta_pooled"].to_numpy()
            errors = 1.96 * subset["se_pooled"].to_numpy()
            ax.bar(
                x,
                values,
                color=["#d97706", "#d97706", PALETTE["neutral"]],
                width=0.62,
            )
            ax.errorbar(
                x, values, yerr=errors, fmt="none", ecolor="#334155", capsize=4, lw=1
            )
            ax.axhline(0, color=PALETTE["axis"], linewidth=1)
            ax.set_xticks(x)
            ax.set_xticklabels(subset["label"], fontsize=8)
            ax.set_ylabel("Pace dot-share jump at over 80")
            clean_axes(ax)
            fig.tight_layout()
            show_figure(fig)
            st.caption(
                "The jump vanishes when the same bowler continues, so it is the return "
                "of the frontline seamers rather than a surface change."
            )
        else:
            section("What this means", "A precise gradient with an attribution problem")
            format_runs = format_phase[
                format_phase["outcome"].eq("runs_per_ball")
            ].set_index("phase")
            runs_high = format_runs["coefficient"].idxmax()
            runs_low = format_runs["coefficient"].idxmin()
            dot_note = (
                " Dot share reverses as well."
                if
                format_phase[format_phase["outcome"].eq("dot_share")][
                    "coefficient"
                ].min()
                < 0
                < format_phase[format_phase["outcome"].eq("dot_share")][
                    "coefficient"
                ].max()
                else ""
            )
            st.markdown(
                f"""
                - **Precision:** the dot gradient is {dot_bias['bootstrap_z']:.1f}
                  bootstrap standard deviations from zero.
                - **Phase reversal:** runs per ball is
                  {format_runs.loc[runs_high, 'coefficient']:+.5f} in the
                  {PHASE_LABEL[runs_high].lower()} and
                  {format_runs.loc[runs_low, 'coefficient']:+.5f} in the
                  {PHASE_LABEL[runs_low].lower()}.{dot_note}
                - **Composition:** in the delivery-level decomposition the composition
                  term is {composition_share:+.0%} of its own coefficient.
                - **Persistence:** the per-match slope does not carry across innings.
                """
            )

    with st.expander("Pooled coefficients and the decomposition"):
        regression = dynamics["regression"]
        subset = regression[regression["match_type"].eq(match_type)].copy()
        subset = subset[subset["term"].eq("Extra change per over for spin")]
        st.dataframe(
            subset[
                ["outcome", "coefficient", "std_error", "ci_low", "ci_high", "p_value"]
            ].rename(
                columns={
                    "outcome": "Outcome",
                    "coefficient": "Age × spin coefficient",
                    "std_error": "Clustered SE",
                    "ci_low": "95% low",
                    "ci_high": "95% high",
                    "p_value": "p",
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "From `wear_regression.csv`, which covers runs per ball and wicket rate only. "
            "It is a separate specification from the headline cell-panel estimate and "
            "from the delivery-level decomposition, so the magnitudes are not directly "
            "comparable; the table is shown for completeness."
        )

# ============================= Model card ================================ #
with tab_model:
    section(
        "What the match model is",
        "XGBoost on pre-match context; the target is a rule-derived behavioural proxy",
    )
    st.caption(
        f"Deployment refit on {card['training_rows']:,} labelled matches "
        f"({card['training_start']} to {card['training_end']}), "
        f"{len(card['feature_columns'])} features, {window}-match venue window. "
        "Performance is not measured on this refit; it is the rolling-origin estimate below."
    )

    rolling = cv[cv["variant"].eq("xgb:ctx") & cv["match_type"].eq(match_type)].iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rolling macro-F1", f"{rolling['pooled_macro_f1']:.3f}")
    m2.metric(
        "95% interval",
        f"{rolling['pooled_macro_f1_ci_low']:.3f} to {rolling['pooled_macro_f1_ci_high']:.3f}",
    )
    m3.metric("Accuracy", f"{rolling['pooled_accuracy']:.3f}")
    m4.metric("Calibration (ECE)", f"{rolling['pooled_ece']:.3f}")

    left, right = st.columns(2)
    with left:
        section("Out-of-fold class report", "Rolling-origin predictions")
        predictions = load_predictions(match_type)
        report = classification_report(
            predictions["y_true"],
            predictions["y_pred"],
            labels=list(CLASS_ORDER),
            output_dict=True,
            zero_division=0,
        )
        report_table = pd.DataFrame(report).T.loc[list(CLASS_ORDER)][
            ["precision", "recall", "f1-score", "support"]
        ]
        report_table.index = [CLASS_SHORT[index] for index in report_table.index]
        report_table.columns = ["Precision", "Recall", "F1", "Support"]
        st.dataframe(
            report_table,
            width="stretch",
            column_config={
                "Precision": st.column_config.NumberColumn(format="%.3f"),
                "Recall": st.column_config.NumberColumn(format="%.3f"),
                "F1": st.column_config.NumberColumn(format="%.3f"),
                "Support": st.column_config.NumberColumn(format="%.0f"),
            },
        )
        st.caption(
            f"Out-of-fold n = {len(predictions):,}. "
            "The rolling folds train only on earlier matches."
        )

    with right:
        section("Confusion matrix", "Rows are observed, columns predicted")
        matrix = confusion_matrix(
            predictions["y_true"], predictions["y_pred"], labels=list(CLASS_ORDER)
        )
        matrix_table = pd.DataFrame(
            matrix,
            index=[f"Observed {label}" for label in CLASS_ORDER],
            columns=[f"Predicted {label}" for label in CLASS_ORDER],
        )
        st.dataframe(matrix_table, width="stretch")

    section("Calibration", "Does a 70% prediction happen 70% of the time?")
    confidence = predictions[
        ["probability_batting", "probability_pace", "probability_spin"]
    ].max(axis=1)
    correct = predictions["y_true"].eq(predictions["y_pred"])
    bins = np.linspace(0.3, 1.0, 8)
    bucket = pd.cut(confidence, bins=bins, include_lowest=True)
    calibration = (
        pd.DataFrame({"confidence": confidence, "correct": correct, "bucket": bucket})
        .groupby("bucket", observed=True)
        .agg(mean_confidence=("confidence", "mean"), accuracy=("correct", "mean"), n=("correct", "size"))
        .reset_index(drop=True)
    )
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.plot([0.3, 1.0], [0.3, 1.0], linestyle="--", color=PALETTE["muted"], linewidth=1)
    ax.plot(
        calibration["mean_confidence"],
        calibration["accuracy"],
        marker="o",
        color=PALETTE["axis"],
        linewidth=2,
    )
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_xlim(0.25, 1.02)
    ax.set_ylim(0.0, 1.02)
    clean_axes(ax)
    fig.tight_layout()
    show_figure(fig)
    st.caption(
        f"Points below the diagonal are over-confident. ECE = {rolling['pooled_ece']:.3f}. "
        "The y-axis starts at zero."
    )

    section("Global feature importance", "Gain-based, aggregated back to raw inputs")
    estimator = model.named_steps["model"]
    preprocessing = model.named_steps["preprocessing"]
    names = list(preprocessing.get_feature_names_out())
    importance = pd.Series(estimator.feature_importances_, index=names)
    grouped = importance.groupby(importance.index.map(raw_feature)).sum()
    grouped = grouped.sort_values(ascending=False).head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    ax.barh([pretty_feature(name) for name in grouped.index], grouped, color=PALETTE["axis"], height=0.62)
    ax.set_xlabel("Relative importance")
    clean_axes(ax)
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    show_figure(fig)
    st.caption(
        "Global importance says what the model uses across all matches; it does not "
        "explain a single prediction. Use the local drivers on the Match brief tab for that."
    )

    with st.expander("Deployment details"):
        st.json(
            {
                "match_type": card["match_type"],
                "venue_window": card["venue_window"],
                "features": len(card["feature_columns"]),
                "training_rows": card["training_rows"],
                "label_counts": card["label_counts"],
                "rolling_cv": card["rolling_cv"],
                "provenance": card["provenance"],
            }
        )

# ============================= Method ==================================== #
with tab_method:
    section(
        "What this prototype is — and is not",
        "A decision aid built on a proxy label and an audit, not a pitch sensor",
    )
    st.markdown(
        """
        The match model predicts **agreement with a rule-derived behavioural label**:
        *Batting*, *Pace* or *Spin*, assigned from first-innings run rate, early-over
        dot percentage and the spin share of bowler-credited wickets. Agreement with
        those rules is not evidence about the physical surface. The dynamics panel
        estimates a **relative** age × spin coefficient from legal spin and pace
        deliveries. It is not a wear rate in millimetres, degrees or revolutions.

        This is a Phase-I research prototype. It is not a forecast, a betting tool,
        a selection tool or a broadcast claim.
        """
    )

    section("Proxy label rules")
    st.dataframe(
        pd.DataFrame(
            [
                ["T20", 6, 8.5, 35.0],
                ["ODI", 10, 5.5, 60.0],
                ["Test", 10, 3.8, 75.0],
            ],
            columns=[
                "Format",
                "Early overs",
                "Batting if first-innings run rate ≥",
                "and early dot % ≤",
            ],
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Otherwise *Spin* if at least two spinner wickets were taken and the spin-to-pace "
        "wicket ratio is at least 1.25 (or no pace wickets were taken); otherwise *Pace*. "
        "Wides and no-balls are excluded from legal-ball denominators."
    )

    section("The dynamics estimand")
    st.markdown(
        """
        - **Unit:** an over × bowling-style cell inside a match innings (spin or pace).
        - **Outcomes:** runs per ball, dot share, wicket rate, boundary share.
        - **Regressor:** ball age × spin indicator, with match fixed effects (and a
          stricter match × innings specification), standard errors clustered by match.
        - **Sample:** 6,776,668 deliveries; 5,116,165 legal spin/pace deliveries;
          855,659 over × style cells.
        - **Interpretation:** how spin's outcomes move relative to pace's as the ball
          ages. Never a physical pitch measurement.
        """
    )

    section("The six checks")
    t20_dot = dynamics["decomposition"][
        dynamics["decomposition"]["match_type"].eq("T20")
        & dynamics["decomposition"]["outcome"].eq("dot")
        & dynamics["decomposition"]["scenario"].eq("smooth_gradient")
    ].iloc[0]
    t20_trim = dynamics["robustness"][
        dynamics["robustness"]["variant"].eq("drop_first_and_last_over")
        & dynamics["robustness"]["match_type"].eq("T20")
        & dynamics["robustness"]["outcome"].eq("runs_per_ball")
    ].iloc[0]
    t20_runs_before = dynamics["specifications"][
        dynamics["specifications"]["spec"].eq("match_fe")
        & dynamics["specifications"]["match_type"].eq("T20")
        & dynamics["specifications"]["outcome"].eq("runs_per_ball")
    ].iloc[0]
    t20_persist = dynamics["persistence"][
        dynamics["persistence"]["match_type"].eq("T20")
        & dynamics["persistence"]["outcome"].eq("runs_per_ball")
    ].iloc[0]
    t20_gain = dynamics["gain"][
        dynamics["gain"]["match_type"].eq("T20")
        & dynamics["gain"]["outcome"].eq("runs_per_ball")
    ].iloc[0]

    placebo_dot = dynamics["placebo"][
        dynamics["placebo"]["outcome"].eq("dot_share")
    ].set_index("match_type")
    z_text = " / ".join(f"{placebo_dot.loc[fmt, 'bootstrap_z']:.1f}" for fmt in FORMATS)
    phase_dot = dynamics["phase"][
        dynamics["phase"]["outcome"].eq("dot_share")
    ].set_index(["match_type", "phase"])
    t20_phase = phase_dot.loc["T20"]
    phase_runs = dynamics["phase"][
        dynamics["phase"]["outcome"].eq("runs_per_ball")
    ]
    reversing = [
        fmt
        for fmt in FORMATS
        if phase_runs[phase_runs["match_type"].eq(fmt)]["coefficient"].min() < 0
        < phase_runs[phase_runs["match_type"].eq(fmt)]["coefficient"].max()
    ]
    runs_text = (
        "all three formats" if len(reversing) == len(FORMATS) else ", ".join(reversing)
    )
    decomposition_dot = dynamics["decomposition"][
        dynamics["decomposition"]["outcome"].eq("dot")
    ].set_index("match_type")
    share = {
        fmt: decomposition_dot.loc[fmt, "composition"]
        / decomposition_dot.loc[fmt, "beta_pooled"]
        for fmt in FORMATS
    }

    audit = pd.DataFrame(
        [
            [
                "1. Correct null",
                "Permutation null is mis-centred; a null-imposed wild cluster bootstrap centres it at zero",
                f"Dot z = {z_text} (T20 / ODI / Test); permutation p-values are not used",
            ],
            [
                "2. Phase",
                "Estimate the same coefficient inside each phase",
                f"Runs per ball reverses sign in {runs_text}; T20 dot share "
                f"{t20_phase.loc['powerplay', 'coefficient']:+.4f} in the powerplay "
                f"and {t20_phase.loc['death', 'coefficient']:+.4f} at the death",
            ],
            [
                "3. Innings boundary",
                "Drop the first and last over of each innings",
                f"T20 runs {t20_runs_before['age_x_spin_coef']:+.5f} → "
                f"{t20_trim['coefficient']:+.5f} (p = {t20_trim['p_value']:.2f})",
            ],
            [
                "4. Bowler composition",
                "Exact variance-weighted within/between decomposition",
                f"Composition is {share['T20']:+.0%} of the T20 coefficient, "
                f"{share['ODI']:+.0%} in ODI and {share['Test']:+.0%} in Test",
            ],
            [
                "5. Persistence",
                "Correlate the first- and second-innings slopes",
                f"T20 r = {t20_persist['pearson_r']:+.3f} (p = {t20_persist['pearson_p']:.2f})",
            ],
            [
                "6. Prediction",
                "Add the per-innings slope to a held-out model",
                f"Relative MSE gain {t20_gain['relative_improvement'] * 100:.3f}% "
                f"(95% CI {t20_gain['ci_low'] * 100:+.3f}% to {t20_gain['ci_high'] * 100:+.3f}%)",
            ],
        ],
        columns=["Check", "What is done", "Result"],
    )
    st.dataframe(audit, hide_index=True, width="stretch")
    st.caption(
        "The verdict is not that the gradient is absent. It is that a precise gradient "
        "does not identify a surface process. The last candidate — the across-innings "
        "Test contrast — is reported as unresolved: its strict interval excludes the "
        "pooled magnitude but includes zero, and two thirds of it is composition."
    )

    section("Toss & line-ups")
    st.markdown(
        """
        - **Question.** If the selected side wins the toss, does batting or fielding
          first have the higher historical win share at this venue?
        - **Estimand.** Among decided matches at the venue with a known toss and a
          known result, the share of toss winners who won after choosing to bat,
          compared with the same share after choosing to field. Ties and no-results
          are excluded.
        - **Small samples.** Each venue rate is shrunk toward the format-wide rate
          for that decision with a Beta prior of 10 pseudo-matches, and the whiskers
          are 95% posterior intervals. A venue needs at least 25 decided matches
          behind each choice before the call uses venue history; otherwise it falls
          back to the format baseline and says so.
        - **Call rule.** The advisor says "no clear edge" when the posterior
          probability that one decision has the higher rate is below 95%, or the
          adjusted difference is under 2 points.
        - **Validation.** A prior-only backtest from 2010
          (`src/backtest_toss_advisor.py`) graded 15,653 matches. The rule made 260
          clear calls (1.7%). Across 89 forward windows at 8 venues the direction
          held 81% match-weighted but 50% venue-balanced; the format baseline scored
          52% on the same windows, and the venue-adjusted rates had a Brier skill of
          -1.6% (95% CI -1.9% to -1.2%) against the format baseline. This does not
          support the venue call as a prediction, so the panel is presented as
          descriptive history.
        - **Line-ups.** Player names and career records come from the Cricsheet
          scorecards and describe the selected XIs. They are not inputs to the toss
          estimate; no player-level win model is fitted.
        - **Not causal.** Toss decisions are made under conditions the data do not
          record, and the comparison does not adjust for team strength or era.
        """
    )

    section("Reproducibility")
    st.code(
        "\n".join(
            [
                "python src\\build_ball_dataset.py",
                "python src\\pitch_dynamics.py",
                "python src\\placebo_bias.py",
                "python src\\wear_robustness.py",
                "python src\\validate_wear.py",
                "python src\\identify_wear.py all",
                "python src\\decompose_wear.py all",
                "python src\\across_innings.py",
                "python src\\power_audit.py",
                "python src\\evaluate_rolling_cv.py --venue-windows T20=5,ODI=5,Test=10",
                "python src\\build_dashboard_data.py",
                "python src\\build_dashboard_model.py",
                "python src\\build_toss_dataset.py",
                "python src\\backtest_toss_advisor.py",
                "streamlit run dashboard/app.py",
            ]
        ),
        language="powershell",
    )

    section("Limitations")
    st.markdown(
        """
        - **Proxy labels.** The match model is graded against rule-derived classes, not
          against expert or instrumented pitch readings.
        - **Relative estimand.** The dynamics coefficient compares spin with pace; it
          does not measure the surface directly.
        - **Observational data.** Innings order, declarations and follow-ons are chosen
          in response to the game, so across-innings contrasts are not random.
        - **Coverage.** The match model uses matches with complete early-window data and
          known bowler styles; role coverage varies by format and era.
        - **Weather.** The app uses month climatology, not a match forecast.
        - **Toss advisor.** The bat/field comparison is a historical association at
          the venue, not a causal effect; it does not adjust for team strength, and
          ODI history mixes two ball-rule eras.
        """
    )

st.divider()
st.markdown(
    '<div class="small-muted">Phase-I prototype · 1RV25ET007 Aravind R · '
    "1RV25EC154 Rishab H R · 1RV25EC229 Ujwal S Nair · RVCE</div>",
    unsafe_allow_html=True,
)
