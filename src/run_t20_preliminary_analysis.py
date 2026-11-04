"""Preliminary T20I analysis that runs without the blocked inputs.

Runs README steps that do not need `all_json.zip` or `match_weather.csv`:
bowler-role table + review queue, data checks, threshold sensitivity, venue
behaviour profiles, and a chronological no-weather model comparison.

Usage (repo root):  python src/run_t20_preliminary_analysis.py
Results are PRELIMINARY: T20 internationals only, no weather, role-complete
matches only. Metrics measure agreement with proxy labels, not pitch physics.
"""
from __future__ import annotations
import json, re, subprocess, sys, tempfile, warnings, zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import train_pitch_models as T
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

warnings.filterwarnings("ignore")
ROOT = Path(".")
RAW = ROOT / "data/raw"
ARCHIVE = RAW / "cricsheet/t20s_json.zip"
PROC = ROOT / "data/processed"
MODELS = ROOT / "models/pitch_proxy_t20i_preliminary"
CL = list(T.CLASS_ORDER)
BOUNDARY = {4, 6}

def _norm(s): return T._normalise_name(s)
def _initial_key(name):
    parts = _norm(name).split()
    if len(parts) < 2: return None
    return (parts[0][0], parts[-1])

def role_tables():
    roles, audit = T.build_player_roles(RAW/"cricsheet/people.csv", RAW/"cricsheet/names.csv",
        RAW/"kaggle/international_player_stats.zip", RAW/"kaggle/worldwide_player_profiles.zip")
    wk = defaultdict(int); ids = defaultdict(set)
    per_match = []
    with zipfile.ZipFile(ARCHIVE) as z:
        for n in sorted(x for x in z.namelist() if x.endswith(".json")):
            m = json.load(z.open(n)); reg = m["info"].get("registry", {}).get("people", {})
            bs = set()
            for inn in m.get("innings", []):
                for o in inn.get("overs", []):
                    for d in o.get("deliveries", []):
                        for w in d.get("wickets", []):
                            if str(w.get("kind", "")).lower() in T.WICKET_DISMISSALS:
                                pid = reg.get(d["bowler"]); wk[pid] += 1
                                ids[d["bowler"]].add(pid); bs.add(pid)
            per_match.append((Path(n).stem, bs))
    id2name = {p: b for b, s in ids.items() for p in s}
    rows = [{"bowler": b, "cricsheet_id": p, "style": roles.get(p)} for b, s in ids.items() for p in s]
    t = pd.DataFrame(rows); ambiguous = t.groupby("bowler").cricsheet_id.transform("nunique") > 1
    print("ambiguous bowler names (same name, different Cricsheet IDs):", t.loc[ambiguous, "bowler"].nunique())
    out = t[t["style"].notna() & ~ambiguous].sort_values("bowler")
    (ROOT/"data").mkdir(exist_ok=True)
    out[["bowler", "style"]].to_csv(ROOT/"data/bowler_roles.csv", index=False)

    # Review queue: never auto-applied. Candidates are hints for a human to verify.
    wp = pd.read_csv(zipfile.ZipFile(RAW/"kaggle/worldwide_player_profiles.zip").open("players_data_with_all_info.csv"), dtype=str)
    wp = wp[wp.bowlingstyle.notna()]
    by_full = defaultdict(set); by_init = defaultdict(set)
    for r in wp.itertuples():
        c = f"{r.fullname} [{r.country_name}]: {r.bowlingstyle}"
        by_full[_norm(r.fullname)].add(c)
        k = _initial_key(r.fullname)
        if k: by_init[k].add(c)
    q = [{"bowler": id2name[p], "cricsheet_id": p, "credited_t20i_wickets": n,
          "candidates": " | ".join(sorted(by_full.get(_norm(id2name[p])) or by_init.get(_initial_key(id2name[p]), set()))[:6]),
          "verified_style": "", "source_url": ""}
         for p, n in wk.items() if p not in roles and p]
    q = pd.DataFrame(q).sort_values("credited_t20i_wickets", ascending=False)
    (ROOT/"data/interim").mkdir(parents=True, exist_ok=True)
    q.to_csv(ROOT/"data/interim/bowler_role_review_queue.csv", index=False)

    # How much would curating the top-N unresolved bowlers help?
    order = q.cricsheet_id.tolist(); curve = []
    base = set(roles)
    for N in (0, 50, 100, 200, 400, 800, len(order)):
        known = base | set(order[:N])
        curve.append({"top_n_curated": N,
            "wicket_coverage": sum(v for k, v in wk.items() if k in known) / sum(wk.values()),
            "role_complete_matches": sum(all(b in known for b in bs) for _, bs in per_match)})
    pd.DataFrame(curve).to_csv(PROC/"role_curation_curve.csv", index=False)
    return roles, audit, out, q, curve, wk

def run_phase12(all_matches):
    # phase1_phase2.py joins roles by name and stops on any unknown wicket-taker, by
    # design. Give it only matches whose credited wicket-takers are all in the table.
    keep = set(all_matches.loc[all_matches.role_complete, "match_id"])
    names = set(pd.read_csv(ROOT/"data/bowler_roles.csv").bowler)
    with tempfile.TemporaryDirectory() as td, zipfile.ZipFile(ARCHIVE) as z:
        n = 0
        for name in z.namelist():
            if name.endswith(".json") and Path(name).stem in keep:
                m = json.load(z.open(name))
                ok = all(d["bowler"] in names for i in m.get("innings", []) for o in i.get("overs", []) for d in o.get("deliveries", [])
                         for w in d.get("wickets", []) if str(w.get("kind", "")).strip().lower() in T.WICKET_DISMISSALS)
                if ok:
                    (Path(td)/name).write_bytes(z.read(name)); n += 1
        r = subprocess.run([sys.executable, "src/phase1_phase2.py", "--input", td,
            "--bowler-roles", "data/bowler_roles.csv", "--output-dir", str(PROC)], capture_output=True, text=True)
    if r.returncode: raise RuntimeError(r.stderr[-1500:])
    return n, r.returncode, r.stderr[-600:]

def label(rr, dot, sp, pa, rr_min=8.5, dot_max=35.0, ratio=1.25, min_spin=2):
    bat = (rr >= rr_min) & (dot <= dot_max)
    spin = (sp >= min_spin) & ((pa == 0) | (sp / pa.replace(0, np.nan) >= ratio))
    return np.where(bat, "Batting", np.where(spin, "Spin", "Pace"))

def sensitivity(lab):
    base = label(lab.first_innings_run_rate, lab.early_dot_percent, lab.spin_wickets, lab.pace_wickets)
    rows = []
    grid = [("Baseline", {})] + [(f"Run-rate min {v}", {"rr_min": v}) for v in (8.0, 9.0)] + \
        [(f"Early dot% max {v}", {"dot_max": v}) for v in (30.0, 40.0)] + \
        [(f"Spin:pace ratio {v}", {"ratio": v}) for v in (1.0, 1.5)] + [("Min spin wickets 3", {"min_spin": 3})]
    for n, kw in grid:
        y = label(lab.first_innings_run_rate, lab.early_dot_percent, lab.spin_wickets, lab.pace_wickets, **kw)
        rows.append({"scenario": n, **{c: float((y == c).mean()) for c in CL}, "labels_changed": float((y != base).mean())})
    d = pd.DataFrame(rows); d.to_csv(PROC/"label_threshold_sensitivity.csv", index=False); return d

def venue_profiles(roles):
    rows = []
    with zipfile.ZipFile(ARCHIVE) as z:
        for n in (x for x in z.namelist() if x.endswith(".json")):
            m = json.load(z.open(n)); i = m["info"]; reg = i.get("registry", {}).get("people", {})
            if str(i.get("outcome", {}).get("result", "")).lower() == "no result": continue
            inns = m.get("innings", [])
            if len(inns) < 2: continue
            s = [dict(r=0, b=0, f=0, w=0) for _ in range(2)]; sp = pa = un = 0
            for k, inn in enumerate(inns[:2]):
                for o in inn.get("overs", []):
                    for d in o.get("deliveries", []):
                        e = d.get("extras", {}); s[k]["r"] += d["runs"]["total"]
                        if "wides" not in e and "noballs" not in e: s[k]["b"] += 1
                        if d["runs"].get("batter", 0) in BOUNDARY: s[k]["f"] += 1
                        for w in d.get("wickets", []):
                            s[k]["w"] += 1
                            if str(w.get("kind", "")).lower() in T.WICKET_DISMISSALS:
                                st = roles.get(reg.get(d["bowler"]))
                                sp += st == "spin"; pa += st == "pace"; un += st is None
            if min(s[0]["b"], s[1]["b"]) < 60: continue
            rows.append(dict(match_id=Path(n).stem, date=i["dates"][0], gender=i.get("gender"), venue=i.get("venue"),
                venue_id=i.get("registry", {}).get("venues", {}).get(i.get("venue"), ""), city=i.get("city", ""),
                runs=s[0]["r"]+s[1]["r"], balls=s[0]["b"]+s[1]["b"], bnd=s[0]["f"]+s[1]["f"], wk=s[0]["w"]+s[1]["w"],
                rr1=s[0]["r"]*6/s[0]["b"], rr2=s[1]["r"]*6/s[1]["b"], spin=sp, pace=pa, unknown=un))
    d = pd.DataFrame(rows)
    d.to_csv(PROC/"match_behaviour_stats.csv", index=False)
    g = d.groupby(["gender", "venue_id"]).agg(venue=("venue", lambda s: s.value_counts().index[0]), city=("city", lambda s: s.value_counts().index[0]),
        matches=("runs", "size"), runs=("runs", "sum"), balls=("balls", "sum"), bnd=("bnd", "sum"), wk=("wk", "sum"),
        rr1=("rr1", "mean"), rr2=("rr2", "mean"), spin=("spin", "sum"), pace=("pace", "sum"), unknown=("unknown", "sum")).reset_index()
    g["run_rate"] = g.runs*6/g.balls; g["boundary_pct"] = 100*g.bnd/g.balls; g["wickets_per_20_overs"] = g.wk*120/g.balls
    g["spin_wicket_share"] = g.spin/(g.spin+g.pace).replace(0, np.nan); g["role_known_share"] = (g.spin+g.pace)/(g.spin+g.pace+g.unknown).replace(0, np.nan)
    g["raw_second_minus_first_rr"] = g.rr2-g.rr1
    g.drop(columns=["runs", "balls", "bnd", "wk"]).sort_values("matches", ascending=False).to_csv(PROC/"venue_behaviour_profiles.csv", index=False)
    return d, g

def preproc(num, cat, scale=False):
    steps = [("imp", SimpleImputer(strategy="median", add_indicator=True))] + ([("sc", StandardScaler())] if scale else [])
    return ColumnTransformer([("numeric", Pipeline(steps), num),
        ("categorical", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=True))]), cat)])

def fit_all(tr, te, num, cat):
    cols = num+cat; ytr = tr.y.astype(str); out = {}
    p = np.tile([(ytr == c).mean() for c in CL], (len(te), 1)); out["Majority class"] = p
    # Venue prior: smoothed class shares at the venue from training rows only.
    glob = np.array([(ytr == c).mean() for c in CL]); vp = {}
    for v, s in tr.groupby("venue_id"): vp[v] = (np.array([(s.y == c).sum() for c in CL]) + 5*glob) / (len(s)+5)
    out["Venue prior"] = np.vstack([vp.get(v, glob) for v in te.venue_id])
    sw = compute_sample_weight("balanced", ytr)
    lr = Pipeline([("p", preproc(num, cat, True)), ("m", LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5))]).fit(tr[cols], ytr)
    out["Logistic regression"] = lr.predict_proba(te[cols])[:, [list(lr.classes_).index(c) for c in CL]]
    rf = Pipeline([("p", preproc(num, cat)), ("m", RandomForestClassifier(n_estimators=500, min_samples_leaf=3, class_weight="balanced_subsample", random_state=42, n_jobs=2))]).fit(tr[cols], ytr)
    out["Random forest"] = rf.predict_proba(te[cols])[:, [list(rf.classes_).index(c) for c in CL]]
    x = T._make_pipeline(num, cat, "cpu")
    x.fit(tr[cols], ytr.map({c: i for i, c in enumerate(CL)}).to_numpy(), model__sample_weight=sw)
    out["XGBoost"] = x.predict_proba(te[cols])
    return out, x

def score(y, p):
    pred = np.array(CL)[p.argmax(1)]
    s = T._score_predictions(y.astype(str), pred, p)
    return s

def main():
    PROC.mkdir(parents=True, exist_ok=True); MODELS.mkdir(parents=True, exist_ok=True)
    res = {"scope": "T20 internationals in t20s_json.zip; no weather; role-complete matches only; preliminary"}
    roles, audit, role_csv, queue, curve, wk = role_tables()
    audit.to_csv(MODELS/"player_role_join_audit.csv", index=False)
    all_m, lab = T._parse_matches(ARCHIVE, roles)
    n, rc, err = run_phase12(all_m)
    res["phase1_phase2"] = {"input_matches": n, "returncode": rc, "stderr_tail": err}
    y12 = pd.read_csv(PROC/"y.csv", dtype={"match_id": str}); tcol = [c for c in y12.columns if c != "match_id"][0]
    cmp = lab[["match_id", "y"]].rename(columns={"y": "train_y"}).merge(y12, on="match_id"); res["phase1_vs_train_label_agreement"] = {"compared": len(cmp), "agree": float((cmp.train_y == cmp[tcol]).mean())}
    res["data"] = {"archive_matches": 5849, "parsed_complete_early_window": len(all_m), "date_min": str(all_m.date.min().date()), "date_max": str(all_m.date.max().date()),
        "role_complete": int(all_m.role_complete.sum()), "labelled": len(lab), "labelled_fraction": len(lab)/len(all_m),
        "by_gender": all_m.groupby("gender").agg(parsed=("y", "size"), labelled=("y", "count")).to_dict("index"),
        "class_counts": lab.y.value_counts().to_dict(), "class_share_by_gender": pd.crosstab(lab.gender, lab.y, normalize="index").to_dict("index"),
        "labelled_by_year": lab.groupby(lab.date.dt.year).size().to_dict(),
        "venue_strings": int(all_m.venue.nunique()), "venue_ids": int(all_m.venue_id.nunique()),
        "bowlers_with_credited_wicket": int(len(wk)), "role_table_rows": len(role_csv),
        "wicket_coverage": sum(v for k, v in wk.items() if k in roles)/sum(wk.values()), "role_curation_curve": curve}
    res["missing"] = {c: int(all_m[c].isna().sum()) for c in ("date", "venue", "toss_decision", "first_innings_run_rate", "early_dot_percent")}
    lab.sample(25, random_state=7)[["match_id", "date", "venue", "first_innings_run_rate", "early_dot_percent", "spin_wickets", "pace_wickets", "y"]].to_csv(PROC/"audit_sample_for_manual_review.csv", index=False)
    sens = sensitivity(lab); res["sensitivity"] = sens.to_dict("records")
    _, g = venue_profiles(roles)

    # Features: stub weather so train_pitch_models.build_features can be reused, then drop it.
    all_m["spin_wicket_share"] = np.nan
    stub = PROC/"_stub_weather.csv"; pd.DataFrame({"match_id": all_m.match_id, "temperature_2m_mean": np.nan}).to_csv(stub, index=False)
    fe = T.build_features(all_m, stub).drop(columns="temperature_2m_mean"); stub.unlink()
    d = lab.merge(fe, on=["match_id", "date", "match_type"], suffixes=("_l", "")).sort_values(["date", "match_id"])
    excl = {"match_id", "date", "match_type", "y"}
    fc = [c for c in fe.columns if c not in excl and d[c].notna().any()]
    fc = [c for c in fc if d[c].nunique(dropna=False) > 1]
    cat = [c for c in ("venue", "venue_id", "city", "gender", "team_type", "toss_decision") if c in fc]; num = [c for c in fc if c not in cat]
    tr, te, split = T._chronological_split(d); res["split"] = split; res["n_train"] = len(tr); res["n_test"] = len(te)
    res["features"] = fc; res["test_classes"] = te.y.value_counts().to_dict(); res["train_classes"] = tr.y.value_counts().to_dict()
    # venue_id may have been dropped by the uniqueness filter; venue prior needs it.
    probs, xp = fit_all(tr, te, num, cat)
    rng = np.random.default_rng(42); ytrue = te.y.astype(str).to_numpy(); tab = []; res["models"] = {}
    for name, p in probs.items():
        s = score(te.y, p); res["models"][name] = s
        idx = [rng.integers(0, len(te), len(te)) for _ in range(1000)]
        from sklearn.metrics import f1_score
        pred = np.array(CL)[p.argmax(1)]
        boots = [f1_score(ytrue[i], pred[i], labels=CL, average="macro", zero_division=0) for i in idx]
        tab.append(dict(model=name, accuracy=s["accuracy"], balanced_accuracy=s["balanced_accuracy"], macro_f1=s["macro_f1"],
            macro_f1_ci_low=np.percentile(boots, 2.5), macro_f1_ci_high=np.percentile(boots, 97.5), log_loss=s["multiclass_log_loss"], brier=s["multiclass_brier_score"]))
    pd.DataFrame(tab).to_csv(MODELS/"model_comparison.csv", index=False)
    pd.DataFrame({"match_id": te.match_id.values, "date": te.date.dt.strftime("%Y-%m-%d").values, "y_true": ytrue, "y_pred": np.array(CL)[probs["XGBoost"].argmax(1)],
        **{f"p_{c.lower()}": probs["XGBoost"][:, i] for i, c in enumerate(CL)}}).to_csv(MODELS/"xgboost_test_predictions.csv", index=False)

    # Calibration of the top-class probability.
    for name in ("XGBoost", "Random forest", "Logistic regression"):
        p = probs[name]; conf = p.max(1); ok = np.array(CL)[p.argmax(1)] == ytrue; rows = []
        for lo, hi in ((0, .5), (.5, .6), (.6, .7), (.7, .8), (.8, 1.01)):
            m = (conf >= lo) & (conf < hi)
            if m.sum(): rows.append(dict(bin=f"{lo:.1f}-{min(hi,1):.1f}", n=int(m.sum()), mean_conf=float(conf[m].mean()), accuracy=float(ok[m].mean())))
        res.setdefault("calibration", {})[name] = rows
        res.setdefault("ece", {})[name] = float(sum(r["n"]*abs(r["mean_conf"]-r["accuracy"]) for r in rows)/len(te))

    # Expanding-window, time-aware CV (abstract's evaluation design).
    dates = np.array(sorted(d.date.dt.normalize().unique())); folds = []
    for k in range(5):
        a = dates[int(len(dates)*(.5+.1*k))]; b = dates[int(len(dates)*(.6+.1*k))-1] if k < 4 else dates[-1]
        trf = d[d.date <= a]; tef = d[(d.date > a) & (d.date <= b)]
        pr, _ = fit_all(trf, tef, num, cat)
        for name, p in pr.items():
            s = score(tef.y, p); folds.append(dict(fold=k+1, train=len(trf), test=len(tef), model=name, macro_f1=s["macro_f1"], balanced_accuracy=s["balanced_accuracy"], accuracy=s["accuracy"]))
    fd = pd.DataFrame(folds); fd.to_csv(MODELS/"time_cv_folds.csv", index=False)
    res["time_cv"] = fd.groupby("model")[["macro_f1", "balanced_accuracy", "accuracy"]].agg(["mean", "std"]).round(4).to_json()

    # Explainability: mean |SHAP| via XGBoost's built-in TreeSHAP, grouped to source feature.
    prep = xp.named_steps["preprocessing"]; bo = xp.named_steps["model"].get_booster()
    X = prep.transform(te[num+cat]); names = prep.get_feature_names_out()
    c = bo.predict(xgb.DMatrix(X), pred_contribs=True)  # n, class, feat+1
    imp = np.abs(c[:, :, :-1]).mean(axis=(0, 1))
    s = pd.Series(imp, index=names)
    def grp(n):
        n = n.split("__", 1)[1]
        n = n.replace("missingindicator_", "")
        if n in num: return n
        for k in sorted(cat, key=len, reverse=True):
            if n.startswith(k+"_"): return k
        return n
    sg = s.groupby([grp(n) for n in names]).sum().sort_values(ascending=False)
    sg.rename("mean_abs_shap").to_csv(MODELS/"xgboost_shap_importance.csv"); res["shap_importance"] = sg.round(4).to_dict()

    # Same model with no gender feature: is the score only gender separation?
    if "gender" in cat:
        c2 = [x for x in cat if x != "gender"]
        pr2, _ = fit_all(tr, te, num, c2); res["no_gender_xgb_macro_f1"] = score(te.y, pr2["XGBoost"])["macro_f1"]; res["no_gender_xgb_bal_acc"] = score(te.y, pr2["XGBoost"])["balanced_accuracy"]
    for gn in ("male", "female"):
        mk = te.gender.values == gn
        res.setdefault("xgb_by_gender", {})[gn] = dict(n=int(mk.sum()), accuracy=float((np.array(CL)[probs["XGBoost"][mk].argmax(1)] == ytrue[mk]).mean()), classes=te[mk].y.value_counts().to_dict())

    json.dump(res, open(MODELS/"preliminary_results.json", "w"), indent=2, default=str)
    print(json.dumps({k: v for k, v in res.items() if k not in ("models",)}, indent=1, default=str)[:9000])
    print(pd.DataFrame(tab).round(3).to_string()); print(np.array(res["models"]["XGBoost"]["confusion_matrix"]))
    print(json.dumps(res["models"]["XGBoost"]["classification_report"], indent=0)[:700])

main()
