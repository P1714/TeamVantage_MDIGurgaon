"""Real-world validation on open CGM + clinical datasets.

Synthetic patients let us check the twin against a hidden truth, but a jury will
rightly ask: *does any of this work on real people?* This module re-runs the
risk and forecast models on two open, CC-BY licensed cohorts of **real** people
across the dysglycaemia spectrum (normoglycaemia → prediabetes → T2D), each with
CGM traces *and* clinical/lab records:

* **Hall et al. 2018, PLOS Biology ("Glucotypes")** – 57 adults, ~7 days of
  Dexcom CGM; age, BMI, HbA1c, fasting glucose, OGTT, fasting insulin, SSPG
  (gold-standard insulin resistance), lipids, hs-CRP, diagnosis.
* **Colas et al. 2019, PLOS ONE** – 208 adults at risk of T2D, ~2 days of CGM;
  sex, age, BMI, fasting glycaemia, HbA1c.

Pre-processed copies come from the GlucoBench repository (Sergazinov et al.,
ICLR 2024). Neither dataset has meal logs or smartwatch data, so here we test
the **CGM + EHR fusion arm** of MadhuTwin. Because most participants are not
yet diabetic, the adverse event is a ≥140 mg/dL excursion lasting ≥15 min
within 2 h (the post-prandial threshold used for prediabetes). A ≥180 mg/dL
analysis is reported where events allow.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from . import config as C
from . import evaluation as EV
from . import features as FE

REAL = C.DATA_DIR / "real"
MAX_DAYS = 14
PER_DAY = 288
H = 24
EHR_COLS = ["age", "male", "bmi", "hba1c", "fpg", "ogtt_2h", "fasting_insulin", "sspg", "tg", "hdl", "ldl", "crp"]
PARAMS = dict(objective="binary", learning_rate=0.04, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, verbose=-1, seed=C.SEED, num_threads=0)


# ----------------------------------------------------------------------------- data preparation
def prepare(glucobench_raw_dir) -> None:
    """Convert GlucoBench raw CSVs into compact files under data/real/ (run once)."""
    src = Path(glucobench_raw_dir)
    REAL.mkdir(parents=True, exist_ok=True)
    h = pd.read_csv(src / "hall.csv")
    c = pd.read_csv(src / "colas.csv", index_col=0)
    h_static = h.groupby("id").first().reset_index()
    hs = pd.DataFrame({"patient_id": "HALL-" + h_static["id"].astype(str), "dataset": "Hall 2018",
                       "age": h_static.Age, "male": np.nan, "bmi": h_static.BMI, "hba1c": h_static.A1C,
                       "fpg": h_static.FBG, "ogtt_2h": h_static["ogtt.2hr"], "fasting_insulin": h_static.insulin,
                       "sspg": h_static.SSPG, "tg": h_static.Trg, "hdl": h_static.HDL, "ldl": h_static.LDL,
                       "crp": h_static["hs.CRP"], "diagnosis": h_static.diagnosis.map({0: "normoglycaemic",
                                                                                    1: "prediabetes", 2: "T2D"})})
    c_static = c.groupby("id").first().reset_index()
    cs = pd.DataFrame({"patient_id": "COLAS-" + c_static["id"].astype(str), "dataset": "Colas 2019",
                       "age": c_static.age, "male": (c_static.gender == 1).astype(float), "bmi": c_static.BMI,
                       "hba1c": c_static.HbA1c, "fpg": c_static.glycaemia})
    for col in EHR_COLS:
        cs[col] = cs.get(col, np.nan)
    # diagnosis at baseline from labs (ADA): HbA1c >=6.5 or FPG >=126 -> T2D; >=5.7 / >=100 -> prediabetes
    a1c, fpg = cs.hba1c, cs.fpg
    cs["diagnosis"] = np.where((a1c >= 6.5) | (fpg >= 126), "T2D",
                               np.where((a1c >= 5.7) | (fpg >= 100), "prediabetes", "normoglycaemic"))
    static = pd.concat([hs, cs], ignore_index=True)
    static = static.replace(-1, np.nan)
    static.to_csv(REAL / "clinical.csv", index=False)
    ts = pd.concat([pd.DataFrame({"patient_id": "HALL-" + h["id"].astype(str), "time": h.time, "gl": h.gl}),
                    pd.DataFrame({"patient_id": "COLAS-" + c["id"].astype(str), "time": c.time, "gl": c.gl})])
    ts["time"] = pd.to_datetime(ts.time)
    ts.to_parquet(REAL / "cgm.parquet", index=False)


def load_arrays():
    static = pd.read_csv(REAL / "clinical.csv")
    ts = pd.read_parquet(REAL / "cgm.parquet")
    pids = static.patient_id.tolist()
    n = MAX_DAYS * PER_DAY
    cgm = np.full((len(pids), n), np.nan)
    tod = np.zeros((len(pids), n))
    dow = np.zeros((len(pids), n))
    for i, pid in enumerate(pids):
        d = ts[ts.patient_id == pid].sort_values("time")
        t0 = d.time.iloc[0].floor("5min")
        k = ((d.time - t0).dt.total_seconds() // 300).astype(int).to_numpy()
        ok = k < n
        cgm[i, k[ok]] = d.gl.to_numpy()[ok]
        grid = t0 + pd.to_timedelta(np.arange(n) * 5, unit="min")
        tod[i] = grid.hour * 60 + grid.minute
        dow[i] = grid.dayofweek
    return pids, static, cgm, tod, dow


# ----------------------------------------------------------------------------- features
def build(thr=140):
    pids, static, cgm, tod, dow = load_arrays()
    P, n = cgm.shape
    g = FE._ffill(cgm, limit=2)
    F = {"g_now": g}
    for m in (1, 3, 6, 12):
        F[f"g_delta_{m * 5}m"] = g - FE._lag(g, m)
    F["g_accel"] = (g - FE._lag(g, 3)) - (FE._lag(g, 3) - FE._lag(g, 6))
    F["g_mean_1h"] = FE._roll(cgm, 12, "mean")
    F["g_std_1h"] = FE._roll(cgm, 12, "std", 3)
    F["g_max_3h"] = FE._roll(cgm, 36, "max")
    F["g_min_3h"] = FE._roll(cgm, 36, "min")
    F["g_mean_24h"] = FE._roll(cgm, PER_DAY, "mean", 36)
    F["g_cv_24h"] = FE._roll(cgm, PER_DAY, "std", 36) / F["g_mean_24h"]
    F["frac_above_24h"] = FE._roll((cgm >= thr).astype(float), PER_DAY, "mean", 36)
    F["mins_since_above"] = np.minimum(FE._since(cgm >= thr), 2880)
    F["g_yday_same"] = FE._lag(g, PER_DAY)
    fut_max = pd.DataFrame(cgm[:, ::-1].T).rolling(H, min_periods=1).max().to_numpy().T[:, ::-1]
    F["g_yday_max_next2h"] = FE._lag(fut_max, PER_DAY)
    first = np.argmax(~np.isnan(cgm), axis=1)
    F["hours_of_history"] = np.maximum(0, (np.arange(n)[None, :] - first[:, None]) * 5 / 60)
    # Colas timestamps keep true intervals but not the true clock time (GlucoBench re-based
    # each trace to midnight), so clock features are unknown (NaN) for that cohort.
    known = np.array([p.startswith("HALL") for p in pids])[:, None]
    F["tod_sin"] = np.where(known, np.sin(2 * np.pi * tod / 1440), np.nan)
    F["tod_cos"] = np.where(known, np.cos(2 * np.pi * tod / 1440), np.nan)
    F["weekend"] = np.where(known, (dow >= 5).astype(float), np.nan)

    run = FE._run_mask(cgm, thr)
    run180 = FE._run_mask(cgm, 180)
    y = FE._future_any(run, H)
    y180 = FE._future_any(run180, H)
    # personal time-of-day history (previous days only)
    D = n // PER_DAY
    Y = np.where(~np.isnan(g) & (g < thr), y.astype(float), np.nan).reshape(P, D, PER_DAY)
    s_ = np.nancumsum(Y, 1)
    c_ = np.cumsum(~np.isnan(Y), 1)
    ps = np.concatenate([np.zeros((P, 1, PER_DAY)), s_[:, :-1]], 1)
    pc = np.concatenate([np.zeros((P, 1, PER_DAY)), c_[:, :-1]], 1)
    sm = lambda z: np.apply_along_axis(lambda v: np.convolve(np.r_[v[-6:], v, v[:6]], np.ones(13), "valid"), 2, z)
    with np.errstate(all="ignore"):
        F["hist_spike_rate_tod"] = (sm(ps) / sm(pc)).reshape(P, n)

    fut_nan = pd.DataFrame(np.isnan(cgm)[:, ::-1].T.astype(float)).rolling(H, min_periods=1).mean().to_numpy().T[:, ::-1]
    cover = np.concatenate([fut_nan[:, 1:], np.ones((P, 1))], 1)
    k_all = np.arange(n)
    rows = []
    for i in range(P):
        ok = (k_all >= 36) & (k_all < n - H) & ~np.isnan(g[i]) & (g[i] < thr) & (cover[i] < 0.5)
        rows.append((np.full(ok.sum(), i), k_all[ok]))
    ip = np.concatenate([r[0] for r in rows])
    ik = np.concatenate([r[1] for r in rows])
    df = pd.DataFrame({"patient_id": np.array(pids)[ip], "k": ik, "label": y[ip, ik].astype(int),
                       "label180": y180[ip, ik].astype(int)})
    for h in (30, 60, 120):
        df[f"y_g{h}"] = np.concatenate([cgm[:, h // 5:], np.full((P, h // 5), np.nan)], 1)[ip, ik]
    for key, arr in F.items():
        df[key] = np.asarray(arr, float)[ip, ik]
    st = static.set_index("patient_id")
    for col in EHR_COLS:
        df[f"ehr_{col}"] = st.loc[df.patient_id, col].to_numpy(float)
    df["dataset"] = st.loc[df.patient_id, "dataset"].to_numpy()
    df["diagnosis"] = st.loc[df.patient_id, "diagnosis"].to_numpy()
    return df, static


CGM_COLS = ["g_now", "g_delta_5m", "g_delta_15m", "g_delta_30m", "g_delta_60m", "g_accel", "g_mean_1h", "g_std_1h",
            "g_max_3h", "g_min_3h", "g_mean_24h", "g_cv_24h", "frac_above_24h", "mins_since_above", "g_yday_same",
            "g_yday_max_next2h", "hist_spike_rate_tod", "hours_of_history"]
CLOCK = ["tod_sin", "tod_cos", "weekend"]
EHR = [f"ehr_{c}" for c in EHR_COLS]
SETS = {"Clock only": CLOCK, "EHR only": EHR + CLOCK, "CGM only": CGM_COLS + CLOCK,
        "CGM + EHR (MadhuTwin fusion)": CGM_COLS + EHR + CLOCK}


def _fit(tr, cols, label, objective="binary", rounds=400):
    p = dict(PARAMS)
    if objective != "binary":
        p.update(objective=objective)
    return lgb.train(p, lgb.Dataset(tr[cols], tr[label]), rounds)


def evaluate(df, n_splits=5):
    """Patient-level, dataset-stratified cross-validation; every prediction is out-of-fold."""
    pats = df.groupby("patient_id").dataset.first()
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=C.SEED)
    oof = {name: np.zeros(len(df)) for name in SETS}
    oof["fusion180"] = np.zeros(len(df))
    fc = {h: np.full(len(df), np.nan) for h in (30, 60, 120)}
    fc_cgm = {h: np.full(len(df), np.nan) for h in (30, 60, 120)}
    strata = df.dataset.to_numpy()
    for tr_i, te_i in cv.split(df, strata, groups=df.patient_id):
        tr, te = df.iloc[tr_i], df.iloc[te_i]
        for name, cols in SETS.items():
            oof[name][te_i] = _fit(tr, cols, "label").predict(te[cols])
        oof["fusion180"][te_i] = _fit(tr, SETS["CGM + EHR (MadhuTwin fusion)"], "label180").predict(te[SETS["CGM + EHR (MadhuTwin fusion)"]])
        for h in (30, 60, 120):
            ok = tr[f"y_g{h}"].notna()
            t2 = tr[ok].assign(_d=tr.loc[ok, f"y_g{h}"] - tr.loc[ok, "g_now"])
            for store, cols in ((fc, SETS["CGM + EHR (MadhuTwin fusion)"]), (fc_cgm, SETS["CGM only"])):
                m = _fit(t2, cols, "_d", objective="regression_l1", rounds=300)
                store[h][te_i] = te.g_now.to_numpy() + m.predict(te[cols])
    res = {"cohort": {"patients": int(pats.size), "by_dataset": pats.value_counts().to_dict(),
                      "by_diagnosis": df.groupby("patient_id").diagnosis.first().value_counts().to_dict(),
                      "decision_points": int(len(df)), "prevalence_140": float(df.label.mean()),
                      "prevalence_180": float(df.label180.mean())}}
    d = df.assign(**{f"s_{k}": v for k, v in oof.items()})
    trend = d.g_now + d.g_delta_30m.fillna(0) * 2
    d["s_trend"] = trend
    res["auroc_140"] = {"CGM trend extrapolation (baseline)": EV.patient_bootstrap(d, "s_trend")}
    for name in SETS:
        res["auroc_140"][name] = EV.patient_bootstrap(d, f"s_{name}")
    res["auroc_180_fusion"] = {"auroc": float(roc_auc_score(d.label180, d.s_fusion180)),
                               "auprc": float(average_precision_score(d.label180, d.s_fusion180)),
                               "prevalence": float(d.label180.mean())}
    # value of the EHR when CGM history is short (cold start)
    strat = {}
    for lab, mask in {"first 24 h of CGM": d.hours_of_history < 24, "after 24 h": d.hours_of_history >= 24}.items():
        dd = d[mask]
        strat[lab] = {name: float(roc_auc_score(dd.label, dd[f"s_{name}"])) for name in ("CGM only", "CGM + EHR (MadhuTwin fusion)")}
        strat[lab]["rows"] = int(mask.sum())
    res["by_history"] = strat
    by_dx = {}
    for dx, dd in d.groupby("diagnosis"):
        if dd.label.sum() > 30:
            by_dx[dx] = {"patients": int(dd.patient_id.nunique()),
                         "auroc_fusion": float(roc_auc_score(dd.label, dd["s_CGM + EHR (MadhuTwin fusion)"])),
                         "auroc_cgm_only": float(roc_auc_score(dd.label, dd["s_CGM only"]))}
    res["by_diagnosis"] = by_dx
    fcr = {}
    for h in (30, 60, 120):
        y = d[f"y_g{h}"].to_numpy()
        ok = np.isfinite(y)
        fcr[f"{h}min"] = {"mae_fusion": float(np.mean(np.abs(fc[h][ok] - y[ok]))),
                          "mae_cgm_only": float(np.mean(np.abs(fc_cgm[h][ok] - y[ok]))),
                          "mae_persistence": float(np.mean(np.abs(d.g_now.to_numpy()[ok] - y[ok]))),
                          "clarke": EV.clarke_summary(y[ok], fc[h][ok])}
    res["forecast"] = fcr
    d["f30"], d["f60"] = fc[30], fc[60]
    keep = ["patient_id", "k", "label", "label180", "dataset", "diagnosis", "hours_of_history", "g_now", "y_g30",
            "y_g60", "f30", "f60", "s_trend"] + [f"s_{k}" for k in SETS]
    d[keep].to_parquet(C.DATA_DIR / "processed" / "realworld_oof.parquet", index=False)
    return res, d


def insulin_resistance(n_boot=2000):
    """Real-world check of the twin's *fusion-for-physiology* claim: estimate insulin
    resistance measured by the gold-standard insulin suppression test (SSPG, Hall
    cohort) from EHR labs alone, CGM alone, or both (leave-one-out ridge)."""
    from scipy.stats import spearmanr
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import LeaveOneOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    st = pd.read_csv(REAL / "clinical.csv")
    h = st[st.dataset == "Hall 2018"].dropna(subset=["sspg"]).set_index("patient_id")
    cg = pd.read_parquet(REAL / "cgm.parquet")
    cg = cg[cg.patient_id.isin(h.index)]
    f = cg.groupby("patient_id").gl.agg(cgm_mean="mean", cgm_sd="std", cgm_p90=lambda x: x.quantile(0.9),
                                        cgm_time_above_140=lambda x: (x >= 140).mean(), cgm_max="max")
    f["cgm_cv"] = f.cgm_sd / f.cgm_mean
    X = h.join(f)
    X["homa_ir"] = X.fpg * X.fasting_insulin / 405
    X["tg_hdl"] = X.tg / X.hdl
    y = X.sspg.to_numpy()
    ehr = ["bmi", "fpg", "fasting_insulin", "homa_ir", "tg_hdl", "hba1c", "age"]
    cgm = ["cgm_mean", "cgm_sd", "cgm_p90", "cgm_time_above_140", "cgm_max", "cgm_cv"]
    rng = np.random.default_rng(C.SEED)
    out = {"n_patients": int(len(X)), "outcome": "SSPG (steady-state plasma glucose, insulin suppression test)"}
    for name, cols in {"EHR labs only": ehr, "CGM only": cgm, "EHR + CGM (fused)": ehr + cgm}.items():
        pred = np.zeros(len(X))
        for tr, te in LeaveOneOut().split(X):
            m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))
            m.fit(X.iloc[tr][cols], y[tr])
            pred[te] = m.predict(X.iloc[te][cols])
        bs = []
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            bs.append(spearmanr(pred[i], y[i])[0])
        out[name] = {"spearman": float(spearmanr(pred, y)[0]), "lo": float(np.nanpercentile(bs, 2.5)),
                     "hi": float(np.nanpercentile(bs, 97.5))}
    return out


def run(verbose=True):
    df, static = build()
    res, d = evaluate(df)
    res["insulin_resistance"] = insulin_resistance()
    with open(C.REPORT_DIR / "realworld_metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    if verbose:
        for k, v in res["auroc_140"].items():
            print(f"  real-world {k:34s} AUROC {v['auroc']['point']:.3f} [{v['auroc']['lo']:.3f}-{v['auroc']['hi']:.3f}]"
                  f"  AUPRC {v['auprc']['point']:.3f}")
        print("  by history:", res["by_history"])
        print("  insulin resistance (SSPG):", {k: v for k, v in res["insulin_resistance"].items()})
        print("  forecast:", {h: (round(v['mae_fusion'], 1), round(v['mae_persistence'], 1), round(v['clarke']['A+B'], 3))
                              for h, v in res["forecast"].items()})
    return res, d
