"""Export held-out patients, twin parameters, predictions and explanations
into ``dashboard/data.js`` for the doctor console (a static web app)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import alerting as AL
from . import config as C
from . import contract as CT
from . import ehr as E
from . import features as FE
from . import physiology as PH

N_SHOW = 20            # patients shown in the console
# clinical reason categories for plain-language explanations
REASONS = [
    ("pattern", "Usual pattern at this time of day", ("hist_spike_rate_tod", "g_yday_", "tod_", "weekend")),
    ("meal", "Meal due or still being absorbed", ("mins_to_habit", "habit_meal", "mins_since_meal", "last_meal", "carbs_logged", "twin_gut")),
    ("glucose", "Glucose level and trend now", ("g_now", "g_delta", "g_accel", "g_mean_1h", "g_std_1h", "g_max_3h", "g_min_3h", "mins_since_above")),
    ("twin", "Physiology twin projection", ("twin_proj", "twin_insulin_action", "twin_plasma", "twin_ex_boost")),
    ("sleep", "Sleep, stress and heart (HR, HRV)", ("hr_", "hrv_", "sleep_", "night_", "deep_frac", "wake_bouts", "asleep")),
    ("activity", "Physical activity", ("steps_", "mins_since_brisk")),
    ("baseline", "Patient's baseline (record, twin parameters)", ("ehr_", "twin_SI", "twin_sigma", "twin_Gb", "twin_dawn", "twin_fit", "twin_disposition", "g_mean_24h", "g_cv_24h", "tar_24h")),
]


def _reason_index(cols):
    idx = []
    for c in cols:
        hit = next((j for j, (_, _, pref) in enumerate(REASONS) if c.startswith(pref)), len(REASONS) - 1)
        idx.append(hit)
    return np.array(idx)
DAYS_SHOWN = 7         # last week of the record


def _ints(a, scale=1.0):
    a = np.asarray(a, float) * scale
    return [None if not np.isfinite(v) else int(round(v)) for v in a]


def export(A, ehr, tw, habits, te, model, cols, metrics, meal_log, split):
    hidden = pd.read_csv(C.DATA_DIR / "truth" / "cohort_hidden.csv").set_index("patient_id")
    obs = pd.read_csv(C.DATA_DIR / "ehr" / "observations.csv")
    conds = pd.read_csv(C.DATA_DIR / "ehr" / "conditions.csv")
    meds = pd.read_csv(C.DATA_DIR / "ehr" / "medications.csv")
    e = ehr.set_index("patient_id")
    twi = tw.set_index("patient_id")
    n = A["n5"]
    a = n - DAYS_SHOWN * 288
    thr = metrics["final_model"]["threshold"]

    # choose a spread of held-out patients: ranked by time-above-range, evenly sampled
    test = split["test"]
    tar = {p: np.nanmean(A["cgm"][A["pids"].index(p), a:] > 180) for p in test}
    ranked = sorted(test, key=lambda p: -tar[p])
    idx = np.linspace(0, len(ranked) - 1, N_SHOW).round().astype(int)
    show = [ranked[i] for i in idx]
    showcase = metrics.get("showcase", {}).get("patient_id")
    if showcase in test and showcase not in show:
        show[1] = showcase

    # canonical "highest glucose expected in the next 2 h": never below now or any forecast endpoint
    te = te.copy()
    te["peak_reconciled"] = np.nanmax(te[["g_now", "f30", "f60", "f120", "peak_pred"]].to_numpy(float), axis=1)
    alerts_by_pid = {}
    for pid_, g_ in te.groupby("patient_id"):
        i_ = A["pids"].index(pid_)
        r_ = np.full(n, np.nan)
        r_[g_.k.to_numpy()] = g_.risk.to_numpy()
        alerts_by_pid[pid_] = AL.delivered_alerts(r_, A["cgm"][i_], thr, start=C.CALIB_DAYS * 288)
    te["readings_last_hour"] = [AL.readings_last_hour(A["cgm"][A["pids"].index(p_)], int(k_)) for p_, k_ in zip(te.patient_id, te.k)]
    te["alert_delivered"] = [int(k_) in set(alerts_by_pid[p_].tolist()) for p_, k_ in zip(te.patient_id, te.k)]
    groups = FE.feature_groups(cols)
    sick_path = C.DATA_DIR / "processed" / "sickday_test.csv"
    sick = pd.read_csv(sick_path) if sick_path.exists() else pd.DataFrame(columns=["patient_id", "day"])
    day0 = a // 288
    patients = []
    for pid in show:
        i = A["pids"].index(pid)
        g = te[(te.patient_id == pid) & (te.k >= a)].sort_values("k")
        risk = np.full(n - a, np.nan)
        risk250 = np.full(n - a, np.nan)
        risk[g.k - a] = g.risk
        risk250[g.k - a] = g.risk250
        extra = {}
        for col, scale in (("risk_sd", 100), ("tte_pred", 1), ("peak_reconciled", 1), ("twin_gut_carbs_g", 1),
                           ("twin_insulin_action", 100), ("twin_proj_peak_habit", 1), ("twin_proj_peak", 1)):
            arr = np.full(n - a, np.nan)
            if col in g:
                arr[g.k - a] = g[col]
            extra[col] = _ints(arr, scale)
        fc = {}
        for h in (30, 60, 120):
            for suf in ("", "_q10", "_q90"):
                arr = np.full(n - a, np.nan)
                col = f"f{h}{suf}"
                if col in g:
                    arr[g.k - a] = g[col]
                fc[f"f{h}{suf}"] = _ints(arr)
        # explanations every 15 minutes: top-5 SHAP contributions
        sub = g[(g.k % 3) == 0]
        expl = {}
        why = {}
        ridx = _reason_index(cols)
        if len(sub):
            contrib = model.predict(sub[cols], pred_contrib=True)
            for row, kk in zip(contrib, sub.k.to_numpy()):
                why[int(kk - a)] = [round(float(row[:-1][ridx == j].sum()), 3) for j in range(len(REASONS))]
            for row, kk, vals in zip(contrib, sub.k.to_numpy(), sub[cols].to_numpy()):
                c = row[:-1]
                top = np.argsort(-np.abs(c))[:5]
                expl[int(kk - a)] = [[int(t), round(float(c[t]), 3),
                                      None if not np.isfinite(vals[t]) else round(float(vals[t]), 3)] for t in top]
        ml = meal_log[(meal_log.patient_id == pid) & (meal_log.t >= a * 5)]
        h = hidden.loc[pid]
        er = e.loc[pid]
        pm = meds[meds.PATIENT == pid]
        pc = conds[conds.PATIENT == pid]
        po = obs[obs.PATIENT == pid].sort_values("DATE")
        a1c_hist = po[po.KEY == "hba1c"][["DATE", "VALUE"]].values.tolist()
        hb = habits[habits.patient_id == pid]
        t_ = twi.loc[pid]
        patients.append({
            "id": pid, "name": h["name"], "age": int(h.age), "sex": h.sex, "region": h.region, "city": h.city,
            "ehr": {
                "hba1c": float(er.hba1c), "hba1c_hist": a1c_hist, "fpg": round(float(er.fpg)),
                "bmi": float(er.bmi), "waist": float(er.waist), "egfr": float(er.egfr),
                "homa_ir": round(float(er.homa_ir), 2), "homa_b": round(float(er.homa_b)),
                "tg": round(float(er.tg)), "hdl": round(float(er.hdl)), "ldl": round(float(er.ldl)),
                "sbp": round(float(er.sbp)), "dbp": round(float(er.dbp)), "duration": float(er.diabetes_duration_y),
                "tcf7l2": int(er.gen_tcf7l2), "kcnj11": int(er.gen_kcnj11), "fto": int(er.gen_fto),
                "prs": float(er.gen_prs), "weight": float(er.weight),
                "meds": pm.DESCRIPTION.tolist(), "med_classes": pm.MED_CLASS.tolist(),
                "dx": [d for d in pc.DESCRIPTION.tolist()],
            },
            "twin": {"SI": float(t_.twin_SI), "sigma": float(t_.twin_sigma), "Gb": float(t_.twin_Gb),
                     "dawn": float(t_.twin_dawn), "V": float(t_.twin_V), "sglt2": int(t_.twin_sglt2),
                     "rmse": round(float(t_.twin_fit_rmse), 1)},
            "habits": [{"slot": r.slot, "time": round(r.time), "carbs": round(r.carbs), "gi": round(r.gi),
                        "freq": round(r.freq, 2)} for r in hb.itertuples()],
            "cgm": _ints(A["cgm"][i, a:]), "hr": _ints(A["hr"][i, a:]), "hrv": _ints(A["hrv"][i, a:]),
            "steps": _ints(A["steps"][i, a:]), "stage": _ints(A["stage"][i, a:]),
            "risk": _ints(risk, 100), "risk250": _ints(risk250, 100), "fc": fc, "expl": expl, "why": why,
            "risk_sd": extra["risk_sd"], "tte": extra["tte_pred"], "peak": extra["peak_reconciled"],
            "alerts": [int(k - a) for k in alerts_by_pid.get(pid, []) if k >= a],
            "gut": extra["twin_gut_carbs_g"], "ins": extra["twin_insulin_action"], "tpeak": extra["twin_proj_peak_habit"],
            "nightly": ([[int(d - a // 288), round(float(A["nightly"]["si_mult"][i, d]), 2), round(float(A["nightly"]["gb_delta"][i, d]), 1)]
                         for d in range(A["nightly"]["si_mult"].shape[1]) if np.isfinite(A["nightly"]["si_mult"][i, d])]
                        if "nightly" in A else []),
            "meals": [[int(r.t // 5 - a), r.food, int(r.carbs_g), int(r.gi)] for r in ml.itertuples()],
            "sick": [[int(r.day - day0), round(float(r.anomaly_score), 2), round(float(np.nan_to_num(r.night_hr_dev)), 1),
                      round(float(np.nan_to_num(r.night_hrv_dev)), 1)]
                     for r in sick[sick.patient_id == pid].sort_values("day").itertuples() if r.day >= day0 - 6],
        })

    data = {
        "generated": pd.Timestamp.now().isoformat(timespec="minutes"),
        "start": (pd.Timestamp(C.START_DATE) + pd.Timedelta(minutes=a * 5)).isoformat(),
        "step_min": 5, "threshold": thr, "features": cols, "alert_policy": AL.POLICY_VERSION,
        "model_version": CT.MODEL_VERSION, "min_readings_last_hour": AL.MIN_READINGS_LAST_HOUR,
        "safety_note": CT.SAFETY_NOTE, "uncertainty_note": CT.UNCERTAINTY_NOTE, "reasons": [[k, lab] for k, lab, _ in REASONS],
        "feature_groups": [groups[c] for c in cols],
        "patients": patients,
        "foods": [{"key": f.key, "name": f.name, "slot": f.slot, "carbs": f.carbs, "gi": f.gi, "healthy": f.healthy}
                  for f in C.FOODS],
        "physio": {"P2": PH.P2, "N_INS": PH.N_INS, "ALPHA": PH.ALPHA, "K_FAST": PH.K_FAST, "K_SLOW": PH.K_SLOW,
                   "F_BIO": PH.F_BIO, "K_EX": PH.K_EX, "RENAL_THR": PH.RENAL_THR, "RENAL_K": PH.RENAL_K,
                   "SGLT2_THR": PH.SGLT2_THR, "SGLT2_K": PH.SGLT2_K, "TAU_ISF": PH.TAU_ISF, "SG": 0.016},
        "metrics": {k: metrics[k] for k in ("final_model", "event_level", "ablation", "baselines", "forecast",
                                            "twin_recovery", "whatif_validation", "level2_250", "operating_points",
                                            "cold_start_first_48h", "cohort", "split", "sick_day", "real_world",
                                            "subgroups", "twin_value", "uncertainty", "time_to_event", "twin_maturity",
                                            "nightly_tracking", "genetics_ablation", "latency", "whatif_twin_prob") if k in metrics},
    }
    # compact store for the REST API (feature rows of the console patients)
    meta = ["alert_delivered", "readings_last_hour"]
    store = te[te.patient_id.isin(show)][["patient_id", "k"] + [c for c in cols if c != "g_now"] + ["g_now"] + meta].copy()
    num = [c for c in store.columns if c not in ["patient_id", "k"] + meta]
    store[num] = store[num].astype("float32")
    store.to_parquet(C.DATA_DIR / "processed" / "api_store.parquet", index=False)
    # small committed fixture: lets the API end-to-end + contract tests run in CI without the full pipeline.
    # expected values are the console's exported numbers for the same rows (API must reproduce them).
    fx_rows = []
    for pid in show[:3]:
        gp = te[(te.patient_id == pid) & (te.k >= a)].sort_values("k")
        pick = pd.concat([gp[gp.alert_delivered].head(4), gp.iloc[::400]]).drop_duplicates("k").sort_values("k").head(10)
        fx_rows.append(pick)
    fx = pd.concat(fx_rows)
    fdir = C.ROOT / "tests" / "fixtures"
    fdir.mkdir(parents=True, exist_ok=True)
    fxs = store.merge(fx[["patient_id", "k"]], on=["patient_id", "k"])
    fxs.to_parquet(fdir / "api_store_small.parquet", index=False)
    pidx_show = {q["id"]: q for q in patients}
    expected = []
    for r_ in fx.itertuples():
        q = pidx_show[r_.patient_id]
        j = int(r_.k - a)
        expected.append({"patient_id": r_.patient_id, "step": int(r_.k), "risk_pct": q["risk"][j], "risk_sd_pct": q["risk_sd"][j],
                         "forecast_120": q["fc"]["f120"][j], "forecast_30": q["fc"]["f30"][j], "peak": q["peak"][j],
                         "tte": q["tte"][j], "twin_peak": q["tpeak"][j], "alert": bool(j in q["alerts"]),
                         "readings_last_hour": int(r_.readings_last_hour)})
    (fdir / "expected_console_values.json").write_text(json.dumps(expected, indent=1))
    # sample FHIR R4 bundle for the first console patient at its last alert
    from . import fhir as FH
    p0 = patients[0]
    k_alert = max([k for k, r in enumerate(p0["risk"]) if r is not None and r >= thr * 100] or [0])
    ex = p0["expl"].get(k_alert - (k_alert % 3), [])
    drivers = [f"{cols[i]}={v} ({'raises' if c > 0 else 'lowers'} risk)" for i, c, v in ex]
    when = pd.Timestamp(C.START_DATE) + pd.Timedelta(minutes=(a + k_alert) * 5)
    b = FH.bundle(p0["id"], when.to_pydatetime(), p0["cgm"][k_alert], (p0["risk"][k_alert] or 0) / 100, drivers,
                  p0["twin"], thr)
    (C.ROOT / "docs" / "fhir_sample_bundle.json").write_text(FH.dumps(b))
    # what-if probability: map the twin's projected 2-h peak to the chance of an excursion,
    # fitted on validation patients only (a transparent 2-parameter logistic curve)
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        fv = pd.read_parquet(C.DATA_DIR / "processed" / "features.parquet", columns=["patient_id", "label", "twin_proj_peak_habit"])
        fv = fv[fv.patient_id.isin(split["val"])].dropna()
        lr = LogisticRegression().fit(fv[["twin_proj_peak_habit"]], fv.label)
        data["metrics"]["whatif_twin_prob"] = {"a": float(lr.intercept_[0]), "b": float(lr.coef_[0][0]),
                                               "auroc_val": float(roc_auc_score(fv.label, lr.predict_proba(fv[["twin_proj_peak_habit"]])[:, 1]))}
    except Exception as exc:  # pragma: no cover
        print("what-if probability mapping skipped:", exc)
    data["metrics"]["reliability"] = metrics["final_model"].get("reliability", [])
    data["metrics"]["final_model"] = {k: v for k, v in data["metrics"]["final_model"].items() if k != "reliability"}
    C.DASHBOARD_DIR.mkdir(exist_ok=True)
    with open(C.DASHBOARD_DIR / "data.js", "w") as f:
        f.write("window.MADHU_DATA = ")
        json.dump(data, f, separators=(",", ":"))
        f.write(";\n")
