"""Risk model training, ablation study and clinical evaluation."""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from . import config as C
from .features import NON_FEATURES, feature_groups

PARAMS = dict(objective="binary", learning_rate=0.04, num_leaves=63, min_data_in_leaf=150,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0,
              verbose=-1, seed=C.SEED, num_threads=0)

ABLATIONS = {
    "EHR only": ["ehr", "context"],
    "Wearables only": ["cgm", "lifestyle", "context"],
    "EHR + wearables (fusion)": ["ehr", "cgm", "lifestyle", "context"],
    "MadhuTwin (fusion + physiology twin)": ["ehr", "cgm", "lifestyle", "context", "twin"],
}
FULL = "MadhuTwin (fusion + physiology twin)"


def split_patients(pids, seed=C.SEED):
    rng = np.random.default_rng(seed)
    p = np.array(sorted(pids))
    rng.shuffle(p)
    n = len(p)
    a, b = int(0.7 * n), int(0.85 * n)
    return {"train": sorted(p[:a]), "val": sorted(p[a:b]), "test": sorted(p[b:])}


def feature_cols(df, groups):
    g = feature_groups([c for c in df.columns if c not in NON_FEATURES])
    return [c for c, grp in g.items() if grp in groups]


def train_classifier(tr, va, cols, label="label", rounds=3000):
    dtr = lgb.Dataset(tr[cols], tr[label], free_raw_data=False)
    dva = lgb.Dataset(va[cols], va[label], reference=dtr)
    m = lgb.train(PARAMS, dtr, rounds, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    return m


def train_regressor(tr, va, cols, target, alpha=None):
    p = dict(PARAMS)
    p.update(objective="quantile", alpha=alpha) if alpha else p.update(objective="regression_l1")
    ytr = tr[target] - tr["g_now"]
    yva = va[target] - va["g_now"]
    ok_tr, ok_va = ytr.notna(), yva.notna()
    dtr = lgb.Dataset(tr.loc[ok_tr, cols], ytr[ok_tr])
    dva = lgb.Dataset(va.loc[ok_va, cols], yva[ok_va], reference=dtr)
    p.update(learning_rate=0.08)
    return lgb.train(p, dtr, 800, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])


def binary_metrics(y, s):
    return {"auroc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s)),
            "prevalence": float(np.mean(y))}


def choose_threshold(y, p):
    """Threshold that maximises F1 on the validation patients."""
    best, thr = -1, 0.5
    for t in np.linspace(0.05, 0.95, 91):
        pred = p >= t
        tp = np.sum(pred & (y == 1))
        f1 = 2 * tp / max(np.sum(pred) + np.sum(y == 1), 1)
        if f1 > best:
            best, thr = f1, t
    return round(float(thr), 2)


def event_level(df_pred, cgm_arrays, pids_index, thr):
    """Clinical view of alerting on held-out patients, computed ONLY from delivered alerts.

    Uses ``alerting.py`` (the same policy the console and API use): debounced to one alert per hour,
    suppressed when fewer than 9 of the last 12 CGM readings are present. Detection, lead time,
    alert precision and false alerts are all derived from that one alert stream, with one-to-one
    matching of alerts to excursion onsets in the 2 h window before each onset.
    """
    from . import alerting as AL
    H = C.HORIZON_MIN // C.CGM_INTERVAL
    start = C.CALIB_DAYS * 288
    per, all_leads = [], []
    n_alerts = n_true = n_events = 0
    days = 0.0
    for pid, g in df_pred.groupby("patient_id"):
        cgm = cgm_arrays[pids_index[pid]]
        n = len(cgm)
        risk = np.full(n, np.nan)
        risk[g.k.to_numpy()] = g.risk.to_numpy()
        alerts = AL.delivered_alerts(risk, cgm, thr, start=start)
        ev = AL.onsets(cgm)
        ev = ev[ev >= start + H]
        m = AL.match_events(alerts, ev, H, C.CGM_INTERVAL)
        leads = m["leads_min"]
        d = (n - start) / 288
        per.append((len(ev), int(np.sum(leads >= 30)), int(np.sum(~m["true_alert"])), d,
                    int(np.sum(np.isfinite(leads))), len(alerts)))
        all_leads += list(leads[np.isfinite(leads)])
        n_alerts += len(alerts)
        n_true += int(m["true_alert"].sum())
        n_events += len(ev)
        days += d
    per = np.array(per, float)
    leads = np.array(all_leads)
    rng = np.random.default_rng(0)
    bs_det, bs_fa = [], []
    for _ in range(1000):
        i = rng.integers(0, len(per), len(per))
        bs_det.append(per[i, 1].sum() / max(per[i, 0].sum(), 1))
        bs_fa.append(per[i, 2].sum() / per[i, 3].sum())
    return {
        "alert_policy": AL.POLICY_VERSION,
        "events": int(n_events),
        "detected_any_lead": float(np.sum(per[:, 4]) / max(n_events, 1)),
        "detected_lead_ge_30min": float(np.sum(leads >= 30) / max(n_events, 1)),
        "detected_lead_ge_60min": float(np.sum(leads >= 60) / max(n_events, 1)),
        "median_lead_min": float(np.median(leads)) if len(leads) else 0.0,
        "alerts_per_patient_day": float(n_alerts / days),
        "false_alerts_per_patient_day": float((n_alerts - n_true) / days),
        "alert_precision": float(n_true / max(n_alerts, 1)),
        "detected_lead_ge_30min_ci": [float(np.percentile(bs_det, 2.5)), float(np.percentile(bs_det, 97.5))],
        "false_alerts_per_patient_day_ci": [float(np.percentile(bs_fa, 2.5)), float(np.percentile(bs_fa, 97.5))],
    }


def reliability(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b)
        if m.sum() > 50:
            out.append({"bin": float((a + b) / 2), "pred": float(p[m].mean()), "obs": float(y[m].mean()),
                        "n": int(m.sum())})
    return out


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
