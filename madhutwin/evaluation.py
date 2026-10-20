"""Evaluation helpers: Clarke error grid, patient-level bootstrap
confidence intervals, split-conformal prediction intervals and subgroup audits."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


# ----------------------------------------------------------------------------- Clarke error grid
def clarke_zones(ref, pred):
    """Clarke error grid zone (A-E) for each (reference, prediction) pair in mg/dL.

    A: clinically accurate; B: benign errors; C: over-correction; D: dangerous
    failure to detect; E: erroneous treatment (Clarke et al., Diabetes Care 1987).
    """
    ref = np.asarray(ref, float)
    pred = np.asarray(pred, float)
    z = np.full(ref.shape, "B", dtype="<U1")
    a = ((ref <= 70) & (pred <= 70)) | ((pred <= 1.2 * ref) & (pred >= 0.8 * ref))
    e = ((ref >= 180) & (pred <= 70)) | ((ref <= 70) & (pred >= 180))
    c = (((ref >= 70) & (ref <= 290)) & (pred >= ref + 110)) | \
        (((ref >= 130) & (ref <= 180)) & (pred <= (7 / 5) * ref - 182))
    d = ((ref >= 240) & (pred >= 70) & (pred <= 180)) | \
        ((ref <= 175 / 3) & (pred <= 180) & (pred >= 70)) | \
        ((ref >= 175 / 3) & (ref <= 70) & (pred >= (6 / 5) * ref))
    z[d] = "D"
    z[c] = "C"
    z[e] = "E"
    z[a] = "A"
    return z


def clarke_summary(ref, pred):
    ok = np.isfinite(ref) & np.isfinite(pred)
    z = clarke_zones(np.asarray(ref)[ok], np.asarray(pred)[ok])
    out = {k: float(np.mean(z == k)) for k in "ABCDE"}
    out["A+B"] = out["A"] + out["B"]
    out["n"] = int(ok.sum())
    return out


# ----------------------------------------------------------------------------- bootstrap CIs
def patient_bootstrap(df, score_col, label_col="label", group_col="patient_id", n_boot=300, seed=0,
                      metrics=("auroc", "auprc")):
    """95% CI by resampling *patients* with replacement (rows within a patient are correlated)."""
    rng = np.random.default_rng(seed)
    groups = {g: idx for g, idx in df.groupby(group_col).indices.items()}
    keys = list(groups)
    y = df[label_col].to_numpy()
    s = df[score_col].to_numpy()
    fns = {"auroc": roc_auc_score, "auprc": average_precision_score}
    vals = {m: [] for m in metrics}
    for _ in range(n_boot):
        pick = rng.choice(len(keys), len(keys), replace=True)
        idx = np.concatenate([groups[keys[i]] for i in pick])
        if y[idx].min() == y[idx].max():
            continue
        for m in metrics:
            vals[m].append(fns[m](y[idx], s[idx]))
    out = {}
    for m in metrics:
        v = np.array(vals[m])
        out[m] = {"point": float(fns[m](y, s)), "lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5))}
    return out


def bootstrap_mean(values_by_patient: dict, n_boot=1000, seed=0):
    """CI for a per-patient rate (e.g. false alerts/day) by resampling patients."""
    rng = np.random.default_rng(seed)
    v = np.array(list(values_by_patient.values()), float)
    bs = [np.mean(rng.choice(v, len(v), replace=True)) for _ in range(n_boot)]
    return {"point": float(v.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}


# ----------------------------------------------------------------------------- conformal intervals
def conformal_offsets(y_cal, lo_cal, hi_cal, alpha=0.2):
    """Split-conformalised quantile regression (Romano et al., 2019): returns the
    margin q to add to both ends so that [lo - q, hi + q] covers 1-alpha."""
    y_cal, lo_cal, hi_cal = map(np.asarray, (y_cal, lo_cal, hi_cal))
    ok = np.isfinite(y_cal) & np.isfinite(lo_cal) & np.isfinite(hi_cal)
    s = np.maximum(lo_cal[ok] - y_cal[ok], y_cal[ok] - hi_cal[ok])
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    return float(np.sort(s)[min(k, n - 1)])


# ----------------------------------------------------------------------------- subgroup audit
def subgroup_auroc(df, score_col, groups: dict, label_col="label", min_pos=30):
    """AUROC within clinically / socially meaningful subgroups."""
    rows = []
    for name, mask in groups.items():
        d = df[mask]
        if d[label_col].sum() < min_pos or (d[label_col] == 0).sum() < min_pos:
            continue
        ci = patient_bootstrap(d, score_col, label_col, n_boot=200, metrics=("auroc",))["auroc"]
        rows.append({"subgroup": name, "patients": int(d.patient_id.nunique()), "rows": int(len(d)),
                     "prevalence": float(d[label_col].mean()),
                     "auroc": float(roc_auc_score(d[label_col], d[score_col])),
                     "auroc_lo": ci["lo"], "auroc_hi": ci["hi"],
                     "auprc": float(average_precision_score(d[label_col], d[score_col]))})
    return pd.DataFrame(rows)


def paired_bootstrap_delta(df, score_a, score_b, label_col="label", group_col="patient_id", n_boot=300, seed=0):
    """CI for AUROC/AUPRC(b) - AUROC/AUPRC(a), resampling patients (paired)."""
    rng = np.random.default_rng(seed)
    groups = list(df.groupby(group_col).indices.values())
    y, a, b = df[label_col].to_numpy(), df[score_a].to_numpy(), df[score_b].to_numpy()
    out = {}
    for name, fn in (("auroc", roc_auc_score), ("auprc", average_precision_score)):
        pt = fn(y, b) - fn(y, a)
        bs = []
        for _ in range(n_boot):
            idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            if y[idx].min() == y[idx].max():
                continue
            bs.append(fn(y[idx], b[idx]) - fn(y[idx], a[idx]))
        out[name] = {"delta": float(pt), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}
    return out


def expected_calibration_error(y, p, bins=10):
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b) if b < 1 else (p >= a) & (p <= b)
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)
