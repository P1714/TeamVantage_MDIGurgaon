"""Sick-day early warning from a week of heart-rate, HRV and sleep data.

The challenge brief's second example adverse event is "flagging a high-risk
anomaly based on a week of sleep and heart-rate data". For a person with
diabetes the most actionable such anomaly is an **intercurrent illness**:
infections raise resting heart rate, suppress HRV, disturb sleep and, through
stress hormones, push glucose up for days. Clinicians answer it with
"sick-day rules" (more frequent monitoring, hydration, pausing SGLT2
inhibitors / metformin when dehydrated, ketone checks).

Each morning at 12:00 the twin compares last night with the patient's own
previous seven nights (a rolling personal baseline) and produces two scores:

* an **unsupervised anomaly score**: a sum of personal z-scores. It needs no
  labelled illness, so it works from week 2 for any patient.
* a **supervised score**: LightGBM on the same daily features, trained on
  training patients only.

Ground truth is the simulator's hidden illness days, which the twin never sees.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config as C
from . import features as FE

PER_DAY = 288


def daily_features(A):
    """One row per patient-day (day >= 1), using data up to 12:00 of that day."""
    P, n = A["P"], A["n5"]
    D = n // PER_DAY
    stage, hr, hrv, steps, cgm = A["stage"], A["hr"], A["hrv"], A["steps"], A["cgm"]
    rows = []
    for d in range(1, D):
        a, b = (d - 1) * PER_DAY + 216, d * PER_DAY + 144          # 18:00 previous day -> 12:00 today
        asleep = stage[:, a:b] > 0
        with np.errstate(all="ignore"):
            nhr = np.nanmean(np.where(asleep, hr[:, a:b], np.nan), 1)
            nhrv = np.nanmean(np.where(asleep, hrv[:, a:b], np.nan), 1)
            slp = asleep.sum(1) * 5 / 60
            wake = ((asleep[:, 1:].astype(int) - asleep[:, :-1].astype(int)) == -1).sum(1)
            # yesterday's daytime activity and glucose
            y0, y1 = (d - 1) * PER_DAY + 96, (d - 1) * PER_DAY + 264
            steps_y = np.nansum(steps[:, y0:y1], 1)
            day_hr = np.nanmean(np.where(stage[:, y0:y1] == 0, hr[:, y0:y1], np.nan), 1)
            g24 = np.nanmean(cgm[:, b - PER_DAY:b], 1)
        rows.append(pd.DataFrame({"pi": np.arange(P), "day": d, "night_hr": nhr, "night_hrv": nhrv, "sleep_h": slp,
                                  "awakenings": wake, "steps_yday": steps_y, "day_hr_yday": day_hr, "cgm_mean_24h": g24}))
    df = pd.concat(rows, ignore_index=True).sort_values(["pi", "day"])
    base_cols = ["night_hr", "night_hrv", "sleep_h", "awakenings", "steps_yday", "day_hr_yday", "cgm_mean_24h"]
    g = df.groupby("pi")
    for c in base_cols:
        roll = g[c].transform(lambda s: s.shift(1).rolling(7, min_periods=2).median())
        sd = g[c].transform(lambda s: s.shift(1).rolling(7, min_periods=3).std())
        df[f"{c}_dev"] = df[c] - roll
        df[f"{c}_z"] = (df[c] - roll) / (sd + {"night_hr": 1.5, "night_hrv": 2.0, "sleep_h": 0.4, "awakenings": 1.0,
                                               "steps_yday": 800, "day_hr_yday": 1.5, "cgm_mean_24h": 6}[c])
    df["patient_id"] = np.array(A["pids"])[df.pi]
    df["anomaly_score"] = (df.night_hr_z.fillna(0) - df.night_hrv_z.fillna(0) + df.day_hr_yday_z.fillna(0)
                           - 0.5 * df.steps_yday_z.fillna(0) + 0.5 * df.cgm_mean_24h_z.fillna(0))
    return df


def evaluate(A, split, threshold_z=4.0):
    df = daily_features(A)
    truth = pd.read_csv(C.DATA_DIR / "truth" / "daily_context.csv")
    df = df.merge(truth[["patient_id", "day", "ill", "stress"]], on=["patient_id", "day"], how="left")
    df["ill"] = df.ill.astype(bool).astype(int)
    df = df[df.day >= 3]                                    # need a few nights of personal baseline
    feats = [c for c in df.columns if c.endswith(("_z", "_dev"))] + ["night_hr", "night_hrv", "sleep_h"]
    tr = df[df.patient_id.isin(split["train"] + split["val"])]
    te = df[df.patient_id.isin(split["test"])].copy()
    m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=40, verbose=-1,
                       seed=C.SEED, feature_fraction=0.9), lgb.Dataset(tr[feats], tr.ill), 300)
    te["supervised_score"] = m.predict(te[feats])
    all_ = df.copy()
    res = {"days_evaluated": int(len(te)), "ill_days": int(te.ill.sum()), "test_patients": int(te.patient_id.nunique()),
           "unsupervised_auroc_all_patients": float(roc_auc_score(all_.ill, all_.anomaly_score)),
           "unsupervised_auroc_test": float(roc_auc_score(te.ill, te.anomaly_score)),
           "supervised_auroc_test": float(roc_auc_score(te.ill, te.supervised_score)),
           "supervised_auprc_test": float(average_precision_score(te.ill, te.supervised_score)),
           "prevalence_test": float(te.ill.mean())}
    flag = te.anomaly_score >= threshold_z
    res["flag_threshold_z"] = threshold_z
    res["flag_sensitivity"] = float(flag[te.ill == 1].mean())
    res["flag_false_flags_per_patient_week"] = float(flag[te.ill == 0].sum() / te.patient_id.nunique() / (te.day.nunique() / 7))
    # does a flagged day predict worse glucose today? (time above 180 between 12:00 and 24:00)
    tar = []
    for r in te.itertuples():
        i = A["pids"].index(r.patient_id)
        seg = A["cgm"][i, r.day * PER_DAY + 144:(r.day + 1) * PER_DAY]
        tar.append(np.nanmean(seg > 180))
    te["tar_rest_of_day"] = tar
    res["tar_rest_of_day_flagged"] = float(te.loc[flag, "tar_rest_of_day"].mean())
    res["tar_rest_of_day_not_flagged"] = float(te.loc[~flag, "tar_rest_of_day"].mean())
    te[["patient_id", "day", "ill", "anomaly_score", "supervised_score", "night_hr_dev", "night_hrv_dev",
        "sleep_h", "tar_rest_of_day"]].to_csv(C.DATA_DIR / "processed" / "sickday_test.csv", index=False)
    return res, df
