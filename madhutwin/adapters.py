"""Adapters between MadhuTwin's 5-minute table and a HealthKit / Health Connect
style long export (one row per sample: type, start, end, value, unit).

This is the seam where real consumer-wearable exports would enter the twin.
"""
from __future__ import annotations

import pandas as pd

TYPES = {
    "cgm_mgdl": ("HKQuantityTypeIdentifierBloodGlucose", "mg/dL"),
    "hr_bpm": ("HKQuantityTypeIdentifierHeartRate", "count/min"),
    "hrv_rmssd_ms": ("HKQuantityTypeIdentifierHeartRateVariabilityRMSSD", "ms"),
    "steps": ("HKQuantityTypeIdentifierStepCount", "count"),
    "sleep_stage": ("HKCategoryTypeIdentifierSleepAnalysis", "stage"),
}
SLEEP = {0: "HKCategoryValueSleepAnalysisAwake", 1: "HKCategoryValueSleepAnalysisAsleepCore",
         2: "HKCategoryValueSleepAnalysisAsleepDeep", 3: "HKCategoryValueSleepAnalysisAsleepREM"}
SLEEP_INV = {v: k for k, v in SLEEP.items()}


def to_healthkit_long(wear: pd.DataFrame, source="MadhuTwin simulator") -> pd.DataFrame:
    rows = []
    for col, (hk, unit) in TYPES.items():
        d = wear[["patient_id", "timestamp", col]].dropna()
        if col == "sleep_stage":
            d = d[d[col] > 0]
            val = d[col].astype(int).map(SLEEP)
        elif col == "steps":
            d = d[d[col] > 0]
            val = d[col]
        else:
            val = d[col]
        rows.append(pd.DataFrame({"patient_id": d.patient_id, "type": hk, "sourceName": source, "unit": unit,
                                  "startDate": d.timestamp, "endDate": d.timestamp + pd.Timedelta(minutes=5),
                                  "value": val}))
    return pd.concat(rows, ignore_index=True).sort_values(["patient_id", "startDate"])


def from_healthkit_long(df: pd.DataFrame, start: str, days: int) -> pd.DataFrame:
    """Resample a long export back onto the twin's 5-minute grid."""
    grid = pd.date_range(start, periods=days * 288, freq="5min")
    out = []
    inv = {hk: col for col, (hk, _) in TYPES.items()}
    for pid, g in df.groupby("patient_id"):
        w = pd.DataFrame({"patient_id": pid, "timestamp": grid})
        w["t"] = (w.timestamp - grid[0]).dt.total_seconds().div(60).astype(int)
        for hk, gg in g.groupby("type"):
            col = inv[hk]
            s = gg.set_index(pd.to_datetime(gg.startDate)).value
            if col == "sleep_stage":
                s = s.map(SLEEP_INV)
            agg = "sum" if col == "steps" else ("max" if col == "sleep_stage" else "mean")
            r = pd.to_numeric(s).resample("5min").agg(agg).reindex(grid)
            w[col] = r.to_numpy()
        w["steps"] = w.get("steps", 0).fillna(0)
        w["sleep_stage"] = w.get("sleep_stage", 0).fillna(0)
        out.append(w)
    return pd.concat(out, ignore_index=True)
