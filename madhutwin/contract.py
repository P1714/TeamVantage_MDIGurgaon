"""The prediction contract: one definition of every number the console, API and reports show.

Field                         Meaning (layer)
----------------------------  ---------------------------------------------------------------------------
cgm_mgdl                      Latest CGM reading at the decision time (observed)
readings_last_hour            CGM readings present in the last 60 min (of 12) - data quality
data_sufficient               >= 9 of 12 readings present; otherwise alerts are suppressed
risk_2h                       P(excursion >=180 mg/dL for >=15 min starting within 2 h)  (Layer 2, LightGBM)
risk_sd                       Spread of a 5-model bootstrap ensemble around risk_2h (model disagreement,
                              not a calibrated clinical confidence)
confidence                    High (sd<0.04) / Medium (<0.08) / Low; Low whenever data are insufficient
alert_delivered               An alert was sent at this step under alerting.POLICY_VERSION
forecast_mgdl[+30/+60/+120]   Predicted glucose AT that horizon: median forecast (Layer 2)
forecast_band_80[...]         Empirical ~80% band (split-conformal on validation patients; observed
                              coverage on held-out patients is reported, not guaranteed)
predicted_peak_next_2h_mgdl   Highest glucose expected at ANY point in the next 2 h (Layer 2 peak model,
                              reconciled so it is never below the current reading or any forecast endpoint)
time_to_onset_min             Expected minutes until the excursion starts; only reported when risk >= threshold
twin_projected_peak_mgdl      Mechanistic Twin projection (Layer 1): physics only, assumes the patient's usual
                              next meal; a scenario, not the ML forecast
twin_carbs_in_gut_g           Twin state: carbohydrate still being absorbed
"""
from __future__ import annotations

import math

MODEL_VERSION = "MadhuTwin v3.1"
TARGET = "CGM >=180 mg/dL for >=15 min, starting within the next 2 h"
UNCERTAINTY_NOTE = ("risk_sd is ensemble disagreement, not a calibrated clinical confidence; forecast bands show "
                    "empirical ~80% coverage on held-out synthetic patients, not a guarantee.")
SAFETY_NOTE = "Research prototype - decision support only; not for diagnosis or dosing; clinician judgement required."


def _f(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def reconciled_peak(g_now, forecasts, peak_pred):
    """Highest expected glucose over the next 2 h, consistent with the endpoint forecasts."""
    vals = [_f(g_now), _f(peak_pred)] + [_f(v) for v in forecasts]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def confidence_label(risk_sd, data_sufficient=True):
    if not data_sufficient:
        return "Low"
    sd = _f(risk_sd)
    if sd is None:
        return "Medium"
    return "High" if sd < 0.04 else "Medium" if sd < 0.08 else "Low"


def tte_or_none(tte, risk, thr):
    t = _f(tte)
    return None if t is None or _f(risk) is None or risk < thr else max(5, int(round(t / 5) * 5))
