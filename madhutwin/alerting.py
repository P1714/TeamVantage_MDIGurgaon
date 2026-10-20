"""The single alert policy used everywhere: evaluation, console, API.

Policy (one place, so every reported number describes what a clinician would actually receive):

1. A *decision* exists at step k when the current CGM reading is present and below 180 mg/dL.
2. **Data sufficiency:** at least ``MIN_READINGS_LAST_HOUR`` of the last 12 CGM readings (60 min) are present.
   Otherwise the alert is suppressed and the console shows "insufficient recent data".
3. **Threshold:** risk >= operating threshold (chosen on validation patients).
4. **Debounce:** at most one alert per ``REFRACTORY_STEPS`` (60 min).

Event matching (``match_events``):
* An excursion onset is the first reading of a run of >=3 readings (>=15 min) at or above 180 mg/dL.
* An onset is *detected* if a delivered alert falls in the 2 h before it (1..24 steps earlier).
  Matching is one-to-one: each onset takes the earliest delivered alert in its window that no earlier
  onset has used. Lead time = onset time - alert time.
* A delivered alert is a *true alert* if any onset follows within 2 h, otherwise a *false alert*.
  (An alert may precede two onsets; it counts once as true but can be credited to only one onset.)
"""
from __future__ import annotations

import numpy as np

REFRACTORY_STEPS = 12          # 60 min between alerts
HORIZON_STEPS = 24             # 2 h
MIN_READINGS_LAST_HOUR = 9     # of 12
POLICY_VERSION = "alert-policy-v2 (debounce 60 min, >=9/12 readings in last hour, threshold from validation)"


def readings_last_hour(cgm: np.ndarray, k: int) -> int:
    lo = max(0, k - 11)
    return int(np.isfinite(cgm[lo:k + 1]).sum())


def data_sufficient(cgm: np.ndarray, k: int) -> bool:
    return bool(np.isfinite(cgm[k]) and readings_last_hour(cgm, k) >= MIN_READINGS_LAST_HOUR)


def delivered_alerts(risk: np.ndarray, cgm: np.ndarray, thr: float, start: int = 0) -> np.ndarray:
    """Steps at which an alert is delivered. ``risk`` is NaN where no decision is made."""
    out, last = [], -10 ** 9
    cand = np.where(np.isfinite(risk) & (np.nan_to_num(risk, nan=-1) >= thr))[0]
    for k in cand:
        if k < start or k - last < REFRACTORY_STEPS:
            continue
        if not data_sufficient(cgm, k):
            continue
        out.append(k)
        last = k
    return np.array(out, int)


def onsets(cgm: np.ndarray, thr: float = 180.0) -> np.ndarray:
    above = cgm >= thr
    run = np.zeros_like(above)
    run[:-2] = above[:-2] & above[1:-1] & above[2:]
    prev = np.zeros_like(above)
    prev[1:] = above[:-1]
    return np.where(run & ~prev)[0]


def match_events(alerts: np.ndarray, events: np.ndarray, horizon: int = HORIZON_STEPS, step_min: int = 5):
    """One-to-one matching of delivered alerts to excursion onsets (see module docstring)."""
    used = np.zeros(len(alerts), bool)
    leads = np.full(len(events), np.nan)
    for j, o in enumerate(np.sort(events)):
        win = np.where((alerts >= o - horizon) & (alerts < o) & ~used)[0]
        if len(win):
            a = win[0]
            used[a] = True
            leads[j] = (o - alerts[a]) * step_min
    true_alert = np.array([np.any((events > a) & (events <= a + horizon)) for a in alerts], bool)
    return {"leads_min": leads, "true_alert": true_alert, "used": used}
