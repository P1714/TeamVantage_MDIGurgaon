"""The Digital Twin: personalised physiology + state estimation + projection.

Three steps turn a patient's raw data into a living virtual replica:

1. **Personalisation (calibration)** – using only the patient's first week of
   CGM, logged meals and step counts, fit the three parameters that differ
   most between people with T2D: insulin sensitivity ``SI``, beta-cell
   secretory gain ``sigma`` and basal glucose ``Gb`` (plus the size of their
   dawn phenomenon, read directly from overnight CGM). The search is a
   vectorised coarse-to-fine grid: every candidate for every patient is
   simulated simultaneously.
2. **State estimation (observer)** – the calibrated model is run forward in
   real time and nudged toward each new CGM reading, so the twin always holds
   an estimate of *hidden* states: carbohydrate still in the gut, insulin
   action, plasma glucose ahead of the lagging sensor.
3. **Projection** – from the current state the twin simulates the next two
   hours (a) with what has already been eaten and (b) adding the patient's
   habitual next meal learned from their own history. These mechanistic
   projections become inputs to the ML risk model, and the same engine powers
   the doctor's what-if simulator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from . import config as C
from . import physiology as PH

DT = C.CGM_INTERVAL
SG_POP = 0.016
SLOT_WINDOWS = {"breakfast": (300, 660), "lunch": (660, 960), "snack": (960, 1140), "dinner": (1140, 1440)}


# -----------------------------------------------------------------------------
# data -> arrays
# -----------------------------------------------------------------------------
def load_arrays(wear: pd.DataFrame, meal_log: pd.DataFrame, pids: list[str]):
    """Pivot the long 5-minute table into [patient, time] arrays."""
    wear = wear.set_index("patient_id").loc[pids].reset_index()
    P = len(pids)
    n5 = len(wear) // P
    A = {}
    for col, key in [("cgm_mgdl", "cgm"), ("hr_bpm", "hr"), ("hrv_rmssd_ms", "hrv"),
                     ("steps", "steps"), ("sleep_stage", "stage")]:
        A[key] = wear[col].to_numpy(float).reshape(P, n5)
    A["tod"] = (np.arange(n5) * DT) % C.MIN_PER_DAY
    A["day"] = (np.arange(n5) * DT) // C.MIN_PER_DAY
    A["n5"], A["P"], A["pids"] = n5, P, pids
    pidx = {p: i for i, p in enumerate(pids)}
    fast = np.zeros((P, n5))
    slow = np.zeros((P, n5))
    carbs = np.zeros((P, n5))
    gi = np.zeros((P, n5))
    ml = meal_log[meal_log.patient_id.isin(pidx)]
    for r in ml.itertuples():
        i, k = pidx[r.patient_id], int(r.t) // DT
        if k >= n5:
            continue
        ff = float(PH.fast_fraction(r.gi))
        fast[i, k] += r.carbs_g * 1000 * ff
        slow[i, k] += r.carbs_g * 1000 * (1 - ff)
        gi[i, k] = (gi[i, k] * carbs[i, k] + r.gi * r.carbs_g) / (carbs[i, k] + r.carbs_g)
        carbs[i, k] += r.carbs_g
    A.update(fast=fast, slow=slow, carbs=carbs, gi=gi)
    spm = A["steps"] / DT
    A["spm"] = spm
    brisk = np.maximum(spm - 60, 0) / 100.0 * DT
    ex_mem = lfilter([1.0], [1.0, -np.exp(-DT / 360.0)], brisk, axis=1)
    A["ex_boost"] = 1 + 0.3 * np.minimum(1, ex_mem / 25.0)
    return A


# -----------------------------------------------------------------------------
# the twin ODE on the 5-minute grid (rows = patients or candidates)
# -----------------------------------------------------------------------------
def _twin_step(s, prm, gb_t, si_t, spm_t, fast_t, slow_t):
    s = PH.step(s, prm, DT, gb_t, si_t, spm_t, fast_t, slow_t)
    s["IG"] = s["IG"] + DT * (s["G"] - s["IG"]) / PH.TAU_ISF
    return s


def _params(rows_SI, rows_sigma, rows_Gb, rows_V, rows_sglt2):
    return {"SG": np.full_like(rows_SI, SG_POP), "sigma": rows_sigma, "h": rows_Gb, "V": rows_V,
            "sglt2": rows_sglt2, "SI": rows_SI, "Gb": rows_Gb}


def estimate_dawn(A, days):
    """Median rise of overnight CGM from 03:00 to 06:30 (no meals at night)."""
    c = A["cgm"][:, : days * 288].reshape(A["P"], days, 288)
    rise = np.nanmedian(np.nanmean(c[:, :, 75:81], axis=2) - np.nanmean(c[:, :, 33:39], axis=2), axis=1)
    return np.clip(np.nan_to_num(rise, nan=12.0), 0, 40) * 1.1


def _simulate_fit(A, pat, SI, sig, Gb, dawn, V, sglt2, k_end, cgm_burn=12, k_start=0):
    """Open-loop simulation of candidate rows; returns Huber loss vs CGM."""
    n = len(pat)
    prm = _params(SI, sig, Gb, V[pat], sglt2[pat])
    s = PH.init_state(n, Gb)
    s["IG"] = s["G"].copy()
    loss = np.zeros(n)
    cnt = np.zeros(n)
    circ = PH.circadian_si(A["tod"])
    bump = PH.dawn_bump(A["tod"])
    with np.errstate(all="ignore"):
        first = np.nanmean(A["cgm"][:, k_start + 12:k_start + 24], axis=1)
    s["G"] = np.nan_to_num(first, nan=150)[pat]
    s["IG"] = s["G"].copy()
    for k in range(k_start, k_end):
        gb_t = Gb + dawn[pat] * bump[k]
        si_t = SI * circ[k] * A["ex_boost"][pat, k]
        s = _twin_step(s, prm, gb_t, si_t, A["spm"][pat, k], A["fast"][pat, k], A["slow"][pat, k])
        y = A["cgm"][pat, k]
        if k >= k_start + cgm_burn:
            ok = ~np.isnan(y)
            r = np.where(ok, s["IG"] - np.nan_to_num(y), 0.0)
            a = np.abs(r)
            loss += np.where(a < 25, 0.5 * r * r, 25 * (a - 12.5))
            cnt += ok
    return loss / np.maximum(cnt, 1)


def calibrate(A, ehr: pd.DataFrame, days: int = C.CALIB_DAYS, prior_weight: float = 40.0,
              verbose=True) -> pd.DataFrame:
    """Coarse-to-fine vectorised grid search for SI, sigma, Gb per patient."""
    P = A["P"]
    e = ehr.set_index("patient_id").loc[A["pids"]]
    V = 1.7 * e["weight"].to_numpy(float)
    sglt2 = e["rx_sglt2"].to_numpy(float)
    dawn = estimate_dawn(A, days)
    night = A["cgm"][:, : days * 288].reshape(P, days, 288)[:, :, 24:60]   # 02:00-05:00
    gb0 = np.nan_to_num(np.nanmedian(night.reshape(P, -1), axis=1), nan=140)
    k_end = days * 288
    # EHR-informed prior with regularised grid-search fitting (a MAP-style penalty on log-parameters,
    # not full Bayesian posterior inference): HOMA-IR ~ 1/SI, HOMA-B ~ beta-cell function.
    # CGM pins down the product SI*sigma (disposition index); the labs decide
    # how it splits between resistance and secretion.
    homa_ir = np.nan_to_num(e["homa_ir"].to_numpy(float), nan=2.4)
    homa_b = np.nan_to_num(e["homa_b"].to_numpy(float), nan=41.0)
    log_si_p = np.log(2.3e-4 * (np.maximum(homa_ir, 0.3) / 2.4) ** -0.9)
    log_sig_p = np.log(0.035 * (np.maximum(homa_b, 3) / 41.0) ** 0.8)

    def penalised(loss, pat, SI_, sig_):
        return loss + prior_weight * ((np.log(SI_) - log_si_p[pat]) ** 2 + (np.log(sig_) - log_sig_p[pat]) ** 2)

    si_grid = np.geomspace(0.5e-4, 7e-4, 9)
    sig_grid = np.geomspace(0.005, 0.12, 9)
    gb_off = np.array([-20, -10, -3, 3, 10, 20])
    cand = np.array(np.meshgrid(si_grid, sig_grid, gb_off, indexing="ij")).reshape(3, -1).T
    nc = len(cand)
    pat = np.repeat(np.arange(P), nc)
    SI = np.tile(cand[:, 0], P)
    sig = np.tile(cand[:, 1], P)
    Gb = gb0[pat] + np.tile(cand[:, 2], P)
    loss = penalised(_simulate_fit(A, pat, SI, sig, Gb, dawn, V, sglt2, k_end), pat, SI, sig).reshape(P, nc)
    best = loss.argmin(axis=1)
    b_si, b_sig, b_gb = SI.reshape(P, nc)[np.arange(P), best], sig.reshape(P, nc)[np.arange(P), best], \
        Gb.reshape(P, nc)[np.arange(P), best]
    if verbose:
        print(f"    coarse grid: {nc} candidates x {P} patients, done")

    # fine stage: local multiplicative grid around the best coarse point
    f = np.array(np.meshgrid([0.72, 0.85, 1.0, 1.18, 1.38], [0.72, 0.85, 1.0, 1.18, 1.38],
                             [-6, -3, 0, 3, 6], indexing="ij")).reshape(3, -1).T
    nf = len(f)
    pat = np.repeat(np.arange(P), nf)
    SI2 = b_si[pat] * np.tile(f[:, 0], P)
    sig2 = b_sig[pat] * np.tile(f[:, 1], P)
    Gb2 = b_gb[pat] + np.tile(f[:, 2], P)
    fit2 = _simulate_fit(A, pat, SI2, sig2, Gb2, dawn, V, sglt2, k_end)
    loss2 = penalised(fit2, pat, SI2, sig2).reshape(P, nf)
    fit2 = fit2.reshape(P, nf)
    best2 = loss2.argmin(axis=1)
    idx = np.arange(P)
    out = pd.DataFrame({
        "patient_id": A["pids"],
        "twin_SI": SI2.reshape(P, nf)[idx, best2],
        "twin_sigma": sig2.reshape(P, nf)[idx, best2],
        "twin_Gb": Gb2.reshape(P, nf)[idx, best2],
        "twin_dawn": dawn,
        "twin_fit_rmse": np.sqrt(2 * fit2[idx, best2]),
        "twin_V": V,
        "twin_sglt2": sglt2,
    })
    return out


def nightly_update(A, tw: pd.DataFrame, window_days: int = 3, first_day: int = C.CALIB_DAYS,
                   lam_si: float = 30.0, lam_gb: float = 8.0):
    """Re-personalise the twin every night.

    Insulin sensitivity and basal glucose drift with sleep debt, stress, illness
    and adherence. Each midnight we refit an SI multiplier and a basal-glucose
    shift on the last ``window_days`` of data only (no look-ahead), shrunk toward
    the week-1 baseline. Beta-cell function is held fixed (it changes over months).
    Returns per-day arrays [P, D] (NaN before ``first_day``)."""
    P, n5 = A["P"], A["n5"]
    D = n5 // 288
    SIb, sig, Gbb = tw.twin_SI.to_numpy(), tw.twin_sigma.to_numpy(), tw.twin_Gb.to_numpy()
    dawn, V, sglt2 = tw.twin_dawn.to_numpy(), tw.twin_V.to_numpy(), tw.twin_sglt2.to_numpy()
    mults = np.array([0.55, 0.65, 0.75, 0.85, 0.93, 1.0, 1.08, 1.18, 1.3, 1.45])
    shifts = np.array([-14, -7, 0, 7, 14, 22, 32])
    cand = np.array(np.meshgrid(mults, shifts, indexing="ij")).reshape(2, -1).T
    nc = len(cand)
    pat = np.repeat(np.arange(P), nc)
    m_rows, g_rows = np.tile(cand[:, 0], P), np.tile(cand[:, 1], P)
    si_mult = np.full((P, D), np.nan)
    gb_delta = np.full((P, D), np.nan)
    for d in range(first_day, D):
        ks, ke = (d - window_days) * 288, d * 288
        loss = _simulate_fit(A, pat, SIb[pat] * m_rows, sig[pat], Gbb[pat] + g_rows, dawn, V, sglt2, ke, k_start=ks)
        loss = loss + lam_si * np.log(m_rows) ** 2 + lam_gb * (g_rows / 10) ** 2
        best = loss.reshape(P, nc).argmin(1)
        si_mult[:, d] = cand[best, 0]
        gb_delta[:, d] = cand[best, 1]
    return {"si_mult": si_mult, "gb_delta": gb_delta}


def _per_step(daily, n5, fill):
    """Expand per-day [P, D] arrays to per-5-min [P, n5], NaN -> fill."""
    return np.nan_to_num(np.repeat(daily, 288, axis=1)[:, :n5], nan=fill)


def prior_params(A, ehr: pd.DataFrame) -> pd.DataFrame:
    """Day-0 twin for a brand-new patient: physiology from the EHR alone
    (HOMA-IR -> SI, HOMA-B -> sigma, fasting glucose -> Gb, population dawn)."""
    e = ehr.set_index("patient_id").loc[A["pids"]]
    homa_ir = np.nan_to_num(e["homa_ir"].to_numpy(float), nan=2.4)
    homa_b = np.nan_to_num(e["homa_b"].to_numpy(float), nan=41.0)
    return pd.DataFrame({
        "patient_id": A["pids"],
        "twin_SI": 2.3e-4 * (np.maximum(homa_ir, 0.3) / 2.4) ** -0.9,
        "twin_sigma": 0.035 * (np.maximum(homa_b, 3) / 41.0) ** 0.8,
        "twin_Gb": np.nan_to_num(e["fpg"].to_numpy(float), nan=140) - 8,
        "twin_dawn": np.full(len(e), 14.0),
        "twin_fit_rmse": np.full(len(e), np.nan),
        "twin_V": 1.7 * e["weight"].to_numpy(float),
        "twin_sglt2": e["rx_sglt2"].to_numpy(float),
    })


# -----------------------------------------------------------------------------
# observer + projections for every patient and every 5-minute step
# -----------------------------------------------------------------------------
STATE_KEYS = ["G", "X", "I", "Y", "F1", "F2", "S1", "S2", "IG"]


def observe(A, tw: pd.DataFrame, gain: float = 0.35, nightly=None):
    """Closed-loop run over the whole record; returns state arrays [P, n5]
    holding the twin state *after* assimilating reading k (i.e. at time k).
    With ``nightly`` (from :func:`nightly_update`) the twin's insulin
    sensitivity and basal glucose change from day to day."""
    P, n5 = A["P"], A["n5"]
    sim = _per_step(nightly["si_mult"], n5, 1.0) if nightly else np.ones((P, n5))
    gbd = _per_step(nightly["gb_delta"], n5, 0.0) if nightly else np.zeros((P, n5))
    SI, sig, Gb = tw.twin_SI.to_numpy(), tw.twin_sigma.to_numpy(), tw.twin_Gb.to_numpy()
    prm = _params(SI, sig, Gb, tw.twin_V.to_numpy(), tw.twin_sglt2.to_numpy())
    dawn = tw.twin_dawn.to_numpy()
    s = PH.init_state(P, Gb)
    s["IG"] = s["G"].copy()
    circ = PH.circadian_si(A["tod"])
    bump = PH.dawn_bump(A["tod"])
    out = {k: np.zeros((P, n5), np.float32) for k in STATE_KEYS}
    for k in range(n5):
        prm["h"] = Gb + gbd[:, k]
        s = _twin_step(s, prm, Gb + gbd[:, k] + dawn * bump[k], SI * sim[:, k] * circ[k] * A["ex_boost"][:, k],
                       A["spm"][:, k], A["fast"][:, k], A["slow"][:, k])
        y = A["cgm"][:, k]
        ok = ~np.isnan(y)
        innov = np.where(ok, np.nan_to_num(y) - s["IG"], 0.0)
        s["IG"] = s["IG"] + gain * innov
        s["G"] = np.clip(s["G"] + gain * innov, 40, 600)
        for key in STATE_KEYS:
            out[key][:, k] = s[key]
    return out


def learn_habits(A, meal_log: pd.DataFrame, days: int = C.CALIB_DAYS) -> pd.DataFrame:
    """Personal meal habits (time, size, GI, frequency) from the history window."""
    rows = []
    ml = meal_log.copy()
    ml["tod"] = ml.t % C.MIN_PER_DAY
    ml["day"] = ml.t // C.MIN_PER_DAY
    hist = ml[(ml.day < days) & (ml.carbs_g >= 15)]
    for pid in A["pids"]:
        h = hist[hist.patient_id == pid]
        for slot, (a, b) in SLOT_WINDOWS.items():
            m = h[(h.tod >= a) & (h.tod < b)]
            if len(m) == 0:
                rows.append({"patient_id": pid, "slot": slot, "time": (a + b) / 2, "carbs": 0.0, "gi": 60.0, "freq": 0.0})
                continue
            rows.append({"patient_id": pid, "slot": slot, "time": float(m.tod.median()),
                         "carbs": float(m.carbs_g.median()), "gi": float(np.average(m.gi, weights=m.carbs_g)),
                         "freq": min(1.0, m.day.nunique() / days)})
    return pd.DataFrame(rows)


def next_habitual_meal(A, habits: pd.DataFrame):
    """For each [patient, t]: minutes until the next habitual meal that has not
    been logged yet today (NaN if none in the next 3 h), its carbs, GI, freq."""
    P, n5 = A["P"], A["n5"]
    tod = A["tod"]
    mins = np.full((P, n5), np.nan)
    carbs = np.zeros((P, n5))
    gi = np.full((P, n5), 60.0)
    freq = np.zeros((P, n5))
    hb = habits.set_index(["patient_id", "slot"])
    logged = A["carbs"] >= 15
    for i, pid in enumerate(A["pids"]):
        for slot, (a, b) in SLOT_WINDOWS.items():
            r = hb.loc[(pid, slot)]
            if r.freq < 0.25:
                continue
            # has a meal been logged in this slot's window today, up to now?
            in_win = (tod >= a) & (tod < b)
            D = n5 // 288
            flag = (np.cumsum((in_win & logged[i]).reshape(D, 288), axis=1) > 0).reshape(-1)
            dt = r.time - tod
            cand = (dt > 0) & (dt <= 180) & ~flag
            better = cand & (np.isnan(mins[i]) | (dt < mins[i]))
            mins[i] = np.where(better, dt, mins[i])
            carbs[i] = np.where(better, r.carbs, carbs[i])
            gi[i] = np.where(better, r.gi, gi[i])
            freq[i] = np.where(better, r.freq, freq[i])
    return {"mins": mins, "carbs": carbs, "gi": gi, "freq": freq}


def project(A, tw: pd.DataFrame, states: dict, horizon: int = C.HORIZON_MIN, habit=None, nightly=None):
    """Forward-simulate every [patient, t] state for ``horizon`` minutes with no
    new logged meals and no activity (and optionally the habitual next meal).
    Returns dict of [P, n5] arrays: g30, g60, g120, peak, t_peak."""
    P, n5 = A["P"], A["n5"]
    steps = horizon // DT
    rep = lambda v: np.repeat(np.asarray(v, float), n5)
    SI, Gb = rep(tw.twin_SI), rep(tw.twin_Gb)
    if nightly:
        SI = SI * _per_step(nightly["si_mult"], n5, 1.0).reshape(-1)
        Gb = Gb + _per_step(nightly["gb_delta"], n5, 0.0).reshape(-1)
    prm = _params(SI, rep(tw.twin_sigma), Gb, rep(tw.twin_V), rep(tw.twin_sglt2))
    dawn = rep(tw.twin_dawn)
    s = {k: states[k].reshape(-1).astype(float).copy() for k in STATE_KEYS}
    tod0 = np.tile(A["tod"], P)
    exb = A["ex_boost"].reshape(-1)
    zero = np.zeros(P * n5)
    if habit is not None:
        h_step = np.round(np.nan_to_num(habit["mins"], nan=1e9) / DT).reshape(-1)
        h_mg = (habit["carbs"] * 1000).reshape(-1)
        h_ff = PH.fast_fraction(habit["gi"]).reshape(-1)
    peak = s["IG"].copy()
    t_peak = np.zeros(P * n5)
    res = {}
    for j in range(1, steps + 1):
        tod = (tod0 + j * DT) % C.MIN_PER_DAY
        fast = slow = zero
        if habit is not None:
            hit = h_step == j
            fast = np.where(hit, h_mg * h_ff, 0.0)
            slow = np.where(hit, h_mg * (1 - h_ff), 0.0)
        s = _twin_step(s, prm, Gb + dawn * PH.dawn_bump(tod), SI * PH.circadian_si(tod) * exb, zero, fast, slow)
        better = s["IG"] > peak
        peak = np.where(better, s["IG"], peak)
        t_peak = np.where(better, j * DT, t_peak)
        if j * DT in (30, 60, 120):
            res[f"g{j * DT}"] = s["IG"].reshape(P, n5).astype(np.float32)
    res["peak"] = peak.reshape(P, n5).astype(np.float32)
    res["t_peak"] = t_peak.reshape(P, n5).astype(np.float32)
    return res
