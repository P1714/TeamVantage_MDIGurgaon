"""Stream fusion: turn EHR + wearables + twin states into one feature row per
patient per decision time (every 5 minutes), strictly using information
available *at or before* that time.

Feature groups (used for the ablation study and for explanations)
    ehr       static record: demographics, labs, genotype, diagnoses, meds
    cgm       glucose level, trend, variability, personal time-of-day history
    lifestyle meals logged, steps, heart rate, HRV, sleep
    context   clock / weekday
    twin      personalised physiology + mechanistic 2-hour projections
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

DT = C.CGM_INTERVAL
H = C.HORIZON_MIN // DT             # 24 steps = 2 h
PER_DAY = C.MIN_PER_DAY // DT        # 288


# ----------------------------------------------------------------------------- helpers
def _roll(a, win, how="mean", minp=1):
    r = pd.DataFrame(a.T).rolling(win, min_periods=minp)
    return getattr(r, how)().to_numpy().T


def _lag(a, k):
    out = np.full_like(a, np.nan, dtype=float)
    out[:, k:] = a[:, :-k]
    return out


def _ffill(a, limit=None):
    return pd.DataFrame(a.T).ffill(limit=limit).to_numpy().T


def _since(mask):
    """Minutes since mask was last True (NaN before first)."""
    P, n = mask.shape
    idx = np.where(mask, np.arange(n)[None, :], np.nan)
    last = _ffill(idx)
    return (np.arange(n)[None, :] - last) * DT


def _future_any(mask, h):
    """any(mask[k+1 .. k+h]) for each k."""
    m = mask.astype(float)
    cs = np.concatenate([np.zeros((m.shape[0], 1)), np.cumsum(m, axis=1)], axis=1)
    n = m.shape[1]
    k = np.arange(n)
    hi = np.minimum(k + h, n - 1) + 1
    lo = np.minimum(k + 1, n)
    return (cs[:, hi] - cs[:, lo]) > 0


def _first_future(mask, h):
    """steps until first True in mask[k+1 .. k+h] (NaN if none)."""
    P, n = mask.shape
    idx = np.where(mask, np.arange(n)[None, :], np.nan)
    nxt = pd.DataFrame(idx.T).bfill().to_numpy().T      # next index >= k
    nxt_after = np.full_like(nxt, np.nan)
    nxt_after[:, :-1] = nxt[:, 1:]                      # next index >= k+1
    d = nxt_after - np.arange(n)[None, :]
    return np.where(d <= h, d, np.nan)


def _run_mask(cgm, thr):
    """True at j when readings j, j+1, j+2 are all >= thr (a >=15-min excursion)."""
    above = cgm >= thr
    run = np.zeros_like(above)
    run[:, :-2] = above[:, :-2] & above[:, 1:-1] & above[:, 2:]
    return run


def excursion_onsets(cgm, thr=C.HYPER_L1):
    """True where a >=15-min excursion above ``thr`` starts."""
    run = _run_mask(cgm, thr)
    prev = np.zeros_like(run)
    prev[:, 1:] = (cgm >= thr)[:, :-1]
    return run & ~prev


def labels(cgm):
    """Targets at time k (current reading < 180): does a >=15-min excursion
    >=180 (resp. >=250) begin within the next 2 hours, and how soon?"""
    run180 = _run_mask(cgm, C.HYPER_L1)
    run250 = _run_mask(cgm, C.HYPER_L2)
    return _future_any(run180, H), _future_any(run250, H), _first_future(run180, H) * DT


# ----------------------------------------------------------------------------- sleep windows
def night_features(A):
    """Stats of the most recent *complete* night (18:00 -> 12:00) and the
    running totals of the current night, all leak-free."""
    P, n = A["P"], A["n5"]
    D = n // PER_DAY
    stage = A["stage"]
    asleep = stage > 0
    feats = {k: np.full((P, n), np.nan) for k in
             ["sleep_last_h", "deep_frac_last", "wake_bouts_last", "night_hr_last", "night_hrv_last",
              "sleep_debt_3n", "night_hrv_z", "night_hr_rel", "asleep_now", "sleep_tonight_h"]}
    nights = []
    for d in range(D):
        a = max(0, (d - 1) * PER_DAY + 216)
        b = d * PER_DAY + 144
        sl = asleep[:, a:b]
        st = stage[:, a:b]
        mins = sl.sum(1) * DT
        deep = (st == 2).sum(1) / np.maximum(sl.sum(1), 1)
        trans = ((sl[:, 1:].astype(int) - sl[:, :-1].astype(int)) == -1).sum(1)
        hr = np.where(sl, A["hr"][:, a:b], np.nan)
        hrv = np.where(sl, A["hrv"][:, a:b], np.nan)
        with np.errstate(all="ignore"):
            nights.append((mins / 60, deep, np.maximum(trans - 1, 0), np.nanmean(hr, 1), np.nanmean(hrv, 1)))
    nights = [np.stack(x, 1) for x in zip(*nights)]          # each [P, D]
    sl_h, deep, bouts, nhr, nhrv = nights
    for d in range(D):
        # complete night ending at 12:00 of day d is available from 12:00 of day d until 12:00 of day d+1
        a, b = d * PER_DAY + 144, min((d + 1) * PER_DAY + 144, n)
        prev = slice(max(0, d - 6), d + 1)
        with np.errstate(all="ignore"):
            base_hrv = np.nanmean(nhrv[:, prev], 1)
            sd_hrv = np.nanstd(nhrv[:, prev], 1) + 2
            base_hr = np.nanmean(nhr[:, prev], 1)
        for key, val in [("sleep_last_h", sl_h[:, d]), ("deep_frac_last", deep[:, d]),
                         ("wake_bouts_last", bouts[:, d]), ("night_hr_last", nhr[:, d]),
                         ("night_hrv_last", nhrv[:, d]),
                         ("sleep_debt_3n", np.maximum(0, 7 - sl_h[:, max(0, d - 2): d + 1]).sum(1)),
                         ("night_hrv_z", (nhrv[:, d] - base_hrv) / sd_hrv),
                         ("night_hr_rel", nhr[:, d] - base_hr)]:
            feats[key][:, a:b] = val[:, None]
    feats["asleep_now"] = asleep.astype(float)
    # cumulative sleep since 18:00 for the current night
    cum = np.zeros((P, n))
    for d in range(D + 1):
        a = max(0, (d - 1) * PER_DAY + 216)
        b = min(d * PER_DAY + 144, n)
        if b > a:
            cum[:, a:b] = np.cumsum(asleep[:, a:b], 1) * DT / 60
    feats["sleep_tonight_h"] = cum
    return feats, nhr


# ----------------------------------------------------------------------------- main builder
def build(A, ehr: pd.DataFrame, tw: pd.DataFrame, states, proj, proj_h, habit,
          patient_step: dict | None = None, day_range=(C.CALIB_DAYS, C.N_DAYS)):
    """Return the fused feature table.

    ``patient_step`` maps patient_id -> subsampling stride (1 = every 5 min)."""
    P, n = A["P"], A["n5"]
    cgm = A["cgm"]
    g = _ffill(cgm, limit=2)
    F = {}
    # ---- CGM dynamics
    F["g_now"] = g
    for m in (1, 3, 6, 12):
        F[f"g_delta_{m * DT}m"] = g - _lag(g, m)
    F["g_accel"] = (g - _lag(g, 3)) - (_lag(g, 3) - _lag(g, 6))
    F["g_mean_1h"] = _roll(cgm, 12, "mean")
    F["g_std_1h"] = _roll(cgm, 12, "std", 3)
    F["g_max_3h"] = _roll(cgm, 36, "max")
    F["g_min_3h"] = _roll(cgm, 36, "min")
    F["g_mean_24h"] = _roll(cgm, PER_DAY, "mean", 60)
    F["g_cv_24h"] = _roll(cgm, PER_DAY, "std", 60) / F["g_mean_24h"]
    F["tar_24h"] = _roll((cgm > 180).astype(float), PER_DAY, "mean", 60)
    F["mins_since_above180"] = np.minimum(_since(cgm >= 180), 2880)
    F["g_yday_same"] = _lag(g, PER_DAY)
    fut_max = pd.DataFrame(cgm[:, ::-1].T).rolling(H, min_periods=1).max().to_numpy().T[:, ::-1]
    F["g_yday_max_next2h"] = _lag(fut_max, PER_DAY)

    y, y250, tte = labels(cgm)
    valid_hist = ~np.isnan(g)
    # personal time-of-day spike history from previous days (known by now)
    D = n // PER_DAY
    Y = np.where(valid_hist & (g < 180), y.astype(float), np.nan).reshape(P, D, PER_DAY)
    s = np.nancumsum(Y, 1)
    c = np.cumsum(~np.isnan(Y), 1)
    prev_s = np.concatenate([np.zeros((P, 1, PER_DAY)), s[:, :-1]], 1)
    prev_c = np.concatenate([np.zeros((P, 1, PER_DAY)), c[:, :-1]], 1)
    ker = np.ones(5)
    sm = lambda z: np.apply_along_axis(lambda v: np.convolve(np.r_[v[-2:], v, v[:2]], ker, "valid"), 2, z)
    with np.errstate(all="ignore"):
        F["hist_spike_rate_tod"] = (sm(prev_s) / sm(prev_c)).reshape(P, n)

    # ---- meals (logged)
    meal = A["carbs"] >= 5
    F["mins_since_meal"] = np.minimum(_since(meal), 1440)
    F["last_meal_carbs"] = _ffill(np.where(meal, A["carbs"], np.nan))
    F["last_meal_gi"] = _ffill(np.where(meal, A["gi"], np.nan))
    F["carbs_logged_3h"] = _roll(A["carbs"], 36, "sum")
    F["carbs_logged_today"] = np.cumsum(A["carbs"].reshape(P, D, PER_DAY), 2).reshape(P, n)
    F["mins_to_habit_meal"] = habit["mins"]
    F["habit_meal_carbs"] = habit["carbs"]
    F["habit_meal_gi"] = habit["gi"]
    F["habit_meal_freq"] = habit["freq"]

    # ---- activity / heart
    st = A["steps"]
    F["steps_15m"] = _roll(st, 3, "sum")
    F["steps_60m"] = _roll(st, 12, "sum")
    F["steps_today"] = np.cumsum(st.reshape(P, D, PER_DAY), 2).reshape(P, n)
    F["mins_since_brisk"] = np.minimum(_since(st >= 300), 1440)
    F["hr_30m"] = _roll(A["hr"], 6, "mean")
    F["hrv_1h"] = _roll(A["hrv"], 12, "mean")
    awake_hrv = np.where(A["stage"] == 0, A["hrv"], np.nan)
    base = _roll(awake_hrv, PER_DAY * 3, "median", 50)
    F["hrv_1h_rel"] = F["hrv_1h"] / base
    nf, nhr = night_features(A)
    F.update(nf)
    with np.errstate(all="ignore"):
        # running mean of *completed* nights only (no look-ahead), available from 12:00 each day
        cnt = np.cumsum(~np.isnan(nhr), 1)
        cm = np.nancumsum(nhr, 1) / np.maximum(cnt, 1)
        cm[cnt == 0] = np.nan
        night_idx = (np.arange(n) - 144) // PER_DAY
        night_base = np.where(night_idx[None, :] >= 0, cm[:, np.clip(night_idx, 0, None)], np.nan)
    F["hr_30m_rel"] = F["hr_30m"] - night_base  # heart rate above the patient's resting (sleeping) baseline

    # ---- clock
    tod = np.broadcast_to(A["tod"][None, :], (P, n)).astype(float)
    F["tod_sin"] = np.sin(2 * np.pi * tod / 1440)
    F["tod_cos"] = np.cos(2 * np.pi * tod / 1440)
    dow = (pd.Timestamp(C.START_DATE).dayofweek + A["day"]) % 7
    F["weekend"] = np.broadcast_to((dow >= 5)[None, :], (P, n)).astype(float)

    # ---- twin
    F["twin_proj_g30"] = proj["g30"]
    F["twin_proj_g60"] = proj["g60"]
    F["twin_proj_g120"] = proj["g120"]
    F["twin_proj_peak"] = proj["peak"]
    F["twin_proj_peak_habit"] = proj_h["peak"]
    F["twin_proj_g120_habit"] = proj_h["g120"]
    F["twin_gut_carbs_g"] = (states["F1"] + states["F2"] + states["S1"] + states["S2"]) / 1000
    F["twin_insulin_action"] = states["X"] * 1e3
    F["twin_plasma_minus_sensor"] = states["G"] - states["IG"]
    F["twin_ex_boost"] = A["ex_boost"]
    twc = tw.set_index("patient_id").loc[A["pids"]]
    for col in ["twin_SI", "twin_sigma", "twin_Gb", "twin_dawn", "twin_fit_rmse"]:
        F[col] = np.broadcast_to(twc[col].to_numpy(float)[:, None], (P, n))
    F["twin_disposition_index"] = F["twin_SI"] * F["twin_sigma"] * 1e5

    # ---- targets
    fut_y = {f"y_g{m}": np.concatenate([cgm[:, m // DT:], np.full((P, m // DT), np.nan)], 1) for m in (30, 60, 120)}
    fut_nan = pd.DataFrame(np.isnan(cgm)[:, ::-1].T.astype(float)).rolling(H, min_periods=1).mean() \
        .to_numpy().T[:, ::-1]
    fut_cover = np.concatenate([fut_nan[:, 1:], np.ones((P, 1))], 1)

    # ---- choose decision points
    day = A["day"]
    rows = []
    k_all = np.arange(n)
    for i, pid in enumerate(A["pids"]):
        stride = (patient_step or {}).get(pid, 1)
        ok = (day >= day_range[0]) & (day < day_range[1]) & (k_all < n - H) & ~np.isnan(g[i]) & (g[i] < C.HYPER_L1) & (fut_cover[i] < 0.5)
        ks = k_all[ok][::stride]
        rows.append((i, ks))
    idx_p = np.concatenate([np.full(len(ks), i) for i, ks in rows])
    idx_k = np.concatenate([ks for _, ks in rows])
    out = pd.DataFrame({"patient_id": np.array(A["pids"])[idx_p], "k": idx_k, "t": idx_k * DT})
    out["label"] = y[idx_p, idx_k].astype(int)
    out["label250"] = y250[idx_p, idx_k].astype(int)
    out["tte_min"] = tte[idx_p, idx_k]
    peak_next = np.concatenate([fut_max[:, 1:], np.full((P, 1), np.nan)], 1)
    out["y_peak2h"] = peak_next[idx_p, idx_k]
    for key, arr in fut_y.items():
        out[key] = arr[idx_p, idx_k]
    feat_df = pd.DataFrame({key: np.asarray(arr, float)[idx_p, idx_k] for key, arr in F.items()})
    out = pd.concat([out, feat_df], axis=1)
    e = ehr.set_index("patient_id")
    ehr_cols = [c for c in e.columns]
    out = out.merge(e[ehr_cols].add_prefix("ehr_"), left_on="patient_id", right_index=True, how="left")
    return out


def feature_groups(columns):
    """Map every feature column to its group."""
    g = {}
    for c in columns:
        if c.startswith("ehr_gen_"):
            g[c] = "genetics"      # long-term baseline only; not used by the 2-h risk model (no measurable gain)
        elif c.startswith("ehr_"):
            g[c] = "ehr"
        elif c.startswith("twin_"):
            g[c] = "twin"
        elif c in ("tod_sin", "tod_cos", "weekend"):
            g[c] = "context"
        elif c.startswith(("g_", "tar_", "mins_since_above", "hist_spike")):
            g[c] = "cgm"
        else:
            g[c] = "lifestyle"
    return g


NON_FEATURES = {"patient_id", "k", "t", "label", "label250", "tte_min", "y_g30", "y_g60", "y_g120", "y_peak2h"}
