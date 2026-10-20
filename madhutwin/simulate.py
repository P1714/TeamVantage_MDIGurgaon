"""Virtual-patient simulator: behaviour -> physiology -> sensors.

For every patient in the cohort we simulate ``N_DAYS`` of free-living life at
1-minute resolution:

* **Behaviour**: region-specific Indian meals with personal favourites and
  habitual timing, unlogged morning chai, festival days, morning / post-meal
  walks, office commutes, sleep with 90-minute cycles, stressful days,
  occasional illness and missed medication doses.
* **Physiology**: the minimal-model ODEs in :mod:`madhutwin.physiology`, with
  insulin sensitivity and basal glucose modulated by sleep debt, stress,
  illness, recent exercise, the circadian clock and the dawn phenomenon.
* **Sensors**: a 5-minute CGM with interstitial lag, sensor noise, drift and
  dropouts; a smartwatch stream (heart rate, HRV/RMSSD, steps, sleep stage);
  and a food-logging app with incomplete, under-estimated entries.

Only the sensor streams and the EHR are visible to the digital twin. Hidden
truths are written to ``data/truth`` for validation only.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from . import config as C
from . import ehr as E
from . import physiology as PH

D = C.N_DAYS
T = D * C.MIN_PER_DAY
FESTIVAL_DAY = 12


def _food_weights(rng, cohort):
    """Per-patient preference weights over the food library (habits repeat)."""
    W = {}
    for i, r in cohort.iterrows():
        w = {}
        for f in C.FOODS:
            if r.region in f.regions:
                base = 3.0
            elif "*" in f.regions:
                base = 1.0
            else:
                base = 0.15
            disc = r.true_diet_discipline
            base *= (0.25 + 2.6 * disc) if f.healthy else (1.35 - disc)
            if f.key == "sweet":
                base *= 0.3 + 3 * r.true_sweet_tooth
            base *= np.exp(rng.normal(0, 0.7))
            w[f.key] = base
        W[r.patient_id] = w
    return W


def _choose(rng, weights, slot):
    keys = [f.key for f in C.FOODS if f.slot == slot]
    p = np.array([weights[k] for k in keys])
    return keys[rng.choice(len(keys), p=p / p.sum())]


def simulate_behaviour(cohort: pd.DataFrame, seed: int = C.SEED):
    rng = np.random.default_rng(seed + 1)
    P = len(cohort)
    steps = np.zeros((P, T), np.float32)
    stage = np.zeros((P, T), np.int8)          # 0 awake, 1 light, 2 deep, 3 REM
    fast = np.zeros((P, T), np.float32)        # mg carbohydrate entering fast chain
    slow = np.zeros((P, T), np.float32)
    meals, daily = [], []
    weights = _food_weights(rng, cohort)

    for i, r in cohort.reset_index(drop=True).iterrows():
        pid = r.patient_id
        port = np.clip(rng.normal(1.0, 0.12), 0.75, 1.3)
        stress = np.where(rng.random(D) < 0.18 + 0.45 * r.true_stress_prone,
                          rng.uniform(0.4, 1.0, D), rng.uniform(0.0, 0.25, D))
        weekend = np.array([(pd.Timestamp(C.START_DATE) + pd.Timedelta(days=d)).dayofweek >= 5 for d in range(D)])
        stress[weekend] *= 0.5
        ill = np.zeros(D, bool)
        if r.true_illness_day >= 0:
            ill[int(r.true_illness_day): int(r.true_illness_day) + int(rng.integers(2, 4))] = True
        adherent = rng.random(D) < r.true_med_adherence
        festival = rng.random() < 0.7

        # ---- sleep: nights -1 .. D-1 (night n starts on the evening of day n) ----
        wake_min = np.zeros(D + 1)
        night_stats = {}
        for n in range(-1, D):
            dn = max(n, 0)
            fri_sat = (pd.Timestamp(C.START_DATE) + pd.Timedelta(days=n)).dayofweek in (4, 5)
            dur_h = np.clip(r.true_sleep_mean_h + rng.normal(0, 0.65) - 0.9 * stress[dn] * (n >= 0)
                            + 0.5 * fri_sat + 0.8 * ill[dn] * (n >= 0), 3.5, 9.8)
            onset = n * 1440 + r.true_sleep_onset_min + rng.normal(0, 35) + 25 * stress[dn] + 45 * fri_sat
            a, b = int(max(onset, 0)), int(min(onset + dur_h * 60, T))
            deep_m, awake_m = 0, 0
            if b > a:
                seg = np.ones(b - a, np.int8)
                L = b - a
                cyc = 90 + rng.normal(0, 8)
                for c0 in np.arange(0, L, cyc):
                    frac = c0 / max(L, 1)
                    deep_len = max(0, int((38 - 45 * frac) * np.exp(rng.normal(0, 0.25))))
                    rem_len = int(8 + 26 * frac + rng.normal(0, 4))
                    s0 = int(c0 + 18)
                    seg[s0: s0 + deep_len] = 2
                    r0 = int(c0 + cyc - rem_len)
                    seg[max(r0, s0 + deep_len): int(c0 + cyc)] = 3
                for _ in range(rng.poisson(2 + 2 * stress[dn])):
                    w0 = int(rng.integers(0, L))
                    seg[w0: w0 + int(rng.integers(2, 9))] = 0
                seg[:int(rng.integers(8, 25))] = 1   # sleep latency counted as light/awake-in-bed
                stage[i, a:b] = seg
                deep_m, awake_m = int((seg == 2).sum()), int((seg == 0).sum())
            if n + 1 < D + 1 and n + 1 >= 0:
                wake_min[n + 1] = onset + dur_h * 60 - (n + 1) * 1440  # minute of day n+1
            night_stats[n] = (dur_h, deep_m / max(b - a, 1), awake_m)

        # ---- days ----------------------------------------------------------
        for d in range(D):
            t0 = d * 1440
            wake = int(np.clip(wake_min[d], 240, 720))
            sleep_on = int(np.clip(r.true_sleep_onset_min, 1260, 1560))
            dur_prev, deep_prev, _ = night_stats[d - 1]
            act_scale = 0.4 if ill[d] else 1.0
            # background household / office movement
            awake_idx = np.arange(t0 + wake, min(t0 + sleep_on, T))
            p_act = (0.09 if r.true_office_worker and not weekend[d] else 0.14) * act_scale
            burst = rng.random(len(awake_idx)) < p_act
            steps[i, awake_idx[burst]] += rng.uniform(10, 65, burst.sum())

            def walk(start, dur, spm):
                a = int(t0 + start)
                b = int(min(a + dur, T))
                if 0 <= a < b:
                    steps[i, a:b] = np.maximum(steps[i, a:b], rng.normal(spm, 6, b - a).clip(40, 140))

            morning_walk = (rng.random() < r.true_walk_habit * act_scale)
            if morning_walk:
                walk(wake + rng.uniform(10, 50), rng.uniform(20, 45), rng.normal(106, 6))
            if r.true_office_worker and not weekend[d] and not ill[d]:
                walk(9 * 60 + rng.normal(10, 15), rng.uniform(8, 18), 80)
                walk(18 * 60 + rng.normal(40, 20), rng.uniform(8, 18), 80)

            # meals
            day_meals = []
            if rng.random() < 0.7:  # morning chai with sugar, rarely logged
                day_meals.append(("chai", wake + rng.uniform(5, 25), 10.0, 65.0, "Morning chai", 0.25))
            for slot in ("breakfast", "lunch", "snack", "dinner"):
                if slot == "breakfast" and rng.random() < r.true_skip_breakfast:
                    continue
                if slot == "snack" and rng.random() > 0.5 + 0.4 * r.true_sweet_tooth:
                    continue
                key = _choose(rng, weights[pid], slot)
                if festival and d == FESTIVAL_DAY:
                    key = {"lunch": "thali", "snack": "sweet"}.get(slot, key)
                f = C.FOOD_BY_KEY[key]
                mt = getattr(r, f"true_{slot}_time") + rng.normal(0, 30) + (30 if weekend[d] else 0)
                if rng.random() < 0.12:                      # meetings, travel, guests...
                    mt += rng.choice([-1, 1]) * rng.uniform(60, 120)
                if slot == "breakfast":
                    mt = max(mt, wake + 20)
                if slot == "dinner" and weekend[d]:
                    mt += 25
                carbs = f.carbs * port * np.exp(rng.normal(0, 0.15)) * (1.1 if weekend[d] else 1.0)
                if slot in ("lunch", "dinner") and rng.random() < 0.12:   # eating out
                    key = rng.choice(["veg_biryani", "chicken_biryani", "fried_rice", "thali"])
                    f = C.FOOD_BY_KEY[key]
                    carbs = f.carbs * port * np.exp(rng.normal(0.1, 0.15))
                if festival and d == FESTIVAL_DAY:
                    carbs *= 1.2
                comp = 0.85 if weekend[d] else 1.0
                comp *= 0.6 if (festival and d == FESTIVAL_DAY) else 1.0
                day_meals.append((key, mt, carbs, f.gi, f.name, comp))
                if slot in ("lunch", "dinner") and rng.random() < r.true_postmeal_walk * act_scale:
                    walk(mt + rng.uniform(5, 20), rng.uniform(10, 25), rng.normal(95, 6))
            if rng.random() < 0.12 * act_scale:
                walk(19 * 60 + rng.normal(0, 30), rng.uniform(15, 35), 95)
            if rng.random() < 0.3:   # unplanned snack: office sweets, biscuits, a relative's visit
                k2 = rng.choice(["sweet", "chai_biscuit", "namkeen", "samosa_chai", "banana"])
                f2 = C.FOOD_BY_KEY[k2]
                day_meals.append((k2, rng.uniform(wake + 60, 22 * 60), f2.carbs * np.exp(rng.normal(0, 0.2)),
                                  f2.gi, f2.name, 0.6))

            for key, mt, carbs, gi, name, comp in day_meals:
                tm = int(t0 + np.clip(mt, 0, 1439))
                if tm >= T:
                    continue
                ff = float(PH.fast_fraction(gi))
                fast[i, tm] += carbs * 1000 * ff
                slow[i, tm] += carbs * 1000 * (1 - ff)
                logged = rng.random() < r.true_log_compliance * comp
                meals.append({"patient_id": pid, "day": d, "minute": tm, "food_key": key, "food": name,
                              "carbs_g": round(carbs, 1), "gi": gi, "logged": logged,
                              "logged_minute": int(np.clip(tm + rng.normal(0, 6), 0, T - 1)) if logged else -1,
                              "logged_carbs_g": round(carbs * np.exp(rng.normal(-0.08, 0.25)), 0) if logged else np.nan})
            daily.append({"patient_id": pid, "day": d, "stress": round(stress[d], 3), "ill": bool(ill[d]),
                          "med_adherent": bool(adherent[d]), "weekend": bool(weekend[d]),
                          "festival": bool(festival and d == FESTIVAL_DAY), "morning_walk": bool(morning_walk),
                          "prev_sleep_h": round(dur_prev, 2), "prev_deep_frac": round(deep_prev, 3)})
    return {"steps": steps, "stage": stage, "fast": fast, "slow": slow,
            "meals": pd.DataFrame(meals), "daily": pd.DataFrame(daily)}


def _day_to_min(arr):
    return np.repeat(arr, C.MIN_PER_DAY, axis=1)


def simulate_physiology(cohort: pd.DataFrame, beh: dict):
    """Run the ODEs for all patients in parallel at 1-minute resolution."""
    P = len(cohort)
    prm = E.effective_params(cohort)
    daily = beh["daily"].sort_values(["patient_id", "day"])
    pid_order = cohort["patient_id"].tolist()
    daily = daily.set_index("patient_id").loc[pid_order].reset_index()

    def dmat(col):
        return daily[col].to_numpy(float).reshape(P, D)

    stress, ill, adh = dmat("stress"), dmat("ill"), dmat("med_adherent")
    sleep_h, deep = dmat("prev_sleep_h"), dmat("prev_deep_frac")

    f_sleep = 1 - 0.045 * np.maximum(0, 7 - sleep_h) - 0.07 * (deep < 0.12)
    si_day = f_sleep * (1 - 0.22 * stress) * np.where(ill > 0, 0.62, 1.0)
    gb_day = (prm["gb_untreated"][:, None] - prm["gb_med_drop"][:, None] * np.where(adh > 0, 1.0, 0.3)
              + 14 * stress + 28 * ill)
    tod = np.arange(T) % C.MIN_PER_DAY
    circ = PH.circadian_si(tod)[None, :]
    dawn = PH.dawn_bump(tod)[None, :]

    # exercise memory -> transient insulin-sensitivity boost (hours)
    brisk = np.maximum(beh["steps"].astype(float) - 60, 0) / 100.0
    ex_mem = lfilter([1.0], [1.0, -np.exp(-1 / 360.0)], brisk, axis=1)
    ex_boost = 1 + 0.3 * np.minimum(1, ex_mem / 25.0)

    si_t = prm["SI"][:, None] * _day_to_min(si_day) * circ * ex_boost
    gb_t = _day_to_min(gb_day) + prm["dawn"][:, None] * dawn
    # sulfonylureas stimulate secretion even near normoglycaemia -> hypo risk
    h = prm["gb_untreated"] - prm["gb_med_drop"] - 35 * cohort["sulfonylurea"].to_numpy()
    p = {"SG": prm["SG"], "sigma": prm["sigma"], "h": h, "V": prm["V"], "sglt2": prm["sglt2"]}

    s = PH.init_state(P, gb_t[:, 0])
    IG = s["G"].copy()
    n5 = T // C.CGM_INTERVAL
    G5 = np.zeros((P, n5), np.float32)
    IG5 = np.zeros((P, n5), np.float32)
    gut5 = np.zeros((P, n5), np.float32)
    steps_f = beh["steps"].astype(float)
    for t in range(T):
        s = PH.step(s, p, 1.0, gb_t[:, t], si_t[:, t], steps_f[:, t], beh["fast"][:, t], beh["slow"][:, t])
        IG += (s["G"] - IG) / PH.TAU_ISF
        if t % C.CGM_INTERVAL == 0:
            k = t // C.CGM_INTERVAL
            G5[:, k], IG5[:, k] = s["G"], IG
            gut5[:, k] = s["F1"] + s["F2"] + s["S1"] + s["S2"]
    return {"G5": G5, "IG5": IG5, "gut5": gut5, "si_day": si_day, "gb_day": gb_day}


def simulate_sensors(cohort: pd.DataFrame, beh: dict, phys: dict, seed: int = C.SEED):
    """CGM + smartwatch streams on the 5-minute grid."""
    rng = np.random.default_rng(seed + 2)
    P, n5 = phys["G5"].shape
    k = C.CGM_INTERVAL
    # CGM: interstitial glucose * (1 + drift) + AR(1) noise, dropouts
    drift = rng.normal(0, 0.035, (P, 1)) + rng.normal(0, 0.03, (P, 1)) * np.sin(
        2 * np.pi * np.arange(n5)[None, :] / (288 * rng.uniform(5, 12, (P, 1))) + rng.uniform(0, 6, (P, 1)))
    e = rng.normal(0, 1, (P, n5))
    ar = lfilter([np.sqrt(1 - 0.7 ** 2)], [1, -0.7], e, axis=1)
    sd = cohort["true_cgm_noise"].to_numpy()[:, None]
    cgm = phys["IG5"] * (1 + drift) + ar * sd * (0.6 + 0.4 * phys["IG5"] / 150)
    cgm = np.clip(np.round(cgm), 40, 400).astype(float)
    for i in range(P):
        for _ in range(rng.poisson(n5 * 0.0015)):
            a = rng.integers(0, n5)
            cgm[i, a: a + rng.integers(1, 30)] = np.nan
    # sensor warm-up at start of each 14-day sensor
    cgm[:, 0:12] = np.nan
    cgm[:, 14 * 288: 14 * 288 + 12] = np.nan

    steps5 = beh["steps"].reshape(P, n5, k).sum(axis=2).round()
    stage5 = beh["stage"][:, k // 2::k][:, :n5]
    spm = steps5 / k
    daily = beh["daily"].set_index("patient_id").loc[cohort["patient_id"]].reset_index()
    stress5 = np.repeat(daily["stress"].to_numpy(float).reshape(P, -1), 288, axis=1)
    ill5 = np.repeat(daily["ill"].to_numpy(float).reshape(P, -1), 288, axis=1)
    tod = (np.arange(n5) * k) % 1440
    circ = 3 * np.cos(2 * np.pi * (tod - 16 * 60) / 1440)[None, :]
    rest = cohort["true_resting_hr"].to_numpy()[:, None]
    sleep_off = np.select([stage5 == 1, stage5 == 2, stage5 == 3], [-9, -13, -5], 0)
    pp = np.minimum(5, 4 * phys["gut5"] / 50000)
    hr_noise = lfilter([1], [1, -0.6], rng.normal(0, 2.2, (P, n5)), axis=1)
    hr = rest + 0.33 * spm + sleep_off + 6 * stress5 + 10 * ill5 + pp + circ + hr_noise
    hr = np.round(np.clip(hr, 40, 185), 1)

    base = cohort["true_hrv_base"].to_numpy()[:, None]
    fac = np.select([stage5 == 1, stage5 == 2, stage5 == 3], [1.25, 1.5, 1.0], 1.0)
    fac = fac * (1 - 0.5 * np.minimum(1, spm / 100)) * (1 - 0.35 * stress5) * np.where(ill5 > 0, 0.7, 1)
    fac = fac * (1 - 0.1 * np.minimum(1, phys["gut5"] / 50000))
    hrv = np.round(base * fac * np.exp(rng.normal(0, 0.16, (P, n5))), 1)
    # watch off-wrist gaps (charging ~1/day, ~30 min)
    for i in range(P):
        for d in range(C.N_DAYS):
            if rng.random() < 0.8:
                a = d * 288 + int(rng.integers(90, 250))
                L = int(rng.integers(3, 9))
                hr[i, a:a + L] = np.nan
                hrv[i, a:a + L] = np.nan
    return {"cgm": cgm, "hr": hr, "hrv": hrv, "steps5": steps5, "stage5": stage5}


def run(out_dir=C.DATA_DIR, n_patients: int = C.N_PATIENTS, seed: int = C.SEED, verbose=True):
    out_dir = __import__("pathlib").Path(out_dir)
    (out_dir / "ehr").mkdir(parents=True, exist_ok=True)
    (out_dir / "truth").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cohort = E.sample_cohort(n_patients, seed)
    beh = simulate_behaviour(cohort, seed)
    if verbose:
        print(f"  behaviour simulated ({time.time() - t0:.1f}s)")
    phys = simulate_physiology(cohort, beh)
    if verbose:
        print(f"  physiology simulated ({time.time() - t0:.1f}s)")
    sen = simulate_sensors(cohort, beh, phys, seed)

    P, n5 = sen["cgm"].shape
    ts = pd.Timestamp(C.START_DATE) + pd.to_timedelta(np.arange(n5) * C.CGM_INTERVAL, unit="min")
    wear = pd.DataFrame({
        "patient_id": np.repeat(cohort["patient_id"].to_numpy(), n5),
        "t": np.tile(np.arange(n5) * C.CGM_INTERVAL, P).astype(np.int32),
        "timestamp": np.tile(ts.to_numpy(), P),
        "cgm_mgdl": sen["cgm"].ravel().astype(np.float32),
        "hr_bpm": sen["hr"].ravel().astype(np.float32),
        "hrv_rmssd_ms": sen["hrv"].ravel().astype(np.float32),
        "steps": sen["steps5"].ravel().astype(np.int16),
        "sleep_stage": sen["stage5"].ravel().astype(np.int8),
    })
    wear.to_parquet(out_dir / "wearables_5min.parquet", index=False)
    meals = beh["meals"]
    logged = meals[meals.logged].copy()
    logged = logged.assign(t=logged.logged_minute,
                           timestamp=pd.Timestamp(C.START_DATE) + pd.to_timedelta(logged.logged_minute, unit="min"))
    logged[["patient_id", "t", "timestamp", "food", "logged_carbs_g", "gi"]].rename(
        columns={"logged_carbs_g": "carbs_g"}).to_csv(out_dir / "meal_log.csv", index=False)

    mean_g = np.nanmean(sen["cgm"], axis=1)
    pats, conds, meds, obs = E.build_ehr_tables(cohort, mean_g, seed)
    pats.to_csv(out_dir / "ehr" / "patients.csv", index=False)
    conds.to_csv(out_dir / "ehr" / "conditions.csv", index=False)
    meds.to_csv(out_dir / "ehr" / "medications.csv", index=False)
    obs.to_csv(out_dir / "ehr" / "observations.csv", index=False)

    # hidden truth (validation only)
    cohort.to_csv(out_dir / "truth" / "cohort_hidden.csv", index=False)
    meals.to_csv(out_dir / "truth" / "meals_true.csv", index=False)
    beh["daily"].to_csv(out_dir / "truth" / "daily_context.csv", index=False)
    pd.DataFrame({"patient_id": wear["patient_id"], "t": wear["t"],
                  "glucose_true": phys["G5"].ravel().astype(np.float32)}).to_parquet(
        out_dir / "truth" / "glucose_true_5min.parquet", index=False)
    if verbose:
        cg = sen["cgm"]
        tir = np.nanmean((cg >= 70) & (cg <= 180)) * 100
        tar = np.nanmean(cg > 180) * 100
        tbr = np.nanmean(cg < 70) * 100
        print(f"  sensors + EHR written ({time.time() - t0:.1f}s) | mean CGM {np.nanmean(cg):.0f} mg/dL | "
              f"TIR {tir:.0f}%  TAR {tar:.0f}%  TBR {tbr:.1f}% | meals {len(meals)} (logged {meals.logged.mean():.0%})")
    return cohort
