"""Glucose-insulin physiology used by both the *virtual patient* simulator and
the *digital twin*.

Model
-----
An extended Bergman minimal model with a two-speed gut, delayed beta-cell
secretion, exercise-dependent glucose uptake and renal clearance. It is
vectorised across patients (or across parameter candidates) so an entire
cohort advances in lock-step with one NumPy operation per equation.

State (per row)                         Units
    G   plasma glucose                  mg/dL
    X   remote insulin action           1/min
    I   plasma insulin above basal      uU/mL
    Y   beta-cell secretion signal      uU/mL/min
    F1,F2  fast-absorbing gut chain     mg carbohydrate
    S1,S2  slow-absorbing gut chain     mg carbohydrate

    dG/dt  = -(SG + X + k_ex*steps) * G + SG * Gb(t) + Ra/V - renal(G)
    dX/dt  = -p2 * X + p2 * SI(t) * I
    dI/dt  = -n * I + Y
    dY/dt  =  alpha * (sigma * max(G - h, 0) - Y)
    Ra     =  f * (kf * F2 + ks * S2)

A meal of C grams with glycaemic index GI is split between the fast and slow
chains with fast fraction ``fast_fraction(GI)`` so a high-GI breakfast rises
sooner and higher than a millet lunch with the same carbohydrate load.

The same equations are re-implemented in ``dashboard/app.js`` so the doctor's
"what-if" simulator in the browser runs the exact same twin.
"""
from __future__ import annotations

import numpy as np

# fixed population constants
P2 = 0.025          # 1/min   insulin action decay
N_INS = 0.12        # 1/min   insulin clearance
ALPHA = 0.06        # 1/min   secretion delay (~17 min)
K_FAST = 0.028      # 1/min   fast gut absorption
K_SLOW = 0.011      # 1/min   slow gut absorption
F_BIO = 0.9         # carbohydrate bioavailability
K_EX = 2.0e-4       # extra glucose uptake per (step/min) per min
RENAL_THR = 180.0   # mg/dL
RENAL_K = 0.004     # 1/min above threshold
SGLT2_THR = 130.0
SGLT2_K = 0.007
TAU_ISF = 8.0       # min, plasma -> interstitial lag seen by the CGM


def fast_fraction(gi):
    """Fraction of a meal's carbohydrate that is absorbed on the fast chain."""
    return np.clip((np.asarray(gi, dtype=float) - 25.0) / 80.0, 0.0, 1.0)


def init_state(n: int, gb) -> dict:
    gb = np.broadcast_to(np.asarray(gb, dtype=float), (n,)).copy()
    z = np.zeros(n)
    return {"G": gb.copy(), "X": z.copy(), "I": z.copy(), "Y": z.copy(),
            "F1": z.copy(), "F2": z.copy(), "S1": z.copy(), "S2": z.copy()}


def step(s: dict, p: dict, dt: float, gb_t, si_t, steps_t, carbs_fast_mg, carbs_slow_mg):
    """Advance every row by ``dt`` minutes (explicit Euler; stable for dt<=5).

    ``p`` holds per-row arrays: SG, sigma, h, V, sglt2 (0/1).
    ``gb_t`` / ``si_t`` are the *time-varying* basal glucose and insulin
    sensitivity for this step (circadian, sleep, stress and illness effects
    are folded into these by the caller).
    """
    G, X, I, Y = s["G"], s["X"], s["I"], s["Y"]
    F1, F2, S1, S2 = s["F1"], s["F2"], s["S1"], s["S2"]

    ra = F_BIO * (K_FAST * F2 + K_SLOW * S2)                     # mg/min
    thr = np.where(p["sglt2"] > 0, SGLT2_THR, RENAL_THR)
    kr = np.where(p["sglt2"] > 0, SGLT2_K, RENAL_K)
    renal = kr * np.maximum(G - thr, 0.0)

    dG = -(p["SG"] + X + K_EX * steps_t) * G + p["SG"] * gb_t + ra / p["V"] - renal
    dX = -P2 * X + P2 * si_t * I
    dI = -N_INS * I + Y
    dY = ALPHA * (p["sigma"] * np.maximum(G - p["h"], 0.0) - Y)

    s["G"] = np.clip(G + dt * dG, 35.0, 600.0)
    s["X"] = X + dt * dX
    s["I"] = np.maximum(I + dt * dI, 0.0)
    s["Y"] = np.maximum(Y + dt * dY, 0.0)
    # gut chains (meal carbohydrate enters compartment 1 as a bolus)
    s["F1"] = F1 + dt * (-K_FAST * F1) + carbs_fast_mg
    s["F2"] = F2 + dt * (K_FAST * (F1 - F2))
    s["S1"] = S1 + dt * (-K_SLOW * S1) + carbs_slow_mg
    s["S2"] = S2 + dt * (K_SLOW * (S1 - S2))
    return s


def circadian_si(minute_of_day):
    """Insulin sensitivity is highest late morning and lowest late evening."""
    return 1.0 + 0.12 * np.cos(2 * np.pi * (np.asarray(minute_of_day) - 11 * 60) / 1440)


def dawn_bump(minute_of_day):
    """0..1 bump between ~03:30 and ~09:30 (dawn phenomenon, cortisol/GH)."""
    m = np.asarray(minute_of_day, dtype=float)
    return np.exp(-0.5 * ((m - 6.5 * 60) / 70.0) ** 2)
