"""What-if (counterfactual) simulation, and a test of whether it is correct.

A risk model can only say *what will probably happen*. A digital twin can also
answer *what would happen if...*: "if she walked for 15 minutes after lunch",
"if he swapped white rice for foxtail millet". Because our virtual patients
come from a simulator with known ground truth, we can check those answers:
the true patient is re-simulated under each intervention and compared with
what the twin predicted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import ehr as E
from . import physiology as PH

SCENARIOS = {
    "baseline":        {"meal": ("Rice + dal + sabzi", 85, 72), "walk": None, "label": "Usual lunch: rice + dal + sabzi (85 g carbs)"},
    "walk_20":         {"meal": ("Rice + dal + sabzi", 85, 72), "walk": (15, 20, 105), "label": "Same lunch + 20-min brisk walk, 15 min after eating"},
    "millet_swap":     {"meal": ("Foxtail millet + dal + sabzi", 62, 55), "walk": None, "label": "Swap rice for foxtail millet"},
    "portion_minus30": {"meal": ("Rice + dal + sabzi (smaller)", 60, 72), "walk": None, "label": "30% smaller rice portion"},
}


def run(prm: dict, meal_carbs, meal_gi, walk=None, start_tod=12 * 60 + 30, meal_at=30, minutes=300):
    """Vectorised 1-minute simulation of a single-meal scenario.

    prm: dict of per-row arrays SG, SI, sigma, h, Gb, dawn, V, sglt2.
    Returns glucose trajectories [rows, minutes] (interstitial, as a CGM sees it).
    """
    n = len(prm["SI"])
    s = PH.init_state(n, prm["Gb"])
    IG = s["G"].copy()
    ex_mem = np.zeros(n)
    out = np.zeros((n, minutes))
    ff = float(PH.fast_fraction(meal_gi))
    for t in range(minutes):
        tod = (start_tod + t) % 1440
        spm = np.zeros(n)
        if walk is not None and meal_at + walk[0] <= t < meal_at + walk[0] + walk[1]:
            spm[:] = walk[2]
        ex_mem = ex_mem * np.exp(-1 / 360) + np.maximum(spm - 60, 0) / 100
        si_t = prm["SI"] * PH.circadian_si(tod) * (1 + 0.3 * np.minimum(1, ex_mem / 25))
        gb_t = prm["Gb"] + prm["dawn"] * PH.dawn_bump(tod)
        mg = np.full(n, meal_carbs * 1000.0) if t == meal_at else np.zeros(n)
        s = PH.step(s, prm, 1.0, gb_t, si_t, spm, mg * ff, mg * (1 - ff))
        IG += (s["G"] - IG) / PH.TAU_ISF
        out[:, t] = IG
    return out


def summarise(traj, meal_at=30):
    pre = traj[:, meal_at]
    return {"peak": traj.max(1), "rise": traj.max(1) - pre, "tar_min": (traj > C.HYPER_L1).sum(1)}


def true_params(pids):
    h = pd.read_csv(C.DATA_DIR / "truth" / "cohort_hidden.csv").set_index("patient_id").loc[pids].reset_index()
    ep = E.effective_params(h)
    gb = ep["gb_untreated"] - ep["gb_med_drop"]
    return {"SG": ep["SG"], "SI": ep["SI"], "sigma": ep["sigma"], "Gb": gb,
            "h": gb - 35 * h["sulfonylurea"].to_numpy(), "dawn": ep["dawn"], "V": ep["V"], "sglt2": ep["sglt2"]}


def twin_params(tw: pd.DataFrame, sg=0.016):
    return {"SG": np.full(len(tw), sg), "SI": tw.twin_SI.to_numpy(), "sigma": tw.twin_sigma.to_numpy(),
            "Gb": tw.twin_Gb.to_numpy(), "h": tw.twin_Gb.to_numpy(), "dawn": tw.twin_dawn.to_numpy(),
            "V": tw.twin_V.to_numpy(), "sglt2": tw.twin_sglt2.to_numpy()}


def validate(tw_personal: pd.DataFrame, tw_prior: pd.DataFrame, pids):
    """Compare twin-predicted intervention effects with the hidden truth."""
    tw_personal = tw_personal.set_index("patient_id").loc[pids].reset_index()
    tw_prior = tw_prior.set_index("patient_id").loc[pids].reset_index()
    pop = tw_personal.copy()
    for c in ["twin_SI", "twin_sigma", "twin_Gb", "twin_dawn"]:
        pop[c] = tw_personal[c].median()
    variants = {"truth": true_params(pids), "Personalised twin (EHR + CGM)": twin_params(tw_personal),
                "EHR-only twin (day 0)": twin_params(tw_prior), "Population-average model": twin_params(pop)}
    res = {v: {} for v in variants}
    for v, prm in variants.items():
        for sc, spec in SCENARIOS.items():
            res[v][sc] = summarise(run(prm, spec["meal"][1], spec["meal"][2], spec["walk"]))
    out = {"n_patients": len(pids), "scenarios": {k: v["label"] for k, v in SCENARIOS.items()}, "models": {}}
    truth = res["truth"]
    for v in variants:
        if v == "truth":
            continue
        m = {}
        m["baseline_rise_mae"] = float(np.mean(np.abs(res[v]["baseline"]["rise"] - truth["baseline"]["rise"])))
        m["baseline_rise_rank_corr"] = float(pd.Series(res[v]["baseline"]["rise"]).corr(
            pd.Series(truth["baseline"]["rise"]), method="spearman"))
        m["baseline_peak_rank_corr"] = float(pd.Series(res[v]["baseline"]["peak"]).corr(
            pd.Series(truth["baseline"]["peak"]), method="spearman"))
        for sc in ("walk_20", "millet_swap", "portion_minus30"):
            d_true = truth[sc]["peak"] - truth["baseline"]["peak"]
            d_pred = res[v][sc]["peak"] - res[v]["baseline"]["peak"]
            m[f"{sc}_effect_mae"] = float(np.mean(np.abs(d_pred - d_true)))
            m[f"{sc}_effect_corr"] = float(np.corrcoef(d_pred, d_true)[0, 1]) if np.std(d_pred) > 1e-6 else 0.0
            m[f"{sc}_true_mean_effect"] = float(d_true.mean())
            m[f"{sc}_pred_mean_effect"] = float(d_pred.mean())
            tar_t = truth[sc]["tar_min"] - truth["baseline"]["tar_min"]
            tar_p = res[v][sc]["tar_min"] - res[v]["baseline"]["tar_min"]
            m[f"{sc}_tar_effect_mae_min"] = float(np.mean(np.abs(tar_p - tar_t)))
        out["models"][v] = m
    return out, res
