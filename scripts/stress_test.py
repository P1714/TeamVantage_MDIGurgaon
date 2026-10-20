"""Model-mismatch stress test -> reports/stress_test.json

The main synthetic benchmark generates patients with the same physiology equations the Twin uses, which
flatters the Twin. Here the *generator* is deliberately different from the Twin's assumptions:

* gut absorption: fast chain x0.65, slow chain x1.5, a different GI -> fast/slow split
* insulin kinetics: clearance x1.3, secretion delay x0.7 (slower response)
* sensor: interstitial lag 13 min instead of 8; CGM noise x1.6
* insulin-sensitivity rhythm: amplitude 0.25 (vs 0.12), peak at 08:00 (vs 11:00); wider dawn rise
* behaviour: meal-logging compliance x0.7 (more missing meals)

The Twin and all models are used *unchanged* (models trained on the matched cohort are transferred, not retrained).
We report what degrades: risk discrimination and alerting, physiology recovery and what-if accuracy.
"""
import json
import shutil
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from madhutwin import config as C  # noqa: E402
from madhutwin import ehr as E  # noqa: E402
from madhutwin import features as FE  # noqa: E402
from madhutwin import model as M  # noqa: E402
from madhutwin import physiology as PH  # noqa: E402
from madhutwin import simulate as S  # noqa: E402
from madhutwin import twin as TW  # noqa: E402
from madhutwin import whatif as W  # noqa: E402

N_PAT = 90
SEED = C.SEED + 101
OUT = ROOT / "data" / "stress"
ORIG = {k: getattr(PH, k) for k in ("K_FAST", "K_SLOW", "N_INS", "ALPHA", "TAU_ISF")}
ORIG_FN = {k: getattr(PH, k) for k in ("circadian_si", "dawn_bump", "fast_fraction")}
MISMATCH = {"K_FAST": ORIG["K_FAST"] * 0.65, "K_SLOW": ORIG["K_SLOW"] * 1.5, "N_INS": ORIG["N_INS"] * 1.3,
            "ALPHA": ORIG["ALPHA"] * 0.7, "TAU_ISF": 13.0}


def generator_physics(on: bool):
    for k, v in (MISMATCH if on else ORIG).items():
        setattr(PH, k, v)
    if on:
        PH.circadian_si = lambda m: 1.0 + 0.25 * np.cos(2 * np.pi * (np.asarray(m) - 8 * 60) / 1440)
        PH.dawn_bump = lambda m: np.exp(-0.5 * ((np.asarray(m, float) - 6.0 * 60) / 110.0) ** 2)
        PH.fast_fraction = lambda gi: np.clip((np.asarray(gi, float) - 15.0) / 70.0, 0.0, 1.0)
    else:
        for k, f in ORIG_FN.items():
            setattr(PH, k, f)


def main():
    t0 = time.time()
    if OUT.exists():
        shutil.rmtree(OUT)
    sample = E.sample_cohort

    def mismatched_cohort(n, seed):
        c = sample(n, seed)
        c["true_cgm_noise"] = c["true_cgm_noise"] * 1.6
        c["true_log_compliance"] = c["true_log_compliance"] * 0.7
        return c

    E.sample_cohort = mismatched_cohort
    generator_physics(True)
    try:
        S.run(out_dir=OUT, n_patients=N_PAT, seed=SEED, verbose=False)
    finally:
        generator_physics(False)
        E.sample_cohort = sample
    print(f"[{time.time() - t0:.0f}s] mismatched cohort simulated ({N_PAT} patients)", flush=True)

    # everything below uses the Twin's ORIGINAL assumptions and the models trained on the matched cohort
    data_dir0 = C.DATA_DIR
    C.DATA_DIR = OUT
    try:
        wear = pd.read_parquet(OUT / "wearables_5min.parquet")
        meal_log = pd.read_csv(OUT / "meal_log.csv")
        ehr = E.load_ehr_features(OUT / "ehr")
        pids = ehr.patient_id.tolist()
        A = TW.load_arrays(wear, meal_log, pids)
        tw = TW.calibrate(A, ehr, verbose=False)
        tw_cgm = TW.calibrate(A, ehr, prior_weight=0.0, verbose=False)
        states = TW.observe(A, tw)
        habits = TW.learn_habits(A, meal_log)
        nh = TW.next_habitual_meal(A, habits)
        proj = TW.project(A, tw, states)
        proj_h = TW.project(A, tw, states, habit=nh)
        df = FE.build(A, ehr, tw, states, proj, proj_h, nh)
        print(f"[{time.time() - t0:.0f}s] twin fitted and {len(df):,} decision rows built", flush=True)

        card = json.loads((ROOT / "models" / "model_card.json").read_text())
        cols, thr = card["features"], card["threshold"]
        risk = lgb.Booster(model_file=str(ROOT / "models" / "spike_risk_2h.txt"))
        df["risk"] = risk.predict(df[cols])
        slope = df["g_delta_30m"] / 30
        trend = df.g_now + slope.fillna(0) * 60
        pidx = {p: i for i, p in enumerate(A["pids"])}
        ev = M.event_level(df, A["cgm"], pidx, thr)

        h = pd.read_csv(OUT / "truth" / "cohort_hidden.csv").set_index("patient_id").loc[tw.patient_id]
        r = lambda a, b: float(spearmanr(a, b)[0])
        e2 = ehr.set_index("patient_id").loc[tw.patient_id]
        recovery = {"twin_fused": r(tw.twin_SI, h.true_SI), "cgm_only_twin": r(tw_cgm.twin_SI, h.true_SI),
                    "ehr_only_homa_ir": -r(e2.homa_ir, h.true_SI)}

        # what-if: truth re-simulated with the MISMATCHED physics, Twin prediction with its own physics
        tp = W.true_params(pids)
        generator_physics(True)
        try:
            truth = {sc: W.summarise(W.run(tp, sp["meal"][1], sp["meal"][2], sp["walk"])) for sc, sp in W.SCENARIOS.items()}
        finally:
            generator_physics(False)
        twin_p = W.twin_params(tw.set_index("patient_id").loc[pids].reset_index())
        pred = {sc: W.summarise(W.run(twin_p, sp["meal"][1], sp["meal"][2], sp["walk"])) for sc, sp in W.SCENARIOS.items()}
        whatif = {}
        for sc in ("walk_20", "millet_swap", "portion_minus30"):
            dt = truth[sc]["peak"] - truth["baseline"]["peak"]
            dp = pred[sc]["peak"] - pred["baseline"]["peak"]
            whatif[sc] = {"effect_corr": float(np.corrcoef(dp, dt)[0, 1]), "effect_mae_mgdl": float(np.mean(np.abs(dp - dt))),
                          "true_mean_effect": float(dt.mean()), "pred_mean_effect": float(dp.mean())}
    finally:
        C.DATA_DIR = data_dir0

    matched = json.loads((ROOT / "reports" / "metrics.json").read_text())
    W_m = matched["whatif_validation"]["models"]["Personalised twin (EHR + CGM)"]
    out = {
        "description": __doc__.strip().split("\n\n")[1],
        "mismatches": {"gut_fast_x": 0.65, "gut_slow_x": 1.5, "insulin_clearance_x": 1.3, "secretion_delay_rate_x": 0.7,
                       "sensor_lag_min": 13, "cgm_noise_x": 1.6, "si_rhythm_amplitude": 0.25, "si_rhythm_peak": "08:00",
                       "meal_logging_x": 0.7, "gi_split": "(GI-15)/70 instead of (GI-25)/80"},
        "n_patients": N_PAT, "decision_rows": int(len(df)), "prevalence": float(df.label.mean()),
        "risk_model_transferred": {**M.binary_metrics(df.label, df.risk),
                                   "cgm_trend_baseline_auroc": M.binary_metrics(df.label, trend)["auroc"]},
        "event_level": ev,
        "insulin_sensitivity_recovery_spearman": recovery,
        "whatif_effects": whatif,
        "matched_reference": {
            "auroc": matched["final_model"]["auroc"],
            "cgm_trend_baseline_auroc": matched["baselines"]["CGM trend extrapolation"]["auroc"],
            "detected_lead_ge_30min": matched["event_level"]["detected_lead_ge_30min"],
            "false_alerts_per_patient_day": matched["event_level"]["false_alerts_per_patient_day"],
            "si_recovery_twin_fused": matched["twin_recovery"]["insulin_sensitivity"]["twin_fused"],
            "whatif_effect_corr": {sc: W_m[f"{sc}_effect_corr"] for sc in ("walk_20", "millet_swap", "portion_minus30")}},
    }
    (ROOT / "reports" / "stress_test.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("risk_model_transferred", "insulin_sensitivity_recovery_spearman", "whatif_effects")}, indent=1))
    print(f"[{time.time() - t0:.0f}s] wrote reports/stress_test.json")


if __name__ == "__main__":
    main()
