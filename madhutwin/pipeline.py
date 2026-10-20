"""End-to-end pipeline:  python -m madhutwin.pipeline  [--regen] [--skip-report]

1. simulate   synthetic EHR + wearables for the cohort       -> data/
2. twin       calibrate, observe, learn habits, project        -> data/processed/twin_params.csv
3. features   fuse all streams into decision-time rows          -> data/processed/features.parquet
4. train      ablations, final model, forecasts, evaluation     -> models/, reports/metrics.json
5. report     figures for README / deck                         -> reports/figures/
6. dashboard  export showcase patients for the doctor console   -> dashboard/data.js
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config as C
from . import ehr as E
from . import features as FE
from . import model as M
from . import simulate as S
from . import twin as TW
from . import evaluation as EV
import lightgbm as lgb

PROC = C.DATA_DIR / "processed"


def log(msg, t0):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def build_twin(t0):
    wear = pd.read_parquet(C.DATA_DIR / "wearables_5min.parquet")
    meal_log = pd.read_csv(C.DATA_DIR / "meal_log.csv")
    ehr = E.load_ehr_features(C.DATA_DIR / "ehr")
    pids = ehr.patient_id.tolist()
    A = TW.load_arrays(wear, meal_log, pids)
    log(f"loaded {len(pids)} patients x {A['n5']} CGM readings", t0)
    tw = TW.calibrate(A, ehr)
    log("twin personalised (SI, sigma, Gb, dawn) from week-1 data + EHR priors", t0)
    states = TW.observe(A, tw)
    habits = TW.learn_habits(A, meal_log)
    nh = TW.next_habitual_meal(A, habits)
    proj = TW.project(A, tw, states)
    proj_h = TW.project(A, tw, states, habit=nh)
    log("twin state estimation + 2-hour projections done", t0)
    nightly = TW.nightly_update(A, tw)
    np.savez(PROC / "twin_nightly.npz", **nightly)
    log("nightly re-personalisation of SI and basal glucose done (rolling 3-day windows)", t0)
    A["nightly"] = nightly
    return A, ehr, tw, states, habits, nh, proj, proj_h, meal_log


def twin_recovery(tw, tw_cgm_only=None):
    """How well does the personalised twin recover each patient's hidden physiology?"""
    h = pd.read_csv(C.DATA_DIR / "truth" / "cohort_hidden.csv").set_index("patient_id").loc[tw.patient_id]
    ep = E.effective_params(h.reset_index())
    ehr = E.load_ehr_features(C.DATA_DIR / "ehr").set_index("patient_id").loc[tw.patient_id]
    true_gb = ep["gb_untreated"] - ep["gb_med_drop"]
    r = lambda a, b: float(spearmanr(a, b)[0])
    out = {
        "insulin_sensitivity": {"twin_fused": r(tw.twin_SI, h.true_SI), "ehr_only_homa_ir": -r(ehr.homa_ir, h.true_SI)},
        "beta_cell_function": {"twin_fused": r(tw.twin_sigma, ep["sigma"]), "ehr_only_homa_b": r(ehr.homa_b, ep["sigma"])},
        "basal_glucose": {"twin_fused": r(tw.twin_Gb, true_gb), "ehr_only_fpg": r(ehr.fpg, true_gb)},
        "disposition_index": {"twin_fused": r(tw.twin_SI * tw.twin_sigma, h.true_SI * ep["sigma"]),
                              "ehr_only_homa": r(ehr.homa_b / ehr.homa_ir, h.true_SI * ep["sigma"])},
        "dawn_phenomenon": {"twin_fused": r(tw.twin_dawn, h.true_dawn_amp)},
        "note": "Spearman rank correlation between twin estimate and hidden simulator ground truth, all 300 patients.",
    }
    if tw_cgm_only is not None:
        c = tw_cgm_only.set_index("patient_id").loc[tw.patient_id]
        out["insulin_sensitivity"]["cgm_only_twin"] = r(c.twin_SI, h.true_SI)
        out["beta_cell_function"]["cgm_only_twin"] = r(c.twin_sigma, ep["sigma"])
        out["basal_glucose"]["cgm_only_twin"] = r(c.twin_Gb, true_gb)
        out["disposition_index"]["cgm_only_twin"] = r(c.twin_SI * c.twin_sigma, h.true_SI * ep["sigma"])
    return out


def train_and_evaluate(df, A, split, t0):
    tr = df[df.patient_id.isin(split["train"])]
    va = df[df.patient_id.isin(split["val"])]
    te = df[df.patient_id.isin(split["test"])].copy()
    metrics = {"split": {k: len(v) for k, v in split.items()},
               "rows": {"train": len(tr), "val": len(va), "test": len(te)},
               "prevalence_test": float(te.label.mean())}

    # ---- simple baselines
    base = {}
    slope = te["g_delta_30m"] / 30
    base["CGM trend extrapolation"] = M.binary_metrics(te.label, te.g_now + slope.fillna(0) * 60)
    base["Time-of-day only"] = M.binary_metrics(
        te.label, M.train_classifier(tr, va, ["tod_sin", "tod_cos", "weekend"]).predict(te[["tod_sin", "tod_cos", "weekend"]]))
    base["Physiology twin alone (projected peak)"] = M.binary_metrics(te.label, te.twin_proj_peak_habit)
    metrics["baselines"] = base

    # ---- ablations
    abl, models, preds = {}, {}, {}
    for name, groups in M.ABLATIONS.items():
        cols = M.feature_cols(df, groups)
        m = M.train_classifier(tr, va, cols)
        p = m.predict(te[cols], num_iteration=m.best_iteration)
        early = (te.label == 0) | (te.tte_min >= 60)
        abl[name] = M.binary_metrics(te.label, p) | {"n_features": len(cols), "brier": float(
            np.mean((p - te.label) ** 2)),
            "auroc_early_warning_ge60min": float(M.roc_auc_score(te.label[early], p[early.to_numpy()]))}
        models[name] = (m, cols)
        preds[name] = p
        abl[name]["ci"] = EV.patient_bootstrap(te.assign(_s=p), "_s", n_boot=200)
        log(f"  {name:40s} AUROC {abl[name]['auroc']:.3f}  AUPRC {abl[name]['auprc']:.3f}  "
            f"early(>=60m) AUROC {abl[name]['auroc_early_warning_ge60min']:.3f}", t0)
    metrics["ablation"] = abl

    m, cols = models[M.FULL]
    m.save_model(str(C.MODEL_DIR / "spike_risk_2h.txt"), num_iteration=m.best_iteration)
    pva = m.predict(va[cols], num_iteration=m.best_iteration)
    thr = M.choose_threshold(va.label.to_numpy(), pva)
    te["risk"] = m.predict(te[cols], num_iteration=m.best_iteration)
    yp = te.risk >= thr
    tp, fp = int((yp & (te.label == 1)).sum()), int((yp & (te.label == 0)).sum())
    fn, tn = int((~yp & (te.label == 1)).sum()), int((~yp & (te.label == 0)).sum())
    metrics["final_model"] = {
        "threshold": thr, "auroc": abl[M.FULL]["auroc"], "auprc": abl[M.FULL]["auprc"],
        "brier": abl[M.FULL]["brier"], "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
        "ppv": tp / max(tp + fp, 1), "npv": tn / max(tn + fn, 1), "best_iteration": int(m.best_iteration),
        "reliability": M.reliability(te.label.to_numpy(), te.risk.to_numpy()),
    }
    pidx = {p: i for i, p in enumerate(A["pids"])}
    metrics["event_level"] = M.event_level(te, A["cgm"], pidx, thr)
    # operating points a clinic could choose (alert burden vs. sensitivity)
    metrics["operating_points"] = [{"threshold": float(t)} | M.event_level(te, A["cgm"], pidx, float(t))
                                   for t in (0.25, round(thr, 2), 0.5, 0.6, 0.7, 0.8)]
    log(f"  final model thr={thr:.2f}  events detected {metrics['event_level']['detected_any_lead']:.1%}, "
        f">=30 min lead {metrics['event_level']['detected_lead_ge_30min']:.1%}, "
        f"median lead {metrics['event_level']['median_lead_min']:.0f} min, "
        f"false alerts/day {metrics['event_level']['false_alerts_per_patient_day']:.2f}", t0)

    # ---- what does the physiology twin add? (measured, not asserted)
    nt_name = "EHR + wearables (fusion)"
    te["_p_nt"] = preds[nt_name]
    tv = {"auroc_gain": EV.paired_bootstrap_delta(te, "_p_nt", "risk"),
          "auroc_without_twin": abl[nt_name]["auroc"], "auroc_with_twin": abl[M.FULL]["auroc"],
          "physics_twin_alone_auroc": metrics["baselines"]["Physiology twin alone (projected peak)"]["auroc"]}
    # false alerts when both models are tuned to flag the same share of spikes >=30 min early
    def thr_for(score_col, target):
        lo, hi = 0.02, 0.98
        for _ in range(14):
            mid = (lo + hi) / 2
            ev_ = M.event_level(te.assign(risk=te[score_col]), A["cgm"], pidx, mid)
            lo, hi = (mid, hi) if ev_["detected_lead_ge_30min"] > target else (lo, mid)
        return M.event_level(te.assign(risk=te[score_col]), A["cgm"], pidx, lo)
    tgt = metrics["event_level"]["detected_lead_ge_30min"]
    a_ = thr_for("_p_nt", tgt)
    b_ = M.event_level(te, A["cgm"], pidx, thr)
    tv["matched_sensitivity"] = {"target_detected_ge30": tgt,
                                 "false_alerts_per_day_without_twin": a_["false_alerts_per_patient_day"],
                                 "false_alerts_per_day_with_twin": b_["false_alerts_per_patient_day"],
                                 "median_lead_without_twin": a_["median_lead_min"], "median_lead_with_twin": b_["median_lead_min"]}
    metrics["twin_value"] = tv
    log(f"  twin adds AUROC {tv['auroc_gain']['auroc']['delta']:+.4f} [{tv['auroc_gain']['auroc']['lo']:+.4f}, "
        f"{tv['auroc_gain']['auroc']['hi']:+.4f}]; false alerts/day at equal sensitivity "
        f"{a_['false_alerts_per_patient_day']:.2f} -> {b_['false_alerts_per_patient_day']:.2f}", t0)

    # ---- genetics: do genotype features help a 2-hour prediction? (parsimony check)
    gcols = cols + [c for c in df.columns if c.startswith("ehr_gen_")]
    mg = M.train_classifier(tr, va, gcols)
    metrics["genetics_ablation"] = {"auroc_without_genetics": abl[M.FULL]["auroc"],
                                    "auroc_with_genetics": float(M.roc_auc_score(te.label, mg.predict(te[gcols], num_iteration=mg.best_iteration)))}

    # ---- uncertainty: bootstrap ensemble over training patients -> risk +/- sd
    rng = np.random.default_rng(C.SEED)
    trp = np.array(sorted(tr.patient_id.unique()))
    ens = []
    for b in range(5):
        pick = rng.choice(trp, len(trp), replace=True)
        tb = pd.concat([tr[tr.patient_id == q] for q in pick])
        # light members (15 leaves) so the ensemble can ship with the API and reproduce the console's +/- exactly
        mb = lgb.train(dict(M.PARAMS, num_leaves=15, seed=C.SEED + b), lgb.Dataset(tb[cols], tb.label), int(m.best_iteration))
        mb.save_model(str(C.MODEL_DIR / f"risk_ensemble_{b}.txt"))
        ens.append(mb.predict(te[cols]))
    te["risk_sd"] = np.std(np.vstack(ens), axis=0)
    sd = te.risk_sd.to_numpy()
    conf = np.where(sd < 0.04, "High", np.where(sd < 0.08, "Medium", "Low"))
    metrics["uncertainty"] = {"median_risk_sd": float(np.median(sd)),
                              "share_high_confidence": float(np.mean(conf == "High")),
                              "auroc_high_confidence": float(M.roc_auc_score(te.label[conf == "High"], te.risk[conf == "High"])),
                              "auroc_low_or_medium": float(M.roc_auc_score(te.label[conf != "High"], te.risk[conf != "High"])),
                              "ece": EV.expected_calibration_error(te.label, te.risk)}
    log(f"  uncertainty: median sd {np.median(sd):.3f}, ECE {metrics['uncertainty']['ece']:.3f}", t0)

    # ---- when, and how high? time-to-onset and 2-h peak regressors
    pos_tr, pos_va = tr[tr.label == 1], va[va.label == 1]
    p_ = dict(M.PARAMS, objective="regression_l1", learning_rate=0.08)
    mt = lgb.train(p_, lgb.Dataset(pos_tr[cols], pos_tr.tte_min), 800,
                   valid_sets=[lgb.Dataset(pos_va[cols], pos_va.tte_min)], callbacks=[lgb.early_stopping(50, verbose=False)])
    mt.save_model(str(C.MODEL_DIR / "time_to_onset.txt"), num_iteration=mt.best_iteration)
    te["tte_pred"] = mt.predict(te[cols], num_iteration=mt.best_iteration)
    okp = tr.y_peak2h.notna()
    mp = lgb.train(p_, lgb.Dataset(tr.loc[okp, cols], tr.loc[okp, "y_peak2h"]), 800,
                   valid_sets=[lgb.Dataset(va.loc[va.y_peak2h.notna(), cols], va.loc[va.y_peak2h.notna(), "y_peak2h"])],
                   callbacks=[lgb.early_stopping(50, verbose=False)])
    mp.save_model(str(C.MODEL_DIR / "peak_2h.txt"), num_iteration=mp.best_iteration)
    te["peak_pred"] = mp.predict(te[cols], num_iteration=mp.best_iteration)
    pos = te.label == 1
    okt = te.y_peak2h.notna()
    metrics["time_to_event"] = {
        "onset_mae_min": float(np.abs(te.loc[pos, "tte_pred"] - te.loc[pos, "tte_min"]).mean()),
        "onset_mae_naive_min": float(np.abs(te.loc[pos, "tte_min"].median() - te.loc[pos, "tte_min"]).mean()),
        "onset_within_30min": float((np.abs(te.loc[pos, "tte_pred"] - te.loc[pos, "tte_min"]) <= 30).mean()),
        "peak_mae_mgdl": float(np.abs(te.loc[okt, "peak_pred"] - te.loc[okt, "y_peak2h"]).mean()),
        "peak_mae_twin_physics_mgdl": float(np.abs(te.loc[okt, "twin_proj_peak_habit"] - te.loc[okt, "y_peak2h"]).mean())}
    log(f"  time to onset MAE {metrics['time_to_event']['onset_mae_min']:.0f} min; 2-h peak MAE "
        f"{metrics['time_to_event']['peak_mae_mgdl']:.1f} mg/dL", t0)

    # ---- level-2 hyperglycaemia (>=250)
    m250 = M.train_classifier(tr, va, cols, label="label250")
    m250.save_model(str(C.MODEL_DIR / "spike250_risk_2h.txt"), num_iteration=m250.best_iteration)
    te["risk250"] = m250.predict(te[cols], num_iteration=m250.best_iteration)
    metrics["level2_250"] = M.binary_metrics(te.label250, te.risk250)

    # ---- fairness / subgroup audit (does it work equally for everyone?)
    reg = {r: te[f"ehr_region_{r.lower()}"] == 1 for r in C.REGIONS}
    groups = {"Women": te.ehr_male == 0, "Men": te.ehr_male == 1,
              "Age < 45": te.ehr_age < 45, "Age 45-60": (te.ehr_age >= 45) & (te.ehr_age < 60), "Age >= 60": te.ehr_age >= 60,
              "BMI < 25": te.ehr_bmi < 25, "BMI >= 25 (Asian obesity cut-off)": te.ehr_bmi >= 25,
              "HbA1c < 7.5%": te.ehr_hba1c < 7.5, "HbA1c >= 7.5%": te.ehr_hba1c >= 7.5,
              "On sulfonylurea or insulin": (te.ehr_rx_sulfonylurea == 1) | (te.ehr_rx_basal_insulin == 1),
              "Metformin / DPP-4 / SGLT2 only": (te.ehr_rx_sulfonylurea == 0) & (te.ehr_rx_basal_insulin == 0)}
    groups.update({f"Region: {k}": v for k, v in reg.items()})
    sg = EV.subgroup_auroc(te, "risk", groups)
    metrics["subgroups"] = sg.to_dict(orient="records")
    log(f"  subgroup AUROC range {sg.auroc.min():.3f}-{sg.auroc.max():.3f} across {len(sg)} groups", t0)

    # ---- feature importance (gain) & group importance via SHAP
    contrib = m.predict(te[cols].sample(20000, random_state=1), pred_contrib=True,
                        num_iteration=m.best_iteration)[:, :-1]
    mean_abs = np.abs(contrib).mean(0)
    imp = pd.Series(mean_abs, index=cols).sort_values(ascending=False)
    grp = FE.feature_groups(cols)
    metrics["shap_top20"] = {k: float(v) for k, v in imp.head(20).items()}
    metrics["shap_by_group"] = {g: float(imp[[c for c in cols if grp[c] == g]].sum()) for g in set(grp.values())}

    # ---- glucose trajectory forecasts (+30/+60/+120 min)
    fc = {}
    for h in (30, 60, 120):
        mr = M.train_regressor(tr, va, cols, f"y_g{h}")
        mr.save_model(str(C.MODEL_DIR / f"forecast_{h}m_q50.txt"), num_iteration=mr.best_iteration)
        te[f"f{h}"] = te.g_now + mr.predict(te[cols], num_iteration=mr.best_iteration)
        ok = te[f"y_g{h}"].notna()
        y = te.loc[ok, f"y_g{h}"]
        fc[f"{h}min"] = {
            "mae_madhutwin": float(np.abs(te.loc[ok, f"f{h}"] - y).mean()),
            "mae_twin_physics_only": float(np.abs(te.loc[ok, f"twin_proj_g{h}"] - y).mean()),
            "mae_persistence": float(np.abs(te.loc[ok, "g_now"] - y).mean()),
            "mard_madhutwin_pct": float((np.abs(te.loc[ok, f"f{h}"] - y) / y).mean() * 100),
        }
        qv = {}
        for q in (0.1, 0.9):
            mq = M.train_regressor(tr, va, cols, f"y_g{h}", alpha=q)
            mq.save_model(str(C.MODEL_DIR / f"forecast_{h}m_q{int(q * 100)}.txt"), num_iteration=mq.best_iteration)
            te[f"f{h}_q{int(q * 100)}"] = te.g_now + mq.predict(te[cols], num_iteration=mq.best_iteration)
            qv[q] = va.g_now + mq.predict(va[cols], num_iteration=mq.best_iteration)
        raw_cov = ((y >= te.loc[ok, f"f{h}_q10"]) & (y <= te.loc[ok, f"f{h}_q90"])).mean()
        # split-conformal calibration on the validation patients -> guaranteed ~80% coverage
        qc = EV.conformal_offsets(va[f"y_g{h}"], qv[0.1], qv[0.9], alpha=0.2)
        te[f"f{h}_q10"] -= qc
        te[f"f{h}_q90"] += qc
        fc[f"{h}min"]["pi80_coverage_before_conformal"] = float(raw_cov)
        fc[f"{h}min"]["conformal_margin_mgdl"] = qc
        fc[f"{h}min"]["pi80_width_mgdl"] = float((te.loc[ok, f"f{h}_q90"] - te.loc[ok, f"f{h}_q10"]).median())
        fc[f"{h}min"]["clarke"] = EV.clarke_summary(y.to_numpy(), te.loc[ok, f"f{h}"].to_numpy())
        # ablation: same regressor without the physiology-twin features
        cols_nt = M.feature_cols(tr, ["ehr", "cgm", "lifestyle", "context"])
        mnt = M.train_regressor(tr, va, cols_nt, f"y_g{h}")
        pnt = te.g_now + mnt.predict(te[cols_nt], num_iteration=mnt.best_iteration)
        fc[f"{h}min"]["mae_fusion_without_twin"] = float(np.abs(pnt[ok] - y).mean())
        inb = (y >= te.loc[ok, f"f{h}_q10"]) & (y <= te.loc[ok, f"f{h}_q90"])
        cov = inb.mean()
        fc[f"{h}min"]["pi80_coverage"] = float(cov)
        # rows are correlated within a patient: also report coverage patient by patient
        pc = inb.groupby(te.loc[ok, "patient_id"]).mean()
        fc[f"{h}min"]["pi80_coverage_by_patient"] = {"min": float(pc.min()), "p10": float(pc.quantile(.1)),
                                                     "median": float(pc.median()), "p90": float(pc.quantile(.9)),
                                                     "max": float(pc.max()), "n_patients": int(len(pc)),
                                                     "share_patients_70_90": float(((pc >= .7) & (pc <= .9)).mean())}
        log(f"  forecast +{h}m MAE {fc[f'{h}min']['mae_madhutwin']:.1f} (physics {fc[f'{h}min']['mae_twin_physics_only']:.1f}, "
            f"persistence {fc[f'{h}min']['mae_persistence']:.1f}, no-twin {fc[f'{h}min']['mae_fusion_without_twin']:.1f}) "
            f"PI80 cov {cov:.2f} (raw {raw_cov:.2f}), Clarke A+B {fc[f'{h}min']['clarke']['A+B']:.3f}", t0)
    metrics["forecast"] = fc
    return metrics, te, m, cols


def coldstart(A, ehr, split, t0):
    """First week of a brand-new patient: no personal history, twin built from
    the EHR only. This is where fusing the health record matters most."""
    tw0 = TW.prior_params(A, ehr)
    st0 = TW.observe(A, tw0)
    pr0 = TW.project(A, tw0, st0)
    none = {"mins": np.full((A["P"], A["n5"]), np.nan), "carbs": np.zeros((A["P"], A["n5"])),
            "gi": np.full((A["P"], A["n5"]), 60.0), "freq": np.zeros((A["P"], A["n5"]))}
    stride = {p: (3 if p in set(split["train"]) else 1) for p in A["pids"]}
    df0 = FE.build(A, ehr, tw0, st0, pr0, pr0, none, stride, day_range=(0, 2))
    tr = df0[df0.patient_id.isin(split["train"])]
    va = df0[df0.patient_id.isin(split["val"])]
    te = df0[df0.patient_id.isin(split["test"])]
    out = {}
    for name, groups in M.ABLATIONS.items():
        cols = [c for c in M.feature_cols(df0, groups) if not c.startswith(("mins_to_habit", "habit_"))]
        m = M.train_classifier(tr, va, cols)
        p = m.predict(te[cols], num_iteration=m.best_iteration)
        out[name] = M.binary_metrics(te.label, p)
        log(f"  cold-start {name:40s} AUROC {out[name]['auroc']:.3f}  AUPRC {out[name]['auprc']:.3f}", t0)
    return out


def lowdata_curve(df, split, t0, sizes=((5, 3), (10, 3), (20, 2), (40, 1), (80, 1))):
    """Hybrid-model data efficiency: twin features matter most when few patients
    are available to learn from (the realistic case for a new Indian clinic)."""
    tr_all = df[df.patient_id.isin(split["train"])]
    va = df[df.patient_id.isin(split["val"])]
    te = df[df.patient_id.isin(split["test"])]
    full = M.feature_cols(df, M.ABLATIONS[M.FULL])
    nt = M.feature_cols(df, ["ehr", "cgm", "lifestyle", "context"])
    rng = np.random.default_rng(0)
    rows = []
    for n, reps in sizes:
        for r in range(reps):
            pats = rng.choice(split["train"], n, replace=False)
            tr = tr_all[tr_all.patient_id.isin(pats)]
            row = {"train_patients": n, "rep": r}
            for name, cols in (("without_twin", nt), ("with_twin", full)):
                m = M.train_classifier(tr, va, cols)
                p = m.predict(te[cols], num_iteration=m.best_iteration)
                row[f"auroc_{name}"] = float(M.roc_auc_score(te.label, p))
                row[f"auprc_{name}"] = float(M.average_precision_score(te.label, p))
            rows.append(row)
    out = pd.DataFrame(rows).groupby("train_patients").mean().drop(columns="rep").reset_index()
    log("  low-data curve: " + "; ".join(f"n={r.train_patients}: +{r.auroc_with_twin - r.auroc_without_twin:.3f} AUROC"
                                       for r in out.itertuples()), t0)
    return out.to_dict(orient="records")


def twin_maturity(A, ehr, t0, days=(1, 2, 3, 5, 7)):
    """How fast does the twin learn a new patient? Recovery of hidden physiology
    after 1..7 days of CGM (day 0 = EHR prior only)."""
    h = pd.read_csv(C.DATA_DIR / "truth" / "cohort_hidden.csv").set_index("patient_id").loc[A["pids"]]
    ep = E.effective_params(h.reset_index())
    r = lambda a, b: float(spearmanr(a, b)[0])
    tw0 = TW.prior_params(A, ehr)
    rows = [{"days": 0, "si": r(tw0.twin_SI, h.true_SI), "sigma": r(tw0.twin_sigma, ep["sigma"]),
             "di": r(tw0.twin_SI * tw0.twin_sigma, h.true_SI * ep["sigma"]),
             "gb": r(tw0.twin_Gb, ep["gb_untreated"] - ep["gb_med_drop"])}]
    for d in days:
        twd = TW.calibrate(A, ehr, days=d, verbose=False)
        rows.append({"days": d, "si": r(twd.twin_SI, h.true_SI), "sigma": r(twd.twin_sigma, ep["sigma"]),
                     "di": r(twd.twin_SI * twd.twin_sigma, h.true_SI * ep["sigma"]),
                     "gb": r(twd.twin_Gb, ep["gb_untreated"] - ep["gb_med_drop"])})
    log("  twin maturity: " + ", ".join(f"day {x['days']}: DI rho {x['di']:.2f}" for x in rows), t0)
    return rows


def nightly_tracking(A):
    """Does the nightly-updated twin notice what is happening to the patient?
    Compare refitted SI multiplier / basal shift with hidden illness, stress and missed doses."""
    ng = A["nightly"]
    d = pd.read_csv(C.DATA_DIR / "truth" / "daily_context.csv").set_index(["patient_id", "day"])
    rows = []
    for i, p in enumerate(A["pids"]):
        for dd in range(C.CALIB_DAYS, ng["si_mult"].shape[1]):
            win = d.loc[[(p, x) for x in range(dd - 3, dd)]]
            rows.append({"si": ng["si_mult"][i, dd], "gb": ng["gb_delta"][i, dd], "ill": bool(win.ill.any()),
                         "stress": float(win.stress.mean()), "missed": int((~win.med_adherent.astype(bool)).sum())})
    r = pd.DataFrame(rows)
    grp = lambda g: {"si_mult": float(g.si.mean()), "gb_shift": float(g.gb.mean()), "n": int(len(g))}
    return {"illness_in_window": grp(r[r.ill]), "no_illness": grp(r[~r.ill]),
            "high_stress": grp(r[r.stress >= 0.4]), "low_stress": grp(r[r.stress < 0.2]),
            "missed_doses": {str(k): grp(g) for k, g in r.groupby(r.missed.clip(upper=3))}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen", action="store_true", help="re-simulate the synthetic cohort")
    ap.add_argument("--skip-report", action="store_true")
    ap.add_argument("--skip-real", action="store_true", help="skip the real-world validation (~8 min)")
    a = ap.parse_args(argv)
    t0 = time.time()
    for d in (C.DATA_DIR, PROC, C.MODEL_DIR, C.REPORT_DIR, C.FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    if a.regen or not (C.DATA_DIR / "wearables_5min.parquet").exists():
        log("simulating synthetic cohort ...", t0)
        S.run()
    A, ehr, tw, states, habits, nh, proj, proj_h, meal_log = build_twin(t0)
    tw.to_csv(PROC / "twin_params.csv", index=False)
    habits.to_csv(PROC / "meal_habits.csv", index=False)

    split = M.split_patients(A["pids"])
    stride = {p: (3 if p in set(split["train"]) else 1) for p in A["pids"]}
    df = FE.build(A, ehr, tw, states, proj, proj_h, nh, stride)
    fcols = [c for c in df.columns if c not in ("patient_id", "k", "t")]
    df[fcols] = df[fcols].astype(np.float32)
    df.to_parquet(PROC / "features.parquet", index=False)
    log(f"fused feature table: {len(df):,} rows x {df.shape[1]} columns", t0)

    metrics, te, m, cols = train_and_evaluate(df, A, split, t0)
    tw_cgm = TW.calibrate(A, ehr, prior_weight=0.0, verbose=False)
    metrics["twin_recovery"] = twin_recovery(tw, tw_cgm)
    log("twin recovery of hidden physiology evaluated (CGM-only vs EHR-only vs fused)", t0)
    metrics["cold_start_first_48h"] = coldstart(A, ehr, split, t0)
    from . import whatif as W
    tw0 = TW.prior_params(A, ehr)
    metrics["whatif_validation"], whatif_res = W.validate(tw, tw0, split["test"])
    log("what-if counterfactuals validated against hidden truth", t0)
    from . import sickday as SD
    metrics["sick_day"], _ = SD.evaluate(A, split)
    log(f"sick-day detector AUROC {metrics['sick_day']['unsupervised_auroc_test']:.3f} (unsupervised)", t0)
    if (C.DATA_DIR / "real" / "cgm.parquet").exists() and not a.skip_real:
        from . import realworld as RW
        metrics["real_world"], _ = RW.run(verbose=True)
        log("real-world validation on open CGM + clinical cohorts done", t0)
    elif (C.REPORT_DIR / "realworld_metrics.json").exists():
        # quick mode: keep the last saved open-cohort results (that experiment is independent of the synthetic models)
        import json as _json
        metrics["real_world"] = _json.loads((C.REPORT_DIR / "realworld_metrics.json").read_text())
        log("open-cohort results reused from reports/realworld_metrics.json (run without --skip-real to recompute)", t0)
    metrics["twin_value"]["low_data"] = lowdata_curve(df, split, t0)
    metrics["twin_maturity"] = twin_maturity(A, ehr, t0)
    metrics["nightly_tracking"] = nightly_tracking(A)
    metrics["cohort"] = {"patients": A["P"], "days": C.N_DAYS, "calibration_days": C.CALIB_DAYS,
                         "cgm_readings": int(np.isfinite(A["cgm"]).sum()),
                         "logged_meals": int(len(meal_log))}
    M.save_json(metrics, C.REPORT_DIR / "metrics.json")
    M.save_json({"features": cols, "threshold": metrics["final_model"]["threshold"], "split": split,
                 "conformal_margin_mgdl": {h: metrics["forecast"][f"{h}min"]["conformal_margin_mgdl"] for h in (30, 60, 120)},
                 "ensemble_members": 5},
                C.MODEL_DIR / "model_card.json")
    keep = ["patient_id", "k", "t", "label", "label250", "tte_min", "g_now", "risk", "risk250", "risk_sd",
            "tte_pred", "peak_pred", "y_peak2h", "y_g30", "y_g60", "y_g120"] + [c for c in te.columns if c.startswith("f") and c[1:4].rstrip("_mq").isdigit()]
    te[keep].to_parquet(PROC / "test_predictions.parquet", index=False)
    log("metrics written to reports/metrics.json", t0)

    if not a.skip_report:
        from . import report
        report.make_all(metrics, df, te, A, tw, tw_cgm, whatif_res, split, meal_log)
        report.make_v2(metrics, te, A)
        M.save_json(metrics, C.REPORT_DIR / "metrics.json")
        log("figures written to reports/figures/", t0)
    from . import export_dashboard
    export_dashboard.export(A, ehr, tw, habits, te, m, cols, metrics, meal_log, split)
    import runpy
    runpy.run_path(str(C.ROOT / "scripts" / "build_dashboard.py"))
    log("dashboard data written and console bundled (dashboard/madhutwin_console.html)", t0)


if __name__ == "__main__":
    main()
