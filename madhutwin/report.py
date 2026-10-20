"""Publication-style figures for the README, deck and architecture document."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from . import config as C
from . import features as FE
from . import whatif as W

# palette (validated reference palette; categorical slots in fixed order)
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2, S3, S4, S5, S6, S7 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"
CRIT, GOOD = "#d03b3b", "#0ca30c"
GROUP_COL = {"cgm": S1, "lifestyle": S2, "twin": S3, "ehr": S4, "context": S5}
GROUP_NAME = {"cgm": "CGM", "lifestyle": "Wearable & food log", "twin": "Physiology twin", "ehr": "EHR",
              "context": "Clock"}

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    "lines.linewidth": 2,
})


def _save(fig, name):
    fig.savefig(C.FIG_DIR / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def pick_showcase(te, A):
    """A held-out patient/day with several excursions and a clear early alert."""
    best, best_score = None, -1
    for pid, g in te.groupby("patient_id"):
        g = g.assign(day=g.k // 288)
        for d, gd in g.groupby("day"):
            if d < C.CALIB_DAYS + 1 or len(gd) < 150:
                continue
            ev = gd.label.sum()
            score = min(ev, 60) + 40 * ((gd.risk >= 0.5) == (gd.label == 1)).mean()
            if score > best_score:
                best, best_score = (pid, int(d)), score
    return best


def fig_patient_day(te, A, meal_log, thr, pid, day):
    i = A["pids"].index(pid)
    a, b = day * 288, (day + 1) * 288
    hrs = (np.arange(a, b) - a) * 5 / 60
    cg = A["cgm"][i, a:b]
    risk = np.full(b - a, np.nan)
    g = te[(te.patient_id == pid) & (te.k >= a) & (te.k < b)]
    risk[g.k - a] = g.risk
    fig, ax = plt.subplots(4, 1, figsize=(12, 8.2), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1.6, 1, 0.7], "hspace": 0.12})
    ax[0].axhspan(70, 180, color=GOOD, alpha=0.06, lw=0)
    ax[0].axhline(180, color=CRIT, lw=0.8)
    ax[0].text(0.1, 183, "180 mg/dL", color=CRIT, fontsize=8, va="bottom")
    ax[0].plot(hrs, cg, color=S1, lw=2, label="CGM glucose")
    ml = meal_log[(meal_log.patient_id == pid) & (meal_log.t >= a * 5) & (meal_log.t < b * 5)]
    for r in ml.itertuples():
        x = (r.t - a * 5) / 60
        ax[0].text(x, 44, f"{r.food}\n{r.carbs_g:.0f} g", fontsize=7, color=INK2, ha="center", va="bottom")
        ax[0].plot([x], [66], marker="^", color=S2, ms=6)
    ax[0].set_ylim(40, max(300, np.nanmax(cg) + 20))
    ax[0].axhspan(40, 70, color=SURF, lw=0)
    ax[0].set_ylabel("mg/dL")
    ax[0].set_title(f"Held-out patient {pid}, day {day + 1}: the twin warns before each spike")
    # alerts
    on = risk >= thr
    starts = np.where(on & ~np.r_[False, on[:-1]])[0]
    ax[1].fill_between(hrs, 0, risk, color=S3, alpha=0.25, lw=0)
    ax[1].plot(hrs, risk, color=S3, lw=1.6, label="2-h spike risk")
    ax[1].axhline(thr, color=INK2, lw=0.8)
    ax[1].text(23.9, thr + 0.03, f"alert threshold {thr:.2f}", fontsize=8, color=INK2, ha="right")
    for s_ in starts:
        ax[1].plot(hrs[s_], risk[s_], "v", color=CRIT, ms=8)
    ax[1].set_ylim(0, 1.05)
    ax[1].set_ylabel("risk")
    onsets = np.where(FE.excursion_onsets(A["cgm"][i:i + 1, a:b])[0])[0]
    for o in onsets:
        ax[0].plot(hrs[o], cg[o], "o", color=CRIT, ms=7, mec=SURF, mew=2)
    ax[2].bar(hrs, A["steps"][i, a:b], width=5 / 60, color=S2, lw=0)
    ax[2].set_ylabel("steps/5 min")
    ax2 = ax[2].twinx()
    ax2.grid(False)
    ax2.plot(hrs, A["hr"][i, a:b], color=INK2, lw=0.8)
    ax2.set_ylabel("HR bpm", color=INK2)
    ax2.spines["right"].set_visible(True)
    st = A["stage"][i, a:b]
    cols = {0: SURF, 1: "#9ec5f4", 2: "#1c5cab", 3: S7}
    for sval, cval in cols.items():
        if sval == 0:
            continue
        ax[3].fill_between(hrs, 0, 1, where=st == sval, color=cval, step="mid", lw=0,
                           label={1: "light", 2: "deep", 3: "REM"}[sval])
    ax[3].set_yticks([])
    ax[3].set_ylabel("sleep")
    ax[3].legend(loc="center right", ncol=3, fontsize=8)
    ax[3].grid(False)
    ax[3].set_xticks(range(0, 25, 3))
    ax[3].set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)])
    ax[3].set_xlim(0, 24)
    ax[0].legend(handles=[plt.Line2D([], [], color=S1, lw=2, label="CGM glucose"),
                          plt.Line2D([], [], marker="o", color=CRIT, lw=0, label="excursion onset (>=180 for 15 min)"),
                          plt.Line2D([], [], marker="v", color=CRIT, lw=0, label="twin alert")],
                 loc="upper left", fontsize=8, ncol=3)
    _save(fig, "fig_patient_day.png")


def fig_ablation(metrics):
    abl = metrics["ablation"]
    base = metrics["baselines"]
    names = list(base.keys()) + list(abl.keys())
    au = [base[n]["auroc"] for n in base] + [abl[n]["auroc"] for n in abl]
    ap = [base[n]["auprc"] for n in base] + [abl[n]["auprc"] for n in abl]
    y = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for k, (vals, title) in enumerate([(au, "AUROC"), (ap, "AUPRC")]):
        colors = [MUTED] * len(base) + [S1] * (len(abl) - 1) + [S3]
        ax[k].barh(y, vals, height=0.62, color=colors)
        for yy, v in zip(y, vals):
            ax[k].text(v + 0.005, yy, f"{v:.3f}", va="center", fontsize=8.5, color=INK)
        ax[k].set_xlim(0.4 if k == 0 else 0.2, 1.0)
        ax[k].set_title(title)
        ax[k].grid(axis="y", visible=False)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(names, fontsize=9)
    fig.suptitle("Predicting a >=180 mg/dL excursion within 2 h, 45 held-out patients (gray = baselines)",
                 x=0.01, ha="left", fontsize=11, color=INK2)
    fig.tight_layout()
    _save(fig, "fig_ablation.png")


def fig_roc_pr(te, metrics):
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.4))
    y = te.label.to_numpy()
    series = [("MadhuTwin", te.risk.to_numpy(), S3),
              ("CGM trend extrapolation", (te.g_now + te.g_delta_30m.fillna(0) * 2).to_numpy(), MUTED)]
    for name, s, c in series:
        f, t, _ = roc_curve(y, s)
        ax[0].plot(f, t, color=c, label=name)
        p, r, _ = precision_recall_curve(y, s)
        ax[1].plot(r, p, color=c, label=name)
    ax[0].plot([0, 1], [0, 1], color=AXIS, lw=0.8)
    ax[0].set_xlabel("False positive rate")
    ax[0].set_ylabel("True positive rate")
    ax[0].set_title(f"ROC  (AUROC {metrics['final_model']['auroc']:.3f})")
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].set_title(f"Precision-recall  (AUPRC {metrics['final_model']['auprc']:.3f})")
    ax[1].axhline(y.mean(), color=AXIS, lw=0.8)
    ax[1].legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    _save(fig, "fig_roc_pr.png")


def fig_calibration(metrics):
    rel = pd.DataFrame(metrics["final_model"]["reliability"])
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], color=AXIS, lw=0.8)
    ax.plot(rel.pred, rel.obs, "o-", color=S3, ms=8, mec=SURF, mew=2)
    ax.set_xlabel("Predicted risk")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Calibration (Brier {metrics['final_model']['brier']:.3f})")
    fig.tight_layout()
    _save(fig, "fig_calibration.png")


def fig_operating_points(metrics):
    op = pd.DataFrame(metrics["operating_points"]).sort_values("threshold")
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
    ax[0].plot(op.threshold, op.detected_lead_ge_30min * 100, "o-", color=S1, ms=7, label=">=30 min warning")
    ax[0].plot(op.threshold, op.detected_lead_ge_60min * 100, "o-", color=S2, ms=7, label=">=60 min warning")
    ax[0].set_ylabel("% of excursions warned")
    ax[0].set_xlabel("Alert threshold")
    ax[0].set_ylim(0, 100)
    ax[0].legend(fontsize=9)
    ax[0].set_title("Early warning vs threshold")
    ax[1].plot(op.threshold, op.alerts_per_patient_day, "o-", color=S1, ms=7, label="all alerts")
    ax[1].plot(op.threshold, op.false_alerts_per_patient_day, "o-", color=CRIT, ms=7, label="false alerts")
    ax[1].set_ylabel("per patient per day")
    ax[1].set_xlabel("Alert threshold")
    ax[1].legend(fontsize=9)
    ax[1].set_title("Alert burden vs threshold")
    fig.tight_layout()
    _save(fig, "fig_operating_points.png")


def fig_shap(metrics):
    top = pd.Series(metrics["shap_top20"]).head(15)[::-1]
    grp = FE.feature_groups(top.index)
    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.barh(range(len(top)), top.values, color=[GROUP_COL[grp[c]] for c in top.index], height=0.65)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=9)
    ax.set_xlabel("mean |SHAP| (log-odds)")
    ax.grid(axis="y", visible=False)
    ax.set_title("What drives the 2-hour risk (top 15 features)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COL[g]) for g in GROUP_COL]
    ax.legend(handles, [GROUP_NAME[g] for g in GROUP_COL], loc="lower right", fontsize=8.5)
    fig.tight_layout()
    _save(fig, "fig_shap.png")


def fig_twin_recovery(metrics, tw):
    rec = metrics["twin_recovery"]
    h = pd.read_csv(C.DATA_DIR / "truth" / "cohort_hidden.csv").set_index("patient_id").loc[tw.patient_id]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={"width_ratios": [1, 1.4]})
    ax[0].scatter(h.true_SI * 1e4, tw.twin_SI * 1e4, s=16, color=S1, alpha=0.7, lw=0)
    lim = [0, max(h.true_SI.max(), tw.twin_SI.max()) * 1e4 * 1.05]
    ax[0].plot(lim, lim, color=AXIS, lw=0.8)
    ax[0].set_xlabel("True insulin sensitivity (hidden)  x1e-4")
    ax[0].set_ylabel("Twin estimate  x1e-4")
    ax[0].set_title(f"Twin recovers insulin sensitivity (rho {rec['insulin_sensitivity']['twin_fused']:.2f})")
    keys = [("insulin_sensitivity", "Insulin\nsensitivity", "ehr_only_homa_ir"),
            ("beta_cell_function", "Beta-cell\nfunction", "ehr_only_homa_b"),
            ("disposition_index", "Disposition\nindex", "ehr_only_homa"),
            ("basal_glucose", "Basal\nglucose", "ehr_only_fpg")]
    x = np.arange(len(keys))
    w = 0.26
    for j, (lab, col, src) in enumerate([("EHR labs only", S4, None), ("CGM only (twin)", S1, "cgm_only_twin"),
                                          ("EHR + CGM (fused twin)", S3, "twin_fused")]):
        vals = [rec[k][src] if src else rec[k][e] for k, _, e in keys]
        ax[1].bar(x + (j - 1) * w, vals, width=w - 0.03, color=col, label=lab)
        for xx, v in zip(x + (j - 1) * w, vals):
            ax[1].text(xx, max(v, 0) + 0.02, f"{v:.2f}", ha="center", fontsize=7.5)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([k[1] for k in keys], fontsize=9)
    ax[1].set_ylim(0, 1.08)
    ax[1].set_ylabel("Spearman rho vs hidden truth")
    ax[1].set_title("Fusing EHR with CGM personalises physiology best")
    ax[1].legend(fontsize=8.5, loc="upper left", ncol=3)
    ax[1].grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, "fig_twin_recovery.png")


def fig_whatif(metrics, whatif_res, tw, split):
    pids = split["test"]
    twp = tw.set_index("patient_id").loc[pids].reset_index()
    # choose the patient with the median baseline rise
    j = int(np.argsort(whatif_res["truth"]["baseline"]["rise"])[len(pids) // 2])
    truth = {k: v for k, v in W.true_params(pids).items()}
    pick = lambda d: {k: np.asarray(v)[j:j + 1] for k, v in d.items()}
    tp, pp = pick(truth), pick(W.twin_params(twp))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.35, 1]})
    cols = {"baseline": INK2, "walk_20": S1, "millet_swap": S3, "portion_minus30": S2}
    t = (np.arange(300) - 30) / 60
    for sc, spec in W.SCENARIOS.items():
        tr = W.run(tp, spec["meal"][1], spec["meal"][2], spec["walk"])[0]
        pr = W.run(pp, spec["meal"][1], spec["meal"][2], spec["walk"])[0]
        ax[0].plot(t, pr, color=cols[sc], lw=2, label=spec["label"])
        ax[0].plot(t, tr, color=cols[sc], lw=1, alpha=0.6)
    ax[0].axhline(180, color=CRIT, lw=0.8)
    ax[0].set_xlabel("hours after sitting down to lunch")
    ax[0].set_ylabel("mg/dL")
    ax[0].set_title(f"What-if for patient {pids[j]}: twin (thick) vs true response (thin)")
    ax[0].legend(fontsize=7.5, loc="upper right")
    m = metrics["whatif_validation"]["models"]
    names = list(m.keys())
    scs = [("walk_20", "Post-meal walk"), ("millet_swap", "Rice -> millet"), ("portion_minus30", "Portion -30%")]
    x = np.arange(len(scs))
    w = 0.26
    for k, (nm, c) in enumerate(zip(names, [S3, S4, MUTED])):
        vals = [m[nm][f"{s}_effect_corr"] for s, _ in scs]
        ax[1].bar(x + (k - 1) * w, vals, width=w - 0.03, color=c, label=nm)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([s[1] for s in scs])
    ax[1].set_ylim(0, 1.05)
    ax[1].set_ylabel("corr(predicted, true) peak change")
    ax[1].set_title("Is the personal treatment effect right?")
    ax[1].legend(fontsize=7.5, loc="lower right")
    ax[1].grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, "fig_whatif.png")


def fig_forecast(metrics):
    fc = metrics["forecast"]
    hs = [30, 60, 120]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for key, lab, c in [("mae_persistence", "Persistence (no change)", MUTED),
                        ("mae_twin_physics_only", "Physiology twin alone", S2),
                        ("mae_fusion_without_twin", "ML on EHR + wearables", S1),
                        ("mae_madhutwin", "MadhuTwin (hybrid)", S3)]:
        v = [fc[f"{h}min"][key] for h in hs]
        ax.plot(hs, v, "o-", color=c, ms=7, label=lab, mec=SURF, mew=1.5)
        ax.text(hs[-1] + 3, v[-1], f"{v[-1]:.1f}", va="center", fontsize=8.5, color=INK)
    ax.set_xticks(hs)
    ax.set_xlabel("forecast horizon (min)")
    ax.set_ylabel("mean absolute error (mg/dL)")
    ax.set_title("Glucose trajectory forecast error")
    ax.legend(fontsize=8.5)
    ax.set_xlim(20, 135)
    fig.tight_layout()
    _save(fig, "fig_forecast.png")


def fig_cohort(A, metrics):
    ehr = pd.read_csv(C.DATA_DIR / "ehr" / "observations.csv")
    a1c = ehr[ehr.KEY == "hba1c"].sort_values("DATE").groupby("PATIENT").VALUE.last()
    pats = pd.read_csv(C.DATA_DIR / "ehr" / "patients.csv")
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    ax[0].hist(a1c, bins=25, color=S1, rwidth=0.9)
    ax[0].set_xlabel("latest HbA1c (%)")
    ax[0].set_title("Glycaemic control")
    reg = pats.STATE.value_counts()
    ax[1].barh(reg.index[::-1], reg.values[::-1], color=S1, height=0.6)
    ax[1].set_title("Region of India")
    ax[1].grid(axis="y", visible=False)
    cg = A["cgm"]
    tir = np.nanmean((cg >= 70) & (cg <= 180), 1) * 100
    ax[2].hist(tir, bins=25, color=S3, rwidth=0.9)
    ax[2].set_xlabel("time in range 70-180 (%)")
    ax[2].set_title("CGM time in range")
    fig.suptitle(f"Synthetic cohort: {A['P']} adults with T2D, {C.N_DAYS} days each", x=0.01, ha="left",
                 fontsize=11, color=INK2)
    fig.tight_layout()
    _save(fig, "fig_cohort.png")


def make_all(metrics, df, te, A, tw, tw_cgm, whatif_res, split, meal_log):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    thr = metrics["final_model"]["threshold"]
    pid, day = pick_showcase(te, A)
    metrics["showcase"] = {"patient_id": pid, "day": day}
    fig_patient_day(te, A, meal_log, thr, pid, day)
    fig_ablation(metrics)
    fig_roc_pr(te, metrics)
    fig_calibration(metrics)
    fig_operating_points(metrics)
    fig_shap(metrics)
    fig_twin_recovery(metrics, tw)
    fig_whatif(metrics, whatif_res, tw, split)
    fig_forecast(metrics)
    fig_cohort(A, metrics)


# ----------------------------------------------------------------------------- v2 figures
def _clarke_axes(ax, title):
    for xs, ys in [([0, 400], [0, 400]), ([0, 175 / 3], [70, 70]), ([175 / 3, 400 / 1.2], [70, 400]), ([70, 70], [84, 400]),
                   ([0, 70], [180, 180]), ([70, 290], [180, 400]), ([70, 70], [0, 56]), ([70, 400], [56, 320]),
                   ([180, 180], [0, 70]), ([180, 400], [70, 70]), ([240, 240], [70, 180]), ([240, 400], [180, 180]),
                   ([130, 180], [0, 70])]:
        ax.plot(xs, ys, color=INK2, lw=0.8)
    for txt, x, y in [("A", 30, 15), ("A", 370, 340), ("B", 370, 260), ("B", 280, 370), ("C", 160, 370), ("C", 160, 15),
                      ("D", 30, 140), ("D", 370, 120), ("E", 30, 370), ("E", 370, 15)]:
        ax.text(x, y, txt, fontsize=12, fontweight="bold", color=INK2)
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.set_xlabel("Actual glucose (mg/dL)")
    ax.set_ylabel("Predicted glucose (mg/dL)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(False)


def fig_clarke(te, metrics):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.9))
    rng = np.random.default_rng(0)
    for ax, h in zip(axs, (30, 60, 120)):
        d = te[[f"y_g{h}", f"f{h}"]].dropna()
        d = d.iloc[rng.choice(len(d), min(6000, len(d)), replace=False)]
        ax.scatter(d[f"y_g{h}"], d[f"f{h}"], s=3, alpha=0.25, color=S1, lw=0)
        c = metrics["forecast"][f"{h}min"]["clarke"]
        _clarke_axes(ax, f"+{h} min · A {c['A'] * 100:.1f}% · A+B {c['A+B'] * 100:.1f}%")
    fig.suptitle("Clarke error grid, MadhuTwin forecasts on held-out virtual patients", x=0.01, ha="left", fontsize=11, color=INK2)
    fig.tight_layout()
    _save(fig, "fig_clarke.png")


def fig_realworld(metrics):
    R = metrics.get("real_world")
    if not R:
        return
    oof = pd.read_parquet(C.DATA_DIR / "processed" / "realworld_oof.parquet")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.35, 1, 1]})
    a = R["auroc_140"]
    names = list(a.keys())
    y = np.arange(len(names))[::-1]
    for yy, nm in zip(y, names):
        v = a[nm]["auroc"]
        col = S3 if nm == "CGM only" else (MUTED if "baseline" in nm or "Clock" in nm else S1)
        ax[0].plot([v["lo"], v["hi"]], [yy, yy], color=col, lw=3, solid_capstyle="round")
        ax[0].plot(v["point"], yy, "o", color=col, ms=9, mec=SURF, mew=2)
        ax[0].text(v["hi"] + 0.01, yy, f"{v['point']:.3f}", va="center", fontsize=9)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels([n.replace(" (MadhuTwin fusion)", "") for n in names], fontsize=9)
    ax[0].set_xlim(0.5, 0.9)
    ax[0].set_xlabel("AUROC with 95% CI (patient bootstrap)")
    ax[0].set_title(f"Real people: ≥140 mg/dL within 2 h (n={R['cohort']['patients']})")
    ax[0].grid(axis="y", visible=False)
    d = oof[["y_g30", "f30"]].dropna()
    d = d.sample(min(6000, len(d)), random_state=0)
    ax[1].scatter(d.y_g30, d.f30, s=3, alpha=0.25, color=S1, lw=0)
    c = R["forecast"]["30min"]["clarke"]
    _clarke_axes(ax[1], f"Real 30-min forecasts · A+B {c['A+B'] * 100:.1f}%")
    ir = R["insulin_resistance"]
    ks = ["EHR labs only", "CGM only", "EHR + CGM (fused)"]
    for j, (k, col) in enumerate(zip(ks, [S4, S1, S3])):
        v = ir[k]
        ax[2].bar(j, v["spearman"], color=col, width=0.6)
        ax[2].plot([j, j], [v["lo"], v["hi"]], color=INK2, lw=1.2)
        ax[2].text(j, v["hi"] + 0.03, f"{v['spearman']:.2f}", ha="center", fontsize=9)
    ax[2].set_xticks(range(3))
    ax[2].set_xticklabels(["EHR\nlabs only", "CGM\nonly", "EHR + CGM\nfused"])
    ax[2].set_ylim(-0.1, 1)
    ax[2].set_ylabel("Spearman ρ vs SSPG (95% CI)")
    ax[2].set_title(f"Insulin resistance, gold-standard test (n={ir['n_patients']})")
    ax[2].grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, "fig_realworld.png")


def fig_fairness(metrics):
    sg = pd.DataFrame(metrics.get("subgroups", []))
    if sg.empty:
        return
    sg = sg.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(sg) + 1.2))
    overall = metrics["final_model"]["auroc"]
    ax.axvline(overall, color=AXIS, lw=1)
    ax.text(overall, -0.9, f" overall {overall:.3f}", fontsize=8.5, color=INK2, va="center")
    ax.set_ylim(-1.3, len(sg) - 0.5)
    ax.plot(sg.auroc, range(len(sg)), "o", color=S1, ms=8, mec=SURF, mew=2)
    for i, (a, n) in enumerate(zip(sg.auroc, sg.patients)):
        ax.text(a + 0.004, i, f"{a:.3f} (n={n})", va="center", fontsize=8.5)
    ax.set_yticks(range(len(sg)))
    ax.set_yticklabels(sg.subgroup, fontsize=9)
    ax.set_xlim(min(0.85, sg.auroc.min() - 0.02), 0.97)
    ax.set_xlabel("AUROC, 2-h excursion warning")
    ax.set_title("Fairness audit: performance by subgroup")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, "fig_fairness.png")


def fig_sickday(A, metrics):
    p = C.DATA_DIR / "processed" / "sickday_test.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    ill_p = d[d.ill == 1].patient_id.unique()
    if not len(ill_p):
        return
    pid = ill_p[0]
    g = d[d.patient_id == pid].sort_values("day")
    thr = metrics["sick_day"]["flag_threshold_z"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={"width_ratios": [1.5, 1]})
    for r in g[g.ill == 1].itertuples():
        ax[0].axvspan(r.day - 0.5, r.day + 0.5, color=CRIT, alpha=0.12, lw=0)
    ax[0].bar(g.day, g.anomaly_score, color=[CRIT if v >= thr else S1 for v in g.anomaly_score], width=0.7)
    ax[0].axhline(thr, color=INK2, lw=0.8)
    ax[0].text(g.day.min(), thr + 0.3, "flag threshold", fontsize=8, color=INK2)
    ax[0].set_xlabel("day of record")
    ax[0].set_xticks(range(int(g.day.min()), int(g.day.max()) + 1, 2))
    ax[0].set_ylabel("anomaly score (personal z)")
    ax[0].set_title(f"Patient {pid}: shaded = hidden illness days (never shown to the twin)")
    sd = metrics["sick_day"]
    vals = [sd["tar_rest_of_day_not_flagged"] * 100, sd["tar_rest_of_day_flagged"] * 100]
    ax[1].bar([0, 1], vals, color=[S1, CRIT], width=0.55)
    for i, v in enumerate(vals):
        ax[1].text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=10)
    ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels(["not flagged", "flagged sick-day"])
    ax[1].set_ylabel("% of afternoon/evening above 180")
    ax[1].set_title(f"Flagged days run hotter (AUROC {sd['unsupervised_auroc_test']:.2f})")
    ax[1].grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, "fig_sickday.png")


def make_v2(metrics, te, A):
    fig_clarke(te, metrics)
    fig_realworld(metrics)
    fig_fairness(metrics)
    fig_sickday(A, metrics)
