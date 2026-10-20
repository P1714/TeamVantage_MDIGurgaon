"""Generate docs/model_card.md from reports/metrics.json (Mitchell et al., 2019 model-card format)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = json.loads((ROOT / "reports/metrics.json").read_text())
CARD = json.loads((ROOT / "models/model_card.json").read_text())
LAT = json.loads((ROOT / "reports/latency.json").read_text()) if (ROOT / "reports/latency.json").exists() else None
fm, ev, fc, U, TE, TV = M["final_model"], M["event_level"], M["forecast"], M["uncertainty"], M["time_to_event"], M["twin_value"]
RW, SD = M.get("real_world"), M.get("sick_day")
ci = M["ablation"]["MadhuTwin (fusion + physiology twin)"]["ci"]["auroc"]
sg = M["subgroups"]
pct = lambda v: f"{v * 100:.0f}%"
ST = json.loads((ROOT / "reports/stress_test.json").read_text()) if (ROOT / "reports/stress_test.json").exists() else None
FUS = next((k for k in (RW or {}).get("auroc_140", {}) if k.startswith("CGM + EHR")), None)
worst = min(sg, key=lambda g: g["auroc"])
best = max(sg, key=lambda g: g["auroc"])

md = f"""# Model card: MadhuTwin 2-hour hyperglycaemia warning

*Generated from `reports/metrics.json` by `scripts/build_model_card.py`. Do not edit by hand.*

## Model details
- **System:** MadhuTwin, a hybrid digital twin for Type 2 Diabetes. **Layer 1** is a personalised physiology twin (extended Bergman minimal model). **Layer 2** is a set of LightGBM models that predict from the twin state plus the raw data streams.
- **Models:**
  - `spike_risk_2h.txt`: binary classifier for an excursion ≥180 mg/dL for ≥15 min starting within 2 h;
  - `spike250_risk_2h.txt`: the same for ≥250 mg/dL;
  - quantile forecasters for +30/+60/+120 min;
  - `time_to_onset.txt` and `peak_2h.txt`;
  - a 5-model bootstrap ensemble (`risk_ensemble_*.txt`) whose spread gives the model-disagreement label.
- **Twin fitting:** an EHR-informed prior with regularised grid-search fitting (a MAP-style penalty towards lab-derived values). This is not full Bayesian posterior inference.
- **Inputs:** {len(CARD['features'])} features every 5 min from CGM dynamics, food log, steps, heart rate, HRV, sleep, clock, twin states and projections, and EHR fields. Genotype is excluded because it adds nothing (AUROC {M['genetics_ablation']['auroc_with_genetics']:.4f} with vs {M['genetics_ablation']['auroc_without_genetics']:.4f} without).
- **Operating threshold:** {fm['threshold']:.2f}, chosen to maximise F1 on the validation patients. It is a reproducible demo rule, not a clinically optimised operating point.
- **Alert policy** (`madhutwin/alerting.py`, shared by evaluation, console and API): risk ≥ threshold, at most one alert per hour, and suppressed unless at least 9 of the last 12 CGM readings are present. All alert metrics below count only delivered alerts, matched one-to-one to excursion onsets within the 2 h before them.
- **Output contract:** `madhutwin/contract.py` defines every field: endpoint forecasts, the highest expected glucose in the next 2 h, and the Twin's mechanistic projection.
- **Version / licence:** prototype v3.1, MIT. Built by Team Vantage (MDI Gurgaon) for the Happiest Health Digital Twin Challenge 2026.

## Intended use
- **Users:** diabetologists, physicians and diabetes educators who look after adults with Type 2 Diabetes wearing a CGM.
- **Use:** decision support; in this prototype the CGM data are replayed, not streamed. The system ranks patients, warns of a likely post-meal excursion up to 2 h ahead, explains why (SHAP attributions, which are not proof of causality), and lets the clinician simulate a lifestyle change. Simulations are estimates for discussion, not prescriptions.
- **Not for:**
  - automated insulin dosing or any closed-loop control;
  - Type 1 Diabetes, pregnancy or children;
  - hypoglycaemia prediction;
  - diagnosis;
  - use without a clinician in the loop;
  - medication changes (routed to clinician review; the SGLT2 scenario is a hidden research option).

## Training and evaluation data
- **Synthetic (main):** an India-context simulation of {M['cohort']['patients']} adults with T2D over {M['cohort']['days']} days. Records use the Synthea CSV schema (SNOMED CT, LOINC, RxNorm) and 5-min CGM / HR / HRV / steps / sleep with a realistic, incomplete food log. The cohort is modelled on Indian diets and routines; it is **not** data from Indian patients.
- **Splits:**
  - patient-level 210 train / 45 validation / 45 test;
  - the twin is fitted on days 1–{M['cohort']['calibration_days']} and evaluation runs on days {M['cohort']['calibration_days'] + 1}–{M['cohort']['days']};
  - all features are causal, using data at or before the decision time.
- **Retrospective open-cohort experiment (separate from the main evaluation):**
  - **Cohort:** {RW['cohort']['patients'] if RW else '–'} adults from Hall et al. 2018 (USA, n=57, CC-BY 4.0) and Colas et al. 2019 (Spain, n=208, CC-BY). Diagnosis mix: {', '.join(f'{v} {k}' for k, v in RW['cohort']['by_diagnosis'].items()) if RW else '–'}.
  - **Inputs:** CGM and lab records only.
  - **Method:** the pipeline is **re-trained** on these people with patient-level 5-fold cross-validation; no synthetic-trained weights are transferred.
  - **Target:** ≥140 mg/dL, because most participants are not diabetic.
  - **What it cannot validate:** the full two-stream Twin, meal/wearable fusion and Indian-population performance.

## Performance (45 held-out synthetic patients)
| Metric | Value |
|---|---|
| AUROC | {fm['auroc']:.3f} (95% CI {ci['lo']:.3f}–{ci['hi']:.3f}) |
| AUPRC | {fm['auprc']:.3f} (prevalence {pct(M['prevalence_test'])}) |
| Brier score / ECE | {fm['brier']:.3f} / {U['ece']:.3f} |
| Spikes warned ≥30 min ahead (delivered alerts) | {pct(ev['detected_lead_ge_30min'])} (95% CI {pct(ev['detected_lead_ge_30min_ci'][0])}–{pct(ev['detected_lead_ge_30min_ci'][1])}) |
| Median warning time (delivered alert → onset) | {ev['median_lead_min']:.0f} min |
| Alert precision | {pct(ev['alert_precision'])} of delivered alerts followed by an excursion within 2 h |
| False alerts per patient-day | {ev['false_alerts_per_patient_day']:.2f} (95% CI {ev['false_alerts_per_patient_day_ci'][0]:.2f}–{ev['false_alerts_per_patient_day_ci'][1]:.2f}) |
| Time-to-onset error | {TE['onset_mae_min']:.0f} min MAE |
| 2-h peak error | {TE['peak_mae_mgdl']:.1f} mg/dL MAE |
| Forecast MAE +30 / +60 / +120 min | {fc['30min']['mae_madhutwin']:.1f} / {fc['60min']['mae_madhutwin']:.1f} / {fc['120min']['mae_madhutwin']:.1f} mg/dL |
| Clarke A+B +30 / +60 / +120 min | {fc['30min']['clarke']['A+B'] * 100:.1f}% / {fc['60min']['clarke']['A+B'] * 100:.1f}% / {fc['120min']['clarke']['A+B'] * 100:.1f}% |
| 80% band, empirical coverage (all decisions) | {pct(fc['30min']['pi80_coverage'])} / {pct(fc['60min']['pi80_coverage'])} / {pct(fc['120min']['pi80_coverage'])} |
| 80% band at +2 h, coverage per patient (median, 10th–90th pct) | {pct(fc['120min']['pi80_coverage_by_patient']['median'])} ({pct(fc['120min']['pi80_coverage_by_patient']['p10'])}–{pct(fc['120min']['pi80_coverage_by_patient']['p90'])}) |
| Twin's AUROC gain (210 / {int(TV['low_data'][0]['train_patients'])} training patients) | {TV['auroc_gain']['auroc']['delta']:+.4f} / {TV['low_data'][0]['auroc_with_twin'] - TV['low_data'][0]['auroc_without_twin']:+.3f} |
"""
if RW:
    md += f"| **Open cohorts (retrospective CV, ≥140 mg/dL):** AUROC, **CGM-only** model | {RW['auroc_140']['CGM only']['auroc']['point']:.3f} (95% CI {RW['auroc_140']['CGM only']['auroc']['lo']:.3f}–{RW['auroc_140']['CGM only']['auroc']['hi']:.3f}) |\n"
    md += f"| **Open cohorts (retrospective CV, ≥140 mg/dL):** AUROC, **CGM + EHR** fusion | {RW['auroc_140'][FUS]['auroc']['point']:.3f} (95% CI {RW['auroc_140'][FUS]['auroc']['lo']:.3f}–{RW['auroc_140'][FUS]['auroc']['hi']:.3f}) |\n"
    md += f"| Open cohorts: 30-min forecast Clarke A+B | {RW['forecast']['30min']['clarke']['A+B'] * 100:.1f}% |\n"
if SD:
    md += f"| Sick-day detector AUROC (unsupervised) | {SD['unsupervised_auroc_test']:.3f} |\n"
md += f"""
## Model-mismatch stress test
{"" if not ST else f"Patients generated with deliberately different physiology (gut kinetics, insulin kinetics, 13-min sensor lag, 1.6× noise, a different insulin-sensitivity rhythm, fewer logged meals; n={ST['n_patients']}); models used unchanged. Results: risk AUROC {ST['risk_model_transferred']['auroc']:.3f} (matched {ST['matched_reference']['auroc']:.3f}); spikes warned ≥30 min ahead {pct(ST['event_level']['detected_lead_ge_30min'])}; insulin-sensitivity recovery ρ {ST['insulin_sensitivity_recovery_spearman']['twin_fused']:.2f}. What-if effect correlation stayed {ST['whatif_effects']['walk_20']['effect_corr']:.2f}–{ST['whatif_effects']['millet_swap']['effect_corr']:.2f}, but effect sizes were biased (millet: true {ST['whatif_effects']['millet_swap']['true_mean_effect']:.0f} vs Twin {ST['whatif_effects']['millet_swap']['pred_mean_effect']:.0f} mg/dL)."}

## Subgroup performance (exploratory)
AUROC across {len(sg)} subgroups (sex, age, BMI, HbA1c, medication, region) ranges from {worst['auroc']:.3f} ({worst['subgroup']}, 95% CI {worst.get('auroc_lo', float('nan')):.2f}–{worst.get('auroc_hi', float('nan')):.2f}) to {best['auroc']:.3f} ({best['subgroup']}). Subgroups are small (6–34 synthetic patients), so this is a diagnostic, not evidence of fairness. A release gate (block a model if any subgroup falls more than 0.02 below overall) is a *proposed* governance step.

## Uncertainty
Each risk comes with the SD of a 5-model bootstrap ensemble. This measures model disagreement and is **not** a calibrated clinical confidence. The label is *High* when SD < 0.04 ({pct(U['share_high_confidence'])} of decisions, AUROC {U['auroc_high_confidence']:.3f}), *Medium/Low* otherwise (AUROC {U['auroc_low_or_medium']:.3f}), and always *Low* when recent CGM data are insufficient. Forecast bands are split-conformal calibrated on the validation patients. The coverage shown is what was observed on held-out synthetic patients, where rows within a patient are correlated; it is not a guarantee.

## Limitations and risks
- The headline numbers come from a simulation whose equations the twin shares. Twin-fidelity and what-if results are therefore an upper bound.
- The open-cohort experiment covers CGM + labs only, in 265 mostly non-diabetic people (only {RW['cohort']['by_diagnosis'].get('T2D', 0) if RW else '–'} with T2D) from the USA and Spain. Meal, wearable and Indian-population effects, and the full Twin, are untested on real people.
- There is no live device integration: the console replays recorded synthetic data. FHIR output is a sample; ABDM integration is a proposed path.
- Performance depends on CGM wear and food-log quality. Missing meals lower meal-driven warnings.
- With 200+ training patients the twin adds little AUROC; its value lies in low-data settings, physiology estimates and counterfactuals.
- Alert fatigue: about {ev['false_alerts_per_patient_day']:.1f} false alerts per patient-day at the default threshold. Clinics can raise the threshold (see `operating_points` in `reports/metrics.json`).
- Not a medical device. Clinical use would need prospective validation and the CDSCO SaMD pathway.

## Ethical and privacy considerations
- No real patient data is in the repository; only synthetic and open CC-BY de-identified cohorts are used.
- The production design assumes consent per data stream under the DPDP Act 2023, ABDM consent artefacts, on-premise inference, minimised 5-min aggregates, an audit log, and a human in the loop for every action.
"""
if LAT:
    md += f"""
## Compute
- Measured on a {LAT['hardware']['cores']}-core CPU with no GPU.
- End-to-end alert: {LAT['end_to_end_alert_ms']:.1f} ms per patient per 5 min.
- What-if simulation: {LAT['whatif_5h_simulation_ms']:.1f} ms.
- Total model size: {sum(LAT['model_sizes_mb'].values()):.1f} MB.
- The full rebuild (`python run.py`) runs on a laptop CPU.
"""
(ROOT / "docs/model_card.md").write_text(md)
print("wrote docs/model_card.md")
