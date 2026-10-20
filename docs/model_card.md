# Model card: MadhuTwin 2-hour hyperglycaemia warning

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
- **Inputs:** 107 features every 5 min from CGM dynamics, food log, steps, heart rate, HRV, sleep, clock, twin states and projections, and EHR fields. Genotype is excluded because it adds nothing (AUROC 0.9123 with vs 0.9117 without).
- **Operating threshold:** 0.40, chosen to maximise F1 on the validation patients. It is a reproducible demo rule, not a clinically optimised operating point.
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
- **Synthetic (main):** an India-context simulation of 300 adults with T2D over 21 days. Records use the Synthea CSV schema (SNOMED CT, LOINC, RxNorm) and 5-min CGM / HR / HRV / steps / sleep with a realistic, incomplete food log. The cohort is modelled on Indian diets and routines; it is **not** data from Indian patients.
- **Splits:**
  - patient-level 210 train / 45 validation / 45 test;
  - the twin is fitted on days 1–7 and evaluation runs on days 8–21;
  - all features are causal, using data at or before the decision time.
- **Retrospective open-cohort experiment (separate from the main evaluation):**
  - **Cohort:** 265 adults from Hall et al. 2018 (USA, n=57, CC-BY 4.0) and Colas et al. 2019 (Spain, n=208, CC-BY). Diagnosis mix: 183 prediabetes, 72 normoglycaemic, 10 T2D.
  - **Inputs:** CGM and lab records only.
  - **Method:** the pipeline is **re-trained** on these people with patient-level 5-fold cross-validation; no synthetic-trained weights are transferred.
  - **Target:** ≥140 mg/dL, because most participants are not diabetic.
  - **What it cannot validate:** the full two-stream Twin, meal/wearable fusion and Indian-population performance.

## Performance (45 held-out synthetic patients)
| Metric | Value |
|---|---|
| AUROC | 0.912 (95% CI 0.905–0.918) |
| AUPRC | 0.819 (prevalence 28%) |
| Brier score / ECE | 0.105 / 0.009 |
| Spikes warned ≥30 min ahead (delivered alerts) | 75% (95% CI 70%–79%) |
| Median warning time (delivered alert → onset) | 75 min |
| Alert precision | 66% of delivered alerts followed by an excursion within 2 h |
| False alerts per patient-day | 2.30 (95% CI 2.14–2.46) |
| Time-to-onset error | 17 min MAE |
| 2-h peak error | 18.4 mg/dL MAE |
| Forecast MAE +30 / +60 / +120 min | 8.2 / 13.5 / 21.0 mg/dL |
| Clarke A+B +30 / +60 / +120 min | 99.9% / 98.9% / 98.1% |
| 80% band, empirical coverage (all decisions) | 78% / 79% / 80% |
| 80% band at +2 h, coverage per patient (median, 10th–90th pct) | 80% (76%–85%) |
| Twin's AUROC gain (210 / 5 training patients) | +0.0007 / +0.009 |
| **Open cohorts (retrospective CV, ≥140 mg/dL):** AUROC, **CGM-only** model | 0.809 (95% CI 0.792–0.825) |
| **Open cohorts (retrospective CV, ≥140 mg/dL):** AUROC, **CGM + EHR** fusion | 0.793 (95% CI 0.772–0.814) |
| Open cohorts: 30-min forecast Clarke A+B | 99.2% |
| Sick-day detector AUROC (unsupervised) | 0.966 |

## Model-mismatch stress test
Patients generated with deliberately different physiology (gut kinetics, insulin kinetics, 13-min sensor lag, 1.6× noise, a different insulin-sensitivity rhythm, fewer logged meals; n=90); models used unchanged. Results: risk AUROC 0.918 (matched 0.912); spikes warned ≥30 min ahead 73%; insulin-sensitivity recovery ρ 0.75. What-if effect correlation stayed 0.66–0.87, but effect sizes were biased (millet: true -40 vs Twin -57 mg/dL).

## Subgroup performance (exploratory)
AUROC across 16 subgroups (sex, age, BMI, HbA1c, medication, region) ranges from 0.903 (HbA1c >= 7.5%, 95% CI 0.89–0.91) to 0.932 (Region: Northeast). Subgroups are small (6–34 synthetic patients), so this is a diagnostic, not evidence of fairness. A release gate (block a model if any subgroup falls more than 0.02 below overall) is a *proposed* governance step.

## Uncertainty
Each risk comes with the SD of a 5-model bootstrap ensemble. This measures model disagreement and is **not** a calibrated clinical confidence. The label is *High* when SD < 0.04 (86% of decisions, AUROC 0.928), *Medium/Low* otherwise (AUROC 0.720), and always *Low* when recent CGM data are insufficient. Forecast bands are split-conformal calibrated on the validation patients. The coverage shown is what was observed on held-out synthetic patients, where rows within a patient are correlated; it is not a guarantee.

## Limitations and risks
- The headline numbers come from a simulation whose equations the twin shares. Twin-fidelity and what-if results are therefore an upper bound.
- The open-cohort experiment covers CGM + labs only, in 265 mostly non-diabetic people (only 10 with T2D) from the USA and Spain. Meal, wearable and Indian-population effects, and the full Twin, are untested on real people.
- There is no live device integration: the console replays recorded synthetic data. FHIR output is a sample; ABDM integration is a proposed path.
- Performance depends on CGM wear and food-log quality. Missing meals lower meal-driven warnings.
- With 200+ training patients the twin adds little AUROC; its value lies in low-data settings, physiology estimates and counterfactuals.
- Alert fatigue: about 2.3 false alerts per patient-day at the default threshold. Clinics can raise the threshold (see `operating_points` in `reports/metrics.json`).
- Not a medical device. Clinical use would need prospective validation and the CDSCO SaMD pathway.

## Ethical and privacy considerations
- No real patient data is in the repository; only synthetic and open CC-BY de-identified cohorts are used.
- The production design assumes consent per data stream under the DPDP Act 2023, ABDM consent artefacts, on-premise inference, minimised 5-min aggregates, an audit log, and a human in the loop for every action.

## Compute
- Measured on a 2-core CPU with no GPU.
- End-to-end alert: 8.4 ms per patient per 5 min.
- What-if simulation: 14.5 ms.
- Total model size: 25.7 MB.
- The full rebuild (`python run.py`) runs on a laptop CPU.
