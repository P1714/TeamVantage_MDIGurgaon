# Response to "Solution Readiness & Technical Fix Plan"

Every item in the review, what was changed, and where to verify it. All numbers below are read from `reports/*.json`. In the status column, ✅ means done and ⏳ means only the team can do it.

## P0: release blockers

| Review item | Status | What changed | Verify in |
|---|---|---|---|
| Correct the real-world headline | ✅ | Every mention now gives the model variant, target, cohort and method: **CGM-only 0.809** and **CGM + EHR 0.793**, ≥140 mg/dL, n=265 (183 prediabetes, 72 normoglycaemic, 10 T2D), retrospective CV with re-training. It is described as "not external validation of the synthetic-trained Twin". | README §6.0 and §6.7, model card, console Evidence tab, deck slide 9, video |
| Recompute event-level metrics from the delivered alert stream | ✅ | `madhutwin/alerting.py` holds one policy shared by evaluation, console and API: threshold, ≤1 alert/hour, and ≥9 of 12 readings in the last hour. Matching to onsets in the 2 h window is one-to-one. Detection, lead time, precision and false alerts all come from the same alerts, and the console shows exactly these alerts. Results: 74.6% warned ≥30 min ahead (was 81.1%), median lead 75 min (was 105), 2.30 false alerts/day, precision 66%. | `madhutwin/model.py::event_level`, `tests/test_alerting.py` (8 tests: repeated crossings, debounce, one/multiple excursions, window edges, data sufficiency) |
| Reconcile dashboard forecast and peak semantics | ✅ | `madhutwin/contract.py` defines each field. `forecast_mgdl[+30/+60/+120]` is the ML value *at* that time. `predicted_peak_next_2h_mgdl` is the ML highest value at *any* time, reconciled to be ≥ every forecast and the current reading. `twin_projected_peak_mgdl` is the mechanistic Layer-1 scenario assuming the usual next meal. The console hero labels all three. | `tests/test_api.py::test_api_reproduces_console_values` and `test_contract_semantics` (the API, computed live from the models, matches the console's exported values) |
| Complete repo/video links; correct checklist status | ⏳ team | README marks the video and public-access rows ⏳ until real links exist. `python scripts/set_links.py --video … --repo …` fills them in and regenerates the README with ✅. | README checklist |

## P1: model validation and technical

| Review item | Status | What changed |
|---|---|---|
| Simulator–Twin circularity | ✅ | `scripts/stress_test.py` uses a generator with different gut and insulin kinetics, a 13-min sensor lag, 1.6× noise, a different insulin-sensitivity rhythm and 30% fewer logged meals. Models were not retrained. Risk AUROC was 0.918 (matched 0.912); insulin-sensitivity ρ 0.75 (0.78); what-if correlations 0.66–0.87. Effect sizes became **biased**: millet −40 true vs −57 predicted. This is reported in the README §6.8, model card and console. |
| What the 265-person cohort can and cannot validate | ✅ | Cohort composition and the n=10 T2D limitation are stated beside every real-data result. Meals, wearables, the full Twin and Indian patients are listed as untested. |
| API end-to-end test in CI | ✅ | The committed fixture `tests/fixtures/api_store_small.parquet` lets the API, contract and smoke tests run without the pipeline. Result: 28 passed, 0 skipped. CI fails if any test is skipped. |
| Qualify uncertainty claims | ✅ | Coverage is reported as "empirical", both overall and per patient (+2 h: median 80%, 10th–90th percentile 76–85%). `risk_sd` is described as model disagreement, not calibrated clinical confidence. The confidence label is Low whenever data are insufficient. |
| Terminology for the EHR-informed fit | ✅ | "Bayesian prior" became "EHR-informed prior with regularised grid-search fitting (MAP-style, not full Bayesian inference)" in the code, README, model card, architecture, deck and video. |
| Intervention guardrails | ✅ | A permanent banner reads "research prototype · decision support only · not for diagnosis or dosing · clinician judgement required". What-if results are labelled "simulated estimate for discussion, not a prescription". Actions are labelled "option to discuss · advisory". The sick-day card is advisory, and medication review is a clinician decision. An insufficient-data state suppresses alerts. The SGLT2 scenario stays a hidden research option. |

## Section 5: removed, hidden or reworded

| Wording | Now |
|---|---|
| "Clinical-grade evaluation" | "Evaluation details: held-out synthetic evaluation plus an exploratory retrospective evaluation on open cohorts" |
| "What-if answers you can trust" | "What-if simulator vs synthetic ground truth; real-patient effects remain unvalidated" |
| "AUROC 0.809 on real people" | "CGM-only 0.809 / CGM + EHR 0.793, ≥140 mg/dL, retrospective CV" |
| "Validated on 265 real people" | "Retrospective open-cohort evaluation (n=265; mostly prediabetes/normoglycaemic; CGM + labs only)" |
| "ABDM-ready" | "FHIR R4 sample output; ABDM integration is a proposed production path" |
| "Built to deploy in Indian clinics" | "Research prototype designed with a clinic deployment pathway" |
| Duplicate or conflicting "predicted peak" values | One canonical ML peak, the endpoint forecasts and the Twin projection, each labelled |
| Genotype as a predictive benefit | Not reintroduced; it stays in the record only |
| Automatic medication or dosing instruction | None; the console is advisory only |

## Section 6: additional work

| Item | Status |
|---|---|
| 6.1 Prediction contract | ✅ `madhutwin/contract.py`; the API returns it, the console uses the same values, and a test checks agreement |
| 6.2 Model comparison table | ✅ README §6.0: every row states cohort, target, split and model; the synthetic, stress-test and open-cohort rows are kept separate |
| 6.3 Threshold trade-off | ✅ The operating-point table includes precision, and all figures come from delivered alerts. The default is described as "reproducible demo rule, not clinically optimised" |
| 6.4 Missing data and sensor quality | ✅ Alert suppression below 9/12 readings (tested); an "insufficient recent CGM data" state in the console; feature missingness in `docs/data_dictionary.md` |
| 6.5 Clean-clone verification | ✅ Actually run. A fresh `git clone` ran `python run.py` with exit 0 in 14 min; 28 passed, 0 skipped; every metric was bit-identical (README §9) |
| 6.6 Limitations panel in the console | ✅ A permanent banner, plus the limitations card in the Evidence tab |
| 6.7 Fairness interpretation | ✅ Subgroup AUROC with patient-bootstrap CIs, labelled "exploratory". The release gate is labelled "proposed" |
| 6.8 Minimal end-to-end test path | ✅ `tests/test_api.py::test_smoke_end_to_end_for_one_patient` covers patient → CGM → risk + forecasts → Twin projection → what-if → FHIR bundle checks |

## Independent check
A separate reviewer compared every number in the README, model card, narration, console labels, architecture and deck against `reports/*.json`. It found no numeric mismatches; the wording issues it flagged were fixed.
