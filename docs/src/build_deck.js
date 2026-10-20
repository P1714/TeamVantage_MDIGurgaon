// Builds docs/MadhuTwin_Presentation.pptx from reports/metrics.json + figures.
// Usage: node docs/src/build_deck.js
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..", "..");
const M = JSON.parse(fs.readFileSync(path.join(ROOT, "reports/metrics.json"), "utf8"));
const IMG = f => path.join(ROOT, "docs/img", f);
const TEAM = JSON.parse(fs.readFileSync(path.join(ROOT, "team.json"), "utf8"));

const C = { ink: "15222A", ink2: "4C5A60", muted: "7F8B8F", paper: "FBFBF9", line: "E1E4DE", turmeric: "C9820A", turmericSoft: "F5E9D2",
  teal: "1BAF7A", tealSoft: "E1F5EC", blue: "2A78D6", blueSoft: "E3EEFB", orange: "EB6834", violet: "4A3AA7", red: "D03B3B", gray: "9AA6AB", white: "FFFFFF" };
const HEAD = "Cambria", BODY = "Calibri";

const fm = M.final_model, ev = M.event_level, fc = M.forecast, tr = M.twin_recovery;
const wi = M.whatif_validation.models;
const WP = wi["Personalised twin (EHR + CGM)"], WE = wi["EHR-only twin (day 0)"], WPOP = wi["Population-average model"];

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.author = TEAM.team_name;
pres.title = "MadhuTwin: a digital twin for Type 2 Diabetes in India";

function dot(slide, x, y, d = 0.32) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: C.turmeric }, line: { color: "F3C35B", width: 1.5 } });
}
function title(slide, t, sub, dark = false) {
  slide.addText(t, { x: 0.6, y: 0.38, w: 12.1, h: 0.75, fontFace: HEAD, fontSize: 32, bold: true, color: dark ? C.white : C.ink, margin: 0, isTextBox: true });
  if (sub) slide.addText(sub, { x: 0.6, y: 1.1, w: 12.1, h: 0.45, fontFace: BODY, fontSize: 16, color: dark ? "B3BFC1" : C.ink2, margin: 0, isTextBox: true });
}
function footer(slide, n, dark = false) {
  slide.addText(`MadhuTwin · ${TEAM.team_name}`, { x: 0.6, y: 7.05, w: 6, h: 0.3, fontFace: BODY, fontSize: 10, color: dark ? "7D8A8D" : C.muted, margin: 0, isTextBox: true });
  slide.addText(String(n), { x: 12.2, y: 7.05, w: 0.5, h: 0.3, fontFace: BODY, fontSize: 10, color: dark ? "7D8A8D" : C.muted, align: "right", margin: 0, isTextBox: true });
}
function stat(slide, x, y, w, big, label, color = C.ink, fill = C.white) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 1.35, rectRadius: 0.08, fill: { color: fill }, line: { color: C.line, width: 1 } });
  slide.addText(big, { x: x + 0.2, y: y + 0.12, w: w - 0.4, h: 0.7, fontFace: HEAD, fontSize: 32, bold: true, color, margin: 0, isTextBox: true });
  slide.addText(label, { x: x + 0.2, y: y + 0.8, w: w - 0.4, h: 0.48, fontFace: BODY, fontSize: 12, color: C.ink2, margin: 0, valign: "top", isTextBox: true });
}
function card(slide, x, y, w, h, head, body, fill = C.white, headColor = C.ink) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: C.line, width: 1 } });
  slide.addText(head, { x: x + 0.22, y: y + 0.16, w: w - 0.44, h: 0.4, fontFace: BODY, fontSize: 16, bold: true, color: headColor, margin: 0, isTextBox: true });
  slide.addText(body, { x: x + 0.22, y: y + 0.6, w: w - 0.44, h: h - 0.75, fontFace: BODY, fontSize: 14, color: C.ink2, margin: 0, valign: "top", isTextBox: true, paraSpaceAfter: 6 });
}
const NTESTS = fs.readdirSync(path.join(ROOT, "tests")).filter(f => /^test_.*\.py$/.test(f)).map(f => (fs.readFileSync(path.join(ROOT, "tests", f), "utf8").match(/^def test_/gm) || []).length).reduce((a, b) => a + b, 0);
const bullets = arr => arr.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < arr.length - 1 } }));
const pct = v => `${Math.round(v * 100)}%`;
let n = 0;

// 1 — Title -----------------------------------------------------------------
{
  const s = pres.addSlide(); n++;
  s.background = { color: C.ink };
  dot(s, 0.6, 0.7, 0.55);
  s.addText("MadhuTwin", { x: 0.6, y: 1.7, w: 12, h: 1.2, fontFace: HEAD, fontSize: 60, bold: true, color: C.white, margin: 0, isTextBox: true });
  s.addText("A digital twin for Type 2 Diabetes in India that warns the doctor about a glucose spike two hours before it happens, and shows what would prevent it.",
    { x: 0.6, y: 2.95, w: 10.6, h: 1.1, fontFace: BODY, fontSize: 22, color: "D6DEDF", margin: 0, isTextBox: true });
  s.addText([{ text: "Madhu", options: { bold: true, color: "E3A54A" } }, { text: " (मधु) = honey; madhumeha is the Ayurvedic name for diabetes.", options: { color: "9FB0B3" } }],
    { x: 0.6, y: 4.2, w: 11, h: 0.4, fontFace: BODY, fontSize: 14, margin: 0, isTextBox: true });
  s.addText([{ text: TEAM.team_name, options: { bold: true, breakLine: true } }, { text: TEAM.college, options: { breakLine: true } },
    { text: "Happiest Health Digital Twin Challenge 2026 · Phase 1 submission" }],
    { x: 0.6, y: 5.6, w: 11, h: 1.0, fontFace: BODY, fontSize: 14, color: "B3BFC1", margin: 0, isTextBox: true });
  s.addNotes("Hello, we are presenting MadhuTwin: a digital twin for people with type 2 diabetes in India. It predicts a glucose spike two hours before it happens, explains why, and lets the doctor test what would prevent it, all on a virtual replica of the patient.");
}

// 2 — Problem ---------------------------------------------------------------
{
  const s = pres.addSlide(); n++; s.background = { color: C.paper };
  title(s, "India has 101 million people with diabetes", "and their glucose is managed in 3-monthly snapshots, not in the hours when it actually goes wrong");
  stat(s, 0.6, 1.9, 3.0, "101 M", "adults with diabetes (11.4%)", C.ink);
  stat(s, 3.8, 1.9, 3.0, "136 M", "more with prediabetes", C.ink);
  stat(s, 7.0, 1.9, 5.7, "~70 g+", "carbs in a typical Indian lunch; rice, roti and sweet chai drive steep post-meal spikes", C.turmeric);
  card(s, 0.6, 3.55, 6.0, 3.1, "What a doctor sees today",
    bullets(["An HbA1c every 3 months: an average that hides daily spikes", "A CGM report after the fact, if the patient has a sensor", "Generic diet advice that ignores how this person responds", "No early warning and no way to test a change before prescribing it"]));
  card(s, 6.8, 3.55, 5.9, 3.1, "What MadhuTwin adds",
    bullets(["A virtual replica of each patient, updated every 5 minutes", "A 2-hour warning before glucose crosses 180 mg/dL", "Plain-language reasons: meal habit, sleep, stress, activity", "A what-if simulator: rice → millet, smaller portion, walk after eating"]), C.tealSoft);
  s.addText("Source: ICMR-INDIAB-17, Lancet Diabetes & Endocrinology, 2023.", { x: 0.6, y: 6.75, w: 8, h: 0.3, fontFace: BODY, fontSize: 10, color: C.muted, margin: 0, isTextBox: true });
  footer(s, n);
  s.addNotes("India has about one hundred and one million adults with diabetes, and another one hundred and thirty six million with prediabetes. Yet care is still driven by an HbA1c every three months, an average that hides the daily spikes after high carbohydrate meals. Doctors get no early warning, and no way to test a lifestyle change for a specific patient before recommending it.");
}

// 3 — Use case + approach -----------------------------------------------------
{
  const s = pres.addSlide(); n++; s.background = { color: C.paper };
  title(s, "Use case: predict post-meal spikes 2 hours ahead", "Condition: Type 2 Diabetes · Adverse event: an excursion ≥180 mg/dL lasting ≥15 minutes (International Consensus on Time in Range)");
  const cols = [
    ["Static EHR", "Demographics, diagnoses, meds, labs (HbA1c, HOMA-IR, HOMA-B, eGFR), 12-month trends", C.turmericSoft],
    ["Wearables", "CGM every 5 min, heart rate, HRV, steps, sleep stages, food log", C.blueSoft],
    ["Layer 1 · Twin", "Personal insulin sensitivity, beta-cell response, basal glucose; hidden state every 5 min; 2-h projection", C.tealSoft],
    ["Layer 2 · AI", "LightGBM risk ± confidence, time to event, forecasts; reasons in plain words", "EBE8F8"],
    ["Doctor console", "Who → when → why → what to do → what if I intervene?", "FDE9E0"],
  ];
  cols.forEach((c, i) => {
    const x = 0.6 + i * 2.5;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 2.1, w: 2.3, h: 2.6, rectRadius: 0.1, fill: { color: c[2] }, line: { color: C.line } });
    s.addText(c[0], { x: x + 0.18, y: 2.25, w: 1.95, h: 0.5, fontFace: BODY, fontSize: 16, bold: true, color: C.ink, margin: 0, isTextBox: true });
    s.addText(c[1], { x: x + 0.18, y: 2.8, w: 1.95, h: 1.8, fontFace: BODY, fontSize: 12.5, color: C.ink2, margin: 0, valign: "top", isTextBox: true });
    if (i < 4) s.addText("→", { x: x + 2.28, y: 3.1, w: 0.24, h: 0.5, fontFace: BODY, fontSize: 20, color: C.muted, align: "center", margin: 0, isTextBox: true });
  });
  card(s, 0.6, 5.0, 12.1, 1.75, "Why a hybrid twin, not just a predictor",
    "A pure ML model can say what will probably happen. A mechanistic twin can also answer what would happen if the patient changed something, and it can be personalised from day 0 using the health record. MadhuTwin uses both: physiology for personalisation and counterfactuals, ML for accuracy on messy real-life data.");
  footer(s, n);
  s.addNotes("We picked type 2 diabetes and one precise adverse event: a glucose excursion above 180 milligrams per decilitre lasting at least fifteen minutes, predicted two hours ahead. The system fuses the static health record with wearable streams through a physiology twin and a machine learning layer, and presents everything in a doctor's console. We chose a hybrid design because a pure predictor can't answer what-if questions, and a pure physiology model struggles with messy real-life data.");
}

// 4 — Data ------------------------------------------------------------------
{
  const s = pres.addSlide(); n++; s.background = { color: C.paper };
  title(s, "Data: 300 India-context synthetic patients", "EHRs in the Synthea CSV schema; wearables from a physiology simulator of Indian days · plus a separate retrospective check on 265 adults from open cohorts (USA + Spain)");
  s.addImage({ path: IMG("fig_cohort.png"), x: 0.6, y: 1.75, w: 7.4, h: 2.22, sizing: { type: "contain", w: 7.4, h: 2.22 } });
  card(s, 8.2, 1.75, 4.5, 4.95, "What makes it realistic",
    bullets(["37 regional foods with carbs and GI: idli, poha, rajma chawal, fish-rice, millets", "Late dinners, sugary chai (often unlogged), festival thalis, eating out",
      "Sleep with 90-min cycles, stressful days, illness, missed doses", "CGM lag, noise, drift and dropouts; watch off-wrist while charging",
      "Food log completed only ~56% of the time, carbs under-estimated", "Hidden 'true' physiology kept aside to validate the twin"]));
  const rows = [["Stream", "Resolution", "Size"], ["CGM glucose", "5 min", `${(M.cohort.cgm_readings / 1e6).toFixed(2)} M readings`],
    ["HR · HRV · steps · sleep", "5 min", "6,048 per patient per signal"], ["Food log", "event", `${M.cohort.logged_meals.toLocaleString("en-IN")} meals`],
    ["EHR (Synthea schema)", "static", "labs, SNOMED CT, LOINC, RxNorm"]];
  s.addTable(rows.map((r, i) => r.map(c => ({ text: c, options: { bold: i === 0, color: i === 0 ? C.ink2 : C.ink, fontSize: 12, fontFace: BODY } }))),
    { x: 0.6, y: 4.2, w: 7.4, colW: [2.6, 1.4, 3.4], border: { type: "solid", color: C.line, pt: 0.75 }, fill: { color: C.white }, rowH: 0.42 });
  footer(s, n);
  s.addNotes("Because real patient data is restricted under the DPDP Act and HIPAA, we built three hundred synthetic patients modelled on Indian diets and routines. Health records follow the Synthea schema with standard medical codes. The wearable streams come from a physiology simulator of Indian free-living days: regional meals, late dinners, sugary chai, festivals, stress, poor sleep, illness and missed medicine. Crucially, each virtual patient has a hidden true physiology, which lets us check whether our twin really learned the person.");
}

// 5 — Architecture ----------------------------------------------------------
{
  const s = pres.addSlide(); n++; s.background = { color: C.paper };
  title(s, "Architecture", "Layer 1 keeps a live physiological state per patient; Layer 2 predicts from it. Every arrow runs every 5 minutes.");
  s.addImage({ path: IMG("architecture.png"), x: 0.6, y: 1.6, w: 12.1, h: 5.35, sizing: { type: "contain", w: 12.1, h: 5.35 } });
  footer(s, n);
  s.addNotes("Here is the architecture. Data sources on the left are harmonised onto a five-minute grid. Layer one, the twin core, personalises a glucose-insulin model for each patient, keeps estimating hidden states like carbohydrate still in the gut, and projects two hours ahead. Layer two, the AI, fuses about a hundred features for a LightGBM risk model and glucose forecasts, with SHAP explanations. Everything feeds the doctor's console.");
}

// 6 — The twin --------------------------------------------------------------
{
  const s = pres.addSlide(); n++; s.background = { color: C.paper };
  title(s, "The twin learns each patient's physiology", "Personalised from an EHR-informed prior + week-1 CGM (regularised fit), then checked against the simulator's hidden truth");
  s.addImage({ path: IMG("fig_twin_recovery.png"), x: 0.6, y: 1.7, w: 8.2, h: 3.2, sizing: { type: "contain", w: 8.2, h: 3.2 } });
  card(s, 9.0, 1.7, 3.7, 3.2, "Fusion matters here", `CGM alone cannot separate insulin resistance from weak insulin secretion (rho ${tr.insulin_sensitivity.cgm_only_twin.toFixed(2)}). Labs alone are noisy. Together, the twin recovers insulin sensitivity at rho ${tr.insulin_sensitivity.twin_fused.toFixed(2)} and beta-cell function at ${tr.beta_cell_function.twin_fused.toFixed(2)}.`, C.tealSoft);
  const steps = [["A · Personalise", "Fit SI, β-cell gain, basal glucose, dawn rise"], ["B · Observe", "Track carbs in gut, insulin action, true plasma glucose"], ["C · Project", "Simulate 2 h: food eaten + usual next meal"], ["D · What-if", "Re-simulate with a food swap, smaller portion or walk"]];
  steps.forEach((t, i) => {
    const x = 0.6 + i * 3.05;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 5.15, w: 2.85, h: 1.55, rectRadius: 0.08, fill: { color: C.white }, line: { color: C.teal, width: 1 } });
    s.addText(t[0], { x: x + 0.18, y: 5.27, w: 2.5, h: 0.4, fontFace: BODY, fontSize: 15, bold: true, color: C.ink, margin: 0, isTextBox: true });
    s.addText(t[1], { x: x + 0.18, y: 5.7, w: 2.5, h: 0.9, fontFace: BODY, fontSize: 12.5, color: C.ink2, margin: 0, valign: "top", isTextBox: true });
  });
  footer(s, n);
  s.addNotes("The heart of the system is the physiology twin. From the lab report and the first week of CGM, it fits each patient's insulin sensitivity, beta-cell response, basal glucose and dawn rise. This is where fusing the two data streams clearly pays off: CGM alone cannot tell insulin resistance apart from weak insulin secretion, and labs alone are noisy, but together the twin recovers insulin sensitivity with a rank correlation of about zero point seven eight against the hidden truth.");
}

// 6b — Twin vs ordinary AI ------------------------------------------------------
{
  const s = pres.addSlide(); n++; s.background = { color: C.paper };
  title(s, "A digital twin, not just another risk model", "How the EHR and wearables are fused into one live patient state");
  const rows = [["", "Ordinary AI risk model", "MadhuTwin"],
    ["Answers", "“Will glucose go high?”", "“What state is this person in, what happens next, what if we act?”"],
    ["Per-patient model", "No; shared weights", "Yes: SI, β-cell gain, basal glucose, dawn rise"],
    ["New patient", "Needs history", "Starts from the EHR on day 0; matures over week 1"],
    ["Interventions", "Cannot simulate", "Food swap, portion, walk: simulated estimates"],
    ["Drift", "Retrain", "Nightly re-personalisation"]];
  s.addTable(rows.map((r, i) => r.map((c, j) => ({ text: c, options: { bold: i === 0 || j === 0, fontSize: 12.5, fontFace: BODY,
    color: i === 0 ? C.ink2 : C.ink, fill: { color: j === 2 && i > 0 ? C.tealSoft : C.white } } }))),
    { x: 0.6, y: 1.75, w: 7.0, colW: [1.7, 2.3, 3.0], border: { type: "solid", color: C.line, pt: 0.75 }, rowH: 0.5 });
  card(s, 0.6, 5.0, 7.0, 1.9, "How fusion works", bullets([
    "EHR → informed prior for a regularised fit of the twin's physiology (HOMA-IR → insulin sensitivity, HOMA-B → β-cell gain)",
    "Wearables → the twin's observer every 5 min (CGM), its inputs (meals, steps) and personal baselines (HR, HRV, sleep)",
    "Twin state + raw streams + EHR → Layer 2 AI"]), C.white);
  const sv = [["Patient state vector", "Updated"], ["Glucose (plasma, ahead of sensor)", "5 min"], ["Carbs still in the gut", "5 min"],
    ["Insulin action", "5 min"], ["Activity, HR vs own baseline", "5 min"], ["Sleep debt, HRV vs baseline", "nightly"],
    ["Insulin sensitivity (× baseline)", "nightly"], ["Basal glucose", "nightly"], ["β-cell gain, dawn rise", "weekly"],
    ["HbA1c, eGFR, medicines", "new result"], ["Habitual next meal", "weekly"]];
  s.addTable(sv.map((r, i) => r.map((c, j) => ({ text: c, options: { bold: i === 0, fontSize: 12, fontFace: BODY, color: i === 0 ? C.ink2 : C.ink, align: j ? "center" : "left" } }))),
    { x: 7.85, y: 1.75, w: 4.85, colW: [3.45, 1.4], border: { type: "solid", color: C.line, pt: 0.75 }, fill: { color: C.white }, rowH: 0.43 });
  footer(s, n);
  s.addNotes("What makes this a digital twin rather than another risk model? An ordinary model shares one set of weights across everyone and can only say whether glucose will go high. MadhuTwin keeps a personal physiology for each patient, starts from their health record on day one, keeps a live state vector updated every five minutes, re-personalises every night, and can simulate what would happen if they changed something. The health record enters as a prior on the twin's physiology, and the wearables drive its state.");
}

// 7 — Prediction results ------------------------------------------------------
{
  const s = pres.addSlide(); n++; s.background = { color: C.paper };
  title(s, "It warns before the spike", `45 held-out patients the model never saw · days 8–21 · a decision every 5 minutes`);
  const ciA = M.ablation["MadhuTwin (fusion + physiology twin)"].ci.auroc;
  stat(s, 0.6, 1.7, 2.9, fm.auroc.toFixed(3), `AUROC, ≥180 mg/dL within 2 h (95% CI ${ciA.lo.toFixed(3)}–${ciA.hi.toFixed(3)})`, C.teal);
  stat(s, 3.65, 1.7, 2.9, pct(ev.detected_lead_ge_30min), `of ${ev.events.toLocaleString()} spikes warned ≥30 min ahead by delivered alerts (CI ${pct(ev.detected_lead_ge_30min_ci[0])}–${pct(ev.detected_lead_ge_30min_ci[1])})`, C.ink);
  stat(s, 6.7, 1.7, 2.9, `${ev.median_lead_min} min`, "median warning before onset", C.ink);
  stat(s, 9.75, 1.7, 2.95, ev.false_alerts_per_patient_day.toFixed(1), `false alerts per patient-day (threshold ${fm.threshold.toFixed(2)})`, C.ink);
  s.addImage({ path: IMG("fig_patient_day.png"), x: 0.6, y: 3.25, w: 7.2, h: 3.65, sizing: { type: "contain", w: 7.2, h: 3.65 } });
  const TE = M.time_to_event, U = M.uncertainty;
  card(s, 8.05, 3.25, 4.65, 3.65, `Operating threshold ${fm.threshold.toFixed(2)}: why`,
    bullets([`Chosen on validation patients (best F1), never on test`, `Debounced to ≤1 alert per hour. “We optimise for useful alerts, not maximum sensitivity.”`,
      (() => { const o = M.operating_points.find(o => Math.abs(o.threshold - 0.7) < 1e-6); return `A busy clinic can pick 0.70: ${o.false_alerts_per_patient_day.toFixed(1)} false alerts/day, still ${pct(o.detected_lead_ge_30min)} of spikes ≥30 min early`; })(),
      `When: onset time error ${Math.round(TE.onset_mae_min)} min; 2-h peak error ${TE.peak_mae_mgdl.toFixed(0)} mg/dL`,
      `Calibrated: ECE ${U.ece.toFixed(3)}, Brier ${fm.brier.toFixed(3)}; confidence from a 5-model ensemble`]));
  footer(s, n);
  s.addNotes(`On forty five patients the model never saw, the risk model reaches an AUROC of ${fm.auroc.toFixed(2)}. At the default threshold it flags ${pct(ev.detected_lead_ge_30min)} of glucose spikes at least thirty minutes in advance, with a median warning of about ${ev.median_lead_min} minutes, at around two to three false alerts per patient per day. The chart shows a synthetic held-out day: the risk rises before each meal-related spike. We chose the threshold on validation patients and debounce alerts, because we optimise for useful alerts, not maximum sensitivity. The model also estimates when the spike will start, to within about ${Math.round(M.time_to_event.onset_mae_min)} minutes.`);
}


// 8 — Real people -------------------------------------------------------------
{
  const R = M.real_world, s = pres.addSlide(); n++; s.background = { color: C.paper };
  title(s, "Retrospective check on open cohorts", `${R.cohort.patients} adults (${Object.entries(R.cohort.by_diagnosis).map(([k, v]) => `${v} ${k}`).join(", ")}) · CGM + labs only · re-trained, patient-level CV · target ≥140 mg/dL`);
  s.addImage({ path: IMG("fig_realworld.png"), x: 0.6, y: 1.65, w: 12.1, h: 3.4, sizing: { type: "contain", w: 12.1, h: 3.4 } });
  const a = R.auroc_140["CGM only"].auroc, ir = R.insulin_resistance, f30 = R.forecast["30min"];
  const af = R.auroc_140[Object.keys(R.auroc_140).find(k => k.startsWith("CGM + EHR"))].auroc;
  card(s, 0.6, 5.15, 3.9, 1.35, `AUROC ${a.point.toFixed(3)} / ${af.point.toFixed(3)}`, `CGM-only / CGM + EHR, ≥140 mg/dL in 2 h (trend baseline ${R.auroc_140["CGM trend extrapolation (baseline)"].auroc.point.toFixed(3)}); not the full system`, C.tealSoft);
  card(s, 4.7, 5.15, 3.9, 1.35, `${(f30.clarke["A+B"] * 100).toFixed(1)}% in Clarke zones A+B`, `30-min forecasts: ${f30.mae_fusion.toFixed(1)} mg/dL error vs ${f30.mae_persistence.toFixed(1)} for 'no change'`);
  s.addText("Not external validation of the synthetic-trained twin: the pipeline was re-trained and cross-validated within these cohorts (no person in both train and test). Only 10 people have T2D; meals, wearables, the full twin and Indian patients are untested on real data.",
    { x: 0.6, y: 6.55, w: 12.1, h: 0.45, fontFace: BODY, fontSize: 10.5, color: C.ink2, margin: 0, isTextBox: true });
  card(s, 8.8, 5.15, 3.9, 1.35, "Fusion wins on physiology", `Insulin resistance (gold-standard SSPG): labs + CGM ρ ${ir["EHR + CGM (fused)"].spearman.toFixed(2)} vs labs ${ir["EHR labs only"].spearman.toFixed(2)}, CGM ${ir["CGM only"].spearman.toFixed(2)} (n=${ir.n_patients}); the alarm itself is CGM-driven`);
  footer(s, n);
  s.addNotes(`We also ran a retrospective check on ${R.cohort.patients} adults from two open cohorts, with CGM and lab records only. We re-trained the pipeline there with patient-level cross-validation. For a 140 milligram event, the CGM-only model reached an AUROC of ${a.point.toFixed(2)} and the CGM plus EHR model ${af.point.toFixed(2)}. Only ten participants have type two diabetes, so this is a check of the method, not validation of the full twin.`);
}

// 9 — Rigor -------------------------------------------------------------------
{
  const s = pres.addSlide(); n++; s.background = { color: C.paper };
  title(s, "Evaluation details", "Held-out synthetic evaluation · Clarke error grid · patient-level 95% CIs · empirical forecast bands · exploratory subgroups");
  s.addImage({ path: IMG("fig_clarke.png"), x: 0.6, y: 1.65, w: 7.6, h: 2.65, sizing: { type: "contain", w: 7.6, h: 2.65 } });
  s.addImage({ path: IMG("fig_fairness.png"), x: 8.35, y: 1.55, w: 4.4, h: 5.4, sizing: { type: "contain", w: 4.4, h: 5.4 } });
  const sg = M.subgroups.map(g => g.auroc);
  card(s, 0.6, 4.5, 7.6, 2.35, "How the numbers were checked", bullets([
    `Forecasts: ${(fc["30min"].clarke["A+B"] * 100).toFixed(1)}% / ${(fc["60min"].clarke["A+B"] * 100).toFixed(1)}% / ${(fc["120min"].clarke["A+B"] * 100).toFixed(1)}% in Clarke zones A+B at 30 / 60 / 120 min; zone D+E (dangerous) ${((fc["30min"].clarke.D + fc["30min"].clarke.E) * 100).toFixed(2)}% at 30 min`,
    `Empirical coverage of split-conformal 80% bands: ${pct(fc["60min"].pi80_coverage)} at 60 min (raw quantiles ${pct(fc["60min"].pi80_coverage_before_conformal)}); observed on this test set, not guaranteed`,
    `Subgroups (exploratory, small): AUROC ${Math.min(...sg).toFixed(3)}–${Math.max(...sg).toFixed(3)} across sex, age, BMI, HbA1c, medication and region`,
    "Every headline number carries a 95% CI from resampling patients"]));
  footer(s, n);
  s.addNotes("Forecasts are scored on the Clarke error grid, the standard for glucose prediction. Prediction bands are conformally calibrated, and we report the coverage we actually observed rather than a guarantee. Every headline number has a patient-level confidence interval, and an exploratory subgroup check shows similar performance across sex, age, weight, medication and region.");
}

// 10 — Honest ablation ---------------------------------------------------------
{
  const s = pres.addSlide(); n++; s.background = { color: C.paper };
  title(s, "What each data stream contributes", "AUROC on held-out patients; we report what we found, including where fusion did not help");
  const names = ["CGM trend only", "Time of day only", "Twin physics alone", "EHR only", "Wearables only", "EHR + wearables", "MadhuTwin (all)"];
  const vals = [M.baselines["CGM trend extrapolation"].auroc, M.baselines["Time-of-day only"].auroc, M.baselines["Physiology twin alone (projected peak)"].auroc,
    M.ablation["EHR only"].auroc, M.ablation["Wearables only"].auroc, M.ablation["EHR + wearables (fusion)"].auroc, M.ablation["MadhuTwin (fusion + physiology twin)"].auroc];
  s.addChart(pres.charts.BAR, [{ name: "AUROC", labels: names, values: vals.map(v => +v.toFixed(3)) }], {
    x: 0.6, y: 1.7, w: 7.3, h: 5.1, barDir: "bar", chartColors: [C.gray, C.gray, C.gray, C.blue, C.blue, C.blue, C.teal],
    valAxisMinVal: 0.5, valAxisMaxVal: 1.0, valAxisMajorUnit: 0.1, showValue: true, dataLabelFormatCode: "0.000", dataLabelFontSize: 11, dataLabelColor: C.ink,
    catAxisLabelFontSize: 12, catAxisLabelColor: C.ink, valAxisLabelColor: C.muted, valAxisLabelFontSize: 10, valGridLine: { color: "E6E8E3", size: 0.5 }, catGridLine: { style: "none" },
    showLegend: false, barGapWidthPct: 45, catAxisOrientation: "maxMin",
  });
  card(s, 8.1, 1.7, 4.6, 5.1, "What we learned",
    bullets(["Once a patient has a week of CGM, their own glucose history carries most of the signal", `EHR alone is informative (${M.ablation["EHR only"].auroc.toFixed(3)}) but adds little on top of wearables for the 2-h alarm`,
      `Where the EHR matters: personalising physiology (insulin sensitivity rho ${tr.insulin_sensitivity.cgm_only_twin.toFixed(2)} → ${tr.insulin_sensitivity.twin_fused.toFixed(2)}) and the what-if simulator`,
      `Genotype added nothing (AUROC ${M.genetics_ablation.auroc_with_genetics.toFixed(4)} with vs ${M.genetics_ablation.auroc_without_genetics.toFixed(4)} without), so the risk model leaves it out`, "The twin's value is in low data and counterfactuals: next slide"]));
  footer(s, n);
  s.addNotes("We ran an honest ablation. Time of day alone is surprisingly strong, because meals are habitual. Wearables lift AUROC to about zero point nine one. Adding the health record barely moves the two-hour alarm once a week of CGM exists: the patient's own glucose history already says it. The health record earns its place elsewhere: it is what lets the twin separate insulin resistance from secretion, and that is what the what-if simulator builds on.");
}

// 10b — Measured value of the twin ---------------------------------------------
{
  const s = pres.addSlide(); n++; s.background = { color: C.paper };
  const TV = M.twin_value, LD = TV.low_data, G = TV.auroc_gain.auroc, MS = TV.matched_sensitivity, MT = M.twin_maturity;
  title(s, "What the twin measurably adds", "Same LightGBM, same data, with vs without twin features · held-out patients");
  s.addChart(pres.charts.BAR, [{ name: "AUROC gain from the twin", labels: LD.map(r => `${r.train_patients} pts`),
    values: LD.map(r => +(r.auroc_with_twin - r.auroc_without_twin).toFixed(4)) }], {
    x: 0.6, y: 1.7, w: 6.2, h: 3.4, barDir: "col", chartColors: [C.teal], showValue: true, dataLabelFormatCode: "+0.000;-0.000", dataLabelFontSize: 11,
    catAxisLabelFontSize: 11, valAxisLabelFontSize: 10, valAxisLabelColor: C.muted, valGridLine: { color: "E6E8E3", size: 0.5 }, showLegend: false,
    showTitle: true, title: "AUROC gain from twin features vs training patients", titleFontSize: 13, titleColor: C.ink2 });
  card(s, 0.6, 5.25, 6.2, 1.65, "Honest reading", `With all 210 training patients the gain is within noise (${G.delta >= 0 ? "+" : ""}${G.delta.toFixed(4)}, 95% CI ${G.lo >= 0 ? "+" : ""}${G.lo.toFixed(4)} to +${G.hi.toFixed(4)}); false alerts at equal sensitivity ${MS.false_alerts_per_day_without_twin.toFixed(2)} → ${MS.false_alerts_per_day_with_twin.toFixed(2)}/day. The twin helps most where data are scarce: a new clinic or a new patient.`);
  const m0 = MT[0], m7 = MT[MT.length - 1];
  stat(s, 7.05, 1.7, 2.75, `${tr.insulin_sensitivity.cgm_only_twin.toFixed(2)}→${tr.insulin_sensitivity.twin_fused.toFixed(2)}`, "insulin-sensitivity ρ vs truth: CGM-only → fused twin", C.teal);
  stat(s, 9.95, 1.7, 2.75, `${m0.di.toFixed(2)}→${m7.di.toFixed(2)}`, `disposition index ρ: day 0 (EHR only) → day ${m7.days}`, C.teal);
  stat(s, 7.05, 3.2, 2.75, WP.millet_swap_effect_corr.toFixed(2), "ρ predicted vs true rice→millet effect (ML alone: n/a)", C.ink);
  const NT = M.nightly_tracking;
  stat(s, 9.95, 3.2, 2.75, `×${NT.illness_in_window.si_mult.toFixed(2)}`, `insulin sensitivity tracked nightly during illness (×${NT.no_illness.si_mult.toFixed(2)} when well)`, C.ink);
  card(s, 7.05, 4.7, 5.65, 2.2, "Twin maturity", bullets([`Day 0: personalised from the EHR alone, no CGM history needed`,
    `Days 1–${m7.days}: physiology sharpens (DI ρ ${MT.map(r => r.di.toFixed(2)).join(" → ")})`, "After week 1: nightly re-fit tracks illness, stress and missed doses"]), C.tealSoft);
  footer(s, n);
  s.addNotes(`Judges should ask what the twin really adds, so we measured it. With all two hundred and ten training patients, adding twin features to the same model changes AUROC by less than a thousandth, within noise, and we say so. But with only five or ten training patients, as in a new clinic, the twin adds clearly more. It also recovers each patient's physiology far better than CGM alone, gets sharper day by day, tracks illness night by night, and answers what-if questions that a pure prediction model cannot answer at all.`);
}

// 9 — What-if validated ------------------------------------------------------
{
  const s = pres.addSlide(); n++; s.background = { color: C.paper };
  title(s, "What-if simulator vs synthetic ground truth", "Each intervention was re-run on the hidden 'true' virtual patient · real-patient effects remain unvalidated");
  s.addImage({ path: IMG("fig_whatif.png"), x: 0.6, y: 1.7, w: 8.3, h: 3.05, sizing: { type: "contain", w: 8.3, h: 3.05 } });
  const rows = [["Correlation with true effect", "Personalised twin", "EHR-only twin", "Population model"],
    ["20-min walk after lunch", WP.walk_20_effect_corr, WE.walk_20_effect_corr, WPOP.walk_20_effect_corr],
    ["Rice → foxtail millet", WP.millet_swap_effect_corr, WE.millet_swap_effect_corr, WPOP.millet_swap_effect_corr],
    ["30% smaller portion", WP.portion_minus30_effect_corr, WE.portion_minus30_effect_corr, WPOP.portion_minus30_effect_corr],
    ["Ranking patients by usual post-lunch peak", WP.baseline_peak_rank_corr, WE.baseline_peak_rank_corr, WPOP.baseline_peak_rank_corr]];
  s.addTable(rows.map((r, i) => r.map((c, j) => ({ text: typeof c === "number" ? c.toFixed(2) : c,
    options: { bold: i === 0 || j === 1, color: j === 1 && i > 0 ? "0E7A52" : i === 0 ? C.ink2 : C.ink, fontSize: 12.5, fontFace: BODY, align: j === 0 ? "left" : "center", fill: { color: j === 1 && i > 0 ? C.tealSoft : C.white } } }))),
    { x: 0.6, y: 4.95, w: 8.3, colW: [3.5, 1.6, 1.6, 1.6], border: { type: "solid", color: C.line, pt: 0.75 }, rowH: 0.36 });
  card(s, 9.1, 1.7, 3.6, 5.1, "Why it matters", bullets([`Average peak drop: millet swap ${Math.round(-WP.millet_swap_true_mean_effect)} mg/dL, smaller portion ${Math.round(-WP.portion_minus30_true_mean_effect)}, walk ${Math.round(-WP.walk_20_true_mean_effect)}`, "But responses differ person to person; the personalised twin ranks who benefits most", "Stress test with different physics: rankings held, effect sizes biased", "Shown as simulated estimates for discussion, not prescriptions"]), C.tealSoft);
  footer(s, n);
  s.addNotes("This is the part a pure predictor cannot do. We asked the twin what happens if a patient walks after lunch, swaps rice for foxtail millet, or eats a smaller portion, then re-ran the same change on the hidden true patient. The personalised twin predicts each person's benefit with a correlation of about zero point nine for the food changes, better than an EHR-only or population model. This is a synthetic benchmark: when we changed the simulator's physics, the twin still ranked who benefits but over-estimated some effect sizes, so we show these as estimates for discussion.");
}


// 12 — Sick-day ---------------------------------------------------------------
{
  const S = M.sick_day, s = pres.addSlide(); n++; s.background = { color: C.paper };
  title(s, "A second warning: sick days from heart and sleep data", "The brief's example of a weekly anomaly, made specific to diabetes care");
  s.addImage({ path: IMG("fig_sickday.png"), x: 0.6, y: 1.65, w: 8.2, h: 2.55, sizing: { type: "contain", w: 8.2, h: 2.55 } });
  stat(s, 9.0, 1.65, 3.7, S.unsupervised_auroc_test.toFixed(2), "AUROC for illness days, with no labelled training data", C.teal);
  stat(s, 9.0, 3.1, 3.7, pct(S.flag_sensitivity), `of illness days flagged; ${S.flag_false_flags_per_patient_week.toFixed(2)} false flags per patient-week`, C.ink);
  card(s, 0.6, 4.65, 12.1, 2.25, "How it works and why it matters", bullets([
    "Every morning, last night's sleeping heart rate, HRV, sleep and yesterday's activity are compared with the patient's own previous 7 nights (personal z-scores)",
    `Flagged days ran hotter: ${Math.round(S.tar_rest_of_day_flagged * 100)}% of the rest of the day above 180 mg/dL vs ${Math.round(S.tar_rest_of_day_not_flagged * 100)}% otherwise`,
    "Triggers 'sick-day rules': check glucose more often, fluids, review SGLT2 inhibitor / metformin if dehydrated, call the doctor",
    "Caveat: illness effects in the simulator are clear-cut; real-world infection detection from wearables is weaker but well documented"]));
  footer(s, n);
  s.addNotes("The brief also mentions flagging an anomaly from a week of sleep and heart rate data. For diabetes, the most useful one is a sick day. Each morning the twin compares last night's heart rate, HRV and sleep with the patient's own previous week. It flags hidden illness days well without any labelled training data, and flagged days really do run hotter, so the doctor can trigger sick-day rules early.");
}

// 13 — Console ------------------------------------------------------------------
{
  const s = pres.addSlide(); n++; s.background = { color: C.paper };
  title(s, "The doctor's console", "Static web app: runs offline in any browser, no install");
  s.addImage({ path: IMG("console_now.png"), x: 0.6, y: 1.6, w: 8.2, h: 4.92, sizing: { type: "contain", w: 8.2, h: 4.92 } });
  card(s, 9.0, 1.6, 3.7, 4.92, "A doctor can",
    bullets(["Who: triage by risk × severity × confidence × time to event", "When: “spike likely in ~1 h 20 m, peak ~230 mg/dL”, with confidence", "Why: reasons in plain words, with % share", "What to do: one suggested action", "What if: simulate it → peak, time >180, event probability", "Twin state today vs the patient's own baseline"]));
  footer(s, n);
  s.addNotes("This is the doctor's console, built around five questions. Who needs attention: patients are ranked by risk, severity, confidence and time to event. When: the hero card says when the spike is likely and how high. Why: reasons in plain words with their share. What to do: one suggested action. And what if I intervene: the twin re-simulates the action and shows the new peak, time above range and event probability. Below, the twin state is compared with the patient's own baseline.");
}


// 14 — Deploy -------------------------------------------------------------------
{
  const s = pres.addSlide(); n++; s.background = { color: C.paper };
  title(s, "Designed with a clinic deployment pathway", "Research prototype · interoperable outputs · offline-capable · patient-facing");
  s.addImage({ path: IMG("console_app.png"), x: 0.6, y: 1.6, w: 4.3, h: 5.3, sizing: { type: "contain", w: 4.3, h: 5.3 } });
  card(s, 5.15, 1.6, 3.65, 2.55, "Patient app nudges", bullets(["One message, one action, before the usual meal", "English and हिन्दी today; strings table for more languages", "Sick-day banner with 'call doctor'"]), C.turmericSoft);
  card(s, 9.05, 1.6, 3.65, 2.55, "FHIR R4 sample output", bullets(["Bundle: RiskAssessment + Observations", "Pseudonymous Patient (DPDP data minimisation)", "ABDM integration: proposed production path"]), C.blueSoft);
  card(s, 5.15, 4.35, 3.65, 2.55, "Twin as a service", bullets(["REST API (FastAPI): risk, forecast, what-if, FHIR", "Docker image serves API + console", "Runs on a laptop CPU; no GPU or cloud needed"]), C.tealSoft);
  card(s, 9.05, 4.35, 3.65, 2.55, "Engineering quality", bullets([`${NTESTS} automated tests, none skipped: alert matching, API contract, leakage`, "GitHub Actions CI", "One command: python run.py"]));
  footer(s, n);
  s.addNotes("MadhuTwin is a research prototype designed with a deployment pathway. Patients get one clear nudge before their usual meal, in English or Hindi. The twin produces sample FHIR R4 output, and integration with India's digital health mission is a proposed next step. A REST API and Docker image serve the model and the console, and automated tests run on every commit.");
}

// 15 — Explainability, privacy, safety ---------------------------------------
{
  const s = pres.addSlide(); n++; s.background = { color: C.paper };
  title(s, "Explainable, private, honest about limits", null);
  s.addImage({ path: IMG("fig_shap.png"), x: 0.6, y: 1.35, w: 5.6, h: 3.8, sizing: { type: "contain", w: 5.6, h: 3.8 } });
  card(s, 6.5, 1.35, 3.0, 5.35, "Privacy & compliance", bullets(["Synthetic patients + open de-identified cohorts; no PHI anywhere", "FHIR R4 sample output; ABDM integration is a proposed path", "Designed for on-premise inference; consent-first under the DPDP Act 2023", "Model card and decision-support labelling in the console"]));
  card(s, 9.7, 1.35, 3.0, 5.35, "Limitations", bullets(["Main results are synthetic; open-cohort check: CGM + labs only, 10 with T2D; replayed data, no live devices", "Twin and simulator share equations: fidelity is an upper bound; what-if sizes biased under mismatch", "Twin adds little AUROC once 200+ patients exist", "Not a medical device; decision support only"]), "FDF2EE");
  s.addText("Top drivers of risk (SHAP attributions, not causes): the patient's own time-of-day spike history, heart rate relative to baseline, time of day, and the twin's projected peak including the usual next meal.",
    { x: 0.6, y: 5.3, w: 5.6, h: 1.4, fontFace: BODY, fontSize: 13, color: C.ink2, margin: 0, valign: "top", isTextBox: true });
  footer(s, n);
  s.addNotes("Every alert comes with its reasons. The strongest drivers are the patient's own history at that time of day, heart rate relative to baseline, the time of day, and the twin's projected peak including their usual next meal. We used only synthetic and open de-identified data, the outputs are sample FHIR resources, and we are explicit about limits: the open-cohort check covers CGM and labs only, and twin fidelity on simulated patients is an upper bound.");
}

// 12 — Roadmap ---------------------------------------------------------------
{
  const s = pres.addSlide(); n++; s.background = { color: C.paper };
  title(s, "From prototype to clinic", "What we would build next");
  const phases = [["Now", "300 synthetic patients; 265-person open-cohort check; nightly twin; console, patient app, FHIR sample, API, CI", C.tealSoft], ["Next 3 months", "Validate on an Indian CGM cohort with meals and wearables; hypoglycaemia risk; shadow-mode run in one clinic", C.blueSoft],
    ["6 months", "Live ABDM / Health Connect connectors; more Indian languages; clinician usability study", C.turmericSoft], ["12 months", "Prospective pilot with a diabetes clinic; weekly cardiovascular-risk twin from sleep and HRV; CDSCO SaMD pathway", "FDE9E0"]];
  phases.forEach((p, i) => {
    const x = 0.6 + i * 3.05;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.9, w: 2.85, h: 3.5, rectRadius: 0.1, fill: { color: p[2] }, line: { color: C.line } });
    s.addText(p[0], { x: x + 0.2, y: 2.05, w: 2.5, h: 0.5, fontFace: HEAD, fontSize: 20, bold: true, color: C.ink, margin: 0, isTextBox: true });
    s.addText(p[1], { x: x + 0.2, y: 2.65, w: 2.5, h: 2.6, fontFace: BODY, fontSize: 13.5, color: C.ink2, margin: 0, valign: "top", isTextBox: true });
  });
  card(s, 0.6, 5.65, 12.1, 1.15, "Impact", "The aim, still to be tested in a clinic: turn CGM from a retrospective report into a proactive, personalised coach, with advice tailored to how each patient responds and a screen doctors can act on in a two-minute consult.");
  footer(s, n);
  s.addNotes("Next we would validate on an Indian cohort that also has meals and wearables, re-personalise the twin nightly, and add hypoglycaemia risk. Then connect to ABDM health records and phone health platforms, with nudges in Indian languages, and run a prospective clinic pilot. The goal is to turn CGM from a retrospective report into a proactive, personalised coach.");
}

// 13 — Team / thanks -----------------------------------------------------------
{
  const s = pres.addSlide(); n++; s.background = { color: C.ink };
  dot(s, 0.6, 0.7, 0.45);
  s.addText("Thank you", { x: 0.6, y: 1.4, w: 12, h: 1, fontFace: HEAD, fontSize: 48, bold: true, color: C.white, margin: 0, isTextBox: true });
  s.addText(TEAM.team_name, { x: 0.6, y: 2.6, w: 12, h: 0.5, fontFace: BODY, fontSize: 20, bold: true, color: "E3A54A", margin: 0, isTextBox: true });
  s.addText([{ text: "Team Leader", options: { bold: true, color: "9FB0B3", breakLine: true } }, { text: TEAM.leader, options: { breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "Team Members", options: { bold: true, color: "9FB0B3", breakLine: true } }, ...TEAM.members.map((m, i) => ({ text: m, options: { breakLine: i < TEAM.members.length - 1 } }))],
    { x: 0.6, y: 3.2, w: 12, h: 2.4, fontFace: BODY, fontSize: 18, color: "D6DEDF", margin: 0, valign: "top", isTextBox: true });
  s.addText(TEAM.college, { x: 0.6, y: 5.9, w: 12, h: 0.5, fontFace: BODY, fontSize: 14, color: "B3BFC1", margin: 0, isTextBox: true });
  s.addNotes("Thank you. All code, data generation, models, the console and documentation are in our public GitHub repository, and everything rebuilds with one command: python run.py.");
}

pres.writeFile({ fileName: path.join(ROOT, "docs/MadhuTwin_Presentation.pptx") }).then(f => console.log("wrote", f));
