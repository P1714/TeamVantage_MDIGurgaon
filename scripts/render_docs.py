"""Render docs/src/architecture.html to docs/Architecture_Diagram.pdf (+ PNG preview).

Headline numbers are injected from reports/metrics.json so the PDF never drifts from the code."""
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
m = json.loads((ROOT / "reports/metrics.json").read_text())
fm, ev, fc, tr = m["final_model"], m["event_level"], m["forecast"], m["twin_recovery"]
wi = m["whatif_validation"]["models"]["Personalised twin (EHR + CGM)"]
tiles = [(f"{fm['auroc']:.3f}", "AUROC, 2-h excursion ≥180 mg/dL"),
         (f"{ev['detected_lead_ge_30min'] * 100:.0f}%", "of spikes warned ≥30 min ahead (delivered alerts)"),
         (f"{ev['median_lead_min']:.0f} min", "median warning, alert → onset"),
         (f"{ev['false_alerts_per_patient_day']:.1f}", "false alerts per patient-day"),
         (f"{fc['120min']['mae_madhutwin']:.1f}", f"mg/dL error, 2-h forecast (vs {fc['120min']['mae_persistence']:.1f} persistence)"),
         (f"{tr['insulin_sensitivity']['twin_fused']:.2f}", "rank corr. twin vs true insulin sensitivity"),
         (f"{wi['millet_swap_effect_corr']:.2f}", "corr. predicted vs true rice→millet effect (synthetic truth)")]
rw = m.get("real_world")
if rw:
    fus = next(k for k in rw["auroc_140"] if k.startswith("CGM + EHR"))
    tiles.append((f"{rw['auroc_140']['CGM only']['auroc']['point']:.3f} / {rw['auroc_140'][fus]['auroc']['point']:.3f}",
                  f"AUROC CGM-only / CGM+EHR, {rw['cohort']['patients']} open-cohort adults (≥140, retrospective CV)"))
if m.get("sick_day"):
    tiles.append((f"{m['sick_day']['unsupervised_auroc_test']:.2f}", "sick-day detection AUROC (HR/HRV/sleep)"))
html = ('<div class="card" style="margin-top:8mm"><h2>Headline results · held-out synthetic evaluation (45 virtual patients) + retrospective open-cohort check</h2>'
        f'<div style="display:grid;grid-template-columns:repeat({len(tiles)},1fr);gap:10px">' +
        "".join(f'<div style="border:1px solid var(--line);border-radius:8px;padding:10px"><div style="font:700 21px DejaVu Sans Mono,monospace">{v}</div>'
                f'<div style="font-size:12px;color:var(--ink2);margin-top:4px">{l}</div></div>' for v, l in tiles) + "</div></div>")
src = (ROOT / "docs/src/architecture.html").read_text().replace("<!--RESULTS-->", html)

# ---- page 3: measured twin value + latency (filled from reports, never typed by hand)
tv = m["twin_value"]
ld = tv.get("low_data", [])
ms = tv["matched_sensitivity"]
g = tv["auroc_gain"]["auroc"]
rows = "".join(f"<tr><td>{r['train_patients']} training patients</td><td>{r['auroc_without_twin']:.3f}</td><td>{r['auroc_with_twin']:.3f}</td>"
               f"<td><b>{r['auroc_with_twin'] - r['auroc_without_twin']:+.3f}</b></td></tr>" for r in ld)
cs = m.get("cold_start_first_48h", {})
cs_txt = ""
if cs:
    k = sorted(cs, key=lambda n: cs[n]["auroc"])
    cs_txt = f"<tr><td>First 48 h of a new patient</td><td colspan=3>{k[0]}: {cs[k[0]]['auroc']:.3f} → {k[-1]}: <b>{cs[k[-1]]['auroc']:.3f}</b></td></tr>"
twin_html = (
    '<table><tr><th>Where</th><th>No twin</th><th>With twin</th><th>Gain</th></tr>' + rows +
    f'<tr><td>Full data (210 patients)</td><td>{tv["auroc_without_twin"]:.3f}</td><td>{tv["auroc_with_twin"]:.3f}</td>'
    f'<td>{g["delta"]:+.4f} [{g["lo"]:+.4f}, {g["hi"]:+.4f}]</td></tr>'
    f'<tr><td>False alerts/day at equal sensitivity</td><td>{ms["false_alerts_per_day_without_twin"]:.2f}</td>'
    f'<td>{ms["false_alerts_per_day_with_twin"]:.2f}</td><td></td></tr>'
    f'<tr><td>Insulin-sensitivity estimate (ρ vs truth)</td><td>{tr["insulin_sensitivity"]["cgm_only_twin"]:.2f} CGM only</td>'
    f'<td>{tr["insulin_sensitivity"]["twin_fused"]:.2f}</td><td>fused</td></tr>'
    f'<tr><td>What-if: rice→millet / portion / walk (ρ)</td><td colspan=2>{wi["millet_swap_effect_corr"]:.2f} / {wi["portion_minus30_effect_corr"]:.2f} / {wi["walk_20_effect_corr"]:.2f}</td><td>synthetic truth</td></tr></table>'
    '<p style="font-size:11.5px;color:#4c5a60;margin:6px 0 0">Honest reading: with 210 training patients the ML layer already learns most of what the twin knows, '
    'so the AUROC gain is within noise. The twin earns its place where data are scarce (new clinics, new patients), in estimating physiology, '
    'and in answering “what if?” questions, which a pure prediction model cannot answer at all.</p>')
src = src.replace("<!--TWINVALUE-->", twin_html)
lat_p = ROOT / "reports/latency.json"
if lat_p.exists():
    L = json.loads(lat_p.read_text())
    sz = sum(L["model_sizes_mb"].values())
    lat_html = ('<table>' + "".join(f"<tr><td>{a}</td><td class='k'>{b}</td></tr>" for a, b in [
        ("Twin state update (1 patient)", f"{L['twin_state_update_ms']:.3f} ms"),
        ("2-h twin projection", f"{L['twin_2h_projection_ms']:.2f} ms"),
        ("Risk + SHAP reasons", f"{L['risk_with_shap_explanation_ms']:.2f} ms"),
        ("End-to-end alert (per patient, per 5 min)", f"{L['end_to_end_alert_ms']:.1f} ms"),
        ("What-if 5-h simulation", f"{L['whatif_5h_simulation_ms']:.1f} ms"),
        ("REST API risk request", f"{L['api_risk_request_ms']:.1f} ms" if L.get("api_risk_request_ms") else "n/a"),
        ("All model files", f"{sz:.1f} MB"),
        ("Hardware", f"{L['hardware']['cores']}-core CPU, no GPU")]) + '</table>'
        f'<p style="font-size:11.5px;color:#4c5a60;margin:6px 0 0">One CPU core can refresh about {int(5 * 60 * 1000 / max(L["end_to_end_alert_ms"], 0.01)):,} patients every 5 minutes.</p>')
    src = src.replace("<!--LATENCY-->", lat_html)
n_tests = sum(open(f).read().count("\ndef test_") for f in (ROOT / "tests").glob("test_*.py"))
src = src.replace("__NTESTS__", str(n_tests))
tmp = ROOT / "docs/src/_architecture_rendered.html"
tmp.write_text(src)
with sync_playwright() as pw:
    b = pw.chromium.launch()
    p = b.new_page(viewport={"width": 1587, "height": 1123})
    p.goto(tmp.as_uri())
    p.wait_for_timeout(500)
    p.pdf(path=str(ROOT / "docs/Architecture_Diagram.pdf"), width="420mm", height="297mm", print_background=True)
    p.locator("section.page").first.screenshot(path=str(ROOT / "docs/img/architecture.png"))
    p.locator("section.page").nth(2).screenshot(path=str(ROOT / "docs/img/architecture_p3.png"))
    b.close()
tmp.unlink()
print("wrote docs/Architecture_Diagram.pdf and docs/img/architecture.png")
