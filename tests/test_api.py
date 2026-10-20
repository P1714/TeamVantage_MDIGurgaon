"""API end-to-end, prediction-contract and smoke tests.

These run on a small committed fixture (tests/fixtures/api_store_small.parquet), so they never skip in CI.
``expected_console_values.json`` holds the numbers the console shows for the same rows; the API recomputes
them live from the saved models and must agree.
"""
import json
import os
from pathlib import Path

FIX = Path(__file__).parent / "fixtures"
os.environ.setdefault("MADHUTWIN_API_STORE", str(FIX / "api_store_small.parquet"))

from fastapi.testclient import TestClient  # noqa: E402

from madhutwin import contract as CT  # noqa: E402
from madhutwin.api import app  # noqa: E402

c = TestClient(app)
EXPECTED = json.loads((FIX / "expected_console_values.json").read_text())


def test_health_and_patients():
    h = c.get("/health").json()
    assert h["status"] == "ok" and h["ensemble_members"] == 5 and "not for diagnosis or dosing" in h["notice"]
    pts = c.get("/v1/patients").json()
    assert len(pts) >= 3


def test_api_reproduces_console_values():
    for e in EXPECTED:
        r = c.get(f"/v1/patients/{e['patient_id']}/risk", params={"step": e["step"]}).json()
        assert r["step"] == e["step"]
        assert abs(round(r["risk_2h"] * 100) - e["risk_pct"]) <= 1
        assert abs(round(r["risk_sd"] * 100) - e["risk_sd_pct"]) <= 1
        assert abs(r["forecast_mgdl"]["+120min"] - e["forecast_120"]) <= 1
        assert abs(r["forecast_mgdl"]["+30min"] - e["forecast_30"]) <= 1
        assert abs(r["predicted_peak_next_2h_mgdl"] - e["peak"]) <= 1
        assert abs(r["twin_projected_peak_mgdl"] - e["twin_peak"]) <= 1
        assert r["alert_delivered"] == e["alert"]
        assert r["readings_last_hour"] == e["readings_last_hour"]
        if r["risk_above_threshold"]:
            assert abs(r["time_to_onset_min"] - e["tte"]) <= 3
        else:
            assert r["time_to_onset_min"] is None


def test_contract_semantics():
    for e in EXPECTED:
        r = c.get(f"/v1/patients/{e['patient_id']}/risk", params={"step": e["step"]}).json()
        assert r["predicted_peak_next_2h_mgdl"] >= max(r["forecast_mgdl"].values()) - 1e-6
        assert r["predicted_peak_next_2h_mgdl"] >= r["cgm_mgdl"] - 1e-6
        for lo, hi in r["forecast_band_80"].values():
            assert lo < hi
        assert r["target"] == CT.TARGET and r["model_version"] == CT.MODEL_VERSION
    assert CT.confidence_label(0.01, data_sufficient=False) == "Low"
    assert CT.reconciled_peak(150, [170, 190, 160], 185) == 190
    assert CT.tte_or_none(42, 0.3, 0.4) is None and CT.tte_or_none(42, 0.5, 0.4) == 40


def test_smoke_end_to_end_for_one_patient():
    """Documented path: patient -> current CGM -> risk + forecasts -> Twin projection -> what-if -> FHIR."""
    e = next(x for x in EXPECTED if x["alert"])
    pid, step = e["patient_id"], e["step"]
    r = c.get(f"/v1/patients/{pid}/risk", params={"step": step}).json()
    assert r["cgm_mgdl"] < 180 and 0 <= r["risk_2h"] <= 1 and set(r["forecast_mgdl"]) == {"+30min", "+60min", "+120min"}
    assert len(r["top_reasons"]) == 5 and r["twin_projected_peak_mgdl"] > 0
    w0 = c.post("/v1/whatif", json={"patient_id": pid}).json()
    w1 = c.post("/v1/whatif", json={"patient_id": pid, "walk_minutes": 20}).json()
    assert w1["peak_mgdl"] <= w0["peak_mgdl"] and "not a treatment prescription" in w1["label"]
    f = c.get(f"/v1/patients/{pid}/fhir", params={"step": step}).json()
    assert f["resourceType"] == "Bundle" and f["type"] in ("collection", "transaction", "document")
    kinds = {x["resource"]["resourceType"] for x in f["entry"]}
    assert {"Patient", "Observation", "RiskAssessment"} <= kinds
    ra = next(x["resource"] for x in f["entry"] if x["resource"]["resourceType"] == "RiskAssessment")
    assert abs(ra["prediction"][0]["probabilityDecimal"] - r["risk_2h"]) < 1e-3
    for x in f["entry"]:
        assert x["fullUrl"].startswith("urn:uuid:") and "id" in x["resource"]


def test_unknown_patient_404():
    assert c.get("/v1/patients/NOPE/risk").status_code == 404
