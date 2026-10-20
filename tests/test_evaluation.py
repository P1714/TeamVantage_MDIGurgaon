import numpy as np

from madhutwin import evaluation as EV
from madhutwin import fhir as FH


def test_clarke_zones_reference_points():
    ref = np.array([100, 60, 250, 50, 200, 300])
    pred = np.array([110, 65, 60, 200, 260, 150])
    assert list(EV.clarke_zones(ref, pred)) == ["A", "A", "E", "E", "B", "D"]


def test_conformal_margin_reaches_target_coverage():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 5000)
    lo, hi = np.full(5000, -0.5), np.full(5000, 0.5)       # deliberately too narrow
    q = EV.conformal_offsets(y, lo, hi, alpha=0.2)
    y2 = rng.normal(0, 1, 5000)
    cov = np.mean((y2 >= lo - q) & (y2 <= hi + q))
    assert 0.77 < cov < 0.83


def test_fhir_bundle_is_well_formed():
    from datetime import datetime
    b = FH.bundle("P0001", datetime(2026, 6, 20, 21, 30), 133, 0.93, ["hist_spike_rate_tod=1.0 (raises risk)"],
                  {"SI": 3e-4, "sigma": 0.02, "Gb": 120}, 0.39)
    kinds = [e["resource"]["resourceType"] for e in b["entry"]]
    assert b["resourceType"] == "Bundle" and kinds.count("RiskAssessment") == 1 and "Patient" in kinds
    ra = next(e["resource"] for e in b["entry"] if e["resource"]["resourceType"] == "RiskAssessment")
    assert ra["prediction"][0]["probabilityDecimal"] == 0.93
    assert all(e["fullUrl"].startswith("urn:uuid:") for e in b["entry"])
