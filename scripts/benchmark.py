"""Measure how fast MadhuTwin runs on a plain CPU -> reports/latency.json."""
import json
import os
import platform
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))
from madhutwin import config as C  # noqa: E402
from madhutwin import physiology as PH  # noqa: E402
from madhutwin import whatif as W  # noqa: E402


def timeit(fn, n=200):
    fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1000  # ms


card = json.loads((C.MODEL_DIR / "model_card.json").read_text())
cols = card["features"]
risk = lgb.Booster(model_file=str(C.MODEL_DIR / "spike_risk_2h.txt"))
store = pd.read_parquet(C.DATA_DIR / "processed" / "api_store.parquet")
x = store[cols].iloc[[0]].to_numpy(float)
tw = pd.read_csv(C.DATA_DIR / "processed" / "twin_params.csv").iloc[[0]]
prm = W.twin_params(tw)
s = PH.init_state(1, prm["Gb"])


def twin_step():
    PH.step(s, prm, 5.0, prm["Gb"], prm["SI"], np.zeros(1), np.zeros(1), np.zeros(1))


def projection():
    s2 = {k: v.copy() for k, v in s.items()}
    for _ in range(24):
        PH.step(s2, prm, 5.0, prm["Gb"], prm["SI"], np.zeros(1), np.zeros(1), np.zeros(1))


out = {
    "hardware": {"cpu": platform.processor() or platform.machine(), "cores": os.cpu_count(), "python": platform.python_version(),
                 "gpu": "none"},
    "twin_state_update_ms": timeit(twin_step, 2000),
    "twin_2h_projection_ms": timeit(projection, 500),
    "risk_inference_ms": timeit(lambda: risk.predict(x), 300),
    "risk_with_shap_explanation_ms": timeit(lambda: risk.predict(x, pred_contrib=True), 100),
    "whatif_5h_simulation_ms": timeit(lambda: W.run(prm, 85, 72, (15, 20, 105)), 20),
    "model_sizes_mb": {p.name: round(p.stat().st_size / 1e6, 2) for p in sorted(C.MODEL_DIR.glob("*.txt"))},
}
try:
    from fastapi.testclient import TestClient
    from madhutwin.api import app
    c = TestClient(app)
    pid = c.get("/v1/patients").json()[0]["patient_id"]
    out["api_risk_request_ms"] = timeit(lambda: c.get(f"/v1/patients/{pid}/risk"), 50)
except Exception as exc:  # pragma: no cover
    out["api_risk_request_ms"] = None
    print("API benchmark skipped:", exc)
out["end_to_end_alert_ms"] = round(out["twin_state_update_ms"] + 2 * out["twin_2h_projection_ms"]
                                   + out["risk_with_shap_explanation_ms"], 2)
for k, v in list(out.items()):
    if isinstance(v, float):
        out[k] = round(v, 3)
(C.REPORT_DIR / "latency.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
