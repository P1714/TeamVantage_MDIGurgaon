"""MadhuTwin as a service: a small REST API around the trained twin.

    uvicorn madhutwin.api:app --port 8000      # then open http://localhost:8000/docs

Endpoints
    GET  /health
    GET  /v1/patients                         patients available in the demo store
    GET  /v1/patients/{pid}/risk?step=N       live inference: the full prediction contract (madhutwin/contract.py)
    POST /v1/whatif                           re-simulate the patient's twin with a change
    GET  /v1/patients/{pid}/fhir?step=N       FHIR R4 sample Bundle (RiskAssessment + Observations); ABDM is a proposed path
"""
from __future__ import annotations

import json
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import os

from . import alerting as AL
from . import config as C
from . import contract as CT
from . import fhir as FH
from . import whatif as W

app = FastAPI(title="MadhuTwin API", version="3.1",
              description="Digital twin for Type 2 Diabetes: 2-hour excursion risk, forecasts, what-if, FHIR R4 sample output. "
                          + CT.SAFETY_NOTE)
# MADHUTWIN_API_STORE lets tests/CI point the API at a small committed fixture instead of the full build output
STORE = Path(os.environ.get("MADHUTWIN_API_STORE", C.DATA_DIR / "processed" / "api_store.parquet"))
if not STORE.exists():  # fresh download before `python run.py`: serve the small committed fixture
    STORE = C.ROOT / "tests" / "fixtures" / "api_store_small.parquet"

# serve the offline doctor console from the same container at /console/
if (C.DASHBOARD_DIR / "index.html").exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/console", StaticFiles(directory=str(C.DASHBOARD_DIR), html=True), name="console")


@lru_cache(maxsize=1)
def _load():
    card = json.loads((C.MODEL_DIR / "model_card.json").read_text())
    mdl = lambda f: lgb.Booster(model_file=str(C.MODEL_DIR / f))
    models = {"risk": mdl("spike_risk_2h.txt"), "peak": mdl("peak_2h.txt"), "tte": mdl("time_to_onset.txt"),
              "fc": {h: mdl(f"forecast_{h}m_q50.txt") for h in (30, 60, 120)},
              "lo": {h: mdl(f"forecast_{h}m_q10.txt") for h in (30, 60, 120)},
              "hi": {h: mdl(f"forecast_{h}m_q90.txt") for h in (30, 60, 120)},
              "ens": [mdl(f"risk_ensemble_{b}.txt") for b in range(card.get("ensemble_members", 5))
                      if (C.MODEL_DIR / f"risk_ensemble_{b}.txt").exists()]}
    store = pd.read_parquet(STORE)
    tw = pd.read_csv(C.DATA_DIR / "processed" / "twin_params.csv").set_index("patient_id")
    return card, models, store, tw


def _row(pid, step):
    card, models, store, tw = _load()
    d = store[store.patient_id == pid]
    if d.empty:
        raise HTTPException(404, f"unknown patient {pid}")
    if step is None:
        return d.iloc[-1]
    hit = d[d.k <= step]
    if hit.empty:
        raise HTTPException(404, "no decision point at or before that step")
    return hit.iloc[-1]


@app.get("/health")
def health():
    card, models, *_ = _load()
    return {"status": "ok", "model": CT.MODEL_VERSION, "features": len(card["features"]),
            "threshold": round(card["threshold"], 2), "alert_policy": AL.POLICY_VERSION,
            "ensemble_members": len(models["ens"]), "notice": CT.SAFETY_NOTE}


@app.get("/v1/patients")
def patients():
    _, _, store, _ = _load()
    last = store.sort_values("k").groupby("patient_id").tail(1)
    return [{"patient_id": r.patient_id, "last_step": int(r.k), "glucose_mgdl": float(r.g_now)} for r in last.itertuples()]


def prediction(pid, step=None):
    """The full prediction contract (see madhutwin/contract.py) for one patient at one decision step."""
    card, mdl, store, tw = _load()
    r = _row(pid, step)
    cols = card["features"]
    thr = round(card["threshold"], 2)
    x = r[cols].to_numpy(float)[None, :]
    p = float(mdl["risk"].predict(x)[0])
    sd = float(np.std([m.predict(x)[0] for m in mdl["ens"]])) if mdl["ens"] else None
    contrib = mdl["risk"].predict(x, pred_contrib=True)[0][:-1]
    top = np.argsort(-np.abs(contrib))[:5]
    reasons = [{"feature": cols[i], "value": None if not np.isfinite(x[0, i]) else round(float(x[0, i]), 3),
                "effect": "raises risk" if contrib[i] > 0 else "lowers risk", "log_odds": round(float(contrib[i]), 3)}
               for i in top]
    g = float(r.g_now)
    fc = {h: g + float(mdl["fc"][h].predict(x)[0]) for h in (30, 60, 120)}
    cm = card.get("conformal_margin_mgdl", {})
    band = {f"+{h}min": [round(g + float(mdl["lo"][h].predict(x)[0]) - cm.get(str(h), cm.get(h, 0)), 1),
                         round(g + float(mdl["hi"][h].predict(x)[0]) + cm.get(str(h), cm.get(h, 0)), 1)] for h in (30, 60, 120)}
    peak = CT.reconciled_peak(g, fc.values(), float(mdl["peak"].predict(x)[0]))
    n_read = int(r["readings_last_hour"]) if "readings_last_hour" in r else None
    sufficient = n_read is not None and np.isfinite(g) and n_read >= AL.MIN_READINGS_LAST_HOUR
    when = pd.Timestamp(C.START_DATE) + timedelta(minutes=int(r.k) * 5)
    return {
        "patient_id": pid, "step": int(r.k), "decision_time": when.isoformat(), "model_version": CT.MODEL_VERSION,
        "target": CT.TARGET,
        "cgm_mgdl": g, "readings_last_hour": n_read, "data_sufficient": sufficient,
        "risk_2h": round(p, 4), "risk_sd": None if sd is None else round(sd, 4),
        "confidence": CT.confidence_label(sd, sufficient), "threshold": thr,
        "risk_above_threshold": p >= thr,
        "alert_delivered": bool(r["alert_delivered"]) if "alert_delivered" in r else None,
        "alert_policy": AL.POLICY_VERSION,
        "forecast_mgdl": {f"+{h}min": round(v, 1) for h, v in fc.items()},
        "forecast_band_80": band,
        "predicted_peak_next_2h_mgdl": round(peak, 1),
        "time_to_onset_min": CT.tte_or_none(float(mdl["tte"].predict(x)[0]), p, thr),
        "twin_projected_peak_mgdl": round(float(r["twin_proj_peak_habit"]), 1),
        "twin_carbs_in_gut_g": round(float(r["twin_gut_carbs_g"]), 1),
        "top_reasons": reasons,
        "notes": {"uncertainty": CT.UNCERTAINTY_NOTE, "safety": CT.SAFETY_NOTE,
                  "twin_projection": "Mechanistic Layer-1 scenario assuming the usual next meal; not the ML forecast."},
    }


@app.get("/v1/patients/{pid}/risk")
def risk(pid: str, step: int | None = None):
    return prediction(pid, step)


class WhatIf(BaseModel):
    patient_id: str
    carbs_g: float = Field(85, ge=0, le=250)
    gi: float = Field(72, ge=20, le=100)
    walk_minutes: float = Field(0, ge=0, le=90)
    walk_delay_minutes: float = Field(15, ge=0, le=120)
    meal_time_minutes: int = Field(13 * 60, ge=0, lt=1440, description="minute of day the meal starts")
    sglt2: bool = Field(False, description="research option only; medication changes require clinician review")


@app.post("/v1/whatif")
def whatif(req: WhatIf):
    *_, tw = _load()
    if req.patient_id not in tw.index:
        raise HTTPException(404, f"unknown patient {req.patient_id}")
    t = tw.loc[[req.patient_id]].reset_index()
    prm = W.twin_params(t)
    if req.sglt2:
        prm["sglt2"] = np.ones(1)
    walk = (req.walk_delay_minutes, req.walk_minutes, 105) if req.walk_minutes > 0 else None
    base = W.run(W.twin_params(t), 85, 72, None, start_tod=req.meal_time_minutes - 30)[0]
    traj = W.run(prm, req.carbs_g, req.gi, walk, start_tod=req.meal_time_minutes - 30)[0]
    return {"patient_id": req.patient_id, "label": "Simulated estimate from the patient's Twin, for discussion only; "
            "not a treatment prescription and not validated on real patients", "peak_mgdl": round(float(traj.max()), 1),
            "minutes_above_180": int((traj > 180).sum()), "usual_meal_peak_mgdl": round(float(base.max()), 1),
            "trajectory_every_10min": [round(float(v), 1) for v in traj[::10]]}


@app.get("/v1/patients/{pid}/fhir")
def fhir(pid: str, step: int | None = None):
    """FHIR R4 sample output (RiskAssessment + Observations). Not a live ABDM connection."""
    *_, tw = _load()
    pr = prediction(pid, step)
    t = tw.loc[pid]
    drivers = [f"{x['feature']}={x['value']} ({x['effect']})" for x in pr["top_reasons"]]
    return FH.bundle(pid, pd.Timestamp(pr["decision_time"]).to_pydatetime(), pr["cgm_mgdl"], pr["risk_2h"], drivers,
                     {"SI": t.twin_SI, "sigma": t.twin_sigma, "Gb": t.twin_Gb}, pr["threshold"])
