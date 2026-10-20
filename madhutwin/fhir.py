"""HL7 FHIR R4 export of the twin's outputs, so MadhuTwin can plug into an
ABDM-compliant health record (ABDM's health-information exchange uses FHIR R4).

For one patient and one moment we emit a ``Bundle`` (type ``collection``) with:

* ``Patient``: pseudonymous id only (no names; DPDP data minimisation)
* ``Observation``: latest CGM glucose (LOINC 2339-0, mg/dL)
* ``RiskAssessment``: probability of a ≥180 mg/dL excursion within 2 h, with
  the top drivers in ``prediction.rationale``
* ``Observation`` ×4: the twin's personalised physiology (insulin sensitivity,
  beta-cell response, basal glucose, disposition index), coded in a
  MadhuTwin code system until standard codes exist
* ``Device``: the software, labelled as decision support
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone

CS = "https://github.com/madhutwin/codes"   # project code system for twin-specific concepts
IST = timezone(timedelta(hours=5, minutes=30))


def _id(*parts):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "madhutwin/" + "/".join(map(str, parts))))


def bundle(patient_id: str, when: datetime, glucose_mgdl: float | None, risk: float | None,
           drivers: list[str], twin: dict, threshold: float) -> dict:
    when = when.replace(tzinfo=IST) if when.tzinfo is None else when
    ts = when.isoformat()
    pat_ref = f"Patient/{_id('patient', patient_id)}"
    dev_ref = f"Device/{_id('device')}"
    entries = [
        {"resourceType": "Patient", "id": _id("patient", patient_id),
         "identifier": [{"system": f"{CS}/pseudonymous-id", "value": patient_id}],
         "meta": {"security": [{"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": "PSEUDED"}]}},
        {"resourceType": "Device", "id": _id("device"),
         "deviceName": [{"name": "MadhuTwin digital twin (research prototype)", "type": "user-friendly-name"}],
         "type": {"text": "Clinical decision support software, not a medical device"}},
    ]
    if glucose_mgdl is not None:
        entries.append({"resourceType": "Observation", "id": _id("cgm", patient_id, ts), "status": "final",
                        "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                                  "code": "vital-signs"}]}],
                        "code": {"coding": [{"system": "http://loinc.org", "code": "2339-0",
                                             "display": "Glucose [Mass/volume] in Blood"}],
                                 "text": "Continuous glucose monitor reading"},
                        "subject": {"reference": pat_ref}, "effectiveDateTime": ts,
                        "valueQuantity": {"value": float(glucose_mgdl), "unit": "mg/dL",
                                          "system": "http://unitsofmeasure.org", "code": "mg/dL"}})
    if risk is not None:
        entries.append({"resourceType": "RiskAssessment", "id": _id("risk", patient_id, ts), "status": "final",
                        "subject": {"reference": pat_ref}, "occurrenceDateTime": ts, "performer": {"reference": dev_ref},
                        "method": {"text": "MadhuTwin hybrid physiology twin + LightGBM, v2"},
                        "prediction": [{
                            "outcome": {"text": "Glycaemic excursion >=180 mg/dL lasting >=15 min"},
                            "probabilityDecimal": round(float(risk), 4),
                            "whenRange": {"low": {"value": 0, "unit": "min"}, "high": {"value": 120, "unit": "min"}},
                            "rationale": "; ".join(drivers)}],
                        "note": [{"text": f"Alert threshold {threshold:.2f}. Decision support only."}]})
    for code, name, val, unit in [("SI", "Insulin sensitivity (twin)", twin["SI"] * 1e4, "10^-4/min per uU/mL"),
                                  ("sigma", "Beta-cell secretory gain (twin)", twin["sigma"] * 100, "x100"),
                                  ("Gb", "Basal glucose set-point (twin)", twin["Gb"], "mg/dL"),
                                  ("DI", "Disposition index (twin)", twin["SI"] * twin["sigma"] * 1e5, "1")]:
        entries.append({"resourceType": "Observation", "id": _id("twin", code, patient_id), "status": "preliminary",
                        "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                                  "code": "survey"}]}],
                        "code": {"coding": [{"system": CS, "code": f"twin-{code}", "display": name}]},
                        "subject": {"reference": pat_ref}, "effectiveDateTime": ts, "device": {"reference": dev_ref},
                        "valueQuantity": {"value": round(float(val), 3), "unit": unit}})
    return {"resourceType": "Bundle", "id": _id("bundle", patient_id, ts), "type": "collection",
            "timestamp": datetime.now(IST).isoformat(timespec="seconds"),
            "entry": [{"fullUrl": f"urn:uuid:{e['id']}", "resource": e} for e in entries]}


def dumps(b: dict) -> str:
    return json.dumps(b, indent=2, ensure_ascii=False)
