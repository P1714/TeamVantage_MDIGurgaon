"""Synthetic Electronic Health Records for an Indian Type 2 Diabetes cohort.

Two layers are generated:

1. **Hidden physiology** (``true_*`` columns): insulin sensitivity, beta-cell
   secretory gain, basal glucose, lifestyle traits. These drive the virtual
   patient and are *never* shown to the model; they are kept only to prove the
   twin can recover them.
2. **The visible EHR**: demographics, diagnoses, medications, labs and genetic
   markers, written in the Synthea CSV schema (``patients.csv``,
   ``conditions.csv``, ``medications.csv``, ``observations.csv`` with LOINC
   codes) so real Synthea output can be dropped into the same loader.

Labs are *noisy measurements* of the hidden physiology (e.g. HOMA-IR relates to
insulin sensitivity but is not identical), just as in real clinics. HbA1c is
back-filled from the simulated glucose so that the record and the sensor
stream tell a consistent story.
"""
from __future__ import annotations

import uuid

import numpy as np
import pandas as pd

from . import config as C

FIRST_M = ["Aarav", "Vikram", "Rahul", "Suresh", "Arjun", "Karthik", "Ravi", "Imran", "Sanjay", "Anil",
           "Manoj", "Deepak", "Rohit", "Venkatesh", "Abhishek", "Farhan", "Joseph", "Harpreet", "Prakash", "Sourav"]
FIRST_F = ["Priya", "Lakshmi", "Anita", "Sunita", "Kavya", "Meera", "Fatima", "Deepa", "Pooja", "Asha",
           "Neha", "Radha", "Shalini", "Divya", "Rekha", "Nirmala", "Mary", "Gurpreet", "Ananya", "Swati"]
LAST = {"North": ["Sharma", "Verma", "Singh", "Gupta", "Kapoor", "Khan"],
        "South": ["Iyer", "Reddy", "Nair", "Rao", "Pillai", "Krishnan"],
        "East": ["Das", "Banerjee", "Mukherjee", "Ghosh", "Mohanty", "Sen"],
        "West": ["Patel", "Shah", "Desai", "Kulkarni", "Joshi", "Pawar"],
        "Northeast": ["Bora", "Baruah", "Sangma", "Lyngdoh", "Singha", "Gogoi"]}
CITIES = {"North": ["Delhi", "Lucknow", "Jaipur", "Chandigarh"],
          "South": ["Bengaluru", "Chennai", "Hyderabad", "Kochi"],
          "East": ["Kolkata", "Bhubaneswar", "Patna", "Ranchi"],
          "West": ["Mumbai", "Pune", "Ahmedabad", "Nagpur"],
          "Northeast": ["Guwahati", "Shillong", "Imphal", "Agartala"]}

# SNOMED-CT codes used in conditions.csv
SNOMED = {"t2d": (44054006, "Diabetes mellitus type 2"),
          "htn": (38341003, "Hypertensive disorder"),
          "dyslip": (55822004, "Hyperlipidemia"),
          "ckd": (431855005, "Chronic kidney disease stage 1-3"),
          "nafld": (197321007, "Steatosis of liver"),
          "cad": (53741008, "Coronary arteriosclerosis"),
          "neuropathy": (230572002, "Diabetic peripheral neuropathy"),
          "famhx": (160303001, "Family history: Diabetes mellitus"),
          "obesity": (414916001, "Obesity")}

MEDS = {"metformin": (860975, "Metformin hydrochloride 500 MG Extended Release Oral Tablet"),
        "sulfonylurea": (310534, "Glimepiride 2 MG Oral Tablet"),
        "dpp4": (665033, "Sitagliptin 100 MG Oral Tablet"),
        "sglt2": (1545658, "Dapagliflozin 10 MG Oral Tablet"),
        "basal_insulin": (285018, "Insulin Glargine 100 UNT/ML Injectable Solution"),
        "statin": (617310, "Atorvastatin 20 MG Oral Tablet"),
        "arb": (979492, "Telmisartan 40 MG Oral Tablet")}

# LOINC codes used in observations.csv
LOINC = {"hba1c": ("4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood", "%"),
         "fpg": ("1558-6", "Fasting glucose [Mass/volume] in Serum or Plasma", "mg/dL"),
         "fasting_insulin": ("20448-7", "Insulin [Units/volume] in Serum or Plasma", "uU/mL"),
         "bmi": ("39156-5", "Body mass index (BMI) [Ratio]", "kg/m2"),
         "weight": ("29463-7", "Body weight", "kg"),
         "height": ("8302-2", "Body height", "cm"),
         "waist": ("8280-0", "Waist Circumference at umbilicus by Tape measure", "cm"),
         "sbp": ("8480-6", "Systolic blood pressure", "mm[Hg]"),
         "dbp": ("8462-4", "Diastolic blood pressure", "mm[Hg]"),
         "tg": ("2571-8", "Triglyceride [Mass/volume] in Serum or Plasma", "mg/dL"),
         "hdl": ("2085-9", "Cholesterol in HDL [Mass/volume] in Serum or Plasma", "mg/dL"),
         "ldl": ("18262-6", "Cholesterol in LDL [Mass/volume] in Serum or Plasma by Direct assay", "mg/dL"),
         "creatinine": ("2160-0", "Creatinine [Mass/volume] in Serum or Plasma", "mg/dL"),
         "egfr": ("33914-3", "Glomerular filtration rate/1.73 sq M.predicted", "mL/min/{1.73_m2}"),
         "uacr": ("9318-7", "Albumin/Creatinine [Mass Ratio] in Urine", "mg/g"),
         "alt": ("1742-6", "Alanine aminotransferase [Enzymatic activity/volume] in Serum or Plasma", "U/L"),
         # genetic markers (LOINC 'Discrete genetic variant' panels, genotype as allele count)
         "tcf7l2": ("69548-6", "Genetic variant assessment: TCF7L2 rs7903146 risk-allele (T) count", "{count}"),
         "kcnj11": ("69548-6", "Genetic variant assessment: KCNJ11 rs5219 risk-allele (K) count", "{count}"),
         "fto": ("69548-6", "Genetic variant assessment: FTO rs9939609 risk-allele (A) count", "{count}"),
         "prs_t2d": ("X-PRS-T2D", "Polygenic risk score for T2D (South-Asian calibrated, z)", "{z}")}


def _clip(x, lo, hi):
    return np.clip(x, lo, hi)


def sample_cohort(n: int = C.N_PATIENTS, seed: int = C.SEED) -> pd.DataFrame:
    """Sample hidden physiology, demographics, genotype and lifestyle traits."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"patient_id": [f"P{i + 1:04d}" for i in range(n)]})
    df["sex"] = np.where(rng.random(n) < 0.56, "M", "F")
    male = (df["sex"] == "M").to_numpy()
    df["age"] = _clip(rng.normal(52, 10, n), 30, 78).round().astype(int)
    df["region"] = rng.choice(C.REGIONS, n, p=C.REGION_WEIGHTS)
    df["urban"] = rng.random(n) < 0.68
    df["city"] = [rng.choice(CITIES[r]) for r in df["region"]]
    df["name"] = [f"{rng.choice(FIRST_M if m else FIRST_F)} {rng.choice(LAST[r])}"
                  for m, r in zip(male, df["region"])]

    # genetics (allele frequencies approximate South-Asian cohorts)
    df["tcf7l2_rs7903146"] = rng.binomial(2, 0.30, n)
    df["kcnj11_rs5219"] = rng.binomial(2, 0.35, n)
    df["fto_rs9939609"] = rng.binomial(2, 0.32, n)
    g = (0.45 * (df["tcf7l2_rs7903146"] - 0.6) + 0.25 * (df["kcnj11_rs5219"] - 0.7)
         + 0.2 * (df["fto_rs9939609"] - 0.64))
    df["prs_t2d"] = (g + rng.normal(0, 0.85, n)).round(2)
    df["family_history_t2d"] = rng.random(n) < _clip(0.45 + 0.12 * df["prs_t2d"], 0.1, 0.9)

    # anthropometrics (Asian BMI cut-offs: >=23 overweight, >=25 obese)
    height = np.where(male, rng.normal(168, 7, n), rng.normal(155, 6, n))
    bmi = _clip(rng.normal(26.8, 3.8, n) + 0.5 * df["fto_rs9939609"], 19, 42)
    df["height_cm"] = height.round(1)
    df["bmi"] = bmi.round(1)
    df["weight_kg"] = (bmi * (height / 100) ** 2).round(1)
    df["waist_cm"] = (48 + 1.65 * bmi + np.where(male, 5, 0) + rng.normal(0, 4.5, n)).round(1)

    df["diabetes_duration_y"] = _clip(rng.gamma(2.0, 3.6, n), 0.3, np.maximum(df["age"] - 25, 1)).round(1)
    dur = df["diabetes_duration_y"].to_numpy()

    # ---- hidden physiology ----------------------------------------------------
    log_si = (np.log(2.3) - 0.050 * (bmi - 25) - 0.010 * (df["waist_cm"] - 90)
              - 0.06 * df["fto_rs9939609"] + rng.normal(0, 0.28, n))
    df["true_SI"] = _clip(np.exp(log_si), 0.6, 6.0) * 1e-4           # per min per uU/mL
    log_sig = (np.log(0.040) - 0.045 * dur - 0.17 * df["tcf7l2_rs7903146"]
               - 0.08 * df["kcnj11_rs5219"] + rng.normal(0, 0.25, n))
    df["true_sigma"] = _clip(np.exp(log_sig), 0.006, 0.08)
    df["true_SG"] = _clip(rng.normal(0.016, 0.003, n), 0.008, 0.026)
    gb = (118 + 55 * (1 - df["true_sigma"] / 0.045) + 10 * (2.3e-4 / df["true_SI"] - 1)
          + rng.normal(0, 9, n))
    df["true_Gb_untreated"] = _clip(gb, 100, 230)
    df["true_dawn_amp"] = _clip(rng.normal(14, 7, n), 0, 35)

    # ---- comorbidities --------------------------------------------------------
    df["htn"] = rng.random(n) < _clip(0.25 + 0.006 * (df["age"] - 40) + 0.02 * (bmi - 25), 0.1, 0.85)
    df["dyslipidemia"] = rng.random(n) < 0.58
    df["neuropathy"] = rng.random(n) < _clip(0.04 + 0.018 * dur, 0.02, 0.6)
    df["nafld"] = rng.random(n) < _clip(0.2 + 0.03 * (bmi - 25), 0.05, 0.8)
    df["cad"] = rng.random(n) < _clip(0.02 + 0.004 * (df["age"] - 40) + 0.01 * dur, 0.01, 0.4)
    egfr = rng.normal(96 - 0.7 * (df["age"] - 45) - 0.9 * dur, 13, n)
    df["egfr"] = _clip(egfr, 25, 125).round(0)
    df["ckd"] = df["egfr"] < 60

    # ---- medications ----------------------------------------------------------
    df["metformin"] = (rng.random(n) < 0.84) & (df["egfr"] >= 30)
    df["sulfonylurea"] = rng.random(n) < _clip(0.15 + 0.025 * dur, 0.1, 0.6)
    df["dpp4"] = rng.random(n) < 0.26
    df["sglt2"] = rng.random(n) < _clip(0.12 + 0.25 * (df["cad"] | df["ckd"]), 0, 0.5)
    df["basal_insulin"] = rng.random(n) < _clip(-0.05 + 0.02 * dur + 0.004 * (gb - 150), 0.02, 0.5)
    df["statin"] = df["dyslipidemia"] | df["cad"] | (rng.random(n) < 0.2)
    df["arb"] = df["htn"] & (rng.random(n) < 0.8)
    df["true_med_adherence"] = _clip(rng.beta(8, 2, n), 0.4, 1.0)

    # ---- lifestyle traits (hidden; only their *traces* reach the wearables) ----
    df["true_diet_discipline"] = rng.beta(2.0, 2.4, n)
    df["true_log_compliance"] = _clip(rng.beta(4, 2, n), 0.3, 0.97)
    df["true_walk_habit"] = rng.beta(1.6, 2.4, n)
    df["true_postmeal_walk"] = rng.beta(1.3, 4.0, n)
    df["true_office_worker"] = rng.random(n) < np.where(df["urban"], 0.6, 0.3)
    df["true_sleep_mean_h"] = _clip(rng.normal(6.6, 0.7, n), 4.8, 8.4)
    df["true_sleep_onset_min"] = _clip(rng.normal(23 * 60 + 20, 45, n), 21 * 60 + 30, 25 * 60 + 30)
    df["true_stress_prone"] = rng.beta(2, 4, n)
    df["true_sweet_tooth"] = rng.beta(1.5, 5, n)
    df["true_skip_breakfast"] = rng.beta(1.2, 8, n)
    df["true_resting_hr"] = _clip(rng.normal(73, 7, n) + 4 * df["neuropathy"] + 0.25 * (bmi - 25), 55, 98)
    df["true_hrv_base"] = _clip(np.exp(rng.normal(np.log(30), 0.3, n)) - 0.3 * (df["age"] - 50)
                                - 7 * df["neuropathy"] - 0.25 * dur, 8, 70)
    for slot, (mu, sd) in C.MEAL_SLOTS.items():
        df[f"true_{slot}_time"] = rng.normal(mu, sd, n)
    df["true_illness_day"] = np.where(rng.random(n) < 0.22, rng.integers(3, C.N_DAYS - 2, n), -1)
    df["true_cgm_noise"] = _clip(rng.normal(6, 1.5, n), 3, 11)
    return df


def effective_params(cohort: pd.DataFrame) -> dict:
    """Map hidden physiology + medication list to simulator parameters."""
    sigma = cohort["true_sigma"].to_numpy() * np.where(cohort["sulfonylurea"], 1.35, 1.0) \
        * np.where(cohort["dpp4"], 1.15, 1.0)
    gb = cohort["true_Gb_untreated"].to_numpy().copy()
    gb_med_drop = (np.where(cohort["metformin"], 0.14 * gb, 0) + np.where(cohort["dpp4"], 6, 0)
                   + np.where(cohort["basal_insulin"], 26, 0) + np.where(cohort["sulfonylurea"], 8, 0))
    return {
        "SG": cohort["true_SG"].to_numpy(),
        "SI": cohort["true_SI"].to_numpy(),
        "sigma": sigma,
        "gb_untreated": gb,
        "gb_med_drop": gb_med_drop,
        "V": 1.7 * cohort["weight_kg"].to_numpy(),
        "sglt2": cohort["sglt2"].to_numpy().astype(float),
        "dawn": cohort["true_dawn_amp"].to_numpy(),
    }


# -----------------------------------------------------------------------------
# EHR tables (Synthea CSV schema subset)
# -----------------------------------------------------------------------------
def _uuid(rng) -> str:
    return str(uuid.UUID(int=int(rng.integers(0, 2**63)) << 64 | int(rng.integers(0, 2**63))))


def build_ehr_tables(cohort: pd.DataFrame, mean_glucose: np.ndarray, seed: int = C.SEED):
    """Create Synthea-style tables. ``mean_glucose`` (from the simulated sensor
    period) back-fills HbA1c via the ADAG relation HbA1c = (MG + 46.7) / 28.7."""
    rng = np.random.default_rng(seed + 7)
    start = pd.Timestamp(C.START_DATE)
    pats, conds, meds, obs = [], [], [], []
    for i, r in cohort.reset_index(drop=True).iterrows():
        pid = r.patient_id
        birth = start - pd.Timedelta(days=int(r.age * 365.25 + rng.integers(0, 365)))
        pats.append({"Id": pid, "BIRTHDATE": birth.date(), "DEATHDATE": None, "FIRST": r["name"].split()[0],
                     "LAST": r["name"].split()[1], "GENDER": r.sex, "RACE": "asian", "ETHNICITY": "south_asian",
                     "CITY": r.city, "STATE": r.region, "COUNTY": r.region, "COUNTRY": "IN",
                     "URBAN": bool(r.urban), "HEALTHCARE_EXPENSES": round(float(rng.gamma(3, 9000)), 0)})
        onset = start - pd.Timedelta(days=int(r.diabetes_duration_y * 365.25))
        conds.append({"START": onset.date(), "STOP": None, "PATIENT": pid,
                      "CODE": SNOMED["t2d"][0], "DESCRIPTION": SNOMED["t2d"][1]})
        for flag, key in [("htn", "htn"), ("dyslipidemia", "dyslip"), ("ckd", "ckd"), ("nafld", "nafld"),
                          ("cad", "cad"), ("neuropathy", "neuropathy"), ("family_history_t2d", "famhx")]:
            if r[flag]:
                d = onset + pd.Timedelta(days=int(rng.integers(-900, 900)))
                conds.append({"START": min(d, start).date(), "STOP": None, "PATIENT": pid,
                              "CODE": SNOMED[key][0], "DESCRIPTION": SNOMED[key][1]})
        if r.bmi >= 30:
            conds.append({"START": (start - pd.Timedelta(days=700)).date(), "STOP": None, "PATIENT": pid,
                          "CODE": SNOMED["obesity"][0], "DESCRIPTION": SNOMED["obesity"][1]})
        for m in MEDS:
            if r[m]:
                d = onset + pd.Timedelta(days=int(rng.integers(30, 1500)))
                meds.append({"START": min(d, start - pd.Timedelta(days=60)).date(), "STOP": None, "PATIENT": pid,
                             "CODE": MEDS[m][0], "DESCRIPTION": MEDS[m][1], "MED_CLASS": m})

        # labs: measured values = hidden physiology + measurement noise
        mg = mean_glucose[i]
        a1c_now = (mg + 46.7) / 28.7 + rng.normal(0, 0.25)
        gb_eff = r.true_Gb_untreated - (0.14 * r.true_Gb_untreated if r.metformin else 0) - (26 if r.basal_insulin else 0)
        fpg = gb_eff + r.true_dawn_amp * 0.6 + rng.normal(0, 10)
        fins = np.clip(7 * (2.3e-4 / r.true_SI) ** 0.8 * (r.true_sigma / 0.03) ** 0.5 * np.exp(rng.normal(0, 0.25)), 2, 45)
        vals = {
            "fpg": fpg, "fasting_insulin": fins, "bmi": r.bmi, "weight": r.weight_kg, "height": r.height_cm,
            "waist": r.waist_cm, "sbp": rng.normal(128 + 12 * r.htn, 11), "dbp": rng.normal(80 + 6 * r.htn, 8),
            "tg": np.exp(rng.normal(np.log(150) + 0.25 * (2.3e-4 / r.true_SI - 1), 0.3)),
            "hdl": rng.normal(42 if r.sex == "M" else 48, 8), "ldl": rng.normal(105 - 20 * r.statin, 25),
            "egfr": r.egfr, "creatinine": np.clip(90 / max(r.egfr, 20) + rng.normal(0, 0.08), 0.5, 4),
            "uacr": np.exp(rng.normal(np.log(18) + 0.9 * r.ckd + 0.03 * r.diabetes_duration_y, 0.6)),
            "alt": np.exp(rng.normal(np.log(26) + 0.35 * r.nafld, 0.3)),
            "tcf7l2": r.tcf7l2_rs7903146, "kcnj11": r.kcnj11_rs5219, "fto": r.fto_rs9939609, "prs_t2d": r.prs_t2d,
        }
        for key, v in vals.items():
            code, desc, unit = LOINC[key]
            d = start - pd.Timedelta(days=int(rng.integers(5, 40)))
            if key in ("tcf7l2", "kcnj11", "fto", "prs_t2d"):
                d = start - pd.Timedelta(days=int(rng.integers(200, 900)))
            obs.append({"DATE": d.date(), "PATIENT": pid, "CODE": code, "DESCRIPTION": desc,
                        "VALUE": round(float(v), 2), "UNITS": unit, "TYPE": "numeric", "KEY": key})
        # HbA1c history: last value from the sensor period, earlier values trend back
        code, desc, unit = LOINC["hba1c"]
        for k, months in enumerate([0, 3, 6, 12]):
            drift = rng.normal(0.12 * k, 0.3)
            d = start - pd.Timedelta(days=int(months * 30.4 + rng.integers(2, 10)))
            obs.append({"DATE": d.date(), "PATIENT": pid, "CODE": code, "DESCRIPTION": desc,
                        "VALUE": round(float(np.clip(a1c_now + drift, 5.6, 14)), 1), "UNITS": unit,
                        "TYPE": "numeric", "KEY": "hba1c"})
    return (pd.DataFrame(pats), pd.DataFrame(conds), pd.DataFrame(meds), pd.DataFrame(obs))


def load_ehr_features(ehr_dir) -> pd.DataFrame:
    """Flatten Synthea-schema CSVs into one row of static features per patient.

    Works on our synthetic export and on any Synthea export that contains the
    same LOINC / SNOMED codes (missing items become NaN, which LightGBM handles).
    """
    from pathlib import Path
    ehr_dir = Path(ehr_dir)
    pats = pd.read_csv(ehr_dir / "patients.csv")
    conds = pd.read_csv(ehr_dir / "conditions.csv")
    meds = pd.read_csv(ehr_dir / "medications.csv")
    obs = pd.read_csv(ehr_dir / "observations.csv")
    ref = pd.Timestamp(C.START_DATE)

    f = pd.DataFrame({"patient_id": pats["Id"]})
    f["age"] = ((ref - pd.to_datetime(pats["BIRTHDATE"])).dt.days / 365.25).round(1)
    f["male"] = (pats["GENDER"] == "M").astype(int)
    for reg in C.REGIONS:
        f[f"region_{reg.lower()}"] = (pats["STATE"] == reg).astype(int)
    f["urban"] = pats["URBAN"].astype(int)

    obs = obs.copy()
    obs["DATE"] = pd.to_datetime(obs["DATE"])
    obs["KEY"] = obs["KEY"] if "KEY" in obs else None
    latest = (obs.sort_values("DATE").groupby(["PATIENT", "KEY"])["VALUE"].last().unstack())
    f = f.merge(latest, left_on="patient_id", right_index=True, how="left")
    a1c = obs[obs["KEY"] == "hba1c"].sort_values("DATE")
    trend = a1c.groupby("PATIENT")["VALUE"].agg(lambda v: v.iloc[-1] - v.iloc[0]).rename("hba1c_trend_12m")
    f = f.merge(trend, left_on="patient_id", right_index=True, how="left")
    f["homa_ir"] = f["fpg"] * f["fasting_insulin"] / 405.0
    f["homa_b"] = 360 * f["fasting_insulin"] / np.maximum(f["fpg"] - 63, 5)
    f["tg_hdl"] = f["tg"] / f["hdl"]

    t2d = conds[conds["CODE"] == SNOMED["t2d"][0]].groupby("PATIENT")["START"].min()
    f["diabetes_duration_y"] = f["patient_id"].map(
        lambda p: (ref - pd.Timestamp(t2d[p])).days / 365.25 if p in t2d else np.nan).round(1)
    for key, (code, _) in SNOMED.items():
        if key == "t2d":
            continue
        has = set(conds.loc[conds["CODE"] == code, "PATIENT"])
        f[f"dx_{key}"] = f["patient_id"].isin(has).astype(int)
    for m in MEDS:
        has = set(meds.loc[meds["MED_CLASS"] == m, "PATIENT"])
        f[f"rx_{m}"] = f["patient_id"].isin(has).astype(int)
    f = f.rename(columns={"tcf7l2": "gen_tcf7l2", "kcnj11": "gen_kcnj11", "fto": "gen_fto", "prs_t2d": "gen_prs"})
    return f
