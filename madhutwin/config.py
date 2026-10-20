"""Global configuration, clinical thresholds and the Indian food library.

All numbers here are *simulation assumptions* chosen to be physiologically
plausible for an Indian adult Type 2 Diabetes population. Carbohydrate and
glycaemic-index (GI) values are rounded approximations of typical home-style
servings drawn from published GI tables and Indian nutrition literature; they
are not a substitute for a dietitian-grade food database.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
DASHBOARD_DIR = ROOT / "dashboard"

SEED = 2026

# --- cohort / simulation -----------------------------------------------------
N_PATIENTS = 300
N_DAYS = 21                 # simulated days per patient
CALIB_DAYS = 7              # first week = twin calibration / personal history
MIN_PER_DAY = 1440
CGM_INTERVAL = 5            # minutes (Libre / Dexcom style)
START_DATE = "2026-06-01"   # arbitrary synthetic calendar start

# --- clinical thresholds (International Consensus on Time-in-Range, 2019) -----
HYPER_L1 = 180              # mg/dL  level-1 hyperglycaemia
HYPER_L2 = 250              # mg/dL  level-2 hyperglycaemia
HYPO_L1 = 70
EVENT_MIN_DURATION = 15     # minutes a reading must stay >= threshold
HORIZON_MIN = 120           # prediction horizon: 2 hours
DECISION_EVERY_MIN = 5      # the twin re-evaluates risk every CGM reading

# --- regions of India used for diet patterns ---------------------------------
REGIONS = ["North", "South", "East", "West", "Northeast"]
REGION_WEIGHTS = [0.30, 0.27, 0.17, 0.18, 0.08]


@dataclass(frozen=True)
class Food:
    key: str
    name: str
    slot: str          # breakfast | lunch | snack | dinner
    carbs: float       # grams available carbohydrate per typical serving
    gi: float          # approximate glycaemic index (glucose = 100)
    regions: tuple     # regions where this is a common choice ("*" = all)
    healthy: bool      # low-GI / diet-disciplined option


FOODS: list[Food] = [
    # breakfast
    Food("idli", "Idli (3) + sambar", "breakfast", 45, 70, ("South", "*"), False),
    Food("masala_dosa", "Masala dosa", "breakfast", 60, 75, ("South", "West"), False),
    Food("poha", "Poha", "breakfast", 45, 66, ("West", "North", "East"), False),
    Food("upma", "Rava upma", "breakfast", 45, 68, ("South", "West"), False),
    Food("aloo_paratha", "Aloo paratha (2) + curd", "breakfast", 70, 70, ("North",), False),
    Food("bread_omelette", "Bread + omelette", "breakfast", 32, 72, ("*",), False),
    Food("luchi", "Luchi + aloo", "breakfast", 60, 72, ("East",), False),
    Food("oats", "Vegetable oats", "breakfast", 32, 55, ("*",), True),
    Food("ragi_dosa", "Ragi dosa", "breakfast", 38, 55, ("South",), True),
    Food("moong_chilla", "Moong dal chilla", "breakfast", 25, 40, ("North", "West"), True),
    Food("pongal", "Ven pongal", "breakfast", 55, 75, ("South",), False),
    # lunch
    Food("rice_sambar", "Rice + sambar + curd", "lunch", 95, 75, ("South",), False),
    Food("rice_dal", "Rice + dal + sabzi", "lunch", 85, 72, ("East", "West", "Northeast", "*"), False),
    Food("roti_dal", "Roti (3) + dal + sabzi", "lunch", 68, 60, ("North", "West", "*"), False),
    Food("veg_biryani", "Veg biryani", "lunch", 90, 72, ("*",), False),
    Food("chicken_biryani", "Chicken biryani", "lunch", 80, 68, ("South", "North", "East"), False),
    Food("rajma_chawal", "Rajma chawal", "lunch", 88, 65, ("North",), False),
    Food("fish_rice", "Fish curry + rice", "lunch", 85, 74, ("East", "Northeast", "South"), False),
    Food("millet_meal", "Foxtail millet + dal + sabzi", "lunch", 62, 55, ("South", "*"), True),
    Food("dal_salad", "Dal + salad + 1 roti", "lunch", 32, 42, ("*",), True),
    Food("thali", "Festive thali", "lunch", 120, 72, ("*",), False),
    # snack
    Food("chai_biscuit", "Sweet chai + 2 biscuits", "snack", 22, 75, ("*",), False),
    Food("samosa_chai", "Samosa + chai", "snack", 42, 70, ("North", "West", "East"), False),
    Food("banana", "Banana", "snack", 25, 55, ("*",), True),
    Food("sweet", "Mithai (gulab jamun / jalebi)", "snack", 45, 80, ("*",), False),
    Food("roasted_chana", "Roasted chana", "snack", 15, 30, ("*",), True),
    Food("fruit", "Fruit plate (guava/apple)", "snack", 20, 42, ("*",), True),
    Food("namkeen", "Namkeen / mixture", "snack", 25, 62, ("*",), False),
    Food("bajji", "Bajji / pakora + tea", "snack", 35, 68, ("South", "East"), False),
    # dinner
    Food("roti_sabzi", "Roti (3) + sabzi", "dinner", 55, 60, ("North", "West", "*"), False),
    Food("rice_rasam", "Rice + rasam + poriyal", "dinner", 80, 75, ("South",), False),
    Food("khichdi", "Khichdi + curd", "dinner", 60, 65, ("*",), False),
    Food("dosa_dinner", "Plain dosa (2) + chutney", "dinner", 55, 75, ("South",), False),
    Food("paneer_roti", "Paneer + 2 roti", "dinner", 42, 52, ("North", "West"), True),
    Food("fried_rice", "Fried rice / noodles", "dinner", 80, 75, ("*",), False),
    Food("rice_fish_dinner", "Rice + fish / egg curry", "dinner", 78, 73, ("East", "Northeast"), False),
    Food("millet_roti", "Jowar/bajra roti + sabzi", "dinner", 45, 52, ("West", "*"), True),
]
FOOD_BY_KEY = {f.key: f for f in FOODS}

# meal slots: (mean start minute of day, sd between patients)
MEAL_SLOTS = {
    "breakfast": (8 * 60 + 30, 35),
    "lunch": (13 * 60 + 30, 35),
    "snack": (17 * 60 + 15, 30),
    "dinner": (21 * 60 + 0, 40),   # late dinners are common in urban India
}
