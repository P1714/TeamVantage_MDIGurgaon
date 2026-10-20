import numpy as np

from madhutwin import physiology as PH
from madhutwin import whatif as W


def _prm(n=1, gb=130.0):
    return {"SG": np.full(n, 0.016), "SI": np.full(n, 2e-4), "sigma": np.full(n, 0.03), "Gb": np.full(n, gb),
            "h": np.full(n, gb), "dawn": np.zeros(n), "V": np.full(n, 120.0), "sglt2": np.zeros(n)}


def test_fasting_steady_state_is_stable():
    tr = W.run(_prm(), meal_carbs=0, meal_gi=60)[0]
    assert np.all(np.abs(tr - 130) < 1.0)


def test_meal_raises_then_returns():
    tr = W.run(_prm(), meal_carbs=80, meal_gi=72)[0]
    assert tr.max() > 190 and abs(tr[-1] - 130) < 40


def test_low_gi_and_smaller_portion_lower_the_peak():
    base = W.run(_prm(), 80, 72)[0].max()
    assert W.run(_prm(), 80, 45)[0].max() < base
    assert W.run(_prm(), 56, 72)[0].max() < base


def test_walk_after_meal_lowers_peak():
    base = W.run(_prm(), 80, 72)[0].max()
    walked = W.run(_prm(), 80, 72, walk=(15, 20, 105))[0].max()
    assert walked < base - 3


def test_more_insulin_sensitive_patient_spikes_less():
    p = _prm(2)
    p["SI"] = np.array([1e-4, 4e-4])
    peaks = W.run(p, 80, 72).max(1)
    assert peaks[1] < peaks[0]


def test_fast_fraction_bounds():
    assert PH.fast_fraction(20) == 0 and PH.fast_fraction(110) == 1
    assert 0 < PH.fast_fraction(60) < 1
