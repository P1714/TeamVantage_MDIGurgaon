"""Leakage guards: features at time k must not change when the future changes."""
import numpy as np
import pandas as pd

from madhutwin import features as FE


def _arrays(P=2, days=3, seed=0):
    rng = np.random.default_rng(seed)
    n = days * 288
    cgm = 140 + 40 * np.sin(np.arange(n) / 20)[None, :] + rng.normal(0, 5, (P, n))
    A = {"cgm": cgm, "hr": 70 + rng.normal(0, 3, (P, n)), "hrv": 30 + rng.normal(0, 3, (P, n)),
         "steps": rng.integers(0, 200, (P, n)).astype(float), "stage": np.zeros((P, n)),
         "tod": (np.arange(n) * 5) % 1440, "day": (np.arange(n) * 5) // 1440, "n5": n, "P": P,
         "pids": [f"P{i}" for i in range(P)], "carbs": np.zeros((P, n)), "gi": np.zeros((P, n)),
         "ex_boost": np.ones((P, n))}
    A["stage"][:, :60] = 1
    return A


def test_labels_detect_future_excursion():
    cgm = np.full((1, 100), 120.0)
    cgm[0, 50:55] = 200
    y, y250, tte = FE.labels(cgm)
    assert y[0, 30] and not y[0, 10] and not y250[0, 30]
    assert tte[0, 30] == (50 - 30) * 5


def test_excursion_needs_15_minutes():
    cgm = np.full((1, 60), 120.0)
    cgm[0, 30:32] = 200            # only 10 minutes
    assert not FE.labels(cgm)[0].any()


def test_rolling_cgm_features_ignore_the_future():
    A = _arrays()
    k = 400
    before = {n: f(A["cgm"])[:, k].copy() for n, f in {
        "mean1h": lambda c: FE._roll(c, 12, "mean"), "max3h": lambda c: FE._roll(c, 36, "max"),
        "since": lambda c: FE._since(c >= 180)}.items()}
    A["cgm"][:, k + 1:] = 399.0
    after = {"mean1h": FE._roll(A["cgm"], 12, "mean")[:, k], "max3h": FE._roll(A["cgm"], 36, "max")[:, k],
             "since": FE._since(A["cgm"] >= 180)[:, k]}
    for n in before:
        np.testing.assert_allclose(before[n], after[n], equal_nan=True)


def test_night_features_use_only_completed_nights():
    A = _arrays()
    f1, _ = FE.night_features(A)
    k = 288 + 60                   # 05:00 on day 1: tonight not finished yet
    A["stage"][:, k + 1: 288 + 144] = 2
    f2, _ = FE.night_features(A)
    for key in ("sleep_last_h", "deep_frac_last", "sleep_tonight_h"):
        np.testing.assert_allclose(f1[key][:, k], f2[key][:, k], equal_nan=True)


def test_healthkit_roundtrip():
    from madhutwin import adapters as AD
    ts = pd.date_range("2026-06-01", periods=288, freq="5min")
    w = pd.DataFrame({"patient_id": "PX", "timestamp": ts, "t": np.arange(288) * 5,
                      "cgm_mgdl": np.linspace(100, 200, 288), "hr_bpm": 70.0, "hrv_rmssd_ms": 30.0,
                      "steps": np.arange(288) % 7, "sleep_stage": (np.arange(288) < 80) * 2})
    back = AD.from_healthkit_long(AD.to_healthkit_long(w), "2026-06-01", 1)
    for c in ("cgm_mgdl", "steps", "sleep_stage"):
        np.testing.assert_allclose(back[c].to_numpy(float), w[c].to_numpy(float))
