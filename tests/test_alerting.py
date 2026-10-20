"""The alert policy: detection, lead time and false alerts all come from delivered alerts."""
import numpy as np

from madhutwin import alerting as AL


def flat(n=400, v=120.0):
    return np.full(n, v)


def spike(cgm, at, length=6, v=220.0):
    cgm[at:at + length] = v
    return cgm


def test_repeated_crossings_are_debounced():
    cgm = flat()
    risk = np.full(400, 0.9)                         # above threshold at every step
    a = AL.delivered_alerts(risk, cgm, 0.4)
    assert np.all(np.diff(a) >= AL.REFRACTORY_STEPS)
    assert a[0] == AL.MIN_READINGS_LAST_HOUR - 1        # first alert needs 9 readings of history
    assert len(a) == len(range(a[0], 400, AL.REFRACTORY_STEPS))


def test_raw_crossing_suppressed_by_debounce_does_not_count_as_detection():
    # alert delivered at 100 (no spike follows within 2 h); risk crosses again at 105 (debounced),
    # spike at 140: the raw crossing at 105 is within 2 h of the spike but was never delivered.
    cgm = spike(flat(), 140)
    risk = np.full(400, np.nan)
    risk[100] = 0.9
    risk[105] = 0.9
    a = AL.delivered_alerts(risk, cgm, 0.4)
    assert list(a) == [100]
    m = AL.match_events(a, AL.onsets(cgm))
    assert np.isnan(m["leads_min"]).all()            # 100 is 40 steps before 140: outside the 2 h window
    assert m["true_alert"].tolist() == [False]


def test_one_excursion_lead_time():
    cgm = spike(flat(), 200)
    risk = np.full(400, np.nan)
    risk[188] = 0.8                                  # 12 steps = 60 min before onset
    a = AL.delivered_alerts(risk, cgm, 0.4)
    m = AL.match_events(a, AL.onsets(cgm))
    assert m["leads_min"].tolist() == [60.0] and m["true_alert"].tolist() == [True]


def test_multiple_excursions_one_to_one():
    cgm = spike(spike(flat(), 200), 215)             # two onsets 15 steps apart
    ev = AL.onsets(cgm)
    assert list(ev) == [200, 215]
    a = np.array([190])                              # one alert precedes both
    m = AL.match_events(a, ev)
    assert np.isfinite(m["leads_min"]).sum() == 1    # credited to one excursion only
    assert m["true_alert"].tolist() == [True]


def test_alert_inside_and_outside_window():
    cgm = spike(flat(), 300)
    a = np.array([250, 290])                         # 250 is 50 steps (>2 h) before; 290 is inside
    m = AL.match_events(a, AL.onsets(cgm))
    assert m["leads_min"].tolist() == [50.0]
    assert m["true_alert"].tolist() == [False, True]


def test_alert_after_onset_does_not_detect():
    cgm = spike(flat(), 200)
    m = AL.match_events(np.array([200, 201]), AL.onsets(cgm))
    assert np.isnan(m["leads_min"]).all()


def test_insufficient_recent_data_suppresses_alert():
    cgm = flat()
    cgm[90:99] = np.nan                              # only 3 of the last 12 readings at k=100
    risk = np.full(400, np.nan)
    risk[100] = 0.9
    assert AL.readings_last_hour(cgm, 100) == 3
    assert len(AL.delivered_alerts(risk, cgm, 0.4)) == 0
    cgm2 = flat()
    cgm2[95:97] = np.nan                             # 10 of 12 present: alert allowed
    assert list(AL.delivered_alerts(risk, cgm2, 0.4)) == [100]


def test_event_level_uses_delivered_alerts():
    import pandas as pd
    from madhutwin import config as C
    from madhutwin import model as M
    n = C.CALIB_DAYS * 288 + 600
    cgm = np.full(n, 120.0)
    on = C.CALIB_DAYS * 288 + 300
    cgm[on:on + 6] = 220
    ks = np.arange(C.CALIB_DAYS * 288, on)
    df = pd.DataFrame({"patient_id": "P1", "k": ks, "risk": np.where(ks >= on - 20, 0.9, 0.1)})
    r = M.event_level(df, [cgm], {"P1": 0}, 0.4)
    assert r["events"] == 1 and r["detected_lead_ge_30min"] == 1.0 and r["median_lead_min"] == 100.0
    assert r["false_alerts_per_patient_day"] == 0.0 and r["alert_precision"] == 1.0
