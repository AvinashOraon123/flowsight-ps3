from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flowsight import actions, forecast
from flowsight.simulate import GT_WAIT_THRESHOLD, simulate_day
from flowsight.zones import ZoneSet

ZONES = Path(__file__).parent.parent / "data" / "sim_zones.json"


@pytest.fixture(scope="module")
def zones():
    return ZoneSet.load(ZONES)


@pytest.fixture(scope="module")
def day(zones):
    return simulate_day(7, zones)


def test_simulation_is_deterministic(zones, day):
    again = simulate_day(7, zones)
    assert again["n_shoppers"] == day["n_shoppers"]
    pd.testing.assert_frame_equal(again["tracks"].head(500), day["tracks"].head(500))


def test_tracks_are_well_formed(zones, day):
    tr = day["tracks"]
    assert not tr.duplicated(["track_id", "t"]).any()
    assert tr["t"].is_monotonic_increasing or tr.groupby("track_id")["t"].apply(lambda s: s.is_monotonic_increasing).all()
    assert tr["x"].between(-1, zones.width_m + 1).all() and tr["y"].between(-1, zones.height_m + 1).all()
    assert tr["speed"].quantile(0.99) < 3.0                      # nobody teleports
    assert set(tr["gt_state"]) <= {"moving", "engaged", "blocked"}


def test_planted_dead_zone_is_rarely_visited(zones, day):
    visits = day["tracks"].dropna(subset=["zone"]).groupby("zone")["track_id"].nunique()
    assert visits["Dairy corner"] < 0.15 * visits[["Aisle 1", "Aisle 2", "Aisle 3", "Fresh produce"]].min()


def test_endcap_is_seen_but_ignored(day):
    e = day["endcap"]
    assert e["passers"] > 100 and 0.05 < e["stoppers"] / e["passers"] < 0.25


def test_jams_and_queue_exist_in_ground_truth(day):
    g = day["gt_wait"]
    assert (g["held"] >= g["waiting"]).all()                      # held up = waiting + being squeezed past
    w = g.groupby("zone")[["waiting", "held"]].max()
    assert w.loc["Checkout lane", "waiting"] >= GT_WAIT_THRESHOLD["queue"] or w.loc[["Aisle 1", "Aisle 2", "Aisle 3"], "held"].max() >= GT_WAIT_THRESHOLD["aisle"]
    jammed = {z for z, _, _ in day["windows"]["jams"]}
    assert jammed <= {"Aisle 1", "Aisle 2", "Aisle 3"}


def _ts(n=40, zones=("A", "B")):
    rows = []
    for z in zones:
        rows.append(pd.DataFrame(dict(t=np.arange(n) * 60.0, zone=z, count=np.arange(n, dtype=float), rho=0.0,
                                      v_mean=1.0, C=np.arange(n) / 100.0)))
    return pd.concat(rows, ignore_index=True)


def test_features_lags_and_targets_line_up():
    f = forecast.build_features(_ts(), np.arange(0, 2400, 30.0), ["A", "B"], [5, 10])
    a = f[f["zone"] == "A"].reset_index(drop=True)
    assert a.loc[20, "C_l5"] == pytest.approx(0.15)              # C at minute 15
    assert a.loc[20, "y5"] == pytest.approx(0.25)                # C at minute 25
    assert a.loc[20, "y10"] == pytest.approx(0.30)
    assert np.isnan(a.loc[39, "y5"])                             # no future past the end of the day
    assert a.loc[20, "inflow1"] == 2                             # two arrivals per minute


def test_features_do_not_leak_the_future():
    ts = _ts()
    base = forecast.build_features(ts, np.arange(0, 2400, 30.0), ["A"], [5])
    ts2 = ts.copy()
    ts2.loc[ts2["t"] > 20 * 60, "C"] = 9.9                       # change only what happens after minute 20
    changed = forecast.build_features(ts2, np.arange(0, 2400, 30.0), ["A"], [5])
    feats = [c for c in forecast.FEATURES]
    pd.testing.assert_frame_equal(base.loc[base["t"] <= 20 * 60, feats], changed.loc[changed["t"] <= 20 * 60, feats])


def test_best_threshold_prefers_the_separating_level():
    pred = np.r_[np.full(50, 0.05), np.full(10, 0.2)]
    flag = np.r_[np.zeros(50, bool), np.ones(10, bool)]
    assert 0.05 <= forecast._best_threshold(pred, flag, 0.3) < 0.2     # anything in [0.05, 0.2) separates them
    assert forecast._best_threshold(pred, np.zeros(60, bool), 0.3) == pytest.approx(0.21)


def test_forecast_alert_uses_per_row_level():
    fc = pd.DataFrame(dict(t=[0.0, 60.0, 120.0], zone="A", horizon_min=5, pred_C=[0.1, 0.2, 0.2], count=3.0, thr=0.15))
    a = actions.forecast_alerts(fc, thr=0.9)                     # per-row thr (0.15) wins over the default
    assert len(a) == 1 and a[0].t == 60.0 and a[0].audience == "Shift lead" and "in 5 min" in a[0].reason
