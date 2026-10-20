import json

import numpy as np
import pandas as pd
import pytest

from flowsight import actions, bottleneck, heatmap, journeys, metrics
from flowsight.zones import ZoneSet, segments

FPS = 15.0
P = dict(min_visit_s=3.0, merge_gap_s=1.0, stop_speed_ms=0.2, min_stop_s=2.0, crowd_radius_m=1.5,
         crowd_count=3, ahead_dist_m=1.5, passby_radius_m=2.0, v_free_ms=1.2, heat_sigma_cells=2.0,
         dead_zone_decile=0.1, dead_zone_rel_median=0.3, main_path_pct=80, main_path_decay_m=4.0)


@pytest.fixture
def zones(tmp_path):
    f = tmp_path / "z.json"
    f.write_text(json.dumps({"floor_plan": {"width_m": 30, "height_m": 20, "px_per_m": 10}, "zones": [
        {"name": "A", "kind": "aisle", "priority": 2, "capacity": 10, "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]},
        {"name": "Hot", "kind": "endcap", "priority": 1, "rho_max": 2.0, "polygon": [[4, 4], [6, 4], [6, 6], [4, 6]]},
        {"name": "B", "kind": "aisle", "priority": 3, "capacity": 10, "polygon": [[20, 0], [30, 0], [30, 10], [20, 10]]}]}))
    return ZoneSet.load(f)


def make_track(tid, legs, zones, fps=FPS):
    """legs: [(x0, y0, x1, y1, seconds)] constant-velocity legs -> tracks DataFrame."""
    rows, t = [], 0.0
    for x0, y0, x1, y1, sec in legs:
        n = max(int(round(sec * fps)), 1)
        for k in range(n):
            a = k / n
            rows.append((t, x0 + (x1 - x0) * a, y0 + (y1 - y0) * a, (x1 - x0) / sec, (y1 - y0) / sec))
            t += 1 / fps
    d = pd.DataFrame(rows, columns=["t", "x", "y", "vx", "vy"])
    d["frame"], d["track_id"] = (d["t"] * fps).round().astype(int), tid
    d["speed"] = np.hypot(d.vx, d.vy)
    d["zone"] = zones.locate(d.x, d.y)
    return d


def test_zone_priority_and_effective_area(zones):
    assert zones.locate(5, 5)[0] == "Hot"          # higher priority wins inside the overlap
    assert zones.locate(1, 1)[0] == "A"
    assert zones.locate(15, 5)[0] is None
    assert zones.by_name["A"].area == pytest.approx(100 - 4)   # Hot carved out of A
    assert zones.by_name["A"].rho_max == pytest.approx(10 / 96)


def test_segments_merge_short_gap():
    t = np.arange(10) * 1.0
    runs = segments(["A", "A", None, "A", "A", None, None, None, "B", "B"], t, gap_s=1.0)
    assert [(z, i0, i1) for z, i0, i1 in runs] == [("A", 0, 4), (None, 5, 7), ("B", 8, 9)]


def test_dwell_and_engaged_on_known_track(zones):
    # walk in from x=-3 at 1.2 m/s, stand 10 s at x=2, walk out the far side to x=14
    legs = [(-3, 2, 2, 2, 5 / 1.2), (2, 2, 2, 2, 10), (2, 2, 14, 2, 12 / 1.2)]
    tr = make_track(1, legs, zones)
    tr = metrics.add_motion_state(tr, FPS, P)
    ev = metrics.zone_events(tr, FPS, P)
    v = ev[ev.zone == "A"].iloc[0]
    inside = (tr.zone == "A").sum() / FPS
    assert v.is_visit and v.dwell_s == pytest.approx(inside, abs=1 / FPS)
    assert v.dwell_s == pytest.approx(2 / 1.2 + 10 + 8 / 1.2, abs=0.2)     # walk x=0..2, stand 10 s, walk x=2..10
    assert v.engaged_s == pytest.approx(10.0, abs=0.5)
    assert v.blocked_s == 0.0


def test_short_pass_is_not_a_visit(zones):
    tr = metrics.add_motion_state(make_track(1, [(-1, 2, 5, 2, 5)], zones), FPS, P)     # ~4 s... shrink it
    tr = metrics.add_motion_state(make_track(2, [(-1, 2, 5, 2, 2)], zones), FPS, P)     # 1.7 s in zone, fast
    ev = metrics.zone_events(tr, FPS, P)
    assert not ev[ev.zone == "A"].iloc[0].is_visit


def test_stop_shorter_than_min_stop_is_not_a_stop(zones):
    tr = make_track(1, [(0, 2, 3, 2, 3), (3, 2, 3, 2, 1.0), (3, 2, 8, 2, 4)], zones)
    tr = metrics.add_motion_state(tr, FPS, P)
    assert (tr.state == "moving").all()


def _crowd(zones, extra):
    lone = make_track(1, [(-3, 2, 3, 2, 5), (3, 2, 3, 2, 8)], zones)
    return metrics.add_motion_state(pd.concat([lone] + extra, ignore_index=True), FPS, P)


def test_stop_alone_is_engaged(zones):
    tr = _crowd(zones, [])
    assert (tr[(tr.track_id == 1) & (tr.t > 7)].state == "engaged").all()


def test_stop_in_a_crowd_is_blocked(zones):
    others = [make_track(10 + i, [(3 + dx, 2 + dy, 3 + dx, 2 + dy, 14)], zones) for i, (dx, dy) in
              enumerate([(0.8, 0.5), (-0.6, 0.7), (0.4, -0.9), (1.0, -0.2)])]
    for o in others:
        o["t"] = o["t"] - 0.0
    tr = _crowd(zones, others)
    sel = tr[(tr.track_id == 1) & (tr.t > 8)]
    assert (sel.state == "blocked").all()


def test_stop_with_path_ahead_occupied_is_blocked(zones):
    ahead_person = make_track(10, [(4.0, 2, 4.0, 2, 14)], zones)     # 1 m straight ahead (walking +x)
    tr = _crowd(zones, [ahead_person])
    assert (tr[(tr.track_id == 1) & (tr.t > 8)].state == "blocked").all()


def test_person_behind_does_not_block(zones):
    behind = make_track(10, [(2.0, 2, 2.0, 2, 14)], zones)           # 1 m behind
    tr = _crowd(zones, [behind])
    assert (tr[(tr.track_id == 1) & (tr.t > 8)].state == "engaged").all()


def test_zone_table_pass_by_to_stop(zones):
    walkers = [make_track(i, [(-3, 8, 12, 8, 12)], zones) for i in range(1, 5)]   # pass, never stop
    stopper = make_track(9, [(-3, 5, 3, 5, 5), (3, 5, 3, 5, 6), (3, 5, 12, 5, 8)], zones)
    tr = metrics.add_motion_state(pd.concat(walkers + [stopper], ignore_index=True), FPS, P)
    ev = metrics.zone_events(tr, FPS, P)
    tab = metrics.build_zone_table(tr, ev, zones, P).set_index("zone")
    assert tab.loc["A", "passers"] == 5 and tab.loc["A", "engaged_stoppers"] == 1
    assert tab.loc["A", "stop_ratio"] == pytest.approx(0.2)


def test_congestion_formula():
    assert bottleneck.congestion(2.0, 0.0, 2.0) == pytest.approx(1.0)
    assert bottleneck.congestion(2.0, 1.2, 2.0) == pytest.approx(0.0)
    assert bottleneck.congestion(1.0, 0.6, 2.0) == pytest.approx(0.25)
    assert bottleneck.congestion(9.0, 5.0, 2.0) == 0.0                 # clipped


def _ts(c):
    return pd.DataFrame(dict(t=np.arange(len(c)) * 1.0, zone="A", count=3.0, rho=1.0, v_mean=0.2, C=c))


def test_bottleneck_needs_sustained_congestion():
    c = np.r_[np.zeros(10), np.full(59, 0.6), np.zeros(10)]
    assert len(bottleneck.flag_bottlenecks(_ts(c), 0.3, 60)) == 0
    c = np.r_[np.zeros(10), np.full(60, 0.6), np.zeros(10)]
    bn = bottleneck.flag_bottlenecks(_ts(c), 0.3, 60)
    assert len(bn) == 1 and bn.iloc[0].t_start == 10 and bn.iloc[0].alert_t == 70


def test_zone_timeseries_density_and_speed(zones):
    rows = []
    for tid in range(4):                                  # four people standing in A for 20 s
        d = make_track(tid, [(2 + tid, 2, 2 + tid, 2, 20)], zones)
        rows.append(d)
    ts = bottleneck.zone_timeseries(pd.concat(rows), zones, FPS, 1.0, 1.2, t_end=20)
    a = ts[(ts.zone == "A") & (ts.t.between(5, 15))]
    assert a["count"].mean() == pytest.approx(4, abs=0.3)
    assert a["v_mean"].mean() < 0.05
    assert a["C"].mean() == pytest.approx((4 / 96) / (10 / 96) * 1.0, abs=0.05)    # rho/rho_max = 0.4


def test_heatmap_conserves_person_seconds():
    h = heatmap.HeatGrid(10, 10, 0.25)
    h.add([2, 2, 5], [3, 3, 5], 0.5)
    assert h.total_person_s() == pytest.approx(1.5)
    assert h.smoothed(2.0).sum() == pytest.approx(1.5, rel=0.01)
    h.add([-5], [3], 1.0)                                  # off the plan is ignored
    assert h.total_person_s() == pytest.approx(1.5)


def test_dead_zone_flags_cold_zone_next_to_traffic(zones):
    hg = heatmap.HeatGrid(30, 20, 0.25)
    hg.add(np.full(2000, 12.0), np.full(2000, 5.0), 1)     # main path just outside A, between A and B
    tab = pd.DataFrame(dict(zone=["A", "B", "Hot"], kind=["aisle", "aisle", "endcap"],
                            visits=[1, 40, 30], visits_per_m2=[1 / 96, 40 / 100, 30 / 4]))
    dz = bottleneck.dead_zones(tab, zones, hg, P)
    assert dz.set_index("zone").loc["A", "dead_zone"] and not dz.set_index("zone").loc["B", "dead_zone"]


def test_dead_zone_needs_data(zones):
    tab = pd.DataFrame(dict(zone=["A", "B", "Hot"], kind=["aisle"] * 3, visits=[1, 2, 1], visits_per_m2=[.1, .2, .1]))
    assert not bottleneck.dead_zones(tab, zones, heatmap.HeatGrid(30, 20), P)["dead_zone"].any()


def test_journeys_zone_sequence(zones):
    a = make_track(1, [(2, 2, 2, 2, 4), (2, 2, 25, 2, 6), (25, 2, 25, 2, 4)], zones)
    tr = metrics.add_motion_state(a, FPS, P)
    ev = metrics.zone_events(tr, FPS, P)
    jt = journeys.journey_table(tr, ev)
    assert jt.iloc[0].zones == ["A", "B"]
    sk = journeys.sankey_links(jt)
    assert sk.iloc[0].tolist() == ["A", "B", 1]


# ---------------------------------------------------------------- actions
def test_queue_alert_has_reason_and_audience():
    q = pd.DataFrame(dict(t=[10.0, 11.0], queue_len=[3, 7], service_per_min=[1.1, 1.1], est_wait_min=[2.7, 6.4]))
    a = actions.queue_alerts(q, "Checkout", dict(queue_open_till=5, wait_open_till_min=5.0))
    assert len(a) == 1 and a[0].audience == "Floor staff" and "Queue of 7" in a[0].reason and "6.4" in a[0].reason


def test_rate_limit_and_ranking():
    mk = lambda t, z, imp: actions.Alert(t, z, "queue", "x", "y", imp, "Floor staff", "seconds")
    al = actions.rate_limit([mk(0, "A", 1), mk(5, "A", 9), mk(130, "A", 2), mk(1, "B", 3)], 120)
    assert [(a.zone, a.t) for a in al] == [("A", 0), ("B", 1), ("A", 130)]
    assert [a.impact for a in actions.rank(al)] == [3, 2, 1]


def test_endcap_alert_only_when_seen_but_ignored():
    tab = pd.DataFrame(dict(zone=["E1", "E2"], kind=["endcap", "endcap"], passers=[30, 30], engaged_stoppers=[2, 20],
                            stop_ratio=[2 / 30, 20 / 30]))
    a = actions.endcap_alerts(tab, 100.0, dict(endcap_min_passers=8, endcap_stop_ratio=0.15))
    assert [x.zone for x in a] == ["E1"] and "seen but ignored" in a[0].action


def test_compare_layouts_delta():
    b = pd.DataFrame(dict(zone=["A"], visits=[10], mean_dwell_s=[5.0], blocked_share=[0.5], stop_ratio=[0.1]))
    a = b.assign(visits=[16], blocked_share=[0.1])
    d = actions.compare_layouts(b, a).iloc[0]
    assert d.visits_delta == 6 and d.blocked_share_delta == pytest.approx(-0.4)
