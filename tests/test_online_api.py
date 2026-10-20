from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flowsight.online import OnlineEngine
from flowsight.pipeline import load_config
from flowsight.zones import ZoneSet

ROOT = Path(__file__).parent.parent
CFG = load_config(ROOT / "configs/demo.yaml")
ZONES = ZoneSet.load(ROOT / "data/zones.json")
FPS = 15.0
CLIP = ROOT / "outputs" / "clip"


def engine():
    # identity homography: "image pixels" are floor metres in these synthetic scenes
    return OnlineEngine(ZONES, np.eye(3), FPS, CFG["analytics"], CFG["actions"])


def run(people, seconds):
    """people: {id: (x, y) | callable(t) -> (x, y)}. Returns every event fired."""
    eng, events = engine(), []
    for k in range(int(seconds * FPS)):
        t = k / FPS
        ids = list(people)
        uv = [people[i](t) if callable(people[i]) else people[i] for i in ids]
        events += eng.update(t, ids, uv)["events"]
    return events, eng


def test_lone_browser_is_never_a_bottleneck():
    ev, eng = run({1: (2.0, 9.0)}, 30)                    # one person standing in Aisle 1 for 30 s
    assert ev == [] and eng.tr[1].state == "engaged"


def test_stuck_crowd_raises_a_sustained_bottleneck_once():
    rng = np.random.default_rng(0)
    # 7 people standing 0.5 m apart in Aisle 1 (capacity 10): crowd + path ahead occupied
    people = {i: (1.0 + 0.5 * (i % 4), 8.0 + 0.5 * (i // 4)) for i in range(7)}
    ev, eng = run({i: (lambda t, p=p: (p[0] + rng.normal(0, .02), p[1] + rng.normal(0, .02))) for i, p in people.items()}, 40)
    b = [e for e in ev if e.kind == "bottleneck"]
    assert len(b) == 1 and b[0].zone == "Aisle 1" and "Congestion" in b[0].reason
    assert b[0].t >= CFG["analytics"]["bottleneck_sustain_s"]       # never before the sustain window
    assert eng.state["Aisle 1"]["flag"] and eng.state["Aisle 3"]["flag"] is False
    assert sum(t.state == "blocked" for t in eng.tr.values()) >= 5


def test_standing_by_choice_is_not_a_bottleneck_even_when_crowded():
    # same 7 people but spread 3 m apart (free space, nothing ahead): browsers, not a jam
    people = {i: (0.5 + 1.2 * (i % 3), 6.5 + 2.5 * (i // 3)) for i in range(7)}
    ev, eng = run(people, 40)
    assert [e for e in ev if e.kind == "bottleneck"] == []
    assert sum(t.state == "engaged" for t in eng.tr.values()) == 7


def test_walking_crowd_is_not_a_bottleneck():
    def tri(i):                                            # 1.2 m/s up and down Aisle 1 (y 6..15)
        return lambda t: (0.5 + (0.8 * i) % 4, 6.0 + 9 * abs(((t * 1.2 / 9) % 2) - 1))
    ev, eng = run({i: tri(i) for i in range(7)}, 20)       # longer than the 8 s sustain window
    assert [e for e in ev if e.kind == "bottleneck"] == []
    assert all(t.state == "moving" for t in eng.tr.values())


def test_long_queue_raises_a_queue_alert_with_reason():
    lane = {i: (14.0 + 0.5 * (i % 6), 15.0 + 0.8 * (i // 6)) for i in range(7)}          # 7 people in Checkout lane
    ev, _ = run(lane, 20)
    q = [e for e in ev if e.kind == "queue"]
    assert q and q[0].audience == "Floor staff" and "Queue of 7" in q[0].reason
    assert len(q) == 1                                                                    # rate-limited, not one per second


def test_engine_tracks_zone_counts_like_the_batch_analytics():
    pytest.importorskip("pyarrow")
    if not (CLIP / "tracks_img.parquet").exists():
        pytest.skip("run the pipeline first")
    from flowsight.geometry import load_homography
    ti = pd.read_parquet(CLIP / "tracks_img.parquet")
    ts = pd.read_parquet(CLIP / "zone_timeseries.parquet")
    eng = OnlineEngine(ZONES, load_homography(ROOT / "data/homography.npy"), 15.0, CFG["analytics"], CFG["actions"])
    online = []
    for _, g in ti.groupby("frame"):
        r = eng.update(float(g["t"].iloc[0]), g["track_id"].to_numpy(), g[["u", "v"]].to_numpy())
        online.append((int(r["t"]), {z: s["count"] for z, s in r["zones"].items()}))
    last = {}
    for sec, st in online:
        last[sec] = st                                      # state after the previous second closed
    diffs = []
    for sec, st in last.items():
        for z, c in st.items():
            ref = ts[(ts.zone == z) & (ts.t == sec - 1)]["count"]
            if len(ref):
                diffs.append(abs(c - float(ref.iloc[0])))
    assert np.mean(diffs) < 0.35                            # online (min_track filter off) vs batch counts


# ---------------------------------------------------------------- API
@pytest.fixture(scope="module")
def client():
    if not (CLIP / "zone_table.parquet").exists():
        pytest.skip("run the pipeline first")
    from fastapi.testclient import TestClient

    from flowsight.api import create_app
    return TestClient(create_app())


def test_api_health_metrics_alerts(client):
    assert client.get("/health").json()["ready"] is True
    m = client.get("/metrics").json()
    assert {z["zone"] for z in m["zones"]} == set(ZONES.names) and m["summary"]["n_tracks"] > 0
    a = client.get("/alerts").json()
    assert [x["impact"] for x in a] == sorted((x["impact"] for x in a), reverse=True)
    assert client.get("/alerts", params={"until": -1}).json() == []
    assert all(x["kind"] == "endcap" for x in client.get("/alerts", params={"kind": "endcap"}).json())


def test_api_heatmap(client):
    r = client.get("/heatmap")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    j = client.get("/heatmap.json").json()
    assert j["shape"] == [88, 80] and np.array(j["person_seconds"]).sum() > 0


def test_api_never_serves_video_or_images_of_people(client):
    assert client.get("/video").status_code == 404 and client.get("/frame").status_code == 404


def test_api_websocket_streams_anonymous_coordinates(client):
    with client.websocket_connect("/live?speed=0&max_frames=30") as ws:
        msgs = [ws.receive_json() for _ in range(31)]
    frames = [m for m in msgs if "t" in m]
    assert len(frames) == 30 and msgs[-1] == {"done": True, "frames": 30}
    assert set(frames[0]) == {"t", "zones", "people", "events"}
    assert all(set(p) == {"id", "x", "y", "state", "zone"} for f in frames for p in f["people"])   # no pixels, no images


def test_small_groups_are_suppressed_in_the_api(client):
    from flowsight.privacy import MIN_GROUP
    rows = client.get("/metrics").json()["zones"]
    small = [r for r in rows if r["visitors"] < MIN_GROUP]
    assert small, "the demo clip has a zone with < 5 visitors"
    assert all(r["mean_dwell_s"] is None and r["engaged_s"] is None and r["suppressed"] for r in small)
    assert all(r["mean_dwell_s"] is not None for r in rows if r["visitors"] >= MIN_GROUP)
