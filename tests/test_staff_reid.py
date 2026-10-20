import numpy as np
import pandas as pd

from flowsight.reid import stitch, track_summaries
from flowsight.staff import classify_staff, exclude_staff


def _tracks():
    rows = []
    for tid, u, zone in [(1, .05, "Aisle 1"), (2, .7, "Aisle 1"), (3, .05, "Back office")]:
        rows.append(pd.DataFrame(dict(track_id=tid, t=np.arange(0, 20.0), uniform_frac=u, zone=zone)))
    return pd.concat(rows, ignore_index=True)


def test_no_config_excludes_nobody():
    tr = _tracks()
    out, ids = exclude_staff(tr, None)
    assert ids == set() and len(out) == len(tr)


def test_uniform_and_zone_cues():
    assert classify_staff(_tracks(), dict(uniform_min=0.3)) == {2}
    assert classify_staff(_tracks(), dict(staff_zones=["Back office"])) == {3}
    assert classify_staff(_tracks(), dict(uniform_min=0.3, staff_zones=["Back office"])) == {2, 3}


def test_presence_cue():
    tr = _tracks()
    tr.loc[tr.track_id == 1, "t"] = np.arange(0, 20) * 400.0
    assert 1 in classify_staff(tr, dict(presence_s=3600))


def _cam(tid, t, x0, x1, emb):
    n = len(t)
    return pd.DataFrame(dict(track_id=tid, t=t, x=np.linspace(x0, x1, n), y=5.0, emb=[np.array(emb, float)] * n))


def test_stitch_links_handover_and_uses_appearance():
    red, blue = [1, 0, 0, 0], [0, 0, 0, 1]
    a = pd.concat([_cam(1, np.arange(0, 10.), 5, 15, red), _cam(2, np.arange(0, 10.), 5.5, 15.5, blue)])
    b = pd.concat([_cam(1, np.arange(10, 20.), 15, 25, blue), _cam(2, np.arange(10, 20.), 15.5, 25.5, red)])
    S = pd.concat([track_summaries(a, "A"), track_summaries(b, "B")], ignore_index=True)
    g = stitch(S)
    assert g[("A", 1)] == g[("B", 2)] and g[("A", 2)] == g[("B", 1)]          # same colour, not same lane
    assert g[("A", 1)] != g[("A", 2)]


def test_stitch_rejects_far_or_late():
    a = _cam(1, np.arange(0, 10.), 5, 10, [1, 0])
    b = _cam(1, np.arange(30, 40.), 10, 20, [1, 0])
    S = pd.concat([track_summaries(a, "A"), track_summaries(b, "B")], ignore_index=True)
    g = stitch(S)
    assert g[("A", 1)] != g[("B", 1)]
