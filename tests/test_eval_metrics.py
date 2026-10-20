import numpy as np
import pandas as pd
import pytest

from eval import mot_eval as me
from eval import sim_eval as se


def boxes(rows):
    return pd.DataFrame(rows, columns=["fi", "id", "x", "y", "w", "h"])


def walk(id_, n, x0=0, y0=0, dx=2.0):
    return [(f, id_, x0 + dx * f, y0, 20, 40) for f in range(n)]


NO_IGN = pd.DataFrame(columns=["fi", "id", "x", "y", "w", "h"])


def test_iou():
    a, b = np.array([[0, 0, 10, 10]]), np.array([[5, 0, 10, 10], [0, 0, 10, 10], [50, 50, 5, 5]])
    assert np.allclose(me.iou_xywh(a, b), [[50 / 150, 1.0, 0.0]])


def test_perfect_tracking_scores_one():
    gt = boxes(walk(1, 20) + walk(2, 20, y0=100))
    r = me.mot_metrics(gt.copy(), gt, NO_IGN)
    assert r["idf1"] == pytest.approx(1.0) and r["mota"] == pytest.approx(1.0) and r["id_switches"] == 0


def test_id_switch_hurts_idf1_and_counts_a_switch():
    gt = boxes(walk(1, 20))
    hyp = boxes([(f, 7 if f < 10 else 8, x, y, w, h) for f, _, x, y, w, h in walk(1, 20)])
    r = me.mot_metrics(hyp, gt, NO_IGN)
    assert r["id_switches"] == 1
    assert r["idf1"] == pytest.approx(0.5, abs=0.02)       # only the longer half can be matched
    assert r["mota"] == pytest.approx(1 - 1 / 20)


def test_missed_and_false_tracks():
    gt = boxes(walk(1, 10))
    hyp = boxes(walk(5, 10, y0=500))                       # tracks the wrong place
    r = me.mot_metrics(hyp, gt, NO_IGN)
    assert r["idf1"] == 0.0 and r["recall"] == 0.0 and r["mota"] == pytest.approx(1 - 20 / 10)


def test_tracks_on_ignored_objects_are_not_penalised():
    gt = boxes(walk(1, 10))
    ign = boxes(walk(9, 10, y0=300))
    hyp = boxes(walk(1, 10) + walk(2, 10, y0=300))         # second track sits on the ignored object
    r = me.mot_metrics(hyp, gt, ign)
    assert r["idf1"] == pytest.approx(1.0)


def test_detection_ap_perfect_and_with_false_positives():
    gt = pd.DataFrame(dict(fi=[0, 0, 1], x=[0, 100, 0], y=[0, 0, 0], w=20, h=40))
    good = pd.DataFrame(dict(fi=[0, 0, 1], x=[0, 100, 0], y=[0, 0, 0], w=20, h=40, conf=[.9, .8, .7]))
    assert me.detection_ap(good, gt, NO_IGN)["ap50"] == pytest.approx(1.0)
    noisy = pd.concat([good, pd.DataFrame(dict(fi=[0], x=[300], y=[300], w=20, h=40, conf=[.95]))], ignore_index=True)
    ap = me.detection_ap(noisy, gt, NO_IGN)["ap50"]
    assert 0.5 < ap < 1.0                                  # a confident false positive costs precision


def test_detection_on_ignored_object_is_not_a_false_positive():
    gt = pd.DataFrame(dict(fi=[0], x=[0], y=[0], w=20, h=40))
    ign = pd.DataFrame(dict(fi=[0], x=[200], y=[0], w=20, h=40))
    dets = pd.DataFrame(dict(fi=[0, 0], x=[200, 0], y=[0, 0], w=20, h=40, conf=[.99, .5]))
    assert me.detection_ap(dets, gt, ign)["ap50"] == pytest.approx(1.0)


def test_load_gt_scales_and_filters(tmp_path):
    f = tmp_path / "gt.txt"
    f.write_text("1,1,100,200,40,80,1,1,0.9\n1,2,0,0,10,10,1,7,1.0\n1,3,0,0,10,10,1,1,0.1\n2,1,100,200,40,80,1,1,0.9\n3,1,100,200,40,80,1,1,0.9\n")
    g, ign = me.load_gt(f, stride=2)
    assert list(g["fi"]) == [0, 2] and g.iloc[0][["x", "y", "w", "h"]].tolist() == [50, 100, 20, 40]
    assert sorted(ign["id"]) == [2, 3]                      # static person + barely visible pedestrian are ignored


def test_episodes_and_prf():
    m = np.array([0, 1, 1, 0, 1, 0, 1, 1, 1, 0], bool)
    assert se._episodes(m, 2) == [(1, 2), (6, 8)]
    assert se._prf(8, 2, 2) == pytest.approx(dict(precision=.8, recall=.8, f1=.8))
    assert se._prf(0, 0, 0)["f1"] == 0.0
