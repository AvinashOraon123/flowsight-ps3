"""Measure the two add-ons on SYNTHETIC data (no real multi-camera or staff footage exists in this project).

    python -m eval.extras_eval            -> outputs/extras.json

Stitching: a simulated store day is split between camera A (x < 16 m) and camera B (x >= 14 m, 2 m overlap). Each
camera numbers its own tracks, adds floor-position noise and a camera-specific appearance distortion. We score the
links against the true person ids, with and without appearance.
Staff: simulated shoppers plus synthetic staff tracks (broken into 5-minute fragments like a real tracker would),
with a noisy uniform-colour cue. This tests the rule logic, not a colour detector on real uniforms.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from flowsight.reid import stitch, track_summaries
from flowsight.simulate import simulate_day
from flowsight.staff import classify_staff
from flowsight.zones import ZoneSet

ROOT = Path(__file__).resolve().parent.parent
XA, XB = 16.0, 14.0
DIM = 16


def _two_cameras(tr: pd.DataFrame, rng, pos_noise=0.25, app_noise=0.10):
    base = {p: rng.dirichlet(np.full(DIM, 0.7)) for p in tr["track_id"].unique()}
    gain = {"A": rng.uniform(.8, 1.2, DIM), "B": rng.uniform(.8, 1.2, DIM)}         # lighting / white-balance difference
    out = []
    for cam, sel in (("A", tr["x"] < XA), ("B", tr["x"] >= XB)):
        d = tr[sel].sort_values(["track_id", "t"]).copy()
        brk = (d["track_id"] != d["track_id"].shift()) | (d["t"].diff() > 1.5)          # leaving the view starts a new local track
        d["local"] = brk.cumsum()
        d["x"] += rng.normal(0, pos_noise, len(d))
        d["y"] += rng.normal(0, pos_noise, len(d))
        e = np.stack([base[p] * gain[cam] for p in d["track_id"]]) + rng.normal(0, app_noise / DIM ** .5, (len(d), DIM))
        e = np.clip(e, 0, None)
        d["emb"] = list(e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9))
        d["cam"] = cam
        out.append(d)
    return out


def stitch_eval(seeds=(201, 202), hours=3.0) -> dict:
    zones = ZoneSet.load(ROOT / "data/sim_zones.json")
    res = {"with_appearance": [0, 0, 0], "space_only": [0, 0, 0]}
    for sd in seeds:
        tr = simulate_day(sd, zones)["tracks"]
        tr = tr[(tr["t"] > 3600) & (tr["t"] < 3600 * (1 + hours))]
        rng = np.random.default_rng(sd)
        cams = _two_cameras(tr, rng)
        S, truth = [], {}
        for d in cams:
            s = track_summaries(d.rename(columns={"local": "_l"}).assign(track_id=lambda x: x["_l"]), d["cam"].iloc[0])
            for _l, g in d.groupby("local"):
                truth[(d["cam"].iloc[0], _l)] = g["track_id"].iloc[0]
            S.append(s)
        S = pd.concat(S, ignore_index=True)
        for name, use in (("with_appearance", True), ("space_only", False)):
            gid = stitch(S, use_appearance=use)
            keys = list(gid)
            tid = np.array([hash(truth[k]) for k in keys])
            gg = np.array([gid[k] for k in keys])
            cam = np.array([k[0] for k in keys])
            i, j = np.triu_indices(len(keys), 1)
            cross = cam[i] != cam[j]                                   # pairs of tracks seen by different cameras
            same_p, same_g = (tid[i] == tid[j])[cross], (gg[i] == gg[j])[cross]
            tp, fp, pos = int((same_p & same_g).sum()), int((~same_p & same_g).sum()), int(same_p.sum())
            r = res[name]
            r[0] += tp; r[1] += fp; r[2] += pos
    out = {}
    for name, (tp, fp, pos) in res.items():
        p, r = tp / max(tp + fp, 1), tp / max(pos, 1)
        out[name] = dict(precision=round(p, 3), recall=round(r, 3), f1=round(2 * p * r / max(p + r, 1e-9), 3),
                         true_cross_camera_pairs=int(pos))
    return out


def staff_eval(seed=301, n_staff=3) -> dict:
    zones = ZoneSet.load(ROOT / "data/sim_zones.json")
    tr = simulate_day(seed, zones)["tracks"].copy()
    rng = np.random.default_rng(seed)
    tr["uniform_frac"] = np.clip(rng.normal(.06, .06, len(tr)), 0, 1)
    odd = rng.random(tr["track_id"].max() + 1) < 0.04                      # 4% of shoppers wear a similar colour
    tr.loc[odd[tr["track_id"].to_numpy()], "uniform_frac"] = rng.uniform(.4, .8, int(odd[tr["track_id"].to_numpy()].sum()))
    rows, fid = [], 10_000
    for s in range(n_staff):                                              # staff pace between two aisle points, 5-min fragments
        t = np.arange(0, 12 * 3600, 1.0)
        ph = rng.uniform(0, 6)
        x = 14 + 8 * np.sin(t / 90 + ph)
        y = 8 + 4 * np.sin(t / 130 + ph)
        for k in range(0, len(t), 300):
            sl = slice(k, k + 300)
            rows.append(pd.DataFrame(dict(t=t[sl], track_id=fid, x=x[sl], y=y[sl], gt_state="moving",
                                          uniform_frac=np.clip(rng.normal(.55, .15, len(t[sl])), 0, 1))))
            fid += 1
    st = pd.concat(rows, ignore_index=True)
    st["zone"] = zones.locate(st["x"], st["y"])
    allt = pd.concat([tr, st], ignore_index=True)
    truth = set(st["track_id"].unique())
    pred = classify_staff(allt, dict(uniform_min=0.3))
    tp, fp, fn = len(pred & truth), len(pred - truth), len(truth - pred)
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(precision=round(p, 3), recall=round(r, 3), staff_fragments=len(truth), shoppers=int(tr["track_id"].nunique()),
                shoppers_wrongly_removed=fp)


def main():
    res = dict(stitching=stitch_eval(), staff=staff_eval(), data="SYNTHETIC (simulated store, simulated cameras/uniforms)")
    out = ROOT / "outputs/extras.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
