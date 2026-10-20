"""Evaluate the analytics against SIMULATED ground truth (agent states the detector never sees).

Ground truth for a jam is the simulator's own count of people truly waiting in a line: an aisle gate with >= 3
people waiting, or the checkout lane with >= 5. The detector only sees positions and speeds.
All numbers here are SIMULATED and must be labelled so wherever they are shown.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from flowsight import bottleneck, heatmap, metrics
from flowsight.simulate import AISLES, GT_WAIT_THRESHOLD, simulate_day

EVAL_SEEDS = [200, 201, 202, 203, 204]          # never used for training or tuning
GT_ZONES = ["Checkout lane", *AISLES]


def _minutes(intervals: pd.DataFrame, zone: str, n: int) -> np.ndarray:
    m = np.zeros(n, bool)
    for r in intervals[intervals["zone"] == zone].itertuples():
        m[int(r.t_start // 60):int(np.ceil(r.t_end / 60))] = True
    return m


def _gt_minutes(gt_wait: pd.DataFrame, zone: str, n: int, strict: bool = False) -> np.ndarray:
    """Ground-truth jam minutes. strict=True counts only people waiting (the first, narrower definition)."""
    thr = GT_WAIT_THRESHOLD["queue" if zone == "Checkout lane" else "aisle"]
    col = "waiting" if strict or zone == "Checkout lane" else "held"
    w = gt_wait[gt_wait["zone"] == zone].sort_values("t")[col].to_numpy()
    out = np.zeros(n, bool)
    out[:min(len(w), n)] = w[:n] >= thr
    return out


def _prf(tp, fp, fn) -> dict:
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else 0.0)


def _episodes(m: np.ndarray, min_len: int = 2) -> list[tuple[int, int]]:
    out, i = [], 0
    while i < len(m):
        if m[i]:
            j = i
            while j + 1 < len(m) and m[j + 1]:
                j += 1
            if j - i + 1 >= min_len:
                out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def bottleneck_eval(zones, p: dict, seeds=EVAL_SEEDS, open_h=8, close_h=20) -> dict:
    n = (close_h - open_h) * 60
    tp = fp = fn = 0                       # over all zones: every flag in a zone with no jam concept is a false positive
    gtp = gfp = gfn = 0                    # only the 4 zones that can jam in the simulator
    stp = sfp = sfn = 0                    # first, stricter definition: waiting only (all zones)
    ep_hit = ep_n = 0
    delays = []
    per_zone = {z: dict(tp=0, fp=0, fn=0) for z in zones.names}
    for seed in seeds:
        d = simulate_day(seed, zones, open_h, close_h, 1.0)
        tr = metrics.add_motion_state(d["tracks"], 1.0, p)
        ts = bottleneck.zone_timeseries(tr, zones, 1.0, 60.0, p["v_free_ms"], t_end=(close_h - open_h) * 3600)
        bn = bottleneck.flag_bottlenecks(ts, p["bottleneck_threshold"], p["bottleneck_sustain_s"], 60.0)
        for z in zones.names:
            pred = _minutes(bn, z, n)
            gt = _gt_minutes(d["gt_wait"], z, n) if z in GT_ZONES else np.zeros(n, bool)
            a, b, c = int((pred & gt).sum()), int((pred & ~gt).sum()), int((~pred & gt).sum())
            tp, fp, fn = tp + a, fp + b, fn + c
            if z in GT_ZONES:
                gs = _gt_minutes(d["gt_wait"], z, n, strict=True)
                stp, sfp, sfn = stp + int((pred & gs).sum()), sfp + int((pred & ~gs).sum()), sfn + int((~pred & gs).sum())
            else:
                sfp += b
            per_zone[z]["tp"] += a
            per_zone[z]["fp"] += b
            per_zone[z]["fn"] += c
            if z in GT_ZONES:
                gtp, gfp, gfn = gtp + a, gfp + b, gfn + c
                for i, j in _episodes(gt):
                    ep_n += 1
                    hit = bn[(bn["zone"] == z) & (bn["t_start"] < (j + 1) * 60) & (bn["t_end"] > i * 60)]
                    if len(hit):
                        ep_hit += 1
                        delays.append(float(hit["alert_t"].min() - i * 60) / 60.0)
    return dict(all_zones=_prf(tp, fp, fn), jam_zones_only=_prf(gtp, gfp, gfn), strict_waiting_only=_prf(stp, sfp, sfn),
                episodes=dict(n=ep_n, detected=ep_hit, recall=ep_hit / max(ep_n, 1),
                              median_alert_delay_min=float(np.median(delays)) if delays else float("nan")),
                per_zone={z: {**v, **_prf(v["tp"], v["fp"], v["fn"])} for z, v in per_zone.items()},
                n_days=len(seeds), minutes=n * len(seeds))


def dead_zone_eval(zones, p: dict, seeds=EVAL_SEEDS[:3], planted=("Dairy corner",)) -> dict:
    """Does the dead-zone rule find the planted quiet corner, and nothing else?"""
    rows = []
    for seed in seeds:
        d = simulate_day(seed, zones, 8, 20, 1.0)
        tr = d["tracks"].assign(state="moving", stop_id=-1)
        ev = metrics.zone_events(tr, 1.0, p)
        tab = metrics.build_zone_table(tr.assign(state="moving"), ev, zones, p)
        heat = heatmap.heat_from_tracks(tr, zones.width_m, zones.height_m, 1.0, p["heat_cell_m"])
        dz = bottleneck.dead_zones(tab, zones, heat, p)
        flagged = set(dz.loc[dz["dead_zone"], "zone"])
        rows.append(dict(seed=seed, flagged=sorted(flagged), found=all(z in flagged for z in planted),
                         false_positives=sorted(flagged - set(planted))))
    return dict(days=rows, recall=float(np.mean([r["found"] for r in rows])),
                false_positive_zones=int(sum(len(r["false_positives"]) for r in rows)))


def state_eval(labels: pd.DataFrame, zones) -> dict:
    """Engaged vs blocked stops against what the simulator intended (aisle zones; the queue lane is excluded)."""
    q = {z.name for z in zones.of_kind("queue")}
    s = labels[(labels["gt_state"] != "moving") & ~labels["zone"].isin(q) & labels["zone"].notna()]
    detected = s[s["state"] != "moving"]
    gtb, prb = detected["gt_state"] == "blocked", detected["state"] == "blocked"
    tp, fp, fn = int((gtb & prb).sum()), int((~gtb & prb).sum()), int((gtb & ~prb).sum())
    return dict(n_gt_stop_samples=int(len(s)), stops_detected=float(len(detected) / max(len(s), 1)),
                accuracy=float((gtb == prb).mean()), blocked=_prf(tp, fp, fn), n_gt_blocked=int((s["gt_state"] == "blocked").sum()))
