"""Zone metrics: visits, dwell, engaged vs blocked dwell, pass-by-to-stop ratio, queue length and wait.

Input is a `tracks` DataFrame with one row per (track, analysed frame):
    frame, t (s), track_id, x, y (floor metres, smoothed), vx, vy, speed (m/s), zone (name | None)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .zones import ZoneSet, segments

MOVING, ENGAGED, BLOCKED = "moving", "engaged", "blocked"


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[(i0, i1)] inclusive index ranges of consecutive True values."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def add_motion_state(tracks: pd.DataFrame, fps: float, p: dict) -> pd.DataFrame:
    """Label every sample moving / engaged / blocked.

    A *stop* is speed < stop_speed_ms for >= min_stop_s. A stop is
      BLOCKED  if the person is stuck in a crowd: >= crowd_count neighbours within crowd_radius_m,
               or somebody stands within ahead_dist_m inside a +-35 deg cone of the heading
               the person had before stopping (path ahead occupied);
      ENGAGED  otherwise (slow, free space around, nothing in the way = interest in the zone).
    The label is decided once per stop episode (majority of its samples), so it cannot flicker.
    """
    df = tracks.sort_values(["track_id", "t"]).reset_index(drop=True)
    dt = 1.0 / fps
    moving_fast = df["speed"] > 0.4
    sp = df["speed"].where(moving_fast)
    df["hx"] = (df["vx"].where(moving_fast) / sp).groupby(df["track_id"]).ffill()
    df["hy"] = (df["vy"].where(moving_fast) / sp).groupby(df["track_id"]).ffill()

    xy = df[["x", "y"]].to_numpy()
    n_near = np.zeros(len(df), dtype=int)
    ahead = np.zeros(len(df), dtype=bool)
    cos35 = np.cos(np.radians(35))
    hx, hy = df["hx"].to_numpy(), df["hy"].to_numpy()
    slow = df["speed"].to_numpy() < p["stop_speed_ms"]     # only stop candidates need neighbour checks
    for idx in df.groupby("frame").indices.values():
        cand = np.flatnonzero(slow[idx])
        if len(idx) < 2 or len(cand) == 0:
            continue
        P = xy[idx]
        D = cdist(P[cand], P)
        D[np.arange(len(cand)), cand] = np.inf            # not your own neighbour
        n_near[idx[cand]] = (D <= p["crowd_radius_m"]).sum(1)
        for a, c in enumerate(cand):
            i = idx[c]
            if np.isnan(hx[i]):
                continue
            close = np.where(D[a] <= p["ahead_dist_m"])[0]
            if len(close) == 0:
                continue
            v = P[close] - P[c]
            cos = (v @ np.array([hx[i], hy[i]])) / np.maximum(np.linalg.norm(v, axis=1), 1e-9)
            ahead[i] = bool((cos >= cos35).any())
    df["n_near"], df["ahead"] = n_near, ahead

    state = np.full(len(df), MOVING, dtype=object)
    stop_id = np.full(len(df), -1)
    sid = 0
    tt, spd = df["t"].to_numpy(), df["speed"].to_numpy()
    for tid, idx in df.groupby("track_id").indices.items():
        t = tt[idx]
        stopped = (spd[idx] < p["stop_speed_ms"]) & (t - t[0] >= 0.7)   # filter warm-up is not a stop
        for i0, i1 in runs(stopped):
            if t[i1] - t[i0] + dt < p["min_stop_s"]:
                continue
            sel = idx[i0:i1 + 1]
            blocked = (n_near[sel] >= p["crowd_count"]) | ahead[sel]
            state[sel] = BLOCKED if blocked.mean() > 0.5 else ENGAGED
            stop_id[sel] = sid
            sid += 1
    df["state"], df["stop_id"] = state, stop_id
    return df


def zone_events(df: pd.DataFrame, fps: float, p: dict) -> pd.DataFrame:
    """One row per contiguous stay of a track inside a zone (visit = dwell >= min_visit_s)."""
    dt = 1.0 / fps
    rows = []
    for tid, idx in df.groupby("track_id").indices.items():
        g = df.iloc[idx]
        t, z, st = g["t"].to_numpy(), g["zone"].to_numpy(), g["state"].to_numpy()
        for zone, i0, i1 in segments(z, t, p["merge_gap_s"]):
            if zone is None:
                continue
            n = i1 - i0 + 1
            seg = st[i0:i1 + 1]
            dwell = t[i1] - t[i0] + dt
            rows.append(dict(track_id=int(tid), zone=zone, t_in=float(t[i0]), t_out=float(t[i1] + dt),
                             dwell_s=float(dwell), engaged_s=float((seg == ENGAGED).sum() * dt),
                             blocked_s=float((seg == BLOCKED).sum() * dt),
                             moving_s=float((seg == MOVING).sum() * dt), n_samples=n,
                             is_visit=bool(dwell >= p["min_visit_s"])))
    return pd.DataFrame(rows, columns=["track_id", "zone", "t_in", "t_out", "dwell_s", "engaged_s",
                                       "blocked_s", "moving_s", "n_samples", "is_visit"])


def passers_per_zone(df: pd.DataFrame, zones: ZoneSet, radius: float) -> dict[str, set]:
    """Track ids that came within `radius` m of each zone (inside counts as 0 m)."""
    out = {}
    for z in zones.zones:
        d = zones.distance_to(z.name, df["x"].to_numpy(), df["y"].to_numpy())
        out[z.name] = set(df["track_id"].to_numpy()[d <= radius].tolist())
    return out


def stoppers_per_zone(df: pd.DataFrame, state: str) -> dict[str, set]:
    sub = df[(df["state"] == state) & df["zone"].notna()]
    return {z: set(g["track_id"]) for z, g in sub.groupby("zone")}


def queue_series(ts: pd.DataFrame, events: pd.DataFrame, zone: str, window_s: float = 60.0) -> pd.DataFrame:
    """Queue length (people in the lane polygon) and wait = length / service rate.

    Service rate = people leaving the lane per minute over the trailing window.
    """
    z = ts[ts["zone"] == zone][["t", "count"]].rename(columns={"count": "queue_len"}).reset_index(drop=True)
    ev = events[(events["zone"] == zone) & events["is_visit"]]
    exits = np.sort(ev["t_out"].to_numpy())
    rate, wait = [], []
    for t in z["t"]:
        w = min(window_s, max(t, 1.0))
        r = ((exits > t - w) & (exits <= t)).sum() / w * 60.0
        rate.append(r)
        q = z.loc[z["t"] == t, "queue_len"].iat[0]
        wait.append(q / r if r > 0 else np.nan)
    z["service_per_min"], z["est_wait_min"] = rate, wait
    return z


def build_zone_table(df: pd.DataFrame, events: pd.DataFrame, zones: ZoneSet, p: dict) -> pd.DataFrame:
    visits = events[events["is_visit"]]
    passers = passers_per_zone(df, zones, p["passby_radius_m"])
    eng = stoppers_per_zone(df, ENGAGED)
    blk = stoppers_per_zone(df, BLOCKED)
    rows = []
    for z in zones.zones:
        v = visits[visits["zone"] == z.name]
        eng_s, blk_s = float(v["engaged_s"].sum()), float(v["blocked_s"].sum())
        pas, es = len(passers[z.name]), len(eng.get(z.name, ()))
        rows.append(dict(zone=z.name, kind=z.kind, area_m2=round(z.area, 1), visits=len(v),
                         visitors=int(v["track_id"].nunique()),
                         mean_dwell_s=float(v["dwell_s"].mean()) if len(v) else 0.0,
                         median_dwell_s=float(v["dwell_s"].median()) if len(v) else 0.0,
                         person_seconds=float(v["dwell_s"].sum()), engaged_s=eng_s, blocked_s=blk_s,
                         blocked_share=blk_s / (eng_s + blk_s) if eng_s + blk_s > 0 else 0.0,
                         passers=pas, engaged_stoppers=es, blocked_stoppers=len(blk.get(z.name, ())),
                         stop_ratio=es / pas if pas else 0.0,
                         visits_per_m2=len(v) / z.area))
    return pd.DataFrame(rows)
