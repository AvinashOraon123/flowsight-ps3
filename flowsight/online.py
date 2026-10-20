"""Streaming engine: one frame at a time, alerts as they happen (the live counterpart of pipeline.analyze).

    eng = OnlineEngine(zones, H, fps, cfg["analytics"], cfg["actions"])
    for t, ids, uv in frames:                  # uv = foot points in image pixels
        out = eng.update(t, ids, uv)           # out["events"] holds alerts that fired on this frame

Same rules as the batch analytics (walking-speed congestion score, sustained-bottleneck flag, queue length and
wait, engaged vs blocked stops, per-zone rate limiting) but decided causally, with no look-ahead. Only anonymous
floor coordinates ever leave this class.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from . import actions
from .bottleneck import congestion
from .geometry import image_to_floor
from .track import FloorSmoother
from .zones import ZoneSet

COS35 = np.cos(np.radians(35))


class _Track:
    __slots__ = ("t0", "slow_since", "head", "stop_n", "stop_blocked", "state")

    def __init__(self, t0: float):
        self.t0, self.slow_since, self.head = t0, None, None
        self.stop_n = self.stop_blocked = 0
        self.state = "moving"


class OnlineEngine:
    def __init__(self, zones: ZoneSet, H: np.ndarray, fps: float, p: dict, act: dict):
        self.zones, self.H, self.fps, self.p, self.act = zones, H, fps, p, act
        self.sm, self.tr = FloorSmoother(), {}
        self.sec, self.acc = -1, self._new_acc()
        self.above = {z: 0 for z in zones.names}                 # seconds each zone has been congested
        self.last_alert: dict[tuple[str, str], float] = {}
        self.exits = {z.name: deque() for z in zones.of_kind("queue")}
        self.prev_zone: dict[int, tuple[str | None, float]] = {}
        self.state = {z: dict(count=0.0, speed=p["v_free_ms"], C=0.0, flag=False) for z in zones.names}

    def _new_acc(self):
        return {z: dict(frames=0, people=0, speeds=[]) for z in self.zones.names}

    # ------------------------------------------------------------------
    def update(self, t: float, ids, uv) -> dict:
        ids = [int(i) for i in ids]
        events: list[actions.Alert] = []
        if int(t) > self.sec:                                      # a full second has passed: close it first
            if self.sec >= 0:
                events += self._close_second(float(self.sec))
            self.sec, self.acc = int(t), self._new_acc()
        f = image_to_floor(self.H, np.asarray(uv, float).reshape(-1, 2)) if len(ids) else np.zeros((0, 2))
        pos = np.zeros((len(ids), 4))
        for k, (i, (x, y)) in enumerate(zip(ids, f)):
            if i not in self.tr:
                self.tr[i] = _Track(t)
            pos[k] = self.sm.update(i, t, float(x), float(y))
        xy, v = pos[:, :2], np.hypot(pos[:, 2], pos[:, 3])
        zl = self.zones.locate(xy[:, 0], xy[:, 1]) if len(ids) else np.array([], dtype=object)
        self._stops(t, ids, pos, v)
        people = []
        for k, i in enumerate(ids):
            tk = self.tr[i]
            people.append(dict(id=i, x=round(float(xy[k, 0]), 2), y=round(float(xy[k, 1]), 2), state=tk.state, zone=zl[k]))
            self._exit_tracking(i, zl[k], t)
            if zl[k] is not None:
                a = self.acc[zl[k]]
                a["people"] += 1
                if t - tk.t0 >= 1.0 and tk.state != "engaged":    # walking speed only: browsers are not being held up
                    a["speeds"].append(float(v[k]))
        for z in self.acc.values():
            z["frames"] += 1
        return dict(t=t, people=people, zones=self.state, events=events)

    # ------------------------------------------------------------------
    def _stops(self, t, ids, pos, v):
        """Latch moving / engaged / blocked per stop episode with causal rules (decided once a stop lasts min_stop_s)."""
        p = self.p
        P = pos[:, :2]
        D = cdist(P, P) if len(ids) > 1 else np.zeros((len(ids), len(ids)))
        for k, i in enumerate(ids):
            tk = self.tr[i]
            if v[k] > 0.4:
                tk.head = pos[k, 2:4] / max(v[k], 1e-9)
            if v[k] < p["stop_speed_ms"] and t - tk.t0 >= 0.7:
                tk.slow_since = t if tk.slow_since is None else tk.slow_since
            else:
                tk.slow_since, tk.stop_n, tk.stop_blocked, tk.state = None, 0, 0, "moving"
                continue
            if t - tk.slow_since + 1.0 / self.fps < p["min_stop_s"]:
                continue
            d = D[k].copy()
            d[k] = np.inf
            crowd = (d <= p["crowd_radius_m"]).sum() >= p["crowd_count"]
            ahead = False
            if tk.head is not None:
                close = np.where(d <= p["ahead_dist_m"])[0]
                if len(close):
                    w = P[close] - P[k]
                    cos = (w @ tk.head) / np.maximum(np.linalg.norm(w, axis=1), 1e-9)
                    ahead = bool((cos >= COS35).any())
            tk.stop_n += 1
            tk.stop_blocked += int(crowd or ahead)
            tk.state = "blocked" if tk.stop_blocked / tk.stop_n > 0.5 else "engaged"

    def _exit_tracking(self, i, zone, t):
        """Record when someone leaves a queue lane after a real visit: the lane's service rate."""
        prev = self.prev_zone.get(i)
        if prev is None:
            self.prev_zone[i] = (zone, t)
        elif zone != prev[0]:
            if prev[0] in self.exits and t - prev[1] >= self.p["min_visit_s"]:
                self.exits[prev[0]].append(t)
            self.prev_zone[i] = (zone, t)

    def _close_second(self, t: float) -> list[actions.Alert]:
        p, out = self.p, []
        for z in self.zones.zones:
            a = self.acc[z.name]
            n = a["people"] / max(a["frames"], 1)
            v = float(np.mean(a["speeds"])) if a["speeds"] else p["v_free_ms"]
            c = float(congestion(n / z.area, v, z.rho_max, p["v_free_ms"]))
            flag = c > p["bottleneck_threshold"]
            self.above[z.name] = self.above[z.name] + 1 if flag else 0
            self.state[z.name] = dict(count=round(n, 2), speed=round(v, 2), C=round(c, 3), flag=bool(self.above[z.name] >= p["bottleneck_sustain_s"]))
            if self.above[z.name] == p["bottleneck_sustain_s"]:
                out += actions.bottleneck_alerts(pd.DataFrame([dict(
                    zone=z.name, alert_t=t, peak_C=c, duration_s=self.above[z.name], mean_count=n, mean_speed=v)]))
            if z.name in self.exits:
                ex = self.exits[z.name]
                while ex and ex[0] < t - 60:
                    ex.popleft()
                rate = len(ex)
                q = pd.DataFrame([dict(t=t, queue_len=round(n), service_per_min=rate,
                                       est_wait_min=(round(n) / rate if rate else np.nan))])
                out += actions.queue_alerts(q, z.name, self.act)
        return actions.rate_limit_online(out, self.last_alert, self.act["rate_limit_s"])
