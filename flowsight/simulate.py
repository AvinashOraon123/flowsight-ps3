"""Agent-based store simulator: synthetic tracks with injected bottlenecks and a known dead zone.

Everything produced here is SIMULATED and labelled as such. Tracks are exact floor coordinates at 1 Hz
(no detector/tracker noise), so results on them validate the analytics, forecaster and action engine,
not the vision stack.

What happens each simulated day (08:00-20:00 by default)
  * shoppers arrive as a Poisson process with lunch and after-work peaks and a random day factor
  * most walk past the Promo endcap on the way in (and only ~12 % stop: "seen but ignored")
  * each browses 2-5 zones; Dairy corner is made deliberately unpopular (ground-truth dead zone)
  * checkout is a FIFO pool of 3 tills; in "understaffed" windows only 1 till works -> a queue forms
  * in "promo stack" windows an aisle has a one-at-a-time gate -> shoppers stand blocked in a line

    python -m flowsight.simulate --config configs/sim.yaml        # train forecaster, run analytics, write outputs/sim
"""
from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .zones import ZoneSet

ENTRY = np.array([15.0, 1.0])
BASE_ORDER = ["Fresh produce", "Aisle 1", "Aisle 2", "Aisle 3", "Dairy corner"]
# chance a shopper includes the zone in their trip; Dairy corner is the planted dead zone
INCLUDE_P = {"Fresh produce": .65, "Aisle 1": .65, "Aisle 2": .60, "Aisle 3": .60, "Dairy corner": .03}
DWELL_MEAN = {"Fresh produce": 60, "Aisle 1": 40, "Aisle 2": 40, "Aisle 3": 40, "Dairy corner": 30, "Promo endcap": 8}
CONVERSION = {"Fresh produce": .04, "Aisle 1": .35, "Aisle 2": .35, "Aisle 3": .35, "Dairy corner": .30, "Promo endcap": .50}
P_ENDCAP_PASS, P_ENDCAP_STOP = 0.80, 0.12
P_CHECKOUT = 0.92
AISLES = ["Aisle 1", "Aisle 2", "Aisle 3"]
GATE_Y = 13.0                       # where the promo stack narrows the aisle
LANE_X, LINE_Y0, LINE_STEP, LINE_MAX_K = 24.0, 5.4, 0.75, 9
TILL_X = [22.6, 23.5, 24.5, 25.4]
N_TILLS, N_TILLS_UNDERSTAFFED, PEAK_RATE, BALK_AT, BALK_TILL = 4, 1, 6.0, 8, 14
TILL_Y = 4.6
QUEUE_APPROACH = np.array([24.0, 12.5])
WALK = 1.2
GATE_SERVICE_S, TILL_SERVICE_S = 60.0, 30.0
GT_WAIT_THRESHOLD = {"aisle": 3.0, "queue": 5.0}      # people truly waiting that make a minute a ground-truth jam


def _lognorm(rng, mean: float, sigma: float = 0.4) -> float:
    return float(rng.lognormal(np.log(mean) - sigma ** 2 / 2, sigma))


def _gauss(h, mu, s):
    return np.exp(-0.5 * ((h - mu) / s) ** 2)


def arrival_times(rng, open_s: float, close_s: float, day_factor: float) -> np.ndarray:
    """Poisson arrivals with lunch and after-work peaks (peak ~6 shoppers/min)."""
    out = []
    for m in range(int((close_s - open_s) // 60) - 25):          # stop admitting 25 min before close
        h = (open_s + m * 60) / 3600
        lam = PEAK_RATE * day_factor * (0.20 + 0.55 * _gauss(h, 12.5, 1.0) + 0.85 * _gauss(h, 18.0, 1.2))
        out += list(open_s + m * 60 + rng.uniform(0, 60, rng.poisson(lam)))
    return np.sort(np.array(out))


def make_windows(rng, open_s: float, close_s: float) -> dict:
    def start_h():
        c = rng.choice(3, p=[.35, .35, .30])
        return [rng.normal(12.0, .5), rng.normal(17.5, .5), rng.uniform(9.5, 18.0)][c]

    def clip(h, dur_min):
        a = float(np.clip(h * 3600, open_s + 1800, close_s - 3600))
        return (a, a + dur_min * 60)

    return {"understaffed": [clip(start_h(), rng.uniform(45, 80)) for _ in range(rng.integers(1, 3))],
            "jams": [(str(rng.choice(AISLES)), *clip(start_h(), rng.uniform(30, 60))) for _ in range(rng.integers(1, 4))]}


class Store:
    """Shared state for one day: till pool, aisle gates, logs. Events are handled in global time order."""

    def __init__(self, rng, zones: ZoneSet, windows: dict):
        self.rng, self.zones, self.w = rng, zones, windows
        self.till_free = [0.0] * N_TILLS
        self.gate_free = {a: 0.0 for a in AISLES}
        self.till_log: list[tuple[float, float]] = []                  # (arrival, start)
        self.gate_log = {a: [] for a in AISLES}
        self.sales: dict[str, int] = {}
        self.endcap = {"passers": 0, "stoppers": 0}
        self.engaged_stops: dict[str, int] = {}
        self.abandons = 0

    def tills_active(self, t: float) -> int:
        return N_TILLS_UNDERSTAFFED if any(a <= t < b for a, b in self.w["understaffed"]) else N_TILLS

    def jam_active(self, aisle: str, t: float) -> bool:
        return any(z == aisle and a <= t < b for z, a, b in self.w["jams"])

    def point_in(self, zone: str) -> np.ndarray:
        x0, y0, x1, y1 = self.zones.by_name[zone].poly.bounds
        return np.array([self.rng.uniform(x0 + .5, x1 - .5), self.rng.uniform(y0 + .5, y1 - .5)])


class Shopper:
    def __init__(self, sid: int, t0: float):
        self.sid, self.segs = sid, []        # seg = (t0, t1, x0, y0, x1, y1, gt_state)

    def walk(self, t, p, q, speed=WALK):
        d = float(np.linalg.norm(np.asarray(q) - np.asarray(p)))
        if d < 1e-6:
            return t, np.asarray(q, float)
        t1 = t + d / speed
        self.segs.append((t, t1, p[0], p[1], q[0], q[1], "moving"))
        return t1, np.asarray(q, float)

    def stand(self, t, p, dur, gt):
        if dur <= 0:
            return t
        self.segs.append((t, t + dur, p[0], p[1], p[0], p[1], gt))
        return t + dur


def _line_pos(k: int) -> np.ndarray:
    return np.array([LANE_X, LINE_Y0 + LINE_STEP * (min(k, LINE_MAX_K) + 1)])


def _wait_in_line(sh: Shopper, a: float, s: float, p_now, log, idx, pos_fn, gt: str):
    """Stand in line from arrival `a` to start `s`, shuffling forward as people ahead are served."""
    starts = np.array([x[1] for x in log[:idx]]) if idx else np.array([])
    t, p = a, np.asarray(p_now, float)
    k = int((starts > a).sum()) if len(starts) else 0
    bps = np.sort(starts[(starts > a) & (starts < s)]) if len(starts) else np.array([])
    q = pos_fn(k)
    t, p = sh.walk(t, p, q)
    for b in bps:
        k = max(k - 1, 0)
        t = sh.stand(t, p, max(b - t, 0.0), gt)
        t, p = sh.walk(t, p, pos_fn(k))
    t = sh.stand(t, p, max(s - t, 0.0), gt)
    return t, p


def shopper_proc(st: Store, sh: Shopper, t0: float):
    """Generator: yields the global time at which the next shared-resource decision happens."""
    rng = st.rng
    t, p = t0, ENTRY.copy()
    plan = [z for z in BASE_ORDER if rng.random() < INCLUDE_P[z]] or ["Aisle 1"]
    if rng.random() < 0.25:
        rng.shuffle(plan)

    if rng.random() < P_ENDCAP_PASS:                                   # walk past the promo endcap
        t, p = sh.walk(t, p, st.point_in("Promo endcap"))
        yield t
        st.endcap["passers"] += 1
        if rng.random() < P_ENDCAP_STOP:
            t = sh.stand(t, p, _lognorm(rng, DWELL_MEAN["Promo endcap"]), "engaged")
            st.endcap["stoppers"] += 1
            st.sales["Promo endcap"] = st.sales.get("Promo endcap", 0) + int(rng.random() < CONVERSION["Promo endcap"])
            st.engaged_stops["Promo endcap"] = st.engaged_stops.get("Promo endcap", 0) + 1

    for zone in plan:
        jam = zone in AISLES and st.jam_active(zone, t)
        if jam:
            gx = float(np.mean(st.zones.by_name[zone].poly.bounds[0::2]))
            t, p = sh.walk(t, p, (gx, GATE_Y + 2.0))
            yield t
            log = st.gate_log[zone]
            if sum(1 for _, s_ in log if s_ > t) >= BALK_AT:            # a long line: shopper skips this aisle
                continue
            start = max(t, st.gate_free[zone])
            service = _lognorm(rng, GATE_SERVICE_S, .3)
            st.gate_free[zone] = start + service
            log.append((t, start))
            t, p = _wait_in_line(sh, t, start, p, log, len(log) - 1,
                                 lambda k, gx=gx: np.array([gx, GATE_Y + 0.8 * (k + 1)]), "blocked")
            t, p = sh.walk(t, p, (gx, GATE_Y))
            t = sh.stand(t, p, service, "blocked")                      # squeezing past the stack
            t, p = sh.walk(t, p, (gx, GATE_Y - 2.0))
        else:
            t, p = sh.walk(t, p, st.point_in(zone))
            yield t
            if rng.random() > 0.9:                                      # 10 % just walk through
                continue
        t = sh.stand(t, p, _lognorm(rng, DWELL_MEAN[zone]), "engaged")
        st.engaged_stops[zone] = st.engaged_stops.get(zone, 0) + 1
        st.sales[zone] = st.sales.get(zone, 0) + int(rng.random() < CONVERSION[zone])

    if rng.random() < P_CHECKOUT:
        t, p = sh.walk(t, p, QUEUE_APPROACH)
        yield t
        if sum(1 for _, s_ in st.till_log if s_ > t) >= BALK_TILL:        # queue too long: abandon the basket
            st.abandons += 1
            t, p = sh.walk(t, p, ENTRY)
            yield t
            return
        c = st.tills_active(t)
        i = int(np.argmin(st.till_free[:c]))
        start = max(t, st.till_free[i])
        service = _lognorm(rng, TILL_SERVICE_S, .35)
        st.till_free[i] = start + service
        st.till_log.append((t, start))
        t, p = _wait_in_line(sh, t, start, p, st.till_log, len(st.till_log) - 1, _line_pos, "blocked")
        t, p = sh.walk(t, p, (TILL_X[i], TILL_Y))
        t = sh.stand(t, p, service, "engaged")
    t, p = sh.walk(t, p, ENTRY)
    yield t


def simulate_day(seed: int, zones: ZoneSet, open_h: float = 8, close_h: float = 20, fps: float = 1.0) -> dict:
    rng = np.random.default_rng(seed)
    open_s, close_s = 0.0, (close_h - open_h) * 3600.0           # t = seconds since opening
    windows = make_windows(rng, open_s + open_h * 3600, close_s + open_h * 3600)
    windows = {"understaffed": [(a - open_h * 3600, b - open_h * 3600) for a, b in windows["understaffed"]],
               "jams": [(z, a - open_h * 3600, b - open_h * 3600) for z, a, b in windows["jams"]]}
    st = Store(rng, zones, windows)
    arr = arrival_times(rng, open_s, close_s, float(rng.uniform(0.8, 1.25)))
    shoppers = [Shopper(i + 1, t) for i, t in enumerate(arr)]
    heap = []
    for sh, t0 in zip(shoppers, arr):
        g = shopper_proc(st, sh, float(t0))
        heapq.heappush(heap, (float(t0), sh.sid, g))
    while heap:
        _, sid, g = heapq.heappop(heap)
        try:
            heapq.heappush(heap, (next(g), sid, g))
        except StopIteration:
            pass

    dt = 1.0 / fps
    rows = []
    for sh in shoppers:
        for t0, t1, x0, y0, x1, y1, gt in sh.segs:
            tt = np.arange(np.ceil(t0 / dt) * dt, t1, dt)
            if len(tt) == 0:
                continue
            a = (tt - t0) / max(t1 - t0, 1e-9)
            x, y = x0 + (x1 - x0) * a, y0 + (y1 - y0) * a
            if gt != "moving":
                x, y = x + rng.normal(0, .03, len(tt)), y + rng.normal(0, .03, len(tt))
            rows.append(pd.DataFrame(dict(t=tt, track_id=sh.sid, x=x, y=y, gt_state=gt)))
    tr = pd.concat(rows, ignore_index=True).sort_values(["track_id", "t"]).reset_index(drop=True)
    tr = tr.drop_duplicates(["track_id", "t"])
    g = tr.groupby("track_id")
    tr["vx"] = (g["x"].diff() / g["t"].diff()).fillna(0.0)
    tr["vy"] = (g["y"].diff() / g["t"].diff()).fillna(0.0)
    tr["speed"] = np.hypot(tr["vx"], tr["vy"])
    tr["frame"] = np.round(tr["t"] * fps).astype(int)
    tr["zone"] = zones.locate(tr["x"], tr["y"])
    # ground truth from the simulator's own queues (independent of the detector's congestion formula)
    gt_wait = _waiting_per_minute(st, close_s)
    return dict(tracks=tr, gt_wait=gt_wait, windows=windows, sales=st.sales, endcap=st.endcap,
                engaged_stops=st.engaged_stops, n_shoppers=len(shoppers), abandons=st.abandons)


def _waiting_per_minute(st: Store, close_s: float) -> pd.DataFrame:
    n = int(close_s // 60) + 1
    out = []
    logs = {"Checkout lane": st.till_log, **{a: st.gate_log[a] for a in AISLES}}
    for zone, log in logs.items():
        w = np.zeros(n)
        for a, s in log:
            if s <= a:
                continue
            for m in range(int(a // 60), min(int(s // 60) + 1, n)):
                w[m] += max(min(s, (m + 1) * 60) - max(a, m * 60), 0) / 60.0
        out.append(pd.DataFrame(dict(t=np.arange(n) * 60.0, zone=zone, waiting=w)))
    return pd.concat(out, ignore_index=True)


def main(argv=None) -> None:
    from . import bottleneck, forecast
    from .pipeline import analyze, load_config, rpath

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sim.yaml")
    ap.add_argument("--train-days", type=int, default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    s = cfg["sim"]
    out = rpath(cfg, cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    fdir = rpath(cfg, cfg["forecast"]["model_dir"])
    fdir.mkdir(parents=True, exist_ok=True)
    zones = ZoneSet.load(rpath(cfg, "zones"))
    p = cfg["analytics"]
    n_train = a.train_days or s["train_days"]

    def day_ts(seed):
        d = simulate_day(seed, zones, s["open_h"], s["close_h"], s["fps"])
        tr = d["tracks"]
        ts = bottleneck.zone_timeseries(tr, zones, s["fps"], p["ts_bin_s"], p["v_free_ms"], t_end=(s["close_h"] - s["open_h"]) * 3600)
        arrivals = tr.groupby("track_id")["t"].min().to_numpy()
        return d, ts, arrivals

    print(f"simulating {n_train} training days + {len(s['test_seeds'])} test days ...")
    train = []
    for seed in range(1, n_train + 1):
        d, ts, arr = day_ts(seed)
        train.append((ts, arr))
        print(f"  train day {seed}: {d['n_shoppers']} shoppers, {len(d['tracks'])} samples")
    tests = []
    for i, seed in enumerate(s["test_seeds"]):
        dd, tts, tarr = day_ts(seed)
        tests.append((tts, tarr))
        print(f"  test day {seed}: {dd['n_shoppers']} shoppers, {len(dd['tracks'])} samples")
        if i == 0:
            d = dd                                   # the first test day gets the full analytics + dashboard view

    res = forecast.train_and_evaluate(train, tests, cfg["forecast"]["horizons_min"], cfg["forecast"]["alert_threshold"], zones.names)
    forecast.save(res, fdir)
    print("forecast:", json.dumps(res["meta"]["metrics"], indent=1))

    sales = pd.DataFrame([(z, n) for z, n in d["sales"].items()], columns=["zone", "purchases"])
    r = analyze(cfg, d["tracks"], zones, s["fps"], out, forecast=res["alerts"], sales=sales)
    d["gt_wait"].to_parquet(out / "gt_waiting.parquet")
    r["tracks"][["track_id", "t", "gt_state", "state", "zone"]].to_parquet(out / "state_labels.parquet")
    gt = dict(windows=d["windows"], endcap=d["endcap"], dead_zones=["Dairy corner"], sales=d["sales"],
              engaged_stops=d["engaged_stops"], n_shoppers=d["n_shoppers"], abandons=d["abandons"], gt_wait_threshold=GT_WAIT_THRESHOLD,
              note="SIMULATED ground truth")
    (out / "sim_gt.json").write_text(json.dumps(gt, indent=2))
    print("done ->", out)


if __name__ == "__main__":
    main()
