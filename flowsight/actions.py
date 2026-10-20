"""Action engine: metrics -> ranked, rate-limited alerts. Every alert carries the reason it fired.

impact = person-minutes affected (people involved x minutes they are affected), so a 7-person queue
waiting 6 min (42) outranks a dead-zone note (a few). Rules follow the proposal's section-5 table.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

AUDIENCE_RANK = {"Floor staff": 0, "Nearest staff": 0, "Shift lead": 1, "Store manager": 2,
                 "Category manager": 2, "Visual merchandising": 3}


@dataclass
class Alert:
    t: float              # seconds into the video/day when it fires
    zone: str
    kind: str             # queue | bottleneck | forecast | endcap | dead_zone | engaged_no_sale
    action: str
    reason: str
    impact: float
    audience: str
    timescale: str        # seconds | minutes | daily | weekly

    def asdict(self) -> dict:
        return asdict(self)


def queue_alerts(q: pd.DataFrame, zone: str, p: dict) -> list[Alert]:
    """Open another till when the lane is long or the estimated wait is long. q: queue_series output."""
    out = []
    hit = q[(q["queue_len"] >= p["queue_open_till"]) | (q["est_wait_min"] > p["wait_open_till_min"])]
    for _, r in hit.iterrows():
        wait = r["est_wait_min"]
        w = f", est. wait {wait:.1f} min at {r['service_per_min']:.1f} people/min" if np.isfinite(wait) else \
            ", nobody has been served recently"
        out.append(Alert(float(r["t"]), zone, "queue", f"Open another till at {zone}",
                         f"Queue of {int(r['queue_len'])} people{w}", float(r["queue_len"] * (wait if np.isfinite(wait) else 3.0)),
                         "Floor staff", "seconds"))
    return out


def bottleneck_alerts(bn: pd.DataFrame) -> list[Alert]:
    out = []
    for _, r in bn.iterrows():
        out.append(Alert(float(r["alert_t"]), r["zone"], "bottleneck",
                         f"Clear the blockage in {r['zone']} (move trolleys / pallet / display)",
                         f"Congestion {r['peak_C']:.2f} for {r['duration_s']:.0f} s: ~{r['mean_count']:.1f} people, "
                         f"average speed down to {r['mean_speed']:.2f} m/s",
                         float(r["mean_count"] * r["duration_s"] / 60.0), "Nearest staff", "seconds"))
    return out


def forecast_alerts(fc: pd.DataFrame, thr: float) -> list[Alert]:
    """fc columns: t, zone, horizon_min, pred_C, count. Fires once per (zone, rising edge)."""
    out = []
    for (zone, h), g in fc.groupby(["zone", "horizon_min"]):
        g = g.sort_values("t")
        hot = (g["pred_C"] > thr).to_numpy()
        for i in range(len(g)):
            if hot[i] and (i == 0 or not hot[i - 1]):
                r = g.iloc[i]
                out.append(Alert(float(r["t"]), zone, "forecast",
                                 f"Pre-position staff near {zone}",
                                 f"Forecast: congestion {r['pred_C']:.2f} in {int(h)} min (now {r['count']:.0f} people there)",
                                 float(max(r["count"], 1) * h * 0.8), "Shift lead", "minutes"))
    return out


def endcap_alerts(table: pd.DataFrame, t_end: float, p: dict) -> list[Alert]:
    out = []
    for _, r in table[table["kind"] == "endcap"].iterrows():
        if r["passers"] >= p["endcap_min_passers"] and r["stop_ratio"] < p["endcap_stop_ratio"]:
            out.append(Alert(t_end, r["zone"], "endcap",
                             f"{r['zone']} is seen but ignored: test new signage or product",
                             f"{int(r['passers'])} people passed within 2 m but only {int(r['engaged_stoppers'])} "
                             f"stopped to look (stop ratio {r['stop_ratio']:.0%})",
                             float((r["passers"] - r["engaged_stoppers"]) * 0.5), "Category manager", "daily"))
    return out


def dead_zone_alerts(dz: pd.DataFrame, table: pd.DataFrame, t_end: float) -> list[Alert]:
    out = []
    t = table.set_index("zone")
    for _, r in dz[dz["dead_zone"]].iterrows():
        row = t.loc[r["zone"]]
        out.append(Alert(t_end, r["zone"], "dead_zone",
                         f"Dead zone: move a high-demand item or promo into {r['zone']}",
                         f"{int(row['visits'])} visits on {row['area_m2']:.0f} m2, {r['dist_to_path_m']:.1f} m from the "
                         f"main path; distance-adjusted visit rate is in the bottom decile of the store",
                         float(row["area_m2"] * 0.1), "Visual merchandising", "weekly"))
    return out


def engaged_no_sale_alerts(table: pd.DataFrame, sales: pd.DataFrame | None, t_end: float,
                           min_engaged: int = 10, max_conv: float = 0.1) -> list[Alert]:
    """Needs POS data (zone, purchases). Looked-but-did-not-buy vs never-saw-it."""
    if sales is None or sales.empty:
        return []
    s = sales.set_index("zone")["purchases"]
    out = []
    for _, r in table.iterrows():
        eng = r["engaged_stoppers"]
        if eng >= min_engaged and r["zone"] in s.index and s[r["zone"]] / eng < max_conv:
            out.append(Alert(t_end, r["zone"], "engaged_no_sale",
                             f"Shoppers consider but do not buy in {r['zone']}: check price, stock or add help",
                             f"{int(eng)} engaged stops but only {int(s[r['zone']])} purchases ({s[r['zone']] / eng:.0%})",
                             float(eng * 0.5), "Store manager", "daily"))
    return out


def rate_limit(alerts: list[Alert], min_gap_s: float) -> list[Alert]:
    """Keep at most one alert per (zone, kind) every `min_gap_s`. Alert fatigue kills adoption."""
    last: dict[tuple, float] = {}
    out = []
    for a in sorted(alerts, key=lambda a: a.t):
        k = (a.zone, a.kind)
        if k in last and a.t - last[k] < min_gap_s:
            continue
        last[k] = a.t
        out.append(a)
    return out


def rank(alerts: list[Alert]) -> list[Alert]:
    """Highest impact first; ties go to the faster timescale / closer audience."""
    ts = {"seconds": 0, "minutes": 1, "daily": 2, "weekly": 3}
    return sorted(alerts, key=lambda a: (-a.impact, ts.get(a.timescale, 9), a.t))


def compare_layouts(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """Before/after zone table delta (visits, dwell, blocked share, stop ratio) for layout A/B."""
    cols = ["visits", "mean_dwell_s", "blocked_share", "stop_ratio"]
    m = before.set_index("zone")[cols].join(after.set_index("zone")[cols], lsuffix="_before", rsuffix="_after")
    for c in cols:
        m[f"{c}_delta"] = m[f"{c}_after"] - m[f"{c}_before"]
    return m.reset_index()


def generate(ts: pd.DataFrame, queues: dict[str, pd.DataFrame], bn: pd.DataFrame, table: pd.DataFrame,
             dz: pd.DataFrame, t_end: float, p: dict, forecast: pd.DataFrame | None = None,
             forecast_thr: float = 0.3, sales: pd.DataFrame | None = None) -> list[Alert]:
    """All rules -> rate-limited -> ranked."""
    a: list[Alert] = []
    for zone, q in queues.items():
        a += queue_alerts(q, zone, p)
    a += bottleneck_alerts(bn)
    if forecast is not None and len(forecast):
        a += forecast_alerts(forecast, forecast_thr)
    a += endcap_alerts(table, t_end, p)
    a += dead_zone_alerts(dz, table, t_end)
    a += engaged_no_sale_alerts(table, sales, t_end)
    return rank(rate_limit(a, p["rate_limit_s"]))
