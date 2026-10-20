"""Congestion score, bottleneck flags and dead-zone detection.

    C_z(t) = (rho_z(t) / rho_max) * (1 - v_z(t) / v_free)        both factors clipped to [0, 1]

rho = people per m2 inside the zone, v = their mean walking speed, v_free ~ 1.2 m/s.
High density *and* collapsed speed is what separates a jam from a busy-but-flowing aisle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .heatmap import HeatGrid
from .zones import ZoneSet


def congestion(rho, v, rho_max: float, v_free: float = 1.2):
    rho, v = np.asarray(rho, float), np.asarray(v, float)
    return np.clip(rho / rho_max, 0, 1) * (1 - np.clip(v / v_free, 0, 1))


def zone_timeseries(tracks: pd.DataFrame, zones: ZoneSet, fps: float, bin_s: float = 1.0,
                    v_free: float = 1.2, warmup_s: float = 1.0, t_end: float | None = None,
                    exclude_engaged: bool = True) -> pd.DataFrame:
    """Per zone and time bin: count, rho, mean speed, C. Gap-free (empty bins = 0 people, C = 0).

    rho counts everyone in the zone. The speed term is the mean *walking* speed (as in the model): people who
    stopped by choice (state == "engaged") are left out, because someone reading a label is not being held up.
    Without this a crowd of browsers looks like a jam. Needs a `state` column (metrics.add_motion_state);
    without one every sample counts."""
    tracks = tracks.reset_index(drop=True)
    use_state = exclude_engaged and "state" in tracks.columns
    df = tracks[tracks["zone"].notna()].copy()
    df["bin"] = np.floor(df["t"] / bin_s).astype(int)
    age = tracks.groupby("track_id")["t"].transform("min")
    df["age"] = (tracks["t"] - age).loc[df.index]
    t_max = t_end if t_end is not None else float(tracks["t"].max())
    n_bins = int(np.floor(t_max / bin_s)) + 1
    frames_per_bin = max(bin_s * fps, 1.0)
    rows = []
    for z in zones.zones:
        d = df[df["zone"] == z.name]
        cnt = np.zeros(n_bins)
        c = d.groupby("bin").size()                 # person-samples per bin
        cnt[np.clip(c.index.to_numpy(), 0, n_bins - 1)] = c.to_numpy() / frames_per_bin
        v = np.full(n_bins, v_free)
        walking = d[(d["age"] >= warmup_s) & ((d["state"] != "engaged") if use_state else True)]
        w = walking.groupby("bin")["speed"].mean()
        v[np.clip(w.index.to_numpy(), 0, n_bins - 1)] = w.to_numpy()
        rho = cnt / z.area
        C = congestion(rho, v, z.rho_max, v_free)
        rows.append(pd.DataFrame(dict(t=np.arange(n_bins) * bin_s, zone=z.name, count=cnt, rho=rho,
                                      v_mean=v, C=C)))
    return pd.concat(rows, ignore_index=True)


def flag_bottlenecks(ts: pd.DataFrame, threshold: float, sustain_s: float, bin_s: float = 1.0) -> pd.DataFrame:
    """Intervals where C > threshold for at least `sustain_s` seconds, per zone."""
    out = []
    for zone, g in ts.groupby("zone"):
        g = g.sort_values("t")
        above = (g["C"].to_numpy() > threshold)
        t = g["t"].to_numpy()
        i = 0
        while i < len(above):
            if not above[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(above) and above[j + 1]:
                j += 1
            dur = t[j] - t[i] + bin_s
            if dur >= sustain_s:
                seg = g.iloc[i:j + 1]
                out.append(dict(zone=zone, t_start=float(t[i]), t_end=float(t[j] + bin_s), duration_s=float(dur),
                                alert_t=float(t[i] + sustain_s), peak_C=float(seg["C"].max()),
                                mean_count=float(seg["count"].mean()), peak_count=float(seg["count"].max()),
                                mean_speed=float(seg["v_mean"].mean())))
            i = j + 1
    return pd.DataFrame(out, columns=["zone", "t_start", "t_end", "duration_s", "alert_t", "peak_C",
                                      "mean_count", "peak_count", "mean_speed"])


def dead_zones(table: pd.DataFrame, zones: ZoneSet, heat: HeatGrid, p: dict) -> pd.DataFrame:
    """Zones whose visit rate per m2, corrected for distance from the main path, is in the bottom decile.

    Main path = the hottest cells of the (smoothed) traffic heat map. A zone far from that path is
    *expected* to be quieter, so its visits/m2 is divided by exp(-d / decay) before ranking. A zone is
    flagged only if it is in the bottom `dead_zone_decile` AND well under the median (so a store with
    uniformly healthy traffic reports no dead zone).
    """
    hs = heat.smoothed(p["heat_sigma_cells"])
    nz = hs[hs > 0]
    cand = table[~table["kind"].isin(["entrance", "queue"])].copy()   # not browsing zones
    if len(nz) == 0 or table["visits"].sum() < 10 or len(cand) < 3:
        cand["adj_visit_rate"], cand["dist_to_path_m"], cand["dead_zone"] = np.nan, np.nan, False
        return cand[["zone", "adj_visit_rate", "dist_to_path_m", "dead_zone"]]
    ys, xs = np.where(hs >= np.percentile(nz, p["main_path_pct"]))
    path = np.stack([(xs + .5) * heat.cell, (ys + .5) * heat.cell], 1)
    cent = np.array([zones.by_name[z].centroid for z in cand["zone"]])
    d = cdist(cent, path).min(1)
    expected = np.exp(-d / p["main_path_decay_m"])
    cand["dist_to_path_m"] = d
    cand["adj_visit_rate"] = cand["visits_per_m2"].to_numpy() / np.maximum(expected, 1e-3)
    k = max(int(np.ceil(p["dead_zone_decile"] * len(cand))), 1)
    cutoff = np.sort(cand["adj_visit_rate"].to_numpy())[k - 1]
    med = float(np.median(cand["adj_visit_rate"]))
    cand["dead_zone"] = (cand["adj_visit_rate"] <= cutoff) & (cand["adj_visit_rate"] < p["dead_zone_rel_median"] * med)
    return cand[["zone", "adj_visit_rate", "dist_to_path_m", "dead_zone"]]
