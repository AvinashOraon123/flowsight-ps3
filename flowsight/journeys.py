"""Customer journeys: anonymous floor paths and the sequence of zones each one visited."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd


def journey_table(tracks: pd.DataFrame, events: pd.DataFrame, min_path_m: float = 2.0) -> pd.DataFrame:
    """One row per track: duration, path length (m), and the ordered zone sequence of its visits."""
    vis = events[events["is_visit"]].sort_values(["track_id", "t_in"])
    seq = vis.groupby("track_id")["zone"].apply(_collapse)
    rows = []
    for tid, g in tracks.groupby("track_id"):
        xy = g[["x", "y"]].to_numpy()
        length = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if len(xy) > 1 else 0.0
        if length < min_path_m:
            continue       # tracker fragments / people who barely moved are not journeys
        rows.append(dict(track_id=int(tid), t_start=float(g["t"].iloc[0]), t_end=float(g["t"].iloc[-1]),
                         duration_s=float(g["t"].iloc[-1] - g["t"].iloc[0]), path_m=length,
                         zones=list(seq.get(tid, []))))
    return pd.DataFrame(rows, columns=["track_id", "t_start", "t_end", "duration_s", "path_m", "zones"])


def _collapse(zs) -> list[str]:
    out: list[str] = []
    for z in zs:
        if not out or out[-1] != z:
            out.append(z)
    return out


def top_routes(journeys: pd.DataFrame, n: int = 5) -> list[tuple[tuple[str, ...], int]]:
    c = Counter(tuple(z) for z in journeys["zones"] if len(z) >= 1)
    return c.most_common(n)


def sankey_links(journeys: pd.DataFrame) -> pd.DataFrame:
    """source -> target counts between consecutive zone visits (feed to plotly Sankey)."""
    c: Counter = Counter()
    for zs in journeys["zones"]:
        for a, b in zip(zs[:-1], zs[1:]):
            c[(a, b)] += 1
    return pd.DataFrame([(a, b, n) for (a, b), n in c.items()], columns=["source", "target", "count"])
