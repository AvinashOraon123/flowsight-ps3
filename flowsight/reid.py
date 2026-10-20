"""Cross-camera identity stitching on the shared floor plan.

Every camera projects its tracks to the same floor coordinates. A track that ends in one camera is linked to a
track that starts in another when they are close in time and space (a hand-over in the overlap, or a short walk
between non-overlapping views) and look alike. Linking is a one-to-one assignment (Hungarian) on
    cost = w_space * d / max_dist  +  (1 - w_space) * cosine_distance(appearance)
and a link is kept only when cost < max_cost. Chains of links become one global identity.

Appearance: `colour_embedding` is an HSV histogram of the torso crop (CPU-cheap, but weak: similar clothing
collides). `osnet_embedding` uses OSNet if `torchreid` is installed (not in requirements.txt; untested here).
Evaluated on SYNTHETIC two-camera data only (eval/stitch_eval.py), no real multi-camera footage was available.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def colour_embedding(frame_bgr: np.ndarray, box_xyxy, bins: int = 8) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    h = y2 - y1
    crop = frame_bgr[max(y1 + int(.2 * h), 0):max(y1 + int(.6 * h), 0), max(x1, 0):max(x2, 0)]
    if crop.size == 0:
        return np.zeros(bins * 2)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hh = cv2.calcHist([hsv], [0], None, [bins], [0, 180]).ravel()
    ss = cv2.calcHist([hsv], [1], None, [bins], [0, 256]).ravel()
    e = np.concatenate([hh / max(hh.sum(), 1), ss / max(ss.sum(), 1)])
    return e / (np.linalg.norm(e) + 1e-9)


def osnet_embedding(frame_bgr: np.ndarray, box_xyxy):                      # pragma: no cover (needs torchreid)
    from torchreid.utils import FeatureExtractor
    if not hasattr(osnet_embedding, "fx"):
        osnet_embedding.fx = FeatureExtractor(model_name="osnet_x0_25", device="cpu")
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    f = osnet_embedding.fx([cv2.cvtColor(frame_bgr[max(y1, 0):y2, max(x1, 0):x2], cv2.COLOR_BGR2RGB)])[0].numpy()
    return f / (np.linalg.norm(f) + 1e-9)


def track_summaries(df: pd.DataFrame, cam: str) -> pd.DataFrame:
    """One row per (camera, local track): first/last time and floor position, mean appearance.
    df columns: track_id, t, x, y, optional emb (ndarray per row)."""
    rows = []
    for tid, g in df.sort_values("t").groupby("track_id"):
        r = dict(cam=cam, tid=tid, t0=g["t"].iloc[0], t1=g["t"].iloc[-1], x0=g["x"].iloc[0], y0=g["y"].iloc[0],
                 x1=g["x"].iloc[-1], y1=g["y"].iloc[-1], n=len(g))
        if "emb" in g:
            e = np.mean(np.stack(g["emb"].to_numpy()), axis=0)
            r["emb"] = e / (np.linalg.norm(e) + 1e-9)
        rows.append(r)
    return pd.DataFrame(rows)


def stitch(summ: pd.DataFrame, max_gap_s: float = 4.0, max_dist_m: float = 3.0, w_space: float = 0.5,
           max_cost: float = 0.6, use_appearance: bool = True) -> dict:
    """Return {(cam, tid): global_id}. summ comes from `track_summaries`, concatenated over cameras."""
    S = summ.reset_index(drop=True)
    n = len(S)
    cost = np.full((n, n), np.inf)                               # row = track that ends, col = track that starts
    for i in range(n):
        for j in range(n):
            a, b = S.iloc[i], S.iloc[j]
            if a["cam"] == b["cam"]:
                continue
            gap = b["t0"] - a["t1"]
            if gap < -1.0 or gap > max_gap_s:                    # small negative gap = overlap hand-over
                continue
            d = float(np.hypot(a["x1"] - b["x0"], a["y1"] - b["y0"]))
            if d > max_dist_m:
                continue
            c = d / max_dist_m
            if use_appearance and "emb" in S:
                c = w_space * c + (1 - w_space) * (1 - float(np.dot(a["emb"], b["emb"])))
            cost[i, j] = c
    finite = np.where(np.isfinite(cost), cost, 1e6)
    ri, ci = linear_sum_assignment(finite)
    parent = list(range(n))

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for i, j in zip(ri, ci):
        if np.isfinite(cost[i, j]) and cost[i, j] < max_cost and find(i) != find(j):
            parent[find(j)] = find(i)
    return {(S.at[k, "cam"], S.at[k, "tid"]): find(k) for k in range(n)}
