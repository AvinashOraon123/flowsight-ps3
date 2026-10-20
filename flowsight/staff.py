"""Staff exclusion: keep employees out of shopper dwell, queue and engagement figures.

Three cues, any one marks a track as staff:
  uniform   median `uniform_frac` of the track (share of torso-crop pixels inside the configured uniform colour range,
            computed at detection time by `uniform_fraction`) is at least `uniform_min`
  zone      at least `zone_share` of the track's time is in a zone named in `staff_zones` (behind the counter, back office)
  presence  the stitched identity is on the floor for at least `presence_s` (shoppers do not stay for hours)
Without a configured cue nothing is excluded. Colour cues need a uniform that differs from shopper clothing; this
is a heuristic, not a classifier trained on real staff.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd


def uniform_fraction(frame_bgr: np.ndarray, box_xyxy, hsv_lo, hsv_hi) -> float:
    """Share of torso pixels (upper-middle of the person box) inside an HSV range. Hue wraps are not handled."""
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    h = y2 - y1
    crop = frame_bgr[max(y1 + int(.2 * h), 0):max(y1 + int(.55 * h), 0), max(x1, 0):max(x2, 0)]
    if crop.size == 0:
        return 0.0
    m = cv2.inRange(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV), np.array(hsv_lo, np.uint8), np.array(hsv_hi, np.uint8))
    return float((m > 0).mean())


def classify_staff(tracks: pd.DataFrame, p: dict | None, id_col: str = "track_id") -> set:
    """Return the set of ids judged to be staff. `p` is the `staff:` config block (None or {} = no staff cue)."""
    if not p or tracks.empty:
        return set()
    g = tracks.groupby(id_col)
    staff: set = set()
    if "uniform_frac" in tracks.columns and p.get("uniform_min") is not None:
        staff |= set(g["uniform_frac"].median()[lambda s: s >= p["uniform_min"]].index)
    if p.get("staff_zones"):
        share = tracks.assign(_s=tracks["zone"].isin(p["staff_zones"])).groupby(id_col)["_s"].mean()
        staff |= set(share[share >= p.get("zone_share", 0.5)].index)
    if p.get("presence_s"):
        pres = g["t"].max() - g["t"].min()
        staff |= set(pres[pres >= p["presence_s"]].index)
    return staff


def exclude_staff(tracks: pd.DataFrame, p: dict | None, id_col: str = "track_id") -> tuple[pd.DataFrame, set]:
    s = classify_staff(tracks, p, id_col)
    return (tracks[~tracks[id_col].isin(s)].reset_index(drop=True) if s else tracks), s
