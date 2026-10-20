"""Small-group suppression for anything shown to people (API, dashboard).

A zone seen by fewer than `min_group` distinct people in the data window reports no per-zone dwell/behaviour
figures: with 1-4 people the "average" is a description of individuals. Counts of how many visitors there were are
kept (they are what lets a reader see why a row is blank). Raw outputs on disk are never altered.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_GROUP = 5
HIDE = ["mean_dwell_s", "median_dwell_s", "person_seconds", "engaged_s", "blocked_s", "blocked_share", "stop_ratio",
        "engaged_stoppers", "blocked_stoppers", "visits_per_m2", "adj_visit_rate", "peak_queue", "max_est_wait_min"]


def suppress_small_groups(table: pd.DataFrame, min_group: int = MIN_GROUP) -> pd.DataFrame:
    """Blank behaviour columns for zones with fewer than `min_group` visitors and mark the row."""
    t = table.copy()
    small = t["visitors"] < min_group
    for c in HIDE:
        if c in t.columns:
            t[c] = t[c].astype(float).mask(small, np.nan)
    if "dead_zone" in t.columns:
        t["dead_zone"] = t["dead_zone"].astype(bool) & ~small       # too little data to call a dead zone either way
    t["suppressed"] = small
    return t
