"""Zones = polygons on the floor plan (metres). Entry/exit events per track."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from shapely import contains_xy, distance, points
from shapely.geometry import Polygon
from shapely.ops import unary_union


@dataclass
class Zone:
    name: str
    kind: str
    priority: int
    rho_max: float
    poly: Polygon          # as drawn
    geom: Polygon          # minus higher-priority zones: what actually belongs to this zone

    @property
    def area(self) -> float:
        return float(self.geom.area)

    @property
    def centroid(self) -> tuple[float, float]:
        c = self.geom.centroid
        return float(c.x), float(c.y)


class ZoneSet:
    def __init__(self, zones: list[Zone], width_m: float, height_m: float, px_per_m: float):
        self.zones = sorted(zones, key=lambda z: z.priority)
        self.width_m, self.height_m, self.px_per_m = width_m, height_m, px_per_m
        self.names = [z.name for z in self.zones]
        self.by_name = {z.name: z for z in self.zones}

    @classmethod
    def load(cls, path) -> "ZoneSet":
        d = json.loads(Path(path).read_text())
        raw = sorted(d["zones"], key=lambda z: z["priority"])
        zones, taken = [], []
        for z in raw:
            poly = Polygon(z["polygon"])
            geom = poly.difference(unary_union(taken)) if taken else poly
            zones.append(Zone(z["name"], z.get("kind", "aisle"), z["priority"],
                              float(z.get("rho_max", 4.0)), poly, geom))
            taken.append(poly)
        fp = d["floor_plan"]
        return cls(zones, fp["width_m"], fp["height_m"], fp["px_per_m"])

    def locate(self, x, y) -> np.ndarray:
        """Zone name for each point (object array, None when outside every zone)."""
        x, y = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float))
        out = np.full(len(x), None, dtype=object)
        for z in reversed(self.zones):           # lowest priority first, so higher priority wins
            out[contains_xy(z.poly, x, y)] = z.name
        return out

    def distance_to(self, name: str, x, y) -> np.ndarray:
        return distance(self.by_name[name].poly, points(np.asarray(x, float), np.asarray(y, float)))

    def of_kind(self, kind: str) -> list[Zone]:
        return [z for z in self.zones if z.kind == kind]


def segments(labels, t, gap_s: float = 1.0) -> list[list]:
    """Contiguous runs [zone, i0, i1] of equal zone label along one track.

    A short excursion (<= gap_s) out of a zone and back is merged, so a tracker
    flicker at a polygon edge does not split one visit into two.
    """
    runs: list[list] = []
    for i, z in enumerate(labels):
        if runs and runs[-1][0] == z:
            runs[-1][2] = i
        else:
            runs.append([z, i, i])
    merged = True
    while merged:
        merged = False
        for k in range(1, len(runs) - 1):
            a, b, c = runs[k - 1], runs[k], runs[k + 1]
            if a[0] == c[0] and a[0] is not None and (t[b[2]] - t[b[1]]) <= gap_s:
                runs[k - 1:k + 2] = [[a[0], a[1], c[2]]]
                merged = True
                break
    return runs
