"""Foot-traffic heat map on the floor plan: person-seconds on a 0.25 m grid, Gaussian smoothed."""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


class HeatGrid:
    def __init__(self, width_m: float, height_m: float, cell_m: float = 0.25):
        self.cell = cell_m
        self.nx, self.ny = int(np.ceil(width_m / cell_m)), int(np.ceil(height_m / cell_m))
        self.grid = np.zeros((self.ny, self.nx), dtype=np.float32)

    def add(self, x, y, dt: float) -> None:
        """Add `dt` person-seconds at each (x, y). Points off the plan are ignored."""
        x, y = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float))
        ix, iy = np.floor(x / self.cell).astype(int), np.floor(y / self.cell).astype(int)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        np.add.at(self.grid, (iy[ok], ix[ok]), dt)

    def decay(self, dt: float, tau_s: float) -> None:
        """Exponential forgetting for the live (last ~tau seconds) view."""
        self.grid *= np.exp(-dt / tau_s)

    def smoothed(self, sigma_cells: float = 2.0) -> np.ndarray:
        return gaussian_filter(self.grid, sigma_cells, mode="constant")

    def total_person_s(self) -> float:
        return float(self.grid.sum())


def heat_from_tracks(tracks, width_m, height_m, fps, cell_m=0.25) -> HeatGrid:
    h = HeatGrid(width_m, height_m, cell_m)
    h.add(tracks["x"].to_numpy(), tracks["y"].to_numpy(), 1.0 / fps)
    return h


def render_heat(heat: np.ndarray, base_bgr: np.ndarray, alpha: float = 0.65, vmax: float | None = None,
                floor_frac: float = 0.04) -> np.ndarray:
    """Overlay a smoothed grid on the floor-plan image (low values stay transparent)."""
    vmax = float(vmax if vmax else heat.max())
    if vmax <= 0:
        return base_bgr.copy()
    norm = np.clip(heat / vmax, 0, 1)
    big = cv2.resize(norm, (base_bgr.shape[1], base_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
    big = np.clip(big, 0, 1)
    colour = cv2.applyColorMap((big * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    a = (np.clip((big - floor_frac) / (1 - floor_frac), 0, 1) * alpha)[..., None]
    return (colour * a + base_bgr * (1 - a)).astype(np.uint8)
