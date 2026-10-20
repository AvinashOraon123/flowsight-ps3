"""Multi-object tracking (ByteTrack via supervision) + Kalman smoothing on floor coords."""
from __future__ import annotations

import numpy as np
import supervision as sv


class Tracker:
    def __init__(self, fps: float, track_thresh: float = 0.25, lost_buffer_s: float = 2.0,
                 match_thresh: float = 0.8):
        self.bt = sv.ByteTrack(track_activation_threshold=track_thresh,
                               lost_track_buffer=int(lost_buffer_s * fps),
                               minimum_matching_threshold=match_thresh, frame_rate=int(round(fps)))

    def update(self, detections: sv.Detections) -> sv.Detections:
        return self.bt.update_with_detections(detections)


class FloorKalman:
    """Constant-velocity Kalman filter for one track on the floor plane (metres).

    State [x, y, vx, vy]. Gives smoothed position and a speed that is stable enough
    to threshold at 0.2 m/s for stop detection.
    """

    def __init__(self, x: float, y: float, q: float = 0.5, r: float = 0.35):
        self.s = np.array([x, y, 0.0, 0.0])
        self.P = np.diag([r ** 2, r ** 2, 1.0, 1.0])
        self.q, self.r2 = q, r ** 2
        self.t_last: float | None = None

    def update(self, t: float, x: float, y: float) -> tuple[float, float, float, float]:
        if self.t_last is None:
            self.t_last = t
            return x, y, 0.0, 0.0
        dt = max(t - self.t_last, 1e-3)
        self.t_last = t
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
        Q = G @ G.T * self.q ** 2
        self.s = F @ self.s
        self.P = F @ self.P @ F.T + Q
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0.0]])
        S = H @ self.P @ H.T + np.eye(2) * self.r2
        K = self.P @ H.T @ np.linalg.inv(S)
        self.s = self.s + K @ (np.array([x, y]) - H @ self.s)
        self.P = (np.eye(4) - K @ H) @ self.P
        return float(self.s[0]), float(self.s[1]), float(self.s[2]), float(self.s[3])


class FloorSmoother:
    """One FloorKalman per track id."""

    def __init__(self, **kw):
        self.kw, self.f = kw, {}

    def update(self, tid: int, t: float, x: float, y: float):
        if tid not in self.f:
            self.f[tid] = FloorKalman(x, y, **self.kw)
        return self.f[tid].update(t, x, y)
