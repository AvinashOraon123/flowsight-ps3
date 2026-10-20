"""Image <-> floor-plan projection via a per-camera homography."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def fit_homography(image_pts, floor_pts) -> np.ndarray:
    """H mapping image pixels to floor metres. >=4 pairs; RANSAC only when >4."""
    src = np.asarray(image_pts, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(floor_pts, dtype=np.float64).reshape(-1, 2)
    if len(src) < 4 or len(src) != len(dst):
        raise ValueError("need >= 4 matching image/floor point pairs")
    H, _ = cv2.findHomography(src, dst, method=0 if len(src) == 4 else cv2.RANSAC,
                              ransacReprojThreshold=0.3)
    if H is None:
        raise ValueError("degenerate point set (collinear?)")
    return H


def image_to_floor(H: np.ndarray, pts) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return pts
    return cv2.perspectiveTransform(pts[None], H)[0]


def floor_to_image(H: np.ndarray, pts) -> np.ndarray:
    return image_to_floor(np.linalg.inv(H), pts)


def reprojection_error(H: np.ndarray, image_pts, floor_pts) -> dict:
    """Residual of the control points, in metres on the floor and in pixels in the image."""
    f = image_to_floor(H, image_pts)
    fl = np.asarray(floor_pts, dtype=float).reshape(-1, 2)
    m = np.linalg.norm(f - fl, axis=1)
    back = floor_to_image(H, fl)
    px = np.linalg.norm(back - np.asarray(image_pts, dtype=float).reshape(-1, 2), axis=1)
    return {"floor_rms_m": float(np.sqrt((m ** 2).mean())), "floor_max_m": float(m.max()),
            "image_rms_px": float(np.sqrt((px ** 2).mean())), "n_points": int(len(m))}


def save_calibration(path, H, image_pts, floor_pts, image_size=None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p.with_suffix(".npy"), H)
    meta = {"image_points": np.asarray(image_pts).tolist(),
            "floor_points_m": np.asarray(floor_pts).tolist(), "image_size": image_size}
    p.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def load_homography(path) -> np.ndarray:
    return np.load(path)
