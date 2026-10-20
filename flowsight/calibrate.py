"""Per-camera calibration: >=4 image <-> floor-plan point pairs -> homography.

Interactive: python -m flowsight.calibrate --frame frame.jpg --floor-plan data/floor_plan.png
             --px-per-m 40 --out data/homography
             (click N points on the camera frame, then the same N points on the floor plan)
Scripted:    python -m flowsight.calibrate --points data/calibration_points.json --out data/homography
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from .geometry import fit_homography, reprojection_error, save_calibration


def _click(img: np.ndarray, title: str, n: int) -> list[list[float]]:
    pts: list[list[float]] = []
    vis = img.copy()

    def cb(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < n:
            pts.append([x, y])
            cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(vis, str(len(pts)), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, cb)
    while len(pts) < n:
        cv2.imshow(title, vis)
        if cv2.waitKey(30) == 27:
            raise SystemExit("cancelled")
    cv2.destroyWindow(title)
    return pts


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", help="JSON with image_points and floor_points_m (non-interactive)")
    ap.add_argument("--frame")
    ap.add_argument("--floor-plan")
    ap.add_argument("--px-per-m", type=float, default=40.0)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default="data/homography")
    a = ap.parse_args(argv)
    if a.points:
        d = json.load(open(a.points))
        img_pts, flr_pts, size = d["image_points"], d["floor_points_m"], d.get("image_size")
    else:
        frame, plan = cv2.imread(a.frame), cv2.imread(a.floor_plan)
        img_pts = _click(frame, "1) click points on the camera frame", a.n)
        flr_px = _click(plan, "2) click the SAME points on the floor plan", a.n)
        flr_pts = (np.asarray(flr_px) / a.px_per_m).tolist()
        size = [frame.shape[1], frame.shape[0]]
    H = fit_homography(img_pts, flr_pts)
    err = reprojection_error(H, img_pts, flr_pts)
    save_calibration(a.out, H, img_pts, flr_pts, size)
    print(f"saved {a.out}.npy | control-point residual: {err['floor_rms_m']:.3f} m RMS "
          f"({err['image_rms_px']:.2f} px) over {err['n_points']} points")


if __name__ == "__main__":
    main()
