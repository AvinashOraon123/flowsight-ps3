"""Drawing helpers. Privacy: the head region of every person box is blurred before anything is saved."""
from __future__ import annotations

import colorsys

import cv2
import numpy as np


def blur_heads(frame: np.ndarray, xyxy: np.ndarray, frac: float = 0.20) -> np.ndarray:
    """Pixelate+blur the top `frac` of each person box (faces never leave the edge box)."""
    h, w = frame.shape[:2]
    for x1, y1, x2, y2 in np.asarray(xyxy).reshape(-1, 4):
        hh = max(int((y2 - y1) * frac), 4)
        xa, xb = int(max(x1, 0)), int(min(x2, w))
        ya, yb = int(max(y1, 0)), int(min(y1 + hh, h))
        if xb - xa < 2 or yb - ya < 2:
            continue
        roi = frame[ya:yb, xa:xb]
        small = cv2.resize(roi, (max((xb - xa) // 6, 1), max((yb - ya) // 6, 1)), interpolation=cv2.INTER_LINEAR)
        roi = cv2.resize(small, (xb - xa, yb - ya), interpolation=cv2.INTER_NEAREST)
        frame[ya:yb, xa:xb] = cv2.GaussianBlur(roi, (0, 0), 3)
    return frame


def track_color(tid: int) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb((tid * 0.61803) % 1.0, 0.75, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def draw_tracks(frame: np.ndarray, ids, xyxy, labels=None, blur: bool = True) -> np.ndarray:
    out = frame.copy()
    if blur and len(xyxy):
        out = blur_heads(out, xyxy)
    for i, (tid, (x1, y1, x2, y2)) in enumerate(zip(ids, np.asarray(xyxy).reshape(-1, 4))):
        c = track_color(int(tid))
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
        txt = f"#{int(tid)}" + (f" {labels[i]}" if labels is not None and labels[i] else "")
        cv2.putText(out, txt, (int(x1), max(int(y1) - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    return out
