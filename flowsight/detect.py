"""Person detection (Ultralytics YOLOv8, COCO class 0 only)."""
from __future__ import annotations

import numpy as np
import supervision as sv
from ultralytics import YOLO


def foot_point(xyxy: np.ndarray) -> np.ndarray:
    """Bottom-centre of each box: where the person touches the floor. (N,4)->(N,2)."""
    xyxy = np.asarray(xyxy, dtype=float).reshape(-1, 4)
    return np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2.0, xyxy[:, 3]], axis=1)


class PersonDetector:
    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.25, imgsz: int = 960,
                 device: str | None = None):
        self.model = YOLO(weights)
        self.conf, self.imgsz, self.device = conf, imgsz, device

    def __call__(self, frame: np.ndarray) -> sv.Detections:
        res = self.model.predict(frame, classes=[0], conf=self.conf, imgsz=self.imgsz,
                                 device=self.device, verbose=False)[0]
        return sv.Detections.from_ultralytics(res)
