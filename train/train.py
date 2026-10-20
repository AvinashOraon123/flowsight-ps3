"""Fine-tune YOLOv8 (person only) on MOT17 clips that are not used for evaluation.

    python train/prepare_dataset.py
    python train/train.py --epochs 15            # CPU: slow; on a GPU pass --device 0 and a larger --model
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--name", default="flowsight_n")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if a.resume:
        YOLO(str(ROOT / "outputs/train" / a.name / "weights/last.pt")).train(resume=True)
    else:
        YOLO(a.model).train(data=str(ROOT / "train/dataset/dataset.yaml"), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                            device=a.device, workers=2, project=str(ROOT / "outputs/train"), name=a.name,
                            patience=0, cos_lr=True, close_mosaic=3, plots=False, exist_ok=True, seed=0)
