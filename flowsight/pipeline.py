"""FlowSight pipeline: video -> tracks -> floor coordinates -> zone analytics -> alerts.

    python -m flowsight.pipeline --config configs/demo.yaml            # everything
    python -m flowsight.pipeline --config configs/demo.yaml --reuse    # skip detection, reuse tracks_img.parquet
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml


def load_config(path) -> dict:
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    cfg["_root"] = str(Path(path).resolve().parent.parent)
    return cfg


def rpath(cfg: dict, key_or_path: str) -> Path:
    p = Path(cfg.get(key_or_path, key_or_path))
    return p if p.is_absolute() else Path(cfg["_root"]) / p


# ---------------------------------------------------------------- step 1: vision
def run_vision(cfg: dict, out_dir: Path, log=print) -> pd.DataFrame:
    """Detect + track every `frame_stride`-th frame. Writes tracks_img.{parquet,csv} and annotated.mp4."""
    import imageio_ffmpeg

    from .detect import PersonDetector, foot_point
    from .render import draw_tracks
    from .track import Tracker

    v = cfg["vision"]
    cap = cv2.VideoCapture(str(rpath(cfg, "video")))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = int(v.get("frame_stride", 1))
    fps = src_fps / stride
    W, Hh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    det = PersonDetector(v["weights"], v["conf"], v["imgsz"], v.get("device"))
    trk = Tracker(fps)
    writer = imageio_ffmpeg.write_frames(str(out_dir / "annotated.mp4"), (W, Hh), fps=fps, codec="libx264",
                                         pix_fmt_in="bgr24", pix_fmt_out="yuv420p", macro_block_size=2,
                                         output_params=["-crf", "26", "-preset", "veryfast"])
    writer.send(None)
    rows, lat = [], []
    fi, n_done = -1, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        if fi % stride:
            continue
        if v.get("max_frames") and n_done >= v["max_frames"]:
            break
        t0 = time.perf_counter()
        d = trk.update(det(frame))
        lat.append(time.perf_counter() - t0)
        t = fi / src_fps
        if len(d):
            fp = foot_point(d.xyxy)
            for tid, b, c, p in zip(d.tracker_id, d.xyxy, d.confidence, fp):
                rows.append((fi, t, int(tid), *map(float, b), float(c), float(p[0]), float(p[1])))
        writer.send(np.ascontiguousarray(draw_tracks(frame, d.tracker_id, d.xyxy)))
        n_done += 1
        if n_done % 50 == 0:
            log(f"  frame {fi} ({n_done} analysed) {1 / np.mean(lat[-50:]):.1f} FPS")
    writer.close()
    cap.release()
    df = pd.DataFrame(rows, columns=["frame", "t", "track_id", "x1", "y1", "x2", "y2", "conf", "u", "v"])
    df.attrs.update(fps=fps, src_fps=src_fps)
    df.to_parquet(out_dir / "tracks_img.parquet")
    df.to_csv(out_dir / "tracks_img.csv", index=False)
    pd.Series({"fps": fps, "src_fps": src_fps, "n_frames": n_done, "width": W, "height": Hh,
               "mean_latency_s": float(np.mean(lat)), "p95_latency_s": float(np.percentile(lat, 95)),
               "detect_track_fps": float(1 / np.mean(lat))}).to_json(out_dir / "vision_meta.json")
    log(f"vision: {n_done} frames, {df.track_id.nunique()} tracks, {1 / np.mean(lat):.1f} FPS detect+track")
    return df


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/demo.yaml")
    ap.add_argument("--reuse", action="store_true", help="reuse outputs/tracks_img.parquet")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    out = rpath(cfg, cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if a.reuse and (out / "tracks_img.parquet").exists():
        pd.read_parquet(out / "tracks_img.parquet")
    else:
        run_vision(cfg, out)


if __name__ == "__main__":
    main()
