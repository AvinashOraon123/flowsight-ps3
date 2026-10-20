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


# ------------------------------------------------- step 2: floor coordinates
def project_tracks(df_img: pd.DataFrame, H: np.ndarray, zones, min_track_s: float = 1.0) -> pd.DataFrame:
    """Foot points -> floor metres, Kalman-smoothed per track, with speed and zone label."""
    from .geometry import image_to_floor
    from .track import FloorSmoother

    df = df_img.sort_values(["track_id", "t"]).reset_index(drop=True)
    dur = df.groupby("track_id")["t"].transform(lambda s: s.max() - s.min())
    df = df[dur >= min_track_s].reset_index(drop=True)
    f = image_to_floor(H, df[["u", "v"]].to_numpy())
    df["fx"], df["fy"] = f[:, 0], f[:, 1]
    sm = FloorSmoother()
    out = np.zeros((len(df), 4))
    for i, (tid, t, x, y) in enumerate(zip(df["track_id"], df["t"], df["fx"], df["fy"])):
        out[i] = sm.update(int(tid), float(t), float(x), float(y))
    df["x"], df["y"], df["vx"], df["vy"] = out.T
    df["speed"] = np.hypot(df["vx"], df["vy"])
    df["zone"] = zones.locate(df["x"], df["y"])
    return df


# ------------------------------------------------------- steps 3-4: analytics
def run_analytics(cfg: dict, df_img: pd.DataFrame, out_dir: Path, fps: float, log=print) -> dict:
    import json

    from . import actions, bottleneck, heatmap, journeys, metrics
    from .geometry import load_homography
    from .zones import ZoneSet

    p, zones = cfg["analytics"], ZoneSet.load(rpath(cfg, "zones"))
    H = load_homography(rpath(cfg, "homography"))
    tr = project_tracks(df_img, H, zones, p.get("min_track_s", 1.0))
    tr = metrics.add_motion_state(tr, fps, p)
    events = metrics.zone_events(tr, fps, p)
    table = metrics.build_zone_table(tr, events, zones, p)

    t_end = float(tr["t"].max() + 1.0 / fps)
    ts = bottleneck.zone_timeseries(tr, zones, fps, 1.0, p["v_free_ms"], t_end=t_end)
    bn = bottleneck.flag_bottlenecks(ts, p["bottleneck_threshold"], p["bottleneck_sustain_s"])
    heat = heatmap.heat_from_tracks(tr, zones.width_m, zones.height_m, fps, p["heat_cell_m"])
    dz = bottleneck.dead_zones(table, zones, heat, p)

    peak = ts.groupby("zone").agg(peak_count=("count", "max"), peak_rho=("rho", "max"), peak_C=("C", "max"))
    bn_min = bn.groupby("zone")["duration_s"].sum() / 60.0
    table = table.merge(peak, left_on="zone", right_index=True, how="left")
    table["bottleneck_min"] = table["zone"].map(bn_min).fillna(0.0)
    table = table.merge(dz[["zone", "dead_zone", "adj_visit_rate"]], on="zone", how="left")
    table["dead_zone"] = table["dead_zone"].fillna(False).astype(bool)

    queues = {z.name: metrics.queue_series(ts, events, z.name) for z in zones.of_kind("queue")}
    for name, q in queues.items():
        i = table["zone"] == name
        table.loc[i, "peak_queue"] = q["queue_len"].max()
        table.loc[i, "max_est_wait_min"] = q["est_wait_min"].max()
    jt = journeys.journey_table(tr, events)
    al = actions.generate(ts, queues, bn, table, dz, t_end, cfg["actions"])

    tr.to_parquet(out_dir / "tracks.parquet")
    tr[["frame", "t", "track_id", "x", "y", "speed", "zone", "state"]].to_csv(out_dir / "tracks.csv", index=False)
    events.to_parquet(out_dir / "zone_events.parquet")
    table.to_parquet(out_dir / "zone_table.parquet")
    table.to_csv(out_dir / "zone_table.csv", index=False)
    ts.to_parquet(out_dir / "zone_timeseries.parquet")
    bn.to_parquet(out_dir / "bottlenecks.parquet")
    dz.to_parquet(out_dir / "dead_zones.parquet")
    jt.to_parquet(out_dir / "journeys.parquet")
    for name, q in queues.items():
        q.to_parquet(out_dir / f"queue_{name.replace(' ', '_').lower()}.parquet")
    np.save(out_dir / "heat_grid.npy", heat.grid)
    plan = cv2.imread(str(rpath(cfg, "floor_plan")))
    cv2.imwrite(str(out_dir / "heatmap.png"), heatmap.render_heat(heat.smoothed(p["heat_sigma_cells"]), plan))
    (out_dir / "alerts.json").write_text(json.dumps([a.asdict() for a in al], indent=2))
    summary = dict(duration_s=t_end, fps=fps, n_tracks=int(tr["track_id"].nunique()), n_samples=int(len(tr)),
                   n_visits=int(events["is_visit"].sum()), n_bottlenecks=int(len(bn)),
                   n_dead_zones=int(table["dead_zone"].sum()), n_alerts=len(al),
                   n_stops_engaged=int(tr.loc[tr.state == "engaged", "stop_id"].nunique()),
                   n_stops_blocked=int(tr.loc[tr.state == "blocked", "stop_id"].nunique()),
                   bottleneck_threshold=p["bottleneck_threshold"])
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"analytics: {summary}")
    return dict(tracks=tr, events=events, table=table, ts=ts, bn=bn, dz=dz, alerts=al, queues=queues,
                journeys=jt, heat=heat, zones=zones, summary=summary)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/demo.yaml")
    ap.add_argument("--reuse", action="store_true", help="reuse outputs/tracks_img.parquet (skip detection)")
    ap.add_argument("--no-live", action="store_true", help="skip rendering the side-by-side live.mp4")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    out = rpath(cfg, cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if a.reuse and (out / "tracks_img.parquet").exists():
        df = pd.read_parquet(out / "tracks_img.parquet")
        meta = pd.read_json(out / "vision_meta.json", typ="series")
    else:
        df = run_vision(cfg, out)
        meta = pd.read_json(out / "vision_meta.json", typ="series")
    res = run_analytics(cfg, df, out, float(meta["fps"]))
    if not a.no_live:
        from .live import render_live
        render_live(cfg, out, res, rpath, float(meta["fps"]))


if __name__ == "__main__":
    main()
