"""Side-by-side live view: camera (boxes, IDs, blurred heads) | floor-plan heat map + zone congestion | alert feed.

Rendered offline from the saved tracks so the dashboard can play it in any browser (H.264).
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2
import numpy as np

from .heatmap import HeatGrid, render_heat
from .render import blur_heads, track_color

STATE_BGR = {"moving": (90, 200, 90), "engaged": (0, 190, 255), "blocked": (60, 60, 255)}
KIND_BGR = {"queue": (80, 80, 255), "bottleneck": (60, 120, 255), "forecast": (0, 190, 255), "endcap": (255, 180, 80),
            "dead_zone": (200, 200, 200), "engaged_no_sale": (180, 120, 255)}


def _congestion_colour(c: float, thr: float) -> tuple[int, int, int]:
    return (60, 60, 255) if c >= thr else (0, 170, 255) if c >= 0.5 * thr else (90, 190, 90)


def _text(img, s, org, scale=0.5, col=(240, 240, 240), th=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)


def render_live(cfg: dict, out_dir: Path, res: dict, rpath, fps: float, tau_s: float = 12.0, log=print) -> Path:
    import imageio_ffmpeg

    p = cfg["analytics"]
    tr, zones, ts, alerts = res["tracks"], res["zones"], res["ts"], res["alerts"]
    plan = cv2.imread(str(rpath(cfg, "floor_plan").with_name("floor_plan_blank.png")))
    cap = cv2.VideoCapture(str(rpath(cfg, "video")))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = int(cfg["vision"].get("frame_stride", 1))
    W, Hh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pw = int(plan.shape[1] * Hh / plan.shape[0])
    s_px = Hh / plan.shape[0]                              # floor-plan px -> panel px
    strip = 170
    out_w, out_h = W + pw, Hh + strip
    out_w += out_w % 2
    out_h += out_h % 2
    path = out_dir / "live.mp4"
    writer = imageio_ffmpeg.write_frames(str(path), (out_w, out_h), fps=fps, codec="libx264", pix_fmt_in="bgr24",
                                         pix_fmt_out="yuv420p", macro_block_size=2,
                                         output_params=["-crf", "27", "-preset", "veryfast"])
    writer.send(None)
    heat = HeatGrid(zones.width_m, zones.height_m, p["heat_cell_m"])
    by_frame = {f: g for f, g in tr.groupby("frame")}
    c_lookup = ts.pivot(index="t", columns="zone", values="C")
    n_lookup = ts.pivot(index="t", columns="zone", values="count")
    trails: dict[int, deque] = {}
    al_sorted = sorted(alerts, key=lambda a: a.t)
    sc = zones.px_per_m * s_px
    zone_px = {z.name: (np.array(z.poly.exterior.coords) * sc).astype(np.int32) for z in zones.zones}
    fi, k = -1, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        if fi % stride:
            continue
        t = fi / src_fps
        g = by_frame.get(fi)
        if g is not None:
            frame = blur_heads(frame, g[["x1", "y1", "x2", "y2"]].to_numpy())
            heat.decay(1.0 / fps, tau_s)
            heat.add(g["x"].to_numpy(), g["y"].to_numpy(), 1.0 / fps)
            for r in g.itertuples():
                col = STATE_BGR[r.state]
                cv2.rectangle(frame, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), col, 2)
                lab = f"#{r.track_id}" + (f" {r.state}" if r.state != "moving" else "")
                _text(frame, lab, (int(r.x1), max(int(r.y1) - 4, 12)), 0.42, col)
                trails.setdefault(r.track_id, deque(maxlen=60)).append((r.x, r.y))
        else:
            heat.decay(1.0 / fps, tau_s)
        # ---- floor panel
        panel = cv2.resize(render_heat(heat.smoothed(p["heat_sigma_cells"]), plan, vmax=0.5), (pw, Hh))
        sec = int(t)
        for z in zones.zones:
            c = float(c_lookup.at[sec, z.name]) if sec in c_lookup.index else 0.0
            n = float(n_lookup.at[sec, z.name]) if sec in n_lookup.index else 0.0
            cv2.polylines(panel, [zone_px[z.name]], True, _congestion_colour(c, p["bottleneck_threshold"]), 2, cv2.LINE_AA)
            cx, cy = zone_px[z.name].mean(0).astype(int)
            _text(panel, f"{z.name}", (int(cx - 35), int(cy - 6)), 0.38)
            _text(panel, f"n={n:.0f}  C={c:.2f}", (int(cx - 35), int(cy + 8)), 0.38,
                  _congestion_colour(c, p["bottleneck_threshold"]))
        if g is not None:
            for tid in g["track_id"]:
                pts = np.array(trails[tid]) * sc
                if len(pts) > 1:
                    cv2.polylines(panel, [pts.astype(np.int32)], False, track_color(int(tid)), 1, cv2.LINE_AA)
            for x, y, st in zip(g["x"], g["y"], g["state"]):
                cv2.circle(panel, (int(x * sc), int(y * sc)), 4, STATE_BGR[st], -1, cv2.LINE_AA)
        # ---- alert strip
        bar = np.full((strip, W + pw, 3), 28, np.uint8)
        _text(bar, f"FlowSight   t = {t:5.1f} s   |   green=moving  amber=engaged stop  red=blocked stop",
              (10, 22), 0.55, (255, 255, 255))
        live = [a for a in al_sorted if a.t <= t][-4:]
        for i, a in enumerate(live[::-1]):
            line = f"[{a.t:5.1f}s] {a.kind.upper()} ({a.audience}): {a.action}  -  {a.reason}"
            _text(bar, line[:215], (10, 50 + i * 28), 0.46, KIND_BGR.get(a.kind, (255, 255, 255)))
        if not live:
            _text(bar, "no alerts yet", (10, 50), 0.46, (160, 160, 160))
        canvas = np.full((out_h, out_w, 3), 28, np.uint8)
        canvas[:Hh, :W] = frame
        canvas[:Hh, W:W + pw] = panel
        canvas[Hh:Hh + strip, :W + pw] = bar
        writer.send(np.ascontiguousarray(canvas))
        k += 1
    writer.close()
    cap.release()
    log(f"live view: {k} frames -> {path}")
    return path
