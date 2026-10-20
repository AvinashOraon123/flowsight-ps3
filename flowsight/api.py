"""FlowSight API.

    uvicorn flowsight.api:app --port 8000            # serves outputs/clip; FLOWSIGHT_CONFIG / FLOWSIGHT_OUT override

GET  /health                 liveness
GET  /metrics                zone table + summary (what the manager dashboard shows)
GET  /alerts?until=<t>       ranked actions, optionally only those fired by time t
GET  /heatmap                cumulative heat map PNG
GET  /heatmap.json           smoothed grid (person-seconds per 0.25 m cell) for custom rendering
WS   /live?speed=1&max_frames=N
                             replays the recorded stream through the streaming OnlineEngine at `speed` x real time
                             (speed=0 = as fast as possible). One message per analysed frame:
                             {t, zones:{name:{count,speed,C,flag}}, people:[{id,x,y,state,zone}], events:[alert...]}
Only anonymous floor coordinates are exposed; there is no video or image endpoint.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .geometry import load_homography
from .heatmap import HeatGrid
from .online import OnlineEngine
from .pipeline import load_config, rpath
from .privacy import suppress_small_groups
from .zones import ZoneSet

ROOT = Path(__file__).resolve().parent.parent


def create_app(config: str | None = None, out_dir: str | None = None) -> FastAPI:
    cfg = load_config(config or os.environ.get("FLOWSIGHT_CONFIG", str(ROOT / "configs/demo.yaml")))
    out = Path(out_dir or os.environ.get("FLOWSIGHT_OUT", rpath(cfg, cfg["out_dir"])))
    app = FastAPI(title="FlowSight", version="1.0", description="Dwell-time & bottleneck mapper (anonymous coordinates only)")

    def need(name: str) -> Path:
        f = out / name
        if not f.exists():
            raise HTTPException(503, f"{name} not generated yet; run `python -m flowsight.pipeline --config ...`")
        return f

    @app.get("/health")
    def health():
        return {"status": "ok", "outputs": str(out), "ready": (out / "zone_table.parquet").exists()}

    @app.get("/metrics")
    def metrics():
        t = suppress_small_groups(pd.read_parquet(need("zone_table.parquet")))      # <5 visitors: no per-zone figures
        summ = json.loads(need("summary.json").read_text())
        return {"summary": summ, "zones": json.loads(t.to_json(orient="records"))}

    @app.get("/alerts")
    def alerts(until: float | None = Query(None, description="only alerts that fired at or before this time (s)"),
               kind: str | None = None):
        a = json.loads(need("alerts.json").read_text())
        if until is not None:
            a = [x for x in a if x["t"] <= until]
        if kind:
            a = [x for x in a if x["kind"] == kind]
        return sorted(a, key=lambda x: -x["impact"])

    @app.get("/heatmap")
    def heatmap_png():
        return FileResponse(need("heatmap.png"), media_type="image/png")

    @app.get("/heatmap.json")
    def heatmap_json(smoothed: bool = True):
        g = np.load(need("heat_grid.npy"))
        h = HeatGrid(g.shape[1] * 0.25, g.shape[0] * 0.25, 0.25)
        h.grid = g
        arr = h.smoothed(cfg["analytics"]["heat_sigma_cells"]) if smoothed else g
        return {"cell_m": 0.25, "shape": list(arr.shape), "person_seconds": np.round(arr, 3).tolist()}

    @app.websocket("/live")
    async def live(ws: WebSocket, speed: float = 1.0, max_frames: int | None = None):
        await ws.accept()
        try:
            ti = pd.read_parquet(need("tracks_img.parquet"))
            meta = json.loads(need("vision_meta.json").read_text())
        except HTTPException as e:
            await ws.send_json({"error": e.detail})
            await ws.close()
            return
        zones = ZoneSet.load(rpath(cfg, "zones"))
        eng = OnlineEngine(zones, load_homography(rpath(cfg, "homography")), float(meta["fps"]), cfg["analytics"], cfg["actions"])
        frame_dt = 1.0 / float(meta["fps"])
        n = 0
        try:
            for fi, g in ti.groupby("frame"):
                r = eng.update(float(g["t"].iloc[0]), g["track_id"].to_numpy(), g[["u", "v"]].to_numpy())
                await ws.send_json({"t": r["t"], "zones": r["zones"], "people": r["people"],
                                    "events": [a.asdict() for a in r["events"]]})
                n += 1
                if max_frames and n >= max_frames:
                    break
                if speed > 0:
                    await asyncio.sleep(frame_dt / speed)
            await ws.send_json({"done": True, "frames": n})
            await ws.close()
        except WebSocketDisconnect:
            return

    return app


app = create_app()
