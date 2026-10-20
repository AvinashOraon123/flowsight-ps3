# FlowSight: Dwell-Time & Bottleneck Mapper

Everseen Computer Vision AI Hackathon · E-Cell IIT Bombay · Problem Statement 3 · team YOU_ONLY_LIVE_ONCE

**FlowSight turns a store's existing CCTV into a live map of where shoppers go, linger, get stuck and which zones they ignore, and tells staff what to do within seconds.** It detects and tracks people, projects them onto the floor plan, and computes journeys, heat maps, per-zone dwell time, bottlenecks and dead zones. A rule engine turns those into ranked actions, each with the reason it fired.

What makes it different:
1. **Engaged vs blocked dwell.** Standing still because you are interested (free space, nothing in the way) is told apart from standing still because you are stuck (crowd around you or someone in front). The bottleneck score uses walking speed only, so a crowd of browsers is not a jam.
2. **Bottlenecks forecast 5 and 10 minutes ahead** (LightGBM), not just reported.
3. **Every output is a ranked action with a reason**, addressed to the person who can act.

## What is in this repo (status, honestly)

| PS3 output | Where | Runs on |
|---|---|---|
| Customer journey maps | Journeys tab, `journeys.py` | real clip + simulated store |
| Real-time foot-traffic heat map | Live tab (`live.mp4`), `heatmap.py`, `online.py` | real clip + simulated store |
| Section-by-section dwell time | Zones tab, `metrics.py` | real clip + simulated store |
| Bottleneck and dead-zone flags | Zones/Live tabs, `bottleneck.py`, `actions.py` | **simulated store** (the real clip is an uncrowded street and correctly raises no bottleneck) |

Measured results versus the proposal's targets are in **[METRICS.md](METRICS.md)**. In short, on the real clip detection AP@0.5 is 0.66 (target 0.85), IDF1 0.59 (0.70) and dwell error about 30% (10%), using a stock YOLOv8n with no fine-tuning. Floor-position error was **not measured** (no surveyed ground truth). Jam detection is F1 0.65 on simulated days (target 0.80). Targets are goals, not results.

## Architecture

```
IN STORE (edge box: Jetson / small GPU)
  [CCTV cameras] -> [Detect people] -> [Track people]    -> [Map to floor]
   existing feeds    YOLOv8, foot pt   ByteTrack+Kalman     homography (metres)
   RTSP 10-15 FPS    no faces stored   one ID per person    (cross-camera re-ID: not built)
                                                                  |
                     anonymous floor coordinates only, no video leaves the store
                                                                  v
CLOUD (analytics and decisions)
  [Action engine] <- [Bottleneck model] <- [Zone analytics]  <- [Event stream]
   rules + ranking    congestion score     heat map, dwell      (here: parquet / FastAPI WS;
   alerts w/ reasons  5-10 min forecast    engaged vs blocked    proposal: Redis + TimescaleDB)
        |                                       |        \
        v                                       v         v
PEOPLE WHO ACT
  [Floor staff app]        [Manager dashboard]        [HQ / category teams]
   push alerts in seconds   live heat map, zone dwell  weekly zone scorecards
   open till, clear aisle   daily staffing report      layout A/B (compare_layouts)
```

Core formula: `C_z(t) = (rho_z / rho_max) * (1 - v_z / v_free)`, with `rho` people per m², `v` mean *walking* speed (people stopped by choice excluded), `v_free` = 1.2 m/s, `rho_max = capacity / area` per zone. A zone is flagged when C stays above a threshold for the sustain window (60 s in production, 8 s on the 35 s demo clip, 120 s in the minute-resolution simulation).

## Setup

Python 3.11. A CPU is enough (about 12 FPS per stream at 960 px on a laptop).

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt                          # CPU users: install torch/torchvision from https://download.pytorch.org/whl/cpu first
python data/download_data.py                             # public MOT17-04 clip + labels (about 15 MB)
python data/make_floor_plan.py
python -m flowsight.calibrate --points data/calibration_points.json --out data/homography
```

## Run

```bash
python -m flowsight.pipeline --config configs/demo.yaml        # video -> tracks -> analytics -> outputs/clip (+ live.mp4); about 1.5 min on CPU
python -m flowsight.simulate --config configs/sim.yaml         # simulated store days, forecaster, outputs/sim; about 1.5 min
python -m eval.run_eval                                        # measures everything -> METRICS.md, outputs/metrics.json
streamlit run app.py                                           # dashboard: Live, Zones, Journeys, Forecast, Metrics
uvicorn flowsight.api:app --port 8000                          # API: /metrics /alerts /heatmap and WS /live  (docs at /docs)
python -m flowsight.pipeline --config configs/demo.yaml --live # streaming mode: alerts print as they fire (RTSP: --source rtsp://...)
python -m pytest -q                                            # 60 tests
```

Interactive calibration for your own camera: `python -m flowsight.calibrate --frame frame.jpg --floor-plan floor_plan.png --px-per-m 40` (click 4+ points on the frame, then the same points on the plan).

Docker (CPU): `docker build -t flowsight . && docker run --rm -p 8501:8501 flowsight`. **The Dockerfile has not been built or run** (Docker was not running when it was written).

## Data and sources

| Data | Source | Licence | Use |
|---|---|---|---|
| `MOT17-04` video (960x540, 35 s) and `gt.txt` | MOTChallenge, https://motchallenge.net | CC BY-NC-SA 3.0, research / non-commercial | demo clip, detection and tracking ground truth |
| Simulated store days | `flowsight/simulate.py` | ours | jams, queue, dead zone, forecaster training and testing |

The MOT17-04 scene is a night-time shopping street seen from above, **not a store**. The floor plan, the 4 calibration points (assumed 16 m x 19 m) and the 7 zones are **illustrative**: zones were drawn after looking at the heat map so each output has something to show. A real deployment measures floor points and draws zones from the store plan.

## Privacy by design

- Edge processing; the API exposes anonymous floor coordinates, zone counts and alerts. It has no video or image endpoint (tested).
- No face recognition, no demographics. Saved debug video blurs the head region of every person box (top 20 %).
- Track IDs are per-session integers; nothing links a person across days.
- Only zone aggregates are shown; the proposal's rule of not reporting groups under 5 people is **not implemented** in this prototype.
- Framed for India's DPDP Act 2023 and GDPR; signage and a 30-day retention policy are deployment tasks, not code here.

## Known limitations

- One 35 s public clip; no real store footage; zones illustrative; floor accuracy unmeasured.
- Stock COCO YOLOv8n, not fine-tuned. Tracking breaks (86 ID switches for 79 people) shorten measured dwell.
- Cross-camera stitching and appearance re-ID (OSNet) are **not built**. Single camera only.
- Staff are not excluded: a doorman standing still shows up as an engaged shopper.
- Jam, queue and dead-zone results, and the forecaster, are validated on **simulated** days with exact positions. They test the logic, not the camera.
- The forecaster beats a no-change baseline at 5 minutes but not at 10; jam start times are random in the simulator.
- The POS-based "looked but did not buy" rule runs only on simulated sales data.
- Throughput was measured for one CPU stream only; 8 streams per edge GPU is untested.
- `OnlineEngine` (live) and the batch analytics use the same rules but are separate code paths; counts agree closely, not exactly.

## Layout

```
flowsight/   detect, track, calibrate, geometry, zones, metrics, heatmap, journeys, bottleneck, forecast,
             actions, simulate, pipeline, online, api, live, render
eval/        mot_eval (AP, CLEAR-MOT, IDF1, dwell vs ground truth), sim_eval, run_eval
tests/       60 tests (geometry, zones, dwell, engaged/blocked, congestion, actions, simulator, forecaster, eval metrics, API)
configs/     demo.yaml (real clip), sim.yaml (simulated store)
```
