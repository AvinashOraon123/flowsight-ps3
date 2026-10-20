# FlowSight: measured vs target

_Generated 2026-10-20 11:55 by `python -m eval.run_eval`._

**How to read this.** The proposal's KPIs are *targets*. Every row below is something we actually ran, with the data it ran on. Rows marked *simulated* use the agent-based store simulator: they test the analytics, not the camera. Rows marked *proxy* measure something close to the target, not the target itself. A target we could not measure says *not measured*.

**Data.** *MOT17-04 clip*: public 35 s CCTV-style pedestrian video (MOTChallenge, CC BY-NC-SA 3.0), hand-annotated ground truth; the zones drawn on it are illustrative, it is a street, not a store. *SIMULATED*: `flowsight/simulate.py` store days with planted jams, queues and a dead zone; evaluation days (seeds 200-204) are never used for training or tuning.

## Headline table

| Metric | Target | Result | Status | Data | |
|---|---|---|---|---|---|
| Person detection AP@0.5 | ≥ 0.85 | **0.664** | measured | MOT17-04 clip (35 s) | ❌ not met |
| IDF1 | ≥ 0.70 | **0.592** | measured | MOT17-04 clip (35 s) | ❌ not met |
| Floor position error | ≤ 0.5 m | **not measured** | not measured | MOT17-04 clip (35 s) | - |
| Dwell-time error (mean per zone) | ≤ 10 % | **30.0%** | measured (proxy) | MOT17-04 clip (35 s) | ❌ not met |
| Bottleneck detection F1 | ≥ 0.80 | **0.654** | measured (simulated) | SIMULATED (5 unseen store days) | ❌ not met |
| Alert compute latency (p95) | ≤ 5 s | **0.09 s** | measured | MOT17-04 clip (35 s) | ✅ met |
| Throughput per stream | ≥ 10 FPS × 8 streams / edge GPU | **12.2 FPS** | measured (proxy) | MOT17-04 clip (35 s) | - |
| Forecast MAE, 5 min ahead | (beat naive) | **0.013** | measured (simulated) | SIMULATED, 3 unseen days | ✅ met |
| Forecast congestion-flag F1, 5 min ahead | (beat naive) | **0.500** | measured (simulated) | SIMULATED, 3 unseen days | ✅ met |
| Forecast congestion-flag F1, 10 min ahead | (beat naive) | **0.343** | measured (simulated) | SIMULATED, 3 unseen days | ❌ not met |
| Dead-zone detection (planted corner found) | - | **100%** | measured (simulated) | SIMULATED, 3 unseen days | - |
| Blocked-stop detection F1 | - | **0.541** | measured (simulated) | SIMULATED day (aisles only) | - |
| Pass-by-to-stop ratio error (Promo endcap) | - | **1.5%** | measured (simulated) | SIMULATED day | - |

## Notes per metric

- **Person detection AP@0.5** - Pretrained YOLOv8n (COCO), NOT fine-tuned. Ground-truth boxes with visibility ≥ 25 %. At conf ≥ 0.25: precision 0.87, recall 0.58. The target is for a model fine-tuned on retail/crowd data (CrowdHuman, MOT17/20).
- **IDF1** - ByteTrack on the same detections. MOTA 0.47, ID switches 86, 147 tracks vs 79 ground-truth people.
- **Floor position error** - No surveyed floor positions exist for this clip, and the calibration points are assumed (control-point residual is 0 by construction, which says nothing about accuracy). Needs a store with measured floor marks.
- **Dwell-time error (mean per zone)** - Our tracks vs the MOT17 ground-truth tracks, both pushed through the same projection and zone logic, zones with ≥ 3 ground-truth visits (6 zones; median 36%; total person-seconds error 48%). No hand-timed dwell exists for this clip, so this isolates tracker error only.
- **Bottleneck detection F1** - Minute-level, all 8 zones (flags in zones that cannot jam count as false positives). Precision 0.49, recall 1.00. Only the 4 zones that can jam: F1 0.65. Jam episodes found: 12/12, median alert delay 2.0 min after a jam starts. Ground truth = people truly held up, from the simulator: at an aisle gate those waiting plus the one being squeezed past the stack (≥ 3), at the checkout lane those waiting (≥ 5). Counting only people waiting (aisle ≥ 3) the same flags score F1 0.41 (precision 0.26). History: the first run scored F1 0.32; diagnosis on separate training days showed (1) browsers standing by choice were flagged as a jam, so the congestion speed term now uses walking speed only (`zone_timeseries` in bottleneck.py), and (2) many false positives were real jam minutes the waiting-only truth missed, so the truth was widened to people held up. Both changes were made before the final run on these evaluation days, but after a first look at them, so treat 0.65 as an optimistic estimate.
- **Alert compute latency (p95)** - Per frame on a laptop CPU: detect + track 87 ms + projection/Kalman/zone lookup 1.9 ms. Excludes network and the deliberate 60 s sustain rule.
- **Throughput per stream** - 12.2 FPS for ONE stream, YOLOv8n at 960 px, GPU: none (CPU only). The 8-streams-per-edge-GPU claim was not tested.
- **Forecast MAE, 5 min ahead** - Naive (no change) 0.015.
- **Forecast congestion-flag F1, 5 min ahead** - Naive 0.450. Warned ahead of 42% of congestion onsets (naive 6%).
- **Forecast congestion-flag F1, 10 min ahead** - Naive 0.380. Warned ahead of 19% of onsets (naive 13%). Jam start times are random in the simulator, so onsets are hard to predict.
- **Dead-zone detection (planted corner found)** - False-positive dead zones across the days: 0.
- **Blocked-stop detection F1** - Precision 0.78, recall 0.41 against the simulator's intended labels; 98% of true stop samples were detected as stops. The queue lane is excluded.
- **Pass-by-to-stop ratio error (Promo endcap)** - Measured 9.8% vs generator 11.3% (the simulator counts only people it routed past the endcap, the detector counts everyone within 2 m).

## Dwell per zone: our tracker vs ground-truth tracks (MOT17-04)

| zone | gt_visits | our_visits | gt_mean_dwell_s | our_mean_dwell_s | gt_person_s | our_person_s | mean_dwell_err | person_s_err |
|---|---|---|---|---|---|---|---|---|
| Entrance | 8 | 4 | 7.23 | 4.63 | 57.87 | 18.53 | 0.36 | 0.68 |
| Promo endcap | 6 | 6 | 13.1 | 8.3 | 78.6 | 49.8 | 0.37 | 0.37 |
| Checkout lane | 1 | 1 | 35.0 | 35.0 | 35.0 | 35.0 |  |  |
| Display corner | 6 | 6 | 9.74 | 8.51 | 58.47 | 51.07 | 0.13 | 0.13 |
| Aisle 1 | 27 | 21 | 16.19 | 8.66 | 437.13 | 181.93 | 0.46 | 0.58 |
| Aisle 2 | 8 | 5 | 9.25 | 5.01 | 74.0 | 25.07 | 0.46 | 0.66 |
| Aisle 3 | 5 | 8 | 5.19 | 5.07 | 25.93 | 40.53 | 0.02 | 0.56 |

## What the numbers say

- Detection and tracking run on a stock COCO YOLOv8n with no fine-tuning and sit below the targets. Closing that gap is the fine-tuning plan in the proposal (CrowdHuman, MOT17/20, store footage). Tracking breaks (86 ID switches, 147 tracks for 79 people) also shorten measured dwell.
- Dwell and floor accuracy on real footage cannot be claimed without a store with measured floor marks and hand-timed dwell. The clip is a street, the zones are illustrative.
- The simulator numbers show the *logic* works when positions are good (jams, queues, dead zone, engaged vs blocked). They do not show the system works in a real store.
- The forecaster beats the naive baseline at 5 min (flag F1 0.50 vs 0.45) and does not beat the naive baseline at 10 min (flag F1 0.34 vs 0.38). Jam start times are random in the simulator, so onsets are inherently hard to predict; real value needs real store data.

## Limitations

- One 35 s clip; no store footage; no GPU, no multi-camera, no re-ID run.
- Staff are not excluded (a doorman standing still shows up as an engaged shopper).
- Zones were drawn after looking at the clip's heat map so each output has something to show.
- `motmetrics` is not used (it breaks on NumPy 2); IDF1, MOTA and AP are implemented in `eval/mot_eval.py` and unit-tested.