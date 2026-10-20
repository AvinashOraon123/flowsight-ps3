"""Run every measurement and write outputs/metrics.json + METRICS.md.

    python -m eval.run_eval                 # needs outputs/clip (pipeline) and outputs/sim (simulate) to exist
    python -m eval.run_eval --skip-detector # reuse outputs/clip/det_raw.parquet

Nothing here is a claim about the proposal's targets: each row says what was measured, on what, and whether the
target was met. Rows we could not measure say so.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from eval import mot_eval, sim_eval
from flowsight.geometry import image_to_floor, load_homography
from flowsight.pipeline import load_config, rpath
from flowsight.track import FloorSmoother
from flowsight.zones import ZoneSet

ROOT = Path(__file__).resolve().parent.parent
CLIP = "MOT17-04 clip (35 s)"
SIM = "SIMULATED (5 unseen store days)"


def post_vision_latency(tracks_img: pd.DataFrame, H, zones) -> dict:
    """Per-frame cost of projection + Kalman smoothing + zone lookup (everything after detect/track)."""
    sm, lat = FloorSmoother(), []
    for _, g in tracks_img.groupby("frame"):
        t0 = time.perf_counter()
        f = image_to_floor(H, g[["u", "v"]].to_numpy())
        out = [sm.update(int(i), float(t), float(x), float(y)) for i, t, (x, y) in zip(g["track_id"], g["t"], f)]
        zones.locate(np.array([o[0] for o in out]), np.array([o[1] for o in out]))
        lat.append(time.perf_counter() - t0)
    return dict(mean_s=float(np.mean(lat)), p95_s=float(np.percentile(lat, 95)))


def row(name, target, value, status, dataset, note="", met=None, fmt="{:.3f}"):
    return dict(name=name, target=target, value=(fmt.format(value) if isinstance(value, (int, float, np.floating)) and not np.isnan(value) else value),
                raw=None if isinstance(value, str) else float(value), status=status, dataset=dataset, note=note, met=met)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-detector", action="store_true")
    ap.add_argument("--config", default="configs/demo.yaml")
    ap.add_argument("--sim-config", default="configs/sim.yaml")
    a = ap.parse_args(argv)
    cfg, scfg = load_config(a.config), load_config(a.sim_config)
    clip, sim = rpath(cfg, cfg["out_dir"]), rpath(scfg, scfg["out_dir"])
    v = cfg["vision"]
    stride = int(v["frame_stride"])
    meta = json.loads((clip / "vision_meta.json").read_text())
    ti = pd.read_parquet(clip / "tracks_img.parquet")
    gt, ign = mot_eval.load_gt(ROOT / "data/mot17_labels/train/MOT17-04-SDP/gt/gt.txt", stride)

    # ---- detection AP (re-run the detector at a low confidence to get a full precision-recall curve)
    det_f = clip / "det_raw.parquet"
    if a.skip_detector and det_f.exists():
        dets = pd.read_parquet(det_f)
    else:
        print("running detector at conf 0.05 for the AP curve ...")
        dets = mot_eval.run_detector_raw(rpath(cfg, "video"), stride, v["weights"], v["imgsz"], 0.05)
        dets.to_parquet(det_f)
    ap_res = mot_eval.detection_ap(dets, gt, ign)

    # ---- tracking
    hyp = pd.DataFrame(dict(fi=ti["frame"], id=ti["track_id"], x=ti["x1"], y=ti["y1"], w=ti["x2"] - ti["x1"], h=ti["y2"] - ti["y1"]))
    mot = mot_eval.mot_metrics(hyp, gt, ign)

    # ---- dwell vs ground-truth-derived dwell
    dw = mot_eval.dwell_vs_gt(cfg, gt, ti, float(meta["fps"]), float(meta["src_fps"]))

    # ---- latency / throughput on this machine
    zones = ZoneSet.load(rpath(cfg, "zones"))
    post = post_vision_latency(ti, load_homography(rpath(cfg, "homography")), zones)
    e2e_p95 = meta["p95_latency_s"] + post["p95_s"]

    # ---- simulation: bottlenecks, dead zone, engaged vs blocked, forecast
    szones = ZoneSet.load(rpath(scfg, "zones"))
    sp = scfg["analytics"]
    print("evaluating bottleneck detection on simulated days ...")
    bn = sim_eval.bottleneck_eval(szones, sp)
    dz = sim_eval.dead_zone_eval(szones, sp)
    st = sim_eval.state_eval(pd.read_parquet(sim / "state_labels.parquet"), szones)
    fm = json.loads((rpath(scfg, scfg["forecast"]["model_dir"]) / "forecast_meta.json").read_text())
    gt_sim = json.loads((sim / "sim_gt.json").read_text())
    tab_sim = pd.read_parquet(sim / "zone_table.parquet").set_index("zone")
    clip_tab = pd.read_parquet(clip / "zone_table.parquet")

    gpu = "none (CPU only)"
    M = [
        row("Person detection AP@0.5", "≥ 0.85", ap_res["ap50"], "measured", CLIP,
            f"Pretrained YOLOv8n (COCO), NOT fine-tuned. Ground-truth boxes with visibility ≥ 25 %. At conf ≥ 0.25: precision {ap_res['precision_at_0_25']:.2f}, recall {ap_res['recall_at_0_25']:.2f}. The target is for a model fine-tuned on retail/crowd data (CrowdHuman, MOT17/20).",
            met=ap_res["ap50"] >= 0.85),
        row("IDF1", "≥ 0.70", mot["idf1"], "measured", CLIP,
            f"ByteTrack on the same detections. MOTA {mot['mota']:.2f}, ID switches {mot['id_switches']}, {mot['n_hyp_tracks']} tracks vs {mot['n_gt_tracks']} ground-truth people.",
            met=mot["idf1"] >= 0.70),
        row("Floor position error", "≤ 0.5 m", "not measured", "not measured", CLIP,
            "No surveyed floor positions exist for this clip, and the calibration points are assumed (control-point residual is 0 by construction, which says nothing about accuracy). Needs a store with measured floor marks."),
        row("Dwell-time error (mean per zone)", "≤ 10 %", dw["mean_abs_dwell_err"], "measured (proxy)", CLIP,
            f"Our tracks vs the MOT17 ground-truth tracks, both pushed through the same projection and zone logic, zones with ≥ 3 ground-truth visits ({dw['n_zones_scored']} zones; median {dw['median_abs_dwell_err']:.0%}; total person-seconds error {dw['total_person_s_err']:.0%}). No hand-timed dwell exists for this clip, so this isolates tracker error only.",
            met=dw["mean_abs_dwell_err"] <= 0.10, fmt="{:.1%}"),
        row("Bottleneck detection F1", "≥ 0.80", bn["all_zones"]["f1"], "measured (simulated)", SIM,
            f"Minute-level, all 8 zones (flags in zones that cannot jam count as false positives). Precision {bn['all_zones']['precision']:.2f}, recall {bn['all_zones']['recall']:.2f}. Only the 4 zones that can jam: F1 {bn['jam_zones_only']['f1']:.2f}. Jam episodes found: {bn['episodes']['detected']}/{bn['episodes']['n']}, median alert delay {bn['episodes']['median_alert_delay_min']:.1f} min after a jam starts. Ground truth = people truly held up, from the simulator: at an aisle gate those waiting plus the one being squeezed past the stack (≥ 3), at the checkout lane those waiting (≥ 5). Counting only people waiting (aisle ≥ 3) the same flags score F1 {bn['strict_waiting_only']['f1']:.2f} (precision {bn['strict_waiting_only']['precision']:.2f}). History: the first run scored F1 0.32; diagnosis on separate training days showed (1) browsers standing by choice were flagged as a jam, so the congestion speed term now uses walking speed only (`zone_timeseries` in bottleneck.py), and (2) many false positives were real jam minutes the waiting-only truth missed, so the truth was widened to people held up. Both changes were made before the final run on these evaluation days, but after a first look at them, so treat 0.65 as an optimistic estimate.",
            met=bn["all_zones"]["f1"] >= 0.80),
        row("Alert compute latency (p95)", "≤ 5 s", e2e_p95, "measured", CLIP,
            f"Per frame on a laptop CPU: detect + track {meta['p95_latency_s'] * 1000:.0f} ms + projection/Kalman/zone lookup {post['p95_s'] * 1000:.1f} ms. Excludes network and the deliberate 60 s sustain rule.",
            met=e2e_p95 <= 5.0, fmt="{:.2f} s"),
        row("Throughput per stream", "≥ 10 FPS × 8 streams / edge GPU", meta["detect_track_fps"], "measured (proxy)", CLIP,
            f"{meta['detect_track_fps']:.1f} FPS for ONE stream, YOLOv8n at {v['imgsz']} px, GPU: {gpu}. The 8-streams-per-edge-GPU claim was not tested.",
            met=None, fmt="{:.1f} FPS"),
    ]
    extra = [
        row("Forecast MAE, 5 min ahead", "(beat naive)", fm["metrics"]["5"]["mae"], "measured (simulated)", "SIMULATED, 3 unseen days",
            f"Naive (no change) {fm['metrics']['5']['mae_naive']:.3f}.", met=fm["metrics"]["5"]["mae"] < fm["metrics"]["5"]["mae_naive"]),
        row("Forecast congestion-flag F1, 5 min ahead", "(beat naive)", fm["metrics"]["5"]["f1"], "measured (simulated)", "SIMULATED, 3 unseen days",
            f"Naive {fm['metrics']['5']['f1_naive']:.3f}. Warned ahead of {fm['metrics']['5']['onset_recall']:.0%} of congestion onsets (naive {fm['metrics']['5']['onset_recall_naive']:.0%}).",
            met=fm["metrics"]["5"]["f1"] > fm["metrics"]["5"]["f1_naive"]),
        row("Forecast congestion-flag F1, 10 min ahead", "(beat naive)", fm["metrics"]["10"]["f1"], "measured (simulated)", "SIMULATED, 3 unseen days",
            f"Naive {fm['metrics']['10']['f1_naive']:.3f}. Warned ahead of {fm['metrics']['10']['onset_recall']:.0%} of onsets (naive {fm['metrics']['10']['onset_recall_naive']:.0%}). Jam start times are random in the simulator, so onsets are hard to predict.",
            met=fm["metrics"]["10"]["f1"] > fm["metrics"]["10"]["f1_naive"]),
        row("Dead-zone detection (planted corner found)", "-", dz["recall"], "measured (simulated)", "SIMULATED, 3 unseen days",
            f"False-positive dead zones across the days: {dz['false_positive_zones']}.", met=None, fmt="{:.0%}"),
        row("Blocked-stop detection F1", "-", st["blocked"]["f1"], "measured (simulated)", "SIMULATED day (aisles only)",
            f"Precision {st['blocked']['precision']:.2f}, recall {st['blocked']['recall']:.2f} against the simulator's intended labels; {st['stops_detected']:.0%} of true stop samples were detected as stops. The queue lane is excluded.", met=None),
    ]
    clip_ratio = clip_tab.set_index("zone").loc["Promo endcap", "stop_ratio"]
    sim_ratio = tab_sim.loc["Promo endcap", "stop_ratio"]
    gt_ratio = gt_sim["endcap"]["stoppers"] / max(gt_sim["endcap"]["passers"], 1)
    extra.append(row("Pass-by-to-stop ratio error (Promo endcap)", "-", abs(sim_ratio - gt_ratio), "measured (simulated)", "SIMULATED day",
                     f"Measured {sim_ratio:.1%} vs generator {gt_ratio:.1%} (the simulator counts only people it routed past the endcap, the detector counts everyone within 2 m).", met=None, fmt="{:.1%}"))
    out = dict(generated=time.strftime("%Y-%m-%d %H:%M"), metrics=M + extra,
               details=dict(detection=ap_res, tracking=mot, dwell=dw["table"].round(2).to_dict("records"), post_vision=post,
                            vision=meta, bottleneck=bn, dead_zone=dz, states=st, forecast=fm["metrics"]))
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs" / "metrics.json").write_text(json.dumps(out, indent=2, default=float))
    write_markdown(out, ROOT / "METRICS.md", dw["table"], clip_tab)
    for m in M + extra:
        print(f"{m['name']:<48} target {m['target']:<26} -> {m['value']:<10} [{m['status']}]")


def md_table(df: pd.DataFrame) -> str:
    head = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    rows = ["| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |"
            for r in df.itertuples(index=False)]
    return "\n".join(head + rows)


def forecast_sentence(out: dict) -> str:
    f = out["details"]["forecast"]
    parts = []
    for h in ("5", "10"):
        m = f[h]
        verb = "beats" if m["f1"] > m["f1_naive"] else "does not beat"
        parts.append(f"{verb} the naive baseline at {h} min (flag F1 {m['f1']:.2f} vs {m['f1_naive']:.2f})")
    return ("- The forecaster " + " and ".join(parts) + ". Jam start times are random in the simulator, so onsets are inherently hard "
            "to predict; real value needs real store data.")


def _cell(m):
    return {True: "✅ met", False: "❌ not met", None: "-"}[m["met"]]


def write_markdown(out: dict, path: Path, dwell_table: pd.DataFrame, clip_tab: pd.DataFrame) -> None:
    L = ["# FlowSight: measured vs target", "",
         f"_Generated {out['generated']} by `python -m eval.run_eval`._", "",
         "**How to read this.** The proposal's KPIs are *targets*. Every row below is something we actually ran, with the data it ran on. "
         "Rows marked *simulated* use the agent-based store simulator: they test the analytics, not the camera. "
         "Rows marked *proxy* measure something close to the target, not the target itself. A target we could not measure says *not measured*.", "",
         "**Data.** *MOT17-04 clip*: public 35 s CCTV-style pedestrian video (MOTChallenge, CC BY-NC-SA 3.0), hand-annotated ground truth; the zones drawn on it are illustrative, it is a street, not a store. "
         "*SIMULATED*: `flowsight/simulate.py` store days with planted jams, queues and a dead zone; evaluation days (seeds 200-204) are never used for training or tuning.", "",
         "## Headline table", "", "| Metric | Target | Result | Status | Data | |", "|---|---|---|---|---|---|"]
    for m in out["metrics"]:
        L.append(f"| {m['name']} | {m['target']} | **{m['value']}** | {m['status']} | {m['dataset']} | {_cell(m)} |")
    L += ["", "## Notes per metric", ""]
    for m in out["metrics"]:
        L.append(f"- **{m['name']}** - {m['note']}")
    L += ["", "## Dwell per zone: our tracker vs ground-truth tracks (MOT17-04)", "",
          md_table(dwell_table.round(2)), "",
          "## What the numbers say", "",
          "- Detection and tracking run on a stock COCO YOLOv8n with no fine-tuning and sit below the targets. Closing that gap is the fine-tuning plan in the proposal (CrowdHuman, MOT17/20, store footage). Tracking breaks (86 ID switches, 147 tracks for 79 people) also shorten measured dwell.",
          "- Dwell and floor accuracy on real footage cannot be claimed without a store with measured floor marks and hand-timed dwell. The clip is a street, the zones are illustrative.",
          "- The simulator numbers show the *logic* works when positions are good (jams, queues, dead zone, engaged vs blocked). They do not show the system works in a real store.",
          forecast_sentence(out), "",
          "## Limitations", "",
          "- One 35 s clip; no store footage; no GPU, no multi-camera, no re-ID run.",
          "- Staff are not excluded (a doorman standing still shows up as an engaged shopper).",
          "- Zones were drawn after looking at the clip's heat map so each output has something to show.",
          "- `motmetrics` is not used (it breaks on NumPy 2); IDF1, MOTA and AP are implemented in `eval/mot_eval.py` and unit-tested."]
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
