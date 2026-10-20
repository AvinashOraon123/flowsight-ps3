"""Measure FlowSight's vision + dwell accuracy on the real clip (MOT17-04) against its hand-annotated ground truth.

Own implementations of detection AP, CLEAR-MOT (MOTA / ID switches) and IDF1: `motmetrics` 1.4 breaks on
NumPy 2 (np.asfarray), and the definitions are short.

Ground truth: MOT17-04-SDP gt.txt (1920x1080 annotations, scaled x0.5 to our 960x540 video).
  considered  = class 1 (pedestrian) with visibility >= 0.25
  ignored     = other classes (static person, person on vehicle, distractor, reflection) and pedestrians with
                visibility < 0.25: detections/tracks that overlap these are neither rewarded nor penalised.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

SCALE = 0.5
IOU = 0.5


def iou_xywh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, 0][:, None], b[:, 0][None]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, 1][:, None], b[:, 1][None]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return inter / np.clip(union, 1e-9, None)


def load_gt(path: Path, stride: int, max_frame: int | None = None, min_vis: float = 0.25):
    """-> (considered, ignored) DataFrames with 0-based source frame `fi`, boxes as x, y, w, h in video pixels."""
    g = pd.read_csv(path, header=None, names=["frame", "id", "x", "y", "w", "h", "flag", "cls", "vis"])
    g["fi"] = g["frame"] - 1
    g = g[(g["fi"] % stride == 0) & ((g["fi"] < max_frame) if max_frame else True)]
    for c in "xywh":
        g[c] = g[c] * SCALE
    good = (g["cls"] == 1) & (g["flag"] == 1) & (g["vis"] >= min_vis)
    ign = (g["cls"].isin([2, 7, 8, 12])) | ((g["cls"] == 1) & (g["vis"] < min_vis))
    return g[good].reset_index(drop=True), g[ign].reset_index(drop=True)


def _by_frame(df: pd.DataFrame, cols) -> dict:
    return {f: grp[cols].to_numpy() for f, grp in df.groupby("fi")}


def detection_ap(dets: pd.DataFrame, gt: pd.DataFrame, ign: pd.DataFrame) -> dict:
    """PASCAL-style AP@0.5 (all-point interpolation) of person detections, plus P/R at conf >= 0.25."""
    G, I = _by_frame(gt, ["x", "y", "w", "h"]), _by_frame(ign, ["x", "y", "w", "h"])
    d = dets.sort_values("conf", ascending=False).reset_index(drop=True)
    used = {f: np.zeros(len(v), bool) for f, v in G.items()}
    tp, fp, keep_conf = [], [], []
    for r in d.itertuples():
        box = np.array([[r.x, r.y, r.w, r.h]])
        g = G.get(r.fi)
        hit = False
        if g is not None:
            ious = iou_xywh(box, g)[0]
            ious[used[r.fi]] = 0
            j = int(ious.argmax())
            if ious[j] >= IOU:
                used[r.fi][j] = True
                hit = True
        if hit:
            tp.append(1), fp.append(0), keep_conf.append(r.conf)
        elif r.fi in I and (iou_xywh(box, I[r.fi])[0] >= IOU).any():
            continue                                             # matches an ignored object
        else:
            tp.append(0), fp.append(1), keep_conf.append(r.conf)
    tp, fp, conf = np.array(tp), np.array(fp), np.array(keep_conf)
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    rec, prec = ctp / max(len(gt), 1), ctp / np.maximum(ctp + cfp, 1)
    mrec, mpre = np.r_[0, rec, 1], np.r_[0, prec, 0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())
    k = conf >= 0.25
    p25 = float(tp[k].sum() / max(k.sum(), 1))
    r25 = float(tp[k].sum() / max(len(gt), 1))
    return dict(ap50=ap, precision_at_0_25=p25, recall_at_0_25=r25, n_gt=int(len(gt)), n_dets=int(len(d)))


def mot_metrics(hyp: pd.DataFrame, gt: pd.DataFrame, ign: pd.DataFrame) -> dict:
    """CLEAR-MOT counts (MOTA, ID switches) and IDF1 for tracker output `hyp` (frame fi, id, x, y, w, h)."""
    frames = sorted(set(gt["fi"]) | set(hyp["fi"]))
    G = {f: g for f, g in gt.groupby("fi")}
    H = {f: g for f, g in hyp.groupby("fi")}
    I = _by_frame(ign, ["x", "y", "w", "h"])
    fn = fp = idsw = tp = 0
    last: dict[int, int] = {}
    pair: dict[tuple[int, int], int] = {}
    n_gt = n_hyp = 0
    for f in frames:
        g, h = G.get(f), H.get(f)
        gb = g[["x", "y", "w", "h"]].to_numpy() if g is not None else np.zeros((0, 4))
        hb = h[["x", "y", "w", "h"]].to_numpy() if h is not None else np.zeros((0, 4))
        gi = g["id"].to_numpy() if g is not None else np.zeros(0, int)
        hi = h["id"].to_numpy() if h is not None else np.zeros(0, int)
        if len(hb) and f in I:                                   # drop tracks sitting on ignored objects
            ok = ~(iou_xywh(hb, I[f]) >= IOU).any(1)
            hb, hi = hb[ok], hi[ok]
        n_gt += len(gb)
        n_hyp += len(hb)
        M = iou_xywh(gb, hb)
        for a in range(len(gi)):                                  # potential TP pairs for IDF1
            for b in np.where(M[a] >= IOU)[0]:
                pair[(int(gi[a]), int(hi[b]))] = pair.get((int(gi[a]), int(hi[b])), 0) + 1
        match: dict[int, int] = {}                                # gt row -> hyp row this frame
        for a, gid in enumerate(gi):                              # 1) keep last frame's correspondences
            pg = last.get(int(gid))
            if pg is None:
                continue
            for b in np.where(hi == pg)[0]:
                if M[a, b] >= IOU and int(b) not in match.values():
                    match[a] = int(b)
        rest_a = [a for a in range(len(gi)) if a not in match]
        rest_b = [b for b in range(len(hi)) if b not in match.values()]
        if rest_a and rest_b:                                     # 2) Hungarian on what is left
            sub = M[np.ix_(rest_a, rest_b)]
            ra, rb = linear_sum_assignment(-((sub >= IOU) * sub))
            for x, y in zip(ra, rb):
                if sub[x, y] >= IOU:
                    a, b = rest_a[x], rest_b[y]
                    if int(gi[a]) in last and last[int(gi[a])] != int(hi[b]):
                        idsw += 1
                    match[a] = b
        for a, b in match.items():
            last[int(gi[a])] = int(hi[b])
        tp += len(match)
        fn += len(gi) - len(match)
        fp += len(hi) - len(match)
    mota = 1 - (fn + fp + idsw) / max(n_gt, 1)
    # IDF1: global one-to-one assignment of GT tracks to hypothesis tracks maximising matched frames
    gids = sorted({k[0] for k in pair})
    hids = sorted({k[1] for k in pair})
    allg, allh = gt["id"].nunique(), hyp["id"].nunique()
    idtp = 0
    if gids and hids:
        gi_ix, hi_ix = {v: i for i, v in enumerate(gids)}, {v: i for i, v in enumerate(hids)}
        C = np.zeros((len(gids), len(hids)))
        for (a, b), n in pair.items():
            C[gi_ix[a], hi_ix[b]] = n
        r, c = linear_sum_assignment(-C)
        idtp = int(C[r, c].sum())
    idfn, idfp = n_gt - idtp, n_hyp - idtp
    idf1 = 2 * idtp / max(2 * idtp + idfp + idfn, 1)
    return dict(mota=float(mota), idf1=float(idf1), idp=idtp / max(n_hyp, 1), idr=idtp / max(n_gt, 1),
                id_switches=int(idsw), recall=tp / max(n_gt, 1), precision=tp / max(tp + fp, 1),
                n_gt_tracks=int(allg), n_hyp_tracks=int(allh), n_gt=int(n_gt))


def run_detector_raw(video: Path, stride: int, weights: str, imgsz: int, conf: float = 0.05, max_frames=None):
    import cv2

    from flowsight.detect import PersonDetector

    det = PersonDetector(weights, conf, imgsz)
    cap = cv2.VideoCapture(str(video))
    rows, fi, n = [], -1, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        fi += 1
        if fi % stride:
            continue
        if max_frames and n >= max_frames:
            break
        n += 1
        d = det(fr)
        for b, c in zip(d.xyxy, d.confidence):
            rows.append((fi, b[0], b[1], b[2] - b[0], b[3] - b[1], float(c)))
    return pd.DataFrame(rows, columns=["fi", "x", "y", "w", "h", "conf"])


def dwell_vs_gt(cfg: dict, gt_all: pd.DataFrame, tracks_img: pd.DataFrame, fps: float, src_fps: float) -> dict:
    """Zone dwell from our tracker vs dwell from the ground-truth annotations, through the SAME projection and analytics.

    gt_all: considered GT (fi, id, x, y, w, h). Per zone with >= 3 ground-truth visits we report
    |mean dwell ours - mean dwell GT| / mean dwell GT. Isolates tracker error (no hand-timed dwell exists for this clip).
    """
    from flowsight import metrics
    from flowsight.geometry import load_homography
    from flowsight.pipeline import project_tracks, rpath
    from flowsight.zones import ZoneSet

    p, zones = cfg["analytics"], ZoneSet.load(rpath(cfg, "zones"))
    H = load_homography(rpath(cfg, "homography"))

    def run(df):
        tr = project_tracks(df, H, zones, p.get("min_track_s", 1.0))
        tr = metrics.add_motion_state(tr, fps, p)
        return metrics.zone_events(tr, fps, p)

    g = gt_all.assign(t=gt_all["fi"] / src_fps, track_id=gt_all["id"], frame=gt_all["fi"],
                      u=gt_all["x"] + gt_all["w"] / 2, v=gt_all["y"] + gt_all["h"])[["frame", "t", "track_id", "u", "v"]]
    ev_gt, ev_us = run(g), run(tracks_img[["frame", "t", "track_id", "u", "v"]])
    rows = []
    for z in zones.names:
        a, b = ev_gt[(ev_gt.zone == z) & ev_gt.is_visit], ev_us[(ev_us.zone == z) & ev_us.is_visit]
        rows.append(dict(zone=z, gt_visits=len(a), our_visits=len(b),
                         gt_mean_dwell_s=float(a.dwell_s.mean()) if len(a) else np.nan,
                         our_mean_dwell_s=float(b.dwell_s.mean()) if len(b) else np.nan,
                         gt_person_s=float(a.dwell_s.sum()), our_person_s=float(b.dwell_s.sum())))
    t = pd.DataFrame(rows)
    ok = t[(t.gt_visits >= 3) & t.our_mean_dwell_s.notna()].copy()
    ok["mean_dwell_err"] = (ok.our_mean_dwell_s - ok.gt_mean_dwell_s).abs() / ok.gt_mean_dwell_s
    ok["person_s_err"] = (ok.our_person_s - ok.gt_person_s).abs() / ok.gt_person_s
    return dict(table=t.merge(ok[["zone", "mean_dwell_err", "person_s_err"]], on="zone", how="left"),
                mean_abs_dwell_err=float(ok.mean_dwell_err.mean()) if len(ok) else float("nan"),
                median_abs_dwell_err=float(ok.mean_dwell_err.median()) if len(ok) else float("nan"),
                total_person_s_err=float(abs(t.our_person_s.sum() - t.gt_person_s.sum()) / max(t.gt_person_s.sum(), 1e-9)),
                n_zones_scored=int(len(ok)))
