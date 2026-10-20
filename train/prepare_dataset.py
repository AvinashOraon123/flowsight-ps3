"""Build a single-class (person) YOLO dataset from MOT17 train clips that are NOT used for evaluation.

Held out for evaluation (never used here): MOT17-02, -04, -09.  Training clips: -05, -10, -11, -13.
Frames come from the same compressed .webm videos the pipeline reads, so train/test pixels match.
Licence: MOT17 is CC BY-NC-SA 3.0 (research / non-commercial); weights trained on it inherit that restriction.

    python train/prepare_dataset.py            # -> train/dataset/{images,labels}/{train,val}, dataset.yaml
"""
import zipfile
from pathlib import Path

import cv2
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "train" / "dataset"
CLIPS = ["MOT17-05-SDP", "MOT17-10-SDP", "MOT17-11-SDP", "MOT17-13-SDP"]
STRIDE, VAL_FRAC, MIN_VIS = 3, 0.15, 0.25
GT_SIZE = {"MOT17-05-SDP": (640, 480)}                     # every other MOT17 clip is annotated at 1920x1080


def fetch(name):
    f = DATA / f"{name}-raw.webm"
    if not f.exists():
        r = requests.get(f"https://motchallenge.net/sequenceVideos/{name}-raw.webm", timeout=120)
        r.raise_for_status()
        f.write_bytes(r.content)
    return f


def main():
    z = zipfile.ZipFile(DATA / "MOT17Labels.zip")
    for c in CLIPS:
        for n in z.namelist():
            if c in n and n.endswith("gt.txt"):
                z.extract(n, DATA / "mot17_labels")
    for sp in ("train", "val"):
        (OUT / "images" / sp).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / sp).mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "val": 0}
    for c in CLIPS:
        g = pd.read_csv(DATA / "mot17_labels/train" / c / "gt/gt.txt", header=None,
                        names=["f", "id", "x", "y", "w", "h", "flag", "cls", "vis"])
        g = g[(g.cls == 1) & (g.flag == 1) & (g.vis >= MIN_VIS)]
        gw, gh = GT_SIZE.get(c, (1920, 1080))
        cap = cv2.VideoCapture(str(fetch(c)))
        W, H = cap.get(3), cap.get(4)
        n_frames = int(cap.get(7))
        fi = -1
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            fi += 1
            if fi % STRIDE:
                continue
            sp = "val" if fi >= (1 - VAL_FRAC) * n_frames else "train"
            rows = g[g.f == fi + 1]
            name = f"{c}_{fi:05d}"
            cv2.imwrite(str(OUT / "images" / sp / f"{name}.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 92])
            lab = []
            for r in rows.itertuples():
                cx, cy = (r.x + r.w / 2) / gw, (r.y + r.h / 2) / gh        # normalised by the clip's own annotation size
                lab.append(f"0 {min(max(cx, 0), 1):.6f} {min(max(cy, 0), 1):.6f} {min(r.w / gw, 1):.6f} {min(r.h / gh, 1):.6f}")
            (OUT / "labels" / sp / f"{name}.txt").write_text("\n".join(lab))
            counts[sp] += 1
        print(c, f"{W:.0f}x{H:.0f}", n_frames, "frames")
    (OUT / "dataset.yaml").write_text(f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: person\n")
    print("images:", counts)


if __name__ == "__main__":
    main()
