"""Render data/floor_plan.png (metre ticks + labelled zones) from zones.json."""
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
KIND_BGR = {"entrance": (200, 170, 90), "endcap": (90, 140, 230), "queue": (110, 110, 220),
            "display": (150, 200, 150), "aisle": (205, 205, 205)}


def main() -> None:
    d = json.loads((HERE / "zones.json").read_text())
    fp = d["floor_plan"]
    s = fp["px_per_m"]
    img = np.full((int(fp["height_m"] * s), int(fp["width_m"] * s), 3), 250, np.uint8)
    for z in sorted(d["zones"], key=lambda z: -z["priority"]):
        pts = (np.array(z["polygon"]) * s).astype(np.int32)
        ov = img.copy()
        cv2.fillPoly(ov, [pts], KIND_BGR.get(z.get("kind"), (220, 220, 220)))
        img = cv2.addWeighted(ov, 0.55, img, 0.45, 0)
        cv2.polylines(img, [pts], True, (90, 90, 90), 1, cv2.LINE_AA)
    blank = img.copy()
    for z in d["zones"]:
        pts = (np.array(z["polygon"]) * s)
        cx, cy = pts.mean(0).astype(int)
        (w, _), _ = cv2.getTextSize(z["name"], cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.putText(img, z["name"], (int(cx - w / 2), int(cy)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (40, 40, 40), 1, cv2.LINE_AA)
    for m in range(0, int(fp["width_m"]) + 1, 2):
        cv2.line(img, (m * s, 0), (m * s, 8), (120, 120, 120), 1)
        cv2.putText(img, f"{m}", (m * s + 2, 20), cv2.FONT_HERSHEY_SIMPLEX, .35, (120, 120, 120), 1)
    for m in range(0, int(fp["height_m"]) + 1, 2):
        cv2.line(img, (0, m * s), (8, m * s), (120, 120, 120), 1)
        cv2.putText(img, f"{m}m", (10, m * s + 12), cv2.FONT_HERSHEY_SIMPLEX, .35, (120, 120, 120), 1)
    cv2.imwrite(str(HERE / "floor_plan.png"), img)
    cv2.imwrite(str(HERE / "floor_plan_blank.png"), blank)      # no labels: used under live overlays
    print("wrote floor_plan.png", img.shape)


if __name__ == "__main__":
    main()
