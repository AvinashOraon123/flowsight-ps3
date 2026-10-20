"""Download the public demo clip (MOT17-04) and its ground-truth labels.

Source : MOTChallenge, https://motchallenge.net/  (MOT17, Dendorfer et al. 2017)
Licence: CC BY-NC-SA 3.0 - research / non-commercial use only.
"""
import sys
import zipfile
from pathlib import Path

import requests

HERE = Path(__file__).parent
FILES = {
    "MOT17-04-raw.webm": "https://motchallenge.net/sequenceVideos/MOT17-04-SDP-raw.webm",
    "MOT17Labels.zip": "https://motchallenge.net/data/MOT17Labels.zip",
}


def fetch(name: str, url: str) -> Path:
    dest = HERE / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"have {name}")
        return dest
    print(f"downloading {name} ...")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return dest


def main() -> int:
    for name, url in FILES.items():
        fetch(name, url)
    with zipfile.ZipFile(HERE / "MOT17Labels.zip") as z:
        for n in z.namelist():
            if "MOT17-04-SDP" in n:
                z.extract(n, HERE / "mot17_labels")
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
