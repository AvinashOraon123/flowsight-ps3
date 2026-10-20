"""Live training monitor.   streamlit run train/monitor.py --server.port 8502"""
import re
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "outputs" / "train" / "flowsight_n"
LOG = ROOT / "outputs" / "exp" / "train.log"
BASELINE_AP = 0.664   # stock yolov8n on MOT17-04 (METRICS.md), for reference only: different data, so not comparable to val mAP

st.set_page_config(page_title="FlowSight training", page_icon="📈", layout="wide")
st.title("📈 FlowSight detector fine-tuning")
st.caption("YOLOv8n, person only, MOT17-05/10/11/13 · CPU · evaluation clips 02/04/09 are never used for training")


def hms(s: float) -> str:
    s = int(max(s, 0))
    return f"{s // 3600}h {s % 3600 // 60:02d}m"


def parse_log():
    if not LOG.exists():
        return None
    txt = LOG.read_text(errors="ignore").replace("\r", "\n")
    m = None
    for m in re.finditer(r"\s(\d+)/(\d+)\s+\S+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+\d+\s+\d+:\s+(\d+)%.*?(\d+)/(\d+)\s+([\d.]+)(s/it|it/s)\s+(\S+)<(\S+)", txt):
        pass
    if m is None:
        return dict(done="epochs completed" in txt, text=txt[-400:])
    ep, n_ep, pct, it, n_it, rate, unit, el, eta = m.groups()
    return dict(epoch=int(ep), epochs=int(n_ep), pct=int(pct), it=int(it), n_it=int(n_it), rate=float(rate), unit=unit,
                eta=eta, done="epochs completed" in txt, validating=False)


@st.fragment(run_every=10)
def panel():
    info = parse_log()
    df = None
    f = RUN / "results.csv"
    if f.exists():
        df = pd.read_csv(f)
        df.columns = [c.strip() for c in df.columns]
    if info is None:
        st.info("No training log yet. Start training: `python train/train.py`.")
        return
    done_epochs = 0 if df is None else len(df)
    total = info.get("epochs", 15)
    cur_frac = info.get("pct", 0) / 100 if not info.get("done") else 0
    overall = min((done_epochs + (cur_frac if done_epochs < total else 0)) / total, 1.0)
    st.progress(overall, text=f"Overall {overall:.0%} · epoch {min(done_epochs + 1, total)} of {total}"
                              + (f" · batch {info['it']}/{info['n_it']}" if "it" in info and not info["done"] else ""))
    per_epoch = float(df["time"].diff().fillna(df["time"]).mean()) if df is not None and len(df) else None
    c = st.columns(5)
    c[0].metric("Epochs finished", f"{done_epochs}/{total}")
    c[1].metric("Time per epoch", hms(per_epoch) if per_epoch else "measuring…")
    if per_epoch:
        left = max(total - done_epochs - cur_frac, 0) * per_epoch
        c[2].metric("Time remaining", "done" if info.get("done") else hms(left),
                    None if info.get("done") else time.strftime("finishes ~%H:%M", time.localtime(time.time() + left)))
    else:
        c[2].metric("Time remaining", "after epoch 1")
    if df is not None and len(df):
        last = df.iloc[-1]
        c[3].metric("val mAP@0.5", f"{last['metrics/mAP50(B)']:.3f}", f"{last['metrics/mAP50(B)'] - df['metrics/mAP50(B)'].iloc[0]:+.3f} since epoch 1" if len(df) > 1 else None)
        c[4].metric("val recall", f"{last['metrics/recall(B)']:.3f}")
    else:
        c[3].metric("val mAP@0.5", "after epoch 1")
        c[4].metric("val recall", "after epoch 1")
    if df is None or len(df) == 0:
        st.caption(f"Epoch 1 is running ({info.get('pct', 0)}% of its batches; {info.get('rate', 0):.1f} {info.get('unit', 's/it')}). "
                   "Charts appear when the first epoch finishes.")
        return
    a, b = st.columns(2)
    with a:
        st.markdown("**Training loss**")
        fig = go.Figure()
        for col, name in [("train/box_loss", "box"), ("train/cls_loss", "class"), ("train/dfl_loss", "dfl")]:
            fig.add_scatter(x=df["epoch"], y=df[col], name=name, mode="lines+markers")
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="epoch")
        st.plotly_chart(fig, width="stretch")
    with b:
        st.markdown("**Validation (held-out last 15% of each training clip)**")
        fig = go.Figure()
        for col, name in [("metrics/mAP50(B)", "mAP@0.5"), ("metrics/precision(B)", "precision"), ("metrics/recall(B)", "recall")]:
            fig.add_scatter(x=df["epoch"], y=df[col], name=name, mode="lines+markers")
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="epoch", yaxis_range=[0, 1])
        st.plotly_chart(fig, width="stretch")
    st.caption("Validation frames come from the same clips as training (later frames), so these numbers flatter the model. "
               "The honest test is the AP/IDF1 re-measurement on the never-seen clips MOT17-02/04/09 after training.")
    with st.expander("Per-epoch table"):
        st.dataframe(df[["epoch", "time", "train/box_loss", "train/cls_loss", "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]]
                     .round(4), hide_index=True, width="stretch")


panel()
