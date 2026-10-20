"""FlowSight dashboard.   streamlit run app.py"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from flowsight.heatmap import HeatGrid, render_heat
from flowsight.zones import ZoneSet

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
SOURCES = {"Demo clip (real footage, MOT17-04)": ("clip", "data/zones.json", "data/floor_plan.png"),
           "Simulated store day (synthetic)": ("sim", "data/sim_zones.json", "data/sim_floor_plan.png")}
KIND_COL = {"queue": "#d9534f", "bottleneck": "#f0803c", "forecast": "#e8b100", "endcap": "#3b8ed0",
            "dead_zone": "#8a8f98", "engaged_no_sale": "#8e6bd1"}
STATE_COL = {"engaged": "#e8a200", "blocked": "#d9534f"}

st.set_page_config(page_title="FlowSight", page_icon="🛒", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.4rem} .tag{display:inline-block;padding:1px 8px;border-radius:9px;font-size:.75rem;color:#fff}
.alert{border-left:5px solid #888;padding:6px 10px;margin:6px 0;background:rgba(128,128,128,.10);border-radius:4px}
.small{font-size:.82rem;opacity:.75}</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ loaders
@st.cache_data(show_spinner=False)
def load(src: str, name: str, mtime: float):
    f = OUT / src / name
    if name.endswith(".parquet"):
        return pd.read_parquet(f)
    if name.endswith(".json"):
        return json.loads(f.read_text())
    return None


def get(src: str, name: str):
    f = OUT / src / name
    return load(src, name, f.stat().st_mtime) if f.exists() else None


@st.cache_resource(show_spinner=False)
def zones_of(path: str) -> ZoneSet:
    return ZoneSet.load(ROOT / path)


@st.cache_data(show_spinner=False)
def png_uri(path: str) -> str:
    return "data:image/png;base64," + base64.b64encode((ROOT / path).read_bytes()).decode()


def plan_figure(zones: ZoneSet, plan_png: str, height: int = 640) -> go.Figure:
    ppm = zones.px_per_m
    fig = go.Figure(go.Image(source=png_uri(plan_png), x0=0.5 / ppm, dx=1 / ppm, y0=0.5 / ppm, dy=1 / ppm, hoverinfo="skip"))
    fig.update_xaxes(range=[0, zones.width_m], title="metres", constrain="domain")
    fig.update_yaxes(range=[zones.height_m, 0], scaleanchor="x", title="metres")
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
    return fig


def fmt_t(t: float, src: str) -> str:
    return f"{t:.0f} s" if src == "clip" else f"{int(t // 3600) + 8:02d}:{int(t % 3600 // 60):02d}"


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🛒 FlowSight")
st.sidebar.caption("Dwell-time & bottleneck mapper · Everseen CV AI Hackathon · PS3")
label = st.sidebar.radio("Data source", list(SOURCES))
src, zones_path, plan_png = SOURCES[label]
if src == "clip":
    st.sidebar.warning("Real pedestrian footage (MOT17-04). Zones are **illustrative** - this scene is not a store.")
else:
    st.sidebar.warning("**Simulated** store day from the agent-based simulator. Not real data.")
if not (OUT / src / "zone_table.parquet").exists():
    st.error(f"No outputs for this source yet. Run: `python -m flowsight.pipeline --config configs/{'demo' if src == 'clip' else 'sim'}.yaml`"
             if src == "clip" else "No simulation outputs yet. Run: `python -m flowsight.simulate --out outputs/sim`")
    st.stop()

zones = zones_of(zones_path)
table = get(src, "zone_table.parquet")
ts = get(src, "zone_timeseries.parquet")
alerts = get(src, "alerts.json") or []
bn = get(src, "bottlenecks.parquet")
summ = get(src, "summary.json") or {}
cfg_thr = summ.get("bottleneck_threshold", 0.30)

st.title("FlowSight")
st.caption("Turns existing CCTV into a live map of where shoppers go, linger, get stuck and which zones they ignore, "
           "and tells staff what to do.")
tab_live, tab_zones, tab_j, tab_fc, tab_m = st.tabs(["① Live", "② Zones", "③ Journeys", "④ Forecast", "⑤ Metrics"])

# ================================================================== LIVE
with tab_live:
    T = float(summ.get("duration_s", ts["t"].max() + 1))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("People tracked", summ.get("n_tracks", "-"))
    c2.metric("Zone visits (>= 3 s)", summ.get("n_visits", "-"))
    c3.metric("Bottlenecks flagged", summ.get("n_bottlenecks", "-"))
    c4.metric("Actions raised", summ.get("n_alerts", "-"))
    left, right = st.columns([3, 2])
    with left:
        live = OUT / src / "live.mp4"
        if live.exists():
            st.video(str(live))
            st.caption("Left: camera with tracked people (heads blurred); amber = engaged stop, red = blocked stop. "
                       "Right: floor-plan heat map of the last ~12 s with per-zone congestion C. Bottom: alert feed.")
        else:
            st.info("No video for this source - use the time slider; the heat map shows the last 10 minutes.")
    with right:
        step = 1.0 if src == "clip" else 60.0
        opts = [float(v) for v in np.arange(0, T - step, step)]
        rt = [a for a in alerts if a["timescale"] in ("seconds", "minutes")]
        start = max(rt, key=lambda a: a["impact"])["t"] if rt else T / 2          # open on the biggest live alert
        start = min(opts, key=lambda o: abs(o - start - 120))                   # a couple of minutes after it fired
        now = st.select_slider("Time", options=opts, value=start, key=f"live_t_{src}",
                               format_func=lambda v: fmt_t(v, src))
        st.markdown(f"**State at {fmt_t(now, src)}**")
        cur = ts[(ts["t"] >= now) & (ts["t"] < now + step)].groupby("zone").agg(
            people=("count", "mean"), speed=("v_mean", "mean"), C=("C", "mean")).reindex(zones.names).round(2)
        cur["status"] = np.where(cur["C"] >= cfg_thr, "🔴 congested", np.where(cur["C"] >= cfg_thr / 2, "🟠 busy", "🟢 flowing"))
        st.dataframe(cur.rename(columns={"speed": "speed m/s"}), width="stretch")
        feed = sorted([a for a in alerts if a["t"] <= now], key=lambda a: -a["impact"])
        st.markdown(f"**Action feed** ({len(feed)} so far, ranked by impact)")
        for a in feed[:6]:
            st.markdown(f"<div class='alert' style='border-color:{KIND_COL.get(a['kind'], '#888')}'>"
                        f"<span class='tag' style='background:{KIND_COL.get(a['kind'], '#888')}'>{a['kind']}</span> "
                        f"<b>{a['action']}</b><br><span class='small'>why: {a['reason']}<br>"
                        f"to: {a['audience']} · {a['timescale']} · impact {a['impact']:.1f} person-min · "
                        f"at {fmt_t(a['t'], src)}</span></div>", unsafe_allow_html=True)
        if not feed:
            st.caption("No alerts yet at this time.")
    h1, h2 = st.columns(2)
    with h1:
        st.markdown("#### Cumulative foot-traffic heat map")
        hp = OUT / src / "heatmap.png"
        if hp.exists():
            st.image(str(hp), width=460, caption="person-seconds on a 0.25 m grid, Gaussian smoothed")
    if src == "sim":
        with h2:
            st.markdown(f"#### Heat map of the 10 minutes before {fmt_t(now, src)}")
            trk = get(src, "tracks.parquet")
            if trk is not None:
                import cv2
                w = trk[(trk["t"] >= now - 600) & (trk["t"] < now)]
                hg = HeatGrid(zones.width_m, zones.height_m, 0.25)
                hg.add(w["x"].to_numpy(), w["y"].to_numpy(), 1.0)
                base = cv2.imread(str(ROOT / plan_png))
                img = render_heat(hg.smoothed(2.0), base, vmax=max(float(hg.smoothed(2.0).max()), 1.0))
                st.image(img[:, :, ::-1], width=460, caption="simulated shoppers, last 10 min")

# ================================================================== ZONES
with tab_zones:
    st.subheader("Section-by-section dwell time")
    show = table[["zone", "kind", "visits", "visitors", "mean_dwell_s", "median_dwell_s", "engaged_s", "blocked_s",
                  "blocked_share", "passers", "engaged_stoppers", "stop_ratio", "peak_count", "peak_C",
                  "bottleneck_min", "dead_zone"]].copy()
    show["dead_zone"] = show["dead_zone"].map({True: "⚠ dead zone", False: ""})
    st.dataframe(show, width="stretch", hide_index=True, column_config={
        "mean_dwell_s": st.column_config.NumberColumn("mean dwell s", format="%.1f"),
        "median_dwell_s": st.column_config.NumberColumn("median dwell s", format="%.1f"),
        "engaged_s": st.column_config.NumberColumn("engaged s", format="%.0f"),
        "blocked_s": st.column_config.NumberColumn("blocked s", format="%.0f"),
        "blocked_share": st.column_config.ProgressColumn("blocked share", min_value=0, max_value=1, format="%.0f%%"),
        "stop_ratio": st.column_config.ProgressColumn("pass-by→stop", min_value=0, max_value=1, format="%.0f%%"),
        "peak_C": st.column_config.NumberColumn("peak C", format="%.2f"),
        "bottleneck_min": st.column_config.NumberColumn("bottleneck min", format="%.1f")})
    st.caption("A visit counts only after 3 s in a zone. Engaged = stopped by choice (slow, free space, nothing in the way); "
               "blocked = stopped in a crowd or with the path ahead occupied. Pass-by→stop = engaged stoppers ÷ people passing within 2 m.")
    a, b = st.columns(2)
    with a:
        st.markdown("**Engaged vs blocked dwell** (person-seconds)")
        long = table.melt(id_vars="zone", value_vars=["engaged_s", "blocked_s"], var_name="kind", value_name="seconds")
        long["kind"] = long["kind"].str.replace("_s", "")
        fig = px.bar(long, x="zone", y="seconds", color="kind", barmode="stack",
                     color_discrete_map={"engaged": STATE_COL["engaged"], "blocked": STATE_COL["blocked"]})
        fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), legend_title=None)
        st.plotly_chart(fig, width="stretch")
    with b:
        st.markdown("**Congestion score C over time**")
        tsp = ts.copy()
        if src == "sim":
            tsp = tsp.assign(t=tsp["t"] / 3600 + 8)
        fig = px.line(tsp, x="t", y="C", color="zone")
        fig.add_hline(y=cfg_thr, line_dash="dash", line_color="#d9534f", annotation_text="flag threshold")
        fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), legend_title=None,
                          xaxis_title="seconds" if src == "clip" else "hour of day")
        st.plotly_chart(fig, width="stretch")
    st.markdown("**Flagged bottlenecks**")
    if bn is not None and len(bn):
        st.dataframe(bn.assign(t_start=bn.t_start.map(lambda v: fmt_t(v, src)), t_end=bn.t_end.map(lambda v: fmt_t(v, src)))
                     .drop(columns=["alert_t"]).round(2), width="stretch", hide_index=True)
    else:
        st.info("No zone was congested for long enough to flag in this data.")
    dead = table[table["dead_zone"]]
    st.markdown("**Dead zones**")
    if len(dead):
        for _, r in dead.iterrows():
            st.warning(f"**{r['zone']}**: {int(r['visits'])} visits on {r['area_m2']:.0f} m² - distance-adjusted visit rate in the bottom decile.")
    else:
        st.info("No dead zone: no zone is both in the bottom decile and far under the store median.")
    qs = [z for z in zones.of_kind("queue")]
    for z in qs:
        q = get(src, f"queue_{z.name.replace(' ', '_').lower()}.parquet")
        if q is not None:
            st.markdown(f"**Queue at {z.name}** (length and estimated wait = length ÷ service rate)")
            qq = q.assign(t=q["t"] / 3600 + 8) if src == "sim" else q
            fig = go.Figure()
            fig.add_scatter(x=qq["t"], y=qq["queue_len"], name="queue length", line=dict(color="#d9534f"))
            fig.add_scatter(x=qq["t"], y=qq["est_wait_min"], name="est. wait (min)", line=dict(color="#3b8ed0"), yaxis="y2")
            fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), yaxis2=dict(overlaying="y", side="right"),
                              xaxis_title="seconds" if src == "clip" else "hour of day")
            st.plotly_chart(fig, width="stretch")

# ================================================================== JOURNEYS
with tab_j:
    jt = get(src, "journeys.parquet")
    tr = get(src, "tracks.parquet")
    st.subheader("Customer journeys")
    if jt is None or tr is None or not len(jt):
        st.info("No journeys.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Journeys", len(jt))
        m2.metric("Median path length", f"{jt['path_m'].median():.1f} m")
        m3.metric("Median time in view", f"{jt['duration_s'].median():.0f} s")
        n_show = st.slider("Paths drawn", 5, min(300, len(jt)), min(60, len(jt)))
        ids = jt.sort_values("path_m", ascending=False)["track_id"].head(n_show).tolist()
        fig = plan_figure(zones, plan_png)
        sub = tr[tr["track_id"].isin(ids)].sort_values(["track_id", "t"])
        for tid, g in sub.groupby("track_id"):
            if src == "sim" and len(g) > 600:
                g = g.iloc[::5]
            fig.add_scatter(x=g["x"], y=g["y"], mode="lines", line=dict(width=2), opacity=.75, hoverinfo="text",
                            text=f"person #{tid}")
            fig.add_scatter(x=g["x"].iloc[[-1]], y=g["y"].iloc[[-1]], mode="markers", marker=dict(size=6), hoverinfo="skip")
        st.plotly_chart(fig, width="stretch")
        from flowsight.journeys import sankey_links, top_routes
        sk = sankey_links(jt)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Most common zone routes**")
            st.dataframe(pd.DataFrame([(" → ".join(r) if r else "(no zone visit ≥ 3 s)", n) for r, n in top_routes(jt, 8)],
                                      columns=["route", "people"]), hide_index=True, width="stretch")
        with c2:
            st.markdown("**Flow between zones**")
            if len(sk):
                names = sorted(set(sk["source"]) | set(sk["target"]))
                ix = {n: i for i, n in enumerate(names)}
                f2 = go.Figure(go.Sankey(node=dict(label=names, pad=14, thickness=14),
                                         link=dict(source=sk["source"].map(ix), target=sk["target"].map(ix), value=sk["count"])))
                f2.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(f2, width="stretch")
            else:
                st.info("Nobody visited two zones in a row in this data.")

# ================================================================== FORECAST
with tab_fc:
    st.subheader("Bottleneck forecast, 5 and 10 minutes ahead")
    st.warning("Evaluated on a **simulated** held-out store day: a 35 s clip is too short to forecast minutes ahead.")
    fe = (OUT / "forecast" / "forecast_eval.parquet")
    meta = (OUT / "forecast" / "forecast_meta.json")
    if not fe.exists():
        st.info("Run `python -m flowsight.simulate` to simulate store days and train the forecaster.")
    else:
        fdf, fm = pd.read_parquet(fe), json.loads(meta.read_text())
        zsel = st.selectbox("Zone", sorted(fdf["zone"].unique()), index=0)
        h = st.radio("Horizon", fm["horizons_min"], horizontal=True, format_func=lambda v: f"{v} min")
        g = fdf[(fdf["day"] == 0) & (fdf["zone"] == zsel) & (fdf["horizon_min"] == h)].sort_values("target_t")
        fig = go.Figure()
        hr = g["target_t"] / 3600 + 8
        fig.add_scatter(x=hr, y=g["actual_C"], name="actual C", line=dict(color="#333"))
        fig.add_scatter(x=hr, y=g["pred_C"], name=f"forecast made {h} min earlier", line=dict(color="#f0803c"))
        fig.add_scatter(x=hr, y=g["persist_C"], name="naive: assume no change", line=dict(color="#aaa", dash="dot"))
        fig.add_hline(y=fm["threshold"], line_dash="dash", line_color="#d9534f", annotation_text="congestion threshold")
        fig.add_hline(y=fm["alert_levels"][str(h)], line_dash="dot", line_color="#f0803c", annotation_text="forecast alert level")
        fig.update_layout(height=380, xaxis_title="hour of day (simulated)", yaxis_title="congestion C",
                          margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")
        k = fm["metrics"].get(str(h), {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE (LightGBM)", f"{k.get('mae', float('nan')):.3f}")
        c2.metric("MAE (naive)", f"{k.get('mae_naive', float('nan')):.3f}")
        c3.metric("Flag F1 (LightGBM)", f"{k.get('f1', float('nan')):.2f}")
        c4.metric("Flag F1 (naive)", f"{k.get('f1_naive', float('nan')):.2f}")
        st.caption(f"Trained on {fm['train_days']} simulated days, alert level tuned on {fm['validation_days']}, scored on {fm['test_days']} unseen days "
                   f"(pooled; the chart shows the first). Onsets: {k.get('n_onsets')} zone congestions began; "
                   f"warned {h} min ahead: {k.get('onset_recall', float('nan')):.0%} (naive {k.get('onset_recall_naive', float('nan')):.0%}).")
        fa = [a for a in (get("sim", "alerts.json") or []) if a["kind"] == "forecast"][:5]
        if fa:
            st.markdown("**Forecast-driven actions**")
            for a in fa:
                st.markdown(f"- `{fmt_t(a['t'], 'sim')}` **{a['action']}** - {a['reason']}")

# ================================================================== METRICS
with tab_m:
    st.subheader("Evaluation: measured vs target")
    mf = OUT / "metrics.json"
    if not mf.exists():
        st.info("Run `python -m eval.run_eval` to measure.")
    else:
        md = json.loads(mf.read_text())
        rows = []
        for m in md["metrics"]:
            rows.append({"metric": m["name"], "target": m["target"], "result": m["value"],
                         "vs target": {True: "✅ met", False: "❌ not met", None: "-"}[m["met"]],
                         "status": m["status"], "measured on": m["dataset"]})
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, width="stretch")
        st.caption("`measured` = we ran it. `target` = goal only, not yet measured. Targets are never presented as results. "
                   "Details and caveats: METRICS.md.")
        md_file = ROOT / "METRICS.md"
        if md_file.exists():
            with st.expander("METRICS.md"):
                st.markdown(md_file.read_text(encoding="utf-8"))
