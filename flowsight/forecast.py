"""Congestion forecaster: predict C_z at t+5 and t+10 minutes from recent counts, speeds and entrance inflow.

LightGBM per horizon, trained on SIMULATED days (a 35 s real clip cannot teach a 10-minute forecast).
Always reported against the naive baseline "tomorrow looks like now" (C_{t+h} = C_t): the model has to
beat that to earn its place.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

FEATURES = ["zone_id", "count", "v", "C", "count_l1", "count_l2", "count_l5", "C_l1", "C_l2", "C_l5", "C_l10",
            "C_mean5", "inflow1", "inflow5", "inflow10", "inflow_trend", "tod_sin", "tod_cos"]
LGB_PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=20, bagging_fraction=0.8,
                  bagging_freq=1, feature_fraction=0.8, seed=0, verbose=-1)
N_ROUNDS = 400


def build_features(ts: pd.DataFrame, arrivals: np.ndarray, zone_names: list[str], horizons: list[int],
                   bin_s: float = 60.0, day_s: float = 12 * 3600) -> pd.DataFrame:
    """One row per (zone, minute): lagged state of the zone + store-wide entrance inflow."""
    n = int(ts["t"].max() // bin_s) + 1
    inflow = np.bincount((np.asarray(arrivals) // bin_s).astype(int), minlength=n)[:n].astype(float)
    inf = pd.Series(inflow)
    i5, i10 = inf.rolling(5, min_periods=1).sum(), inf.rolling(10, min_periods=1).sum()
    frames = []
    for zi, z in enumerate(zone_names):
        g = ts[ts["zone"] == z].sort_values("t").reset_index(drop=True)
        f = pd.DataFrame(dict(t=g["t"], zone=z, zone_id=zi, count=g["count"], v=g["v_mean"], C=g["C"]))
        for L in (1, 2, 5):
            f[f"count_l{L}"] = f["count"].shift(L).fillna(0.0)
        for L in (1, 2, 5, 10):
            f[f"C_l{L}"] = f["C"].shift(L).fillna(0.0)
        f["C_mean5"] = f["C"].rolling(5, min_periods=1).mean()
        f["inflow1"], f["inflow5"], f["inflow10"] = inf.iloc[:len(f)].to_numpy(), i5.iloc[:len(f)].to_numpy(), i10.iloc[:len(f)].to_numpy()
        f["inflow_trend"] = f["inflow5"] - f["inflow5"].shift(5).fillna(0.0)
        frac = (f["t"] % day_s) / day_s
        f["tod_sin"], f["tod_cos"] = np.sin(2 * np.pi * frac), np.cos(2 * np.pi * frac)
        for h in horizons:
            f[f"y{h}"] = f["C"].shift(-h)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def _prf(y: np.ndarray, p: np.ndarray) -> dict:
    tp, fp, fn = int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return dict(precision=pr, recall=rc, f1=2 * pr * rc / (pr + rc) if pr + rc else 0.0)


def _best_threshold(pred: np.ndarray, flag: np.ndarray, default: float) -> float:
    """Pick the forecast alert level that maximises F1 on validation data (a regressor shrinks rare peaks,
    so the detection threshold itself is the wrong operating point for the forecast)."""
    if flag.sum() == 0:
        return default * 0.7
    grid = np.arange(0.05, default + 0.0001, 0.01)
    f1 = [_prf(flag, pred > g)["f1"] for g in grid]
    return float(grid[int(np.argmax(f1))])


def train_and_evaluate(train: list[tuple[pd.DataFrame, np.ndarray]], tests: list[tuple[pd.DataFrame, np.ndarray]],
                       horizons: list[int], thr: float, zone_names: list[str]) -> dict:
    """Fit on all but the last training day, tune the alert level on that last day, score on the unseen test day."""
    fit, val = train[:-1], train[-1]
    tr = pd.concat([build_features(ts, arr, zone_names, horizons) for ts, arr in fit], ignore_index=True)
    va = build_features(val[0], val[1], zone_names, horizons)
    te = pd.concat([build_features(ts, arr, zone_names, horizons).assign(day=i) for i, (ts, arr) in enumerate(tests)],
                   ignore_index=True)
    models, ev, metrics, ops = {}, [], {}, {}
    for h in horizons:
        y = f"y{h}"
        trh = tr.dropna(subset=[y])
        m = lgb.train(LGB_PARAMS, lgb.Dataset(trh[FEATURES], trh[y], categorical_feature=["zone_id"]), N_ROUNDS)
        models[h] = m
        vv = va.dropna(subset=[y])
        vv = vv[vv["t"] >= 600]
        op = _best_threshold(np.clip(m.predict(vv[FEATURES]), 0, 1), vv[y].to_numpy() > thr, thr)
        ops[h] = op
        tt = te.dropna(subset=[y])
        tt = tt[tt["t"] >= 600]                                  # need 10 min of history
        pred = np.clip(m.predict(tt[FEATURES]), 0, 1)
        actual, naive = tt[y].to_numpy(), tt["C"].to_numpy()
        yt, yp, yn = actual > thr, pred > op, naive > thr
        # onset recall: of the minutes where a zone *becomes* congested, how many did we warn h minutes earlier?
        prev = (tt.assign(a=actual).groupby(["day", "zone"])["a"].shift(1).fillna(0).to_numpy() > thr)
        onset = yt & ~prev
        metrics[str(h)] = dict(mae=float(np.abs(pred - actual).mean()), mae_naive=float(np.abs(naive - actual).mean()),
                               alert_level=op, **_prf(yt, yp), **{f"{k}_naive": v for k, v in _prf(yt, yn).items()},
                               n_congested_minutes=int(yt.sum()), n_onsets=int(onset.sum()),
                               onset_recall=float((yp & onset).sum() / onset.sum()) if onset.sum() else float("nan"),
                               onset_recall_naive=float((yn & onset).sum() / onset.sum()) if onset.sum() else float("nan"))
        ev.append(pd.DataFrame(dict(day=tt["day"].to_numpy(), zone=tt["zone"].to_numpy(), horizon_min=h, issue_t=tt["t"].to_numpy(),
                                    target_t=tt["t"].to_numpy() + h * 60, pred_C=pred, actual_C=actual, persist_C=naive,
                                    count=tt["count"].to_numpy(), thr=op)))
    ev = pd.concat(ev, ignore_index=True)
    alerts = ev[ev["day"] == 0].rename(columns={"issue_t": "t"})[["t", "zone", "horizon_min", "pred_C", "count", "thr"]]
    meta = dict(horizons_min=horizons, threshold=thr, alert_levels={str(h): v for h, v in ops.items()}, metrics=metrics,
                train_days=len(fit), validation_days=1, test_days=len(tests), features=FEATURES,
                note="trained, tuned and tested on SIMULATED days")
    return dict(models=models, eval=ev, alerts=alerts, meta=meta)


def save(res: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for h, m in res["models"].items():
        m.save_model(str(out / f"lgbm_h{h}.txt"))
    res["eval"].to_parquet(out / "forecast_eval.parquet")
    (out / "forecast_meta.json").write_text(json.dumps(res["meta"], indent=2))
