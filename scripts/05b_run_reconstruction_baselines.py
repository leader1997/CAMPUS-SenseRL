#!/usr/bin/env python
"""Evaluate classical causal reconstruction baselines (and optional non-causal upper bound)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from campus_senserl.evaluation import regression_metrics, event_binary_metrics
from campus_senserl.evaluation.events import high_co2_events
from campus_senserl.models.baselines import (
    _per_series_mask_eval,
    historical_mean_predict,
    historical_median_predict,
    knn_neighbor_predict,
    linear_extrapolation_predict,
    locf_predict,
    make_random_mask,
    noncausal_interpolation_predict,
    seasonal_historical_predict,
    tabular_tree_predict,
)
from campus_senserl.utils import ensure_dir, load_yaml, repo_root, save_json


def _event_metrics(y_true: np.ndarray, y_pred: np.ndarray, thr: float = 1000.0) -> dict:
    return event_binary_metrics(high_co2_events(y_true, thr), high_co2_events(y_pred, thr))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--mask-rate", type=float, default=0.4)
    parser.add_argument("--max-sensors", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = repo_root()
    cfg = load_yaml(root / "configs" / "reconstruction.yaml")
    out_dir = ensure_dir(root / "results" / "reconstruction_baselines")

    wide = pd.read_parquet(root / "data" / "processed" / "co2_wide_observed.parquet")
    panel = pd.read_parquet(root / "data" / "processed" / "co2_panel_15min.parquet")
    slot_split = panel.drop_duplicates("slot").set_index("slot")["split"]
    wide_split = wide.loc[slot_split.reindex(wide.index) == args.split]
    if wide_split.empty:
        raise RuntimeError(f"No wide rows for split={args.split}")

    # Subset sensors for tractable classical baselines
    counts = wide_split.notna().sum().sort_values(ascending=False)
    cols = counts.head(args.max_sensors).index.tolist()
    wide_split = wide_split[cols]

    mask = make_random_mask(wide_split, rate=args.mask_rate, seed=args.seed)

    predictors = {
        "locf": locf_predict,
        "historical_mean": historical_mean_predict,
        "historical_median": historical_median_predict,
        "linear_extrapolation": linear_extrapolation_predict,
        "seasonal_historical": seasonal_historical_predict,
    }
    results = {}
    for name, fn in predictors.items():
        res = _per_series_mask_eval(wide_split, mask, fn)
        res.name = name
        m = regression_metrics(res.y_true, res.y_pred)
        m.update({f"event_{k}": v for k, v in _event_metrics(res.y_true, res.y_pred).items()})
        m["causal"] = True
        results[name] = m
        print(f"  {name}: MAE={m['mae']:.2f} RMSE={m['rmse']:.2f} event_recall={m.get('event_recall', m.get('event_recall', float('nan')))}")

    # Non-causal upper bound (explicitly labeled)
    if cfg.get("evaluation", {}).get("include_noncausal_interpolation_reference", True):
        res = _per_series_mask_eval(wide_split, mask, noncausal_interpolation_predict)
        m = regression_metrics(res.y_true, res.y_pred)
        m.update({f"event_{k}": v for k, v in _event_metrics(res.y_true, res.y_pred).items()})
        m["causal"] = False
        m["note"] = "OFFLINE UPPER BOUND — uses future observations; not for online/RL"
        results["noncausal_interpolation_UPPER_BOUND"] = m
        print(f"  noncausal_interp (UPPER BOUND): MAE={m['mae']:.2f}")

    # KNN neighbours if adjacency available
    adj_path = root / "results" / "graphs" / "adjacency_hybrid.npy"
    order_path = root / "results" / "graphs" / "node_order.json"
    if adj_path.exists() and order_path.exists():
        import json

        node_order = json.loads(order_path.read_text(encoding="utf-8"))["node_order"]
        # Restrict to overlapping columns
        common = [c for c in node_order if c in wide_split.columns]
        if len(common) >= 3:
            A = np.load(adj_path)
            idx = [node_order.index(c) for c in common]
            A_sub = A[np.ix_(idx, idx)]
            res = knn_neighbor_predict(wide_split[common], mask[common], A_sub, common, k=5)
            m = regression_metrics(res.y_true, res.y_pred)
            m.update({f"event_{k}": v for k, v in _event_metrics(res.y_true, res.y_pred).items()})
            m["causal"] = True
            results["knn_neighbors"] = m
            print(f"  knn_neighbors: MAE={m['mae']:.2f}")

    # Tree models on panel features (train fit only)
    feat_cols = [
        c
        for c in [
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "is_weekend",
            "co2_lag1",
            "co2_delta1",
            "co2_rollmean_4",
            "co2_rollmean_8",
            "motion",
            "temperature",
            "humidity",
            "rssi",
            "lsnr",
            "battery",
        ]
        if c in panel.columns
    ]
    train_panel = panel[(panel["split"] == "train") & (panel["deveui"].isin(cols))]
    eval_panel = panel[(panel["split"] == args.split) & (panel["deveui"].isin(cols))]
    # Simulate held-out: evaluate only on randomly masked observed rows
    rng = np.random.default_rng(args.seed)
    eval_obs = eval_panel.dropna(subset=["co2"]).copy()
    hold = rng.random(len(eval_obs)) < args.mask_rate
    eval_hold = eval_obs.loc[hold]
    if feat_cols and len(train_panel.dropna(subset=["co2"])) > 100 and len(eval_hold) > 10:
        for model_name in ["extratrees", "lightgbm"]:
            try:
                res = tabular_tree_predict(train_panel, eval_hold, feat_cols, model_name=model_name)
                m = regression_metrics(res.y_true, res.y_pred)
                m.update({f"event_{k}": v for k, v in _event_metrics(res.y_true, res.y_pred).items()})
                m["causal"] = True
                results[model_name] = m
                print(f"  {model_name}: MAE={m['mae']:.2f}")
            except Exception as e:
                results[model_name] = {"error": str(e)}
                print(f"  {model_name}: FAILED {e}")

    payload = {
        "split": args.split,
        "mask_rate": args.mask_rate,
        "n_sensors": len(cols),
        "seed": args.seed,
        "results": results,
    }
    save_json(payload, out_dir / f"baselines_{args.split}_mask{args.mask_rate}.json")
    # Also CSV
    rows = []
    for name, m in results.items():
        if isinstance(m, dict) and "mae" in m:
            rows.append({"model": name, **{k: v for k, v in m.items() if not isinstance(v, dict)}})
    pd.DataFrame(rows).to_csv(out_dir / f"baselines_{args.split}_mask{args.mask_rate}.csv", index=False)
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
