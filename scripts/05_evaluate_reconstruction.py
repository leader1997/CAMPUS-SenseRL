#!/usr/bin/env python
"""Step 5: Evaluate reconstruction model and baselines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from campus_senserl.models.baselines import (
    _per_series_mask_eval,
    historical_mean_predict,
    locf_predict,
    make_random_mask,
)
from campus_senserl.models.graph_reconstruction import (
    apply_mask_scheme,
    load_reconstruction_model,
    predict_reconstruction,
)
from campus_senserl.models.uncertainty import summarize_uncertainty_metrics
from campus_senserl.utils import ensure_dir, load_yaml, repo_root, save_json


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "n": 0}
    err = y_pred[m] - y_true[m]
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "n": int(m.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate reconstruction")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--mask-rate", type=float, default=0.4)
    parser.add_argument("--max-sensors", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "auto"])
    args = parser.parse_args()

    root = repo_root()
    cfg = load_yaml(root / "configs" / "reconstruction.yaml")
    out_dir = ensure_dir(root / "results" / "models" / "reconstruction" / "eval")

    wide = pd.read_parquet(root / "data" / "processed" / "co2_wide_observed.parquet")
    panel = pd.read_parquet(root / "data" / "processed" / "co2_panel_15min.parquet")
    slot_split = panel.drop_duplicates("slot").set_index("slot")["split"]
    split_mask = slot_split.reindex(wide.index) == args.split
    wide_split = wide.loc[split_mask]
    if wide_split.empty:
        print(f"[warn] No rows for split={args.split}; using all data.")
        wide_split = wide

    if args.max_sensors:
        wide_split = wide_split.iloc[:, : args.max_sensors]

    mask = make_random_mask(wide_split, rate=args.mask_rate, seed=int(cfg.get("seed", 42)))
    locf_res = _per_series_mask_eval(wide_split, mask, locf_predict)
    hm_res = _per_series_mask_eval(wide_split, mask, historical_mean_predict)
    results = {
        "locf": _metrics(locf_res.y_true, locf_res.y_pred),
        "historical_mean": _metrics(hm_res.y_true, hm_res.y_pred),
    }

    ckpt = root / "results" / "models" / "reconstruction" / "best_model.pt"
    if ckpt.exists():
        device = None if args.device in (None, "auto") else args.device
        model, meta = load_reconstruction_model(ckpt, device=device)
        device = meta["device"]
        adj = meta["adjacency"]
        if adj is None:
            adj = np.eye(wide_split.shape[1], dtype=np.float32)
        # Align adjacency to selected columns if needed
        if adj.shape[0] != wide_split.shape[1]:
            adj = np.eye(wide_split.shape[1], dtype=np.float32)
        obs = wide_split.notna().to_numpy()
        cal = panel.drop_duplicates("slot").set_index("slot").reindex(wide_split.index)
        time_feats = np.stack(
            [
                cal["hour_sin"].fillna(0).to_numpy(dtype=np.float32),
                cal["hour_cos"].fillna(0).to_numpy(dtype=np.float32),
                cal["dow_sin"].fillna(0).to_numpy(dtype=np.float32),
                cal["dow_cos"].fillna(0).to_numpy(dtype=np.float32),
            ],
            axis=-1,
        )
        holdout = apply_mask_scheme(obs, "random", rate=args.mask_rate, seed=int(cfg.get("seed", 42)))
        input_mask = obs & ~holdout
        server_values = np.where(input_mask, wide_split.to_numpy(), np.nan)
        server_values = np.nan_to_num(server_values, nan=0.0).astype(np.float32)
        aoi = np.zeros_like(server_values, dtype=np.float32)
        last_seen = np.full(server_values.shape[1], -1, dtype=int)
        for t in range(server_values.shape[0]):
            for s in range(server_values.shape[1]):
                if input_mask[t, s]:
                    last_seen[s] = t
                    aoi[t, s] = 0.0
                elif last_seen[s] >= 0:
                    aoi[t, s] = float(t - last_seen[s])
                else:
                    aoi[t, s] = float(t + 1)
        means, logvars = predict_reconstruction(
            model, server_values, input_mask, aoi, time_feats, adj, device=device
        )
        gt = wide_split.to_numpy(dtype=np.float32)
        eval_mask = holdout[model.history_window - 1 :]
        y_true = gt[model.history_window - 1 :][eval_mask]
        y_pred = means[0][eval_mask]
        std = np.sqrt(np.exp(np.clip(logvars[0], -10, 10)))[eval_mask]
        results["masked_st_graph"] = _metrics(y_true, y_pred)
        results["masked_st_graph_uncertainty"] = summarize_uncertainty_metrics(
            y_true, y_pred, std=std
        )
        results["device"] = device

    save_json(results, out_dir / f"metrics_{args.split}.json")
    print(f"[done] wrote {out_dir / f'metrics_{args.split}.json'}")
    for name, m in results.items():
        if isinstance(m, dict) and "mae" in m:
            print(f"  {name}: mae={m['mae']:.3f} rmse={m['rmse']:.3f} n={m['n']}")


if __name__ == "__main__":
    main()
