"""Fair reconstruction benchmark: identical cohort, masks, and seeds for all models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from campus_senserl.evaluation import event_binary_metrics, regression_metrics
from campus_senserl.environment.event_detector import EventDetector
from campus_senserl.models.baselines import (
    historical_mean_predict,
    linear_extrapolation_predict,
    locf_predict,
    tabular_tree_predict,
)
from campus_senserl.utils import ensure_dir, load_yaml, repo_root, save_json, set_seed


DEFAULT_SEEDS = [42, 123, 2024, 3407, 9999]


def _wide_for_sensors(panel: pd.DataFrame, sensors: list[str], split: str) -> pd.DataFrame:
    sub = panel[(panel["split"] == split) & (panel["deveui"].isin(sensors))].copy()
    wide = sub.pivot_table(index="slot", columns="deveui", values="co2", aggfunc="last")
    return wide.reindex(columns=sensors)


def _observed_mask(panel: pd.DataFrame, sensors: list[str], split: str) -> pd.DataFrame:
    sub = panel[(panel["split"] == split) & (panel["deveui"].isin(sensors))].copy()
    wide = sub.pivot_table(index="slot", columns="deveui", values="observed", aggfunc="last")
    wide = wide.reindex(columns=sensors).fillna(0.0)
    return wide > 0.5


def make_mask_manifest(
    observed: pd.DataFrame,
    *,
    seed: int,
    mask_rate: float = 0.4,
    scheme: str = "random",
) -> pd.DataFrame:
    """Create a boolean hide-mask only on observed positions (True = hidden)."""
    rng = np.random.default_rng(seed)
    obs = observed.to_numpy(dtype=bool)
    hide = np.zeros_like(obs, dtype=bool)
    if scheme == "random":
        cand = np.argwhere(obs)
        n_hide = int(round(mask_rate * len(cand)))
        if n_hide > 0 and len(cand):
            pick = rng.choice(len(cand), size=min(n_hide, len(cand)), replace=False)
            for r, c in cand[pick]:
                hide[r, c] = True
    elif scheme == "block2":
        # Hide contiguous 2-step blocks for random starts
        t, n = obs.shape
        for j in range(n):
            for i in range(0, t - 1):
                if obs[i, j] and obs[i + 1, j] and rng.random() < mask_rate / 2:
                    hide[i, j] = True
                    hide[i + 1, j] = True
    else:
        raise ValueError(scheme)
    return pd.DataFrame(hide, index=observed.index, columns=observed.columns)


def _eval_series_predictor(
    wide: pd.DataFrame,
    hide: pd.DataFrame,
    predict_fn: Callable,
    event_detector: EventDetector,
) -> dict[str, float]:
    preds = []
    trues = []
    true_events = []
    pred_events = []
    # Align on shared index/columns
    cols = [c for c in wide.columns if c in hide.columns]
    idx = wide.index.intersection(hide.index)
    wide = wide.loc[idx, cols]
    hide = hide.loc[idx, cols]
    for col in cols:
        series = wide[col]
        mask = hide[col].astype(bool)
        visible = series.where(~mask)
        pred = predict_fn(visible, mask)
        if not isinstance(pred, pd.Series):
            pred = pd.Series(pred, index=series.index)
        pred = pred.reindex(series.index)
        m = mask.to_numpy(dtype=bool) & np.isfinite(series.to_numpy(dtype=float))
        yt = series.to_numpy(dtype=float)[m]
        yp = pred.to_numpy(dtype=float)[m]
        if len(yt) == 0:
            continue
        preds.append(yp)
        trues.append(yt)
        full_true = series.to_numpy(dtype=float)
        full_pred = pred.to_numpy(dtype=float)
        prev = np.roll(full_true, 1)
        prev[0] = np.nan
        prev_p = np.roll(full_pred, 1)
        prev_p[0] = np.nan
        te = event_detector.detect_step(full_true, prev)
        pe = event_detector.detect_step(full_pred, prev_p)
        true_events.append(te[m])
        pred_events.append(pe[m])

    if not trues:
        return {"mae": float("nan"), "n": 0}
    yt = np.concatenate(trues)
    yp = np.concatenate(preds)
    metrics = regression_metrics(yt, yp)
    te = np.concatenate(true_events) if true_events else np.array([], dtype=bool)
    pe = np.concatenate(pred_events) if pred_events else np.array([], dtype=bool)
    if len(te):
        metrics.update({f"event_{k}": v for k, v in event_binary_metrics(te, pe).items()})
    return metrics


def run_fair_benchmark(
    *,
    sensors: list[str],
    split: str = "val",
    seeds: list[int] | None = None,
    mask_rate: float = 0.4,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    root = repo_root()
    seeds = seeds or DEFAULT_SEEDS
    out_dir = ensure_dir(out_dir or root / "results" / "reconstruction_final" / "fair_benchmark")
    panel = pd.read_parquet(root / "data" / "processed" / "co2_panel_15min.parquet")
    wide = _wide_for_sensors(panel, sensors, split)
    observed = _observed_mask(panel, sensors, split)
    # Align frames
    idx = wide.index.intersection(observed.index)
    cols = [c for c in sensors if c in wide.columns and c in observed.columns]
    wide = wide.loc[idx, cols]
    observed = observed.loc[idx, cols]
    wide = wide.where(observed)

    rl_cfg = load_yaml(root / "configs" / "rl.yaml")
    detector = EventDetector.from_config(rl_cfg)

    # Persist masks for reuse
    masks_dir = ensure_dir(out_dir / "masks")
    results_rows = []

    methods = {
        "locf": lambda s, m: locf_predict(s, m),
        "linear_extrapolation": lambda s, m: linear_extrapolation_predict(s, m),
        "historical_mean": lambda s, m: historical_mean_predict(s, m),
    }

    for seed in seeds:
        set_seed(seed)
        hide = make_mask_manifest(observed, seed=seed, mask_rate=mask_rate, scheme="random")
        hide.to_parquet(masks_dir / f"mask_seed{seed}_rate{mask_rate}.parquet")
        manifest = {
            "seed": seed,
            "mask_rate": mask_rate,
            "split": split,
            "sensors": sensors,
            "n_hidden": int(hide.to_numpy().sum()),
        }
        save_json(manifest, masks_dir / f"manifest_seed{seed}.json")

        for name, fn in methods.items():
            m = _eval_series_predictor(wide, hide, fn, detector)
            results_rows.append({"method": name, "seed": seed, "mask_rate": mask_rate, **m})

        # ExtraTrees / LightGBM — evaluate ONLY on the same hide cells
        for model_name in ("extratrees", "lightgbm"):
            try:
                train_panel = panel[(panel["split"] == "train") & (panel["deveui"].isin(sensors))].copy()
                feat_cols = [
                    c
                    for c in ["co2_lag1", "co2_lag2", "co2_delta1", "hour_sin", "hour_cos", "motion"]
                    if c in train_panel.columns
                ]
                if not feat_cols:
                    raise RuntimeError("no feature columns for tree models")
                # Build eval rows for hidden (slot, deveui) only — vectorized join
                hide_long = hide.stack()
                hide_pairs = hide_long[hide_long].reset_index()
                hide_pairs.columns = ["slot", "deveui", "hidden"]
                hide_pairs = hide_pairs.drop(columns=["hidden"])
                eval_panel = panel.merge(hide_pairs, on=["slot", "deveui"], how="inner")
                eval_panel = eval_panel[eval_panel["deveui"].isin(sensors)]
                if len(eval_panel) == 0:
                    raise RuntimeError("no hidden eval rows")
                res = tabular_tree_predict(
                    train_panel,
                    eval_panel,
                    feature_cols=feat_cols,
                    target="co2",
                    model_name=model_name,
                )
                yt = np.asarray(res.y_true, dtype=float)
                yp = np.asarray(res.y_pred, dtype=float)
                metrics = regression_metrics(yt, yp) if len(yt) else {"mae": float("nan"), "n": 0}
                # Event metrics on same cells
                if len(yt):
                    te = detector.detect_step(yt, None)
                    pe = detector.detect_step(yp, None)
                    metrics.update({f"event_{k}": v for k, v in event_binary_metrics(te, pe).items()})
                results_rows.append({"method": model_name, "seed": seed, "mask_rate": mask_rate, **metrics})
            except Exception as exc:
                results_rows.append(
                    {"method": model_name, "seed": seed, "mask_rate": mask_rate, "error": str(exc), "mae": float("nan")}
                )

        # ST-GNN — real forward on same hide mask (no LOCF relabeling)
        try:
            import torch

            from campus_senserl.models.graph_reconstruction import load_reconstruction_model

            ckpt = root / "results" / "models" / "reconstruction" / "best_model.pt"
            if not ckpt.exists():
                raise FileNotFoundError(str(ckpt))
            model, meta = load_reconstruction_model(ckpt)
            node_order = [str(x) for x in meta["node_order"]]
            if [str(s) for s in sensors] != node_order:
                raise RuntimeError(
                    f"ST-GNN node_order mismatch: ckpt={len(node_order)} cohort={len(sensors)}"
                )
            device = meta.get("device", "cpu")
            adj = meta.get("adjacency")
            if adj is None:
                from campus_senserl.environment.graph_utils import load_adjacency_for_sensors

                adj = load_adjacency_for_sensors("hybrid", sensors)
            w = int(getattr(model, "history_window", 8))
            values = wide.to_numpy(dtype=np.float32)
            obs = observed.to_numpy(dtype=bool)
            hid = hide.to_numpy(dtype=bool)
            # Server-visible: observed and not hidden (never feed GT of hidden cells)
            visible = obs & ~hid
            server_values = values.copy()
            server_values[~visible] = np.nan
            T, N = values.shape
            preds = np.full((T, N), np.nan, dtype=np.float32)
            # time feats from panel calendar
            cal = panel.drop_duplicates("slot").set_index("slot").reindex(wide.index)
            time_feats = np.stack(
                [
                    cal.get("hour_sin", pd.Series(0, index=wide.index)).fillna(0).to_numpy(np.float32),
                    cal.get("hour_cos", pd.Series(0, index=wide.index)).fillna(0).to_numpy(np.float32),
                    cal.get("dow_sin", pd.Series(0, index=wide.index)).fillna(0).to_numpy(np.float32),
                    cal.get("dow_cos", pd.Series(0, index=wide.index)).fillna(0).to_numpy(np.float32),
                ],
                axis=-1,
            )
            aoi = np.zeros((T, N), dtype=np.float32)
            for ti in range(T):
                for si in range(N):
                    if visible[ti, si]:
                        aoi[ti, si] = 0.0
                    else:
                        aoi[ti, si] = aoi[ti - 1, si] + 1.0 if ti > 0 else 8.0
            adj_t = torch.from_numpy(np.asarray(adj, dtype=np.float32)).float().to(device)
            model.eval()
            with torch.no_grad():
                for start in range(0, max(T - w + 1, 1)):
                    sl = slice(start, start + w)
                    if start + w > T:
                        break
                    batch_vals = np.nan_to_num(server_values[sl], nan=0.0)[None]
                    batch_mask = visible[sl][None]
                    batch_aoi = aoi[sl][None]
                    batch_tf = time_feats[sl][None]
                    mu, _lv = model(
                        torch.from_numpy(batch_vals).float().to(device),
                        torch.from_numpy(batch_mask).to(device),
                        torch.from_numpy(batch_aoi).float().to(device),
                        torch.from_numpy(batch_tf).float().to(device),
                        adj_t,
                    )
                    preds[start + w - 1] = mu.squeeze(0).cpu().numpy()
            # Score only hidden & observed cells where prediction exists
            m = hid & obs & np.isfinite(values) & np.isfinite(preds)
            yt = values[m]
            yp = preds[m]
            metrics = regression_metrics(yt, yp) if m.any() else {"mae": float("nan"), "n": 0}
            if m.any():
                te = detector.detect_step(yt, None)
                pe = detector.detect_step(yp, None)
                metrics.update({f"event_{k}": v for k, v in event_binary_metrics(te, pe).items()})
            results_rows.append({"method": "stgnn", "seed": seed, "mask_rate": mask_rate, **metrics})
        except Exception as exc:
            results_rows.append({"method": "stgnn", "seed": seed, "error": str(exc), "mae": float("nan")})

    df = pd.DataFrame(results_rows)
    df.to_csv(out_dir / "fair_results_raw.csv", index=False)

    # Aggregate mean±std
    agg_rows = []
    for method, g in df.groupby("method"):
        for metric in ("mae", "rmse", "smape", "r2", "event_recall", "event_f1"):
            if metric not in g.columns:
                continue
            vals = pd.to_numeric(g[metric], errors="coerce").dropna()
            if len(vals) == 0:
                continue
            agg_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    "n_seeds": int(len(vals)),
                }
            )
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(out_dir / "fair_results_summary.csv", index=False)
    save_json({"seeds": seeds, "mask_rate": mask_rate, "split": split, "n_sensors": len(sensors)}, out_dir / "benchmark_config.json")

    # Decision helper — ignore pending/error placeholders
    mae_table = agg[agg["metric"] == "mae"].copy() if len(agg) else agg
    if len(mae_table):
        mae_table = mae_table[~mae_table["method"].astype(str).str.contains("pending|error", case=False, regex=True)]
        mae_table = mae_table[np.isfinite(mae_table["mean"])]
        mae_table = mae_table.sort_values("mean")
    best = None
    if len(mae_table):
        best = str(mae_table.iloc[0]["method"])
    decision = {
        "best_by_mae": best,
        "mae_ranking": mae_table[["method", "mean", "std"]].to_dict(orient="records") if len(mae_table) else [],
        "rule": "Prefer ST-GNN only with genuine advantage; else use best causal (ExtraTrees/LightGBM/LOCF).",
        "summary_csv": str(out_dir / "fair_results_summary.csv"),
    }
    save_json(decision, out_dir / "reconstructor_decision.json")
    return {"raw": df, "summary": agg, "decision": decision, "out_dir": str(out_dir)}
