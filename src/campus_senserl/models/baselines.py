"""Causal reconstruction baselines (no future leakage).

Non-causal linear interpolation is provided only as an offline upper-bound
reference and is explicitly labeled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor


@dataclass
class BaselineResult:
    name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    causal: bool = True


def _per_series_mask_eval(
    wide: pd.DataFrame,
    mask_missing: pd.DataFrame,
    predict_fn: Callable[[pd.Series, pd.Series], pd.Series],
) -> BaselineResult:
    """mask_missing=True where value should be reconstructed (held out)."""
    trues = []
    preds = []
    for col in wide.columns:
        s = wide[col]
        m = mask_missing[col].astype(bool)
        # Only evaluate where ground truth exists
        eval_idx = m & s.notna()
        if not eval_idx.any():
            continue
        avail = s.where(~m)  # hide held-out from predictor
        pred = predict_fn(avail, m)
        trues.append(s[eval_idx].to_numpy(dtype=float))
        preds.append(pred[eval_idx].to_numpy(dtype=float))
    if not trues:
        return BaselineResult("empty", np.array([]), np.array([]))
    return BaselineResult("", np.concatenate(trues), np.concatenate(preds))


def locf_predict(series: pd.Series, mask: pd.Series) -> pd.Series:
    return series.ffill()


def historical_mean_predict(series: pd.Series, mask: pd.Series) -> pd.Series:
    """Expanding mean using only past available observations."""
    past = series.expanding(min_periods=1).mean()
    # shift so current is excluded when present; for missing, expanding already ignores NaN
    return past.ffill()


def historical_median_predict(series: pd.Series, mask: pd.Series) -> pd.Series:
    return series.expanding(min_periods=1).median().ffill()


def linear_extrapolation_predict(series: pd.Series, mask: pd.Series) -> pd.Series:
    """Linear extrapolation from the last two past observations (causal)."""
    vals = series.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan, dtype=float)
    last_idx = None
    prev_idx = None
    last_val = None
    prev_val = None
    for i, v in enumerate(vals):
        if np.isfinite(v):
            prev_idx, prev_val = last_idx, last_val
            last_idx, last_val = i, v
            out[i] = v
        else:
            if last_idx is not None and prev_idx is not None and last_idx != prev_idx:
                slope = (last_val - prev_val) / (last_idx - prev_idx)
                out[i] = last_val + slope * (i - last_idx)
            elif last_val is not None:
                out[i] = last_val
    return pd.Series(out, index=series.index)


def seasonal_historical_predict(series: pd.Series, mask: pd.Series) -> pd.Series:
    """Same time-of-day mean from past days only (96 slots/day for 15-min)."""
    vals = series.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    # Assume regular index; period = 96 for 15-min
    period = 96
    buckets: list[list[float]] = [[] for _ in range(period)]
    for i, v in enumerate(vals):
        b = i % period
        if buckets[b]:
            out[i] = float(np.mean(buckets[b]))
        elif np.isfinite(v):
            out[i] = v
        else:
            # fallback: global past mean
            past = vals[:i]
            past = past[np.isfinite(past)]
            out[i] = float(np.mean(past)) if len(past) else np.nan
        if np.isfinite(v):
            buckets[b].append(float(v))
    return pd.Series(out, index=series.index)


def noncausal_interpolation_predict(series: pd.Series, mask: pd.Series) -> pd.Series:
    """OFFLINE UPPER BOUND ONLY — uses future observations. Not for online/RL."""
    return series.interpolate(method="linear", limit_direction="both")


def knn_neighbor_predict(
    wide: pd.DataFrame,
    mask_missing: pd.DataFrame,
    adjacency: np.ndarray,
    node_order: list[str],
    k: int = 5,
) -> BaselineResult:
    """Reconstruct using mean of currently available spatial/correlation neighbours."""
    idx = {n: i for i, n in enumerate(node_order)}
    trues, preds = [], []
    cols = [c for c in node_order if c in wide.columns]
    for t in range(len(wide)):
        row = wide.iloc[t]
        mrow = mask_missing.iloc[t]
        available = row.where(~mrow.astype(bool))
        for col in cols:
            if not bool(mrow[col]) or not np.isfinite(row[col]):
                continue
            i = idx[col]
            neigh = np.argsort(-adjacency[i])[: max(k * 3, k)]
            vals = []
            for j in neigh:
                n = node_order[j]
                if n == col or n not in available.index:
                    continue
                v = available[n]
                if np.isfinite(v):
                    vals.append(float(v))
                if len(vals) >= k:
                    break
            if vals:
                pred = float(np.mean(vals))
            else:
                # fallback LOCF-like: previous available in series
                hist = available[col]
                pred = float(hist) if np.isfinite(hist) else np.nan
                # better: use column ffill up to t
                past = wide[col].iloc[: t + 1].where(~mask_missing[col].iloc[: t + 1].astype(bool))
                past_vals = past.dropna()
                pred = float(past_vals.iloc[-1]) if len(past_vals) else np.nan
            trues.append(float(row[col]))
            preds.append(pred)
    return BaselineResult("knn_neighbors", np.asarray(trues), np.asarray(preds), causal=True)


def tabular_tree_predict(
    panel_train: pd.DataFrame,
    panel_eval: pd.DataFrame,
    feature_cols: list[str],
    target: str = "co2",
    model_name: str = "extratrees",
) -> BaselineResult:
    """Train tree model on train rows where target observed; predict masked eval rows.

    Features must not include the current target value.
    """
    train = panel_train.dropna(subset=[target]).copy()
    X_train = train[feature_cols].fillna(0.0).to_numpy(dtype=float)
    y_train = train[target].to_numpy(dtype=float)

    if model_name == "lightgbm":
        try:
            import lightgbm as lgb

            model = lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=-1,
            )
        except Exception:
            model = ExtraTreesRegressor(
                n_estimators=200, random_state=42, n_jobs=-1, min_samples_leaf=2
            )
            model_name = "extratrees_fallback"
    elif model_name == "xgboost":
        try:
            from xgboost import XGBRegressor

            model = XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
        except Exception:
            model = ExtraTreesRegressor(
                n_estimators=200, random_state=42, n_jobs=-1, min_samples_leaf=2
            )
            model_name = "extratrees_fallback"
    else:
        model = ExtraTreesRegressor(
            n_estimators=200, random_state=42, n_jobs=-1, min_samples_leaf=2
        )

    model.fit(X_train, y_train)
    eval_df = panel_eval.dropna(subset=[target]).copy()
    X_eval = eval_df[feature_cols].fillna(0.0).to_numpy(dtype=float)
    y_true = eval_df[target].to_numpy(dtype=float)
    y_pred = model.predict(X_eval)
    return BaselineResult(model_name, y_true, y_pred, causal=True)


def make_random_mask(
    wide: pd.DataFrame,
    rate: float,
    seed: int = 42,
    only_where_present: bool = True,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    present = wide.notna() if only_where_present else pd.DataFrame(True, index=wide.index, columns=wide.columns)
    rand = pd.DataFrame(rng.random(wide.shape), index=wide.index, columns=wide.columns)
    return (rand < rate) & present
