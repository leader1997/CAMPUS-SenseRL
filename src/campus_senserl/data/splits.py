"""Chronological train/validation/test splits. No random shuffling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def chronological_split_bounds(
    times: pd.Series,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
) -> dict[str, pd.Timestamp]:
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    t = pd.to_datetime(pd.Series(times).dropna().unique())
    t = pd.Series(sorted(t))
    n = len(t)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    # Ensure non-empty splits
    i_train = max(1, min(i_train, n - 2))
    i_val = max(i_train + 1, min(i_val, n - 1))
    return {
        "train_end": pd.Timestamp(t.iloc[i_train - 1]),
        "val_end": pd.Timestamp(t.iloc[i_val - 1]),
        "test_end": pd.Timestamp(t.iloc[-1]),
        "t_min": pd.Timestamp(t.iloc[0]),
        "n_unique_times": int(n),
        "i_train": i_train,
        "i_val": i_val,
    }


def assign_split(times: pd.Series, bounds: dict[str, pd.Timestamp]) -> pd.Series:
    ts = pd.to_datetime(times, utc=True)
    train_end = pd.Timestamp(bounds["train_end"])
    val_end = pd.Timestamp(bounds["val_end"])
    if train_end.tzinfo is None:
        train_end = train_end.tz_localize("UTC")
    else:
        train_end = train_end.tz_convert("UTC")
    if val_end.tzinfo is None:
        val_end = val_end.tz_localize("UTC")
    else:
        val_end = val_end.tz_convert("UTC")

    out = np.where(ts <= train_end, "train", np.where(ts <= val_end, "val", "test"))
    return pd.Series(out, index=times.index if hasattr(times, "index") else None)


def split_frame(df: pd.DataFrame, split_col: str = "split") -> dict[str, pd.DataFrame]:
    return {s: df[df[split_col] == s].copy() for s in ["train", "val", "test"]}


def rolling_origin_folds(
    times: pd.Series,
    n_folds: int = 3,
    min_train_frac: float = 0.5,
) -> list[dict[str, Any]]:
    """Expanding-window rolling-origin evaluation folds (chronological)."""
    t = pd.Series(sorted(pd.to_datetime(pd.Series(times).dropna().unique())))
    n = len(t)
    folds = []
    start = int(n * min_train_frac)
    remaining = n - start
    fold_size = max(1, remaining // n_folds)
    for i in range(n_folds):
        train_end_idx = start + i * fold_size - 1
        test_end_idx = min(n - 1, train_end_idx + fold_size)
        if train_end_idx < 1:
            continue
        # val = last 20% of training window
        val_start_idx = int(train_end_idx * 0.8)
        folds.append(
            {
                "fold": i,
                "train_end": pd.Timestamp(t.iloc[val_start_idx]),
                "val_end": pd.Timestamp(t.iloc[train_end_idx]),
                "test_end": pd.Timestamp(t.iloc[test_end_idx]),
                "t_min": pd.Timestamp(t.iloc[0]),
            }
        )
    return folds
