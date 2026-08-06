"""Temporal and rolling features. Rolling windows use PAST data only (no centering)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_calendar_features(df: pd.DataFrame, time_col: str = "slot") -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out[time_col], utc=True)
    hour = ts.dt.hour + ts.dt.minute / 60.0
    dow = ts.dt.dayofweek.astype(float)
    out["hour"] = hour
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow"] = dow
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    out["is_weekend"] = (ts.dt.dayofweek >= 5).astype("int8")
    out["month"] = ts.dt.month.astype("int8")
    # Season as month-angle
    out["month_sin"] = np.sin(2 * np.pi * (ts.dt.month - 1) / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * (ts.dt.month - 1) / 12.0)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    time_col: str,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Past-only rolling mean/std/delta. Uses shift(1) before rolling to exclude current."""
    windows = windows or [4, 8, 16]
    out = df.sort_values([group_col, time_col]).copy()
    g = out.groupby(group_col, sort=False)[value_col]
    # Previous value and rate of change (past only)
    out[f"{value_col}_lag1"] = g.shift(1)
    out[f"{value_col}_lag2"] = g.shift(2)
    # Rate of change from PAST values only (lag1 - lag2), never using current y
    out[f"{value_col}_delta1"] = out[f"{value_col}_lag1"] - out[f"{value_col}_lag2"]
    # For rolling, operate on lagged series so current observation is excluded
    lagged = g.shift(1)
    for w in windows:
        roll = lagged.groupby(out[group_col]).rolling(w, min_periods=1)
        out[f"{value_col}_rollmean_{w}"] = roll.mean().reset_index(level=0, drop=True)
        out[f"{value_col}_rollstd_{w}"] = roll.std().reset_index(level=0, drop=True)
    return out
