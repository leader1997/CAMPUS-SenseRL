"""Important environmental event detection (configurable thresholds)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def high_co2_events(values: np.ndarray, threshold_ppm: float) -> np.ndarray:
    return (np.asarray(values, dtype=float) >= threshold_ppm) & np.isfinite(values)


def rapid_increase_events(
    values: np.ndarray,
    delta_ppm: float,
) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    out = np.zeros(len(v), dtype=bool)
    if len(v) < 2:
        return out
    d = np.diff(v, prepend=np.nan)
    out = (d >= delta_ppm) & np.isfinite(d)
    return out


def persistent_high_events(
    values: np.ndarray,
    threshold_ppm: float,
    min_intervals: int = 3,
) -> np.ndarray:
    high = high_co2_events(values, threshold_ppm)
    out = np.zeros(len(values), dtype=bool)
    run = 0
    for i, h in enumerate(high):
        run = run + 1 if h else 0
        if run >= min_intervals:
            out[i] = True
    return out


def deviation_events(
    values: np.ndarray,
    baseline: np.ndarray,
    z_thresh: float = 2.5,
) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    b = np.asarray(baseline, dtype=float)
    resid = v - b
    m = np.isfinite(resid)
    out = np.zeros(len(v), dtype=bool)
    if m.sum() < 10:
        return out
    mu = np.nanmean(resid)
    sd = np.nanstd(resid)
    if sd < 1e-6:
        return out
    z = (resid - mu) / sd
    out = (np.abs(z) >= z_thresh) & m
    return out


def combine_events(*masks: np.ndarray) -> np.ndarray:
    out = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        out = out | m.astype(bool)
    return out


def detect_events_series(
    values: np.ndarray,
    cfg: dict[str, Any],
    baseline: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    thr = float(cfg.get("primary_threshold_ppm", 1000))
    rapid = float(cfg.get("rapid_increase_ppm", 150))
    persist = int(cfg.get("persistent_high_intervals", 3))
    high = high_co2_events(values, thr)
    rapid_m = rapid_increase_events(values, rapid)
    persist_m = persistent_high_events(values, thr, persist)
    result = {
        "high": high,
        "rapid": rapid_m,
        "persistent": persist_m,
        "any": combine_events(high, rapid_m, persist_m),
    }
    if baseline is not None:
        result["deviation"] = deviation_events(values, baseline)
        result["any"] = combine_events(result["any"], result["deviation"])
    return result
