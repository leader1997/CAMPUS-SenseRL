"""Bootstrap confidence intervals and summary statistics."""

from __future__ import annotations

from typing import Callable

import numpy as np

from campus_senserl.evaluation import mean_std_ci, paired_bootstrap_ci


def bootstrap_ci(
    values: list[float] | np.ndarray,
    *,
    stat_fn: Callable[[np.ndarray], float] | None = None,
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Generic bootstrap confidence interval."""
    if stat_fn is None:
        stat_fn = np.mean
    rng = np.random.default_rng(seed)
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(arr) == 0:
        return {"stat": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    n = len(arr)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats.append(float(stat_fn(arr[idx])))
    stats_arr = np.asarray(stats)
    alpha = (1 - confidence) / 2
    return {
        "stat": float(stat_fn(arr)),
        "ci_low": float(np.quantile(stats_arr, alpha)),
        "ci_high": float(np.quantile(stats_arr, 1 - alpha)),
        "n": int(n),
    }


def bootstrap_mean_ci(
    values: list[float] | np.ndarray,
    *,
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap CI for the mean of a 1-D sample."""
    return bootstrap_ci(values, stat_fn=np.mean, n_boot=n_boot, confidence=confidence, seed=seed)


__all__ = [
    "bootstrap_ci",
    "bootstrap_mean_ci",
    "mean_std_ci",
    "paired_bootstrap_ci",
]
