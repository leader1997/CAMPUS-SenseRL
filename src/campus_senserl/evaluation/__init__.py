"""Evaluation metrics for reconstruction and event preservation."""

from __future__ import annotations

from typing import Any

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[m] - y_pred[m])))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not m.any():
        return float("nan")
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not m.any():
        return float("nan")
    yt, yp = y_true[m], y_pred[m]
    denom = np.abs(yt) + np.abs(yp)
    # Avoid division by zero when both zero
    valid = denom > 1e-8
    if not valid.any():
        return 0.0
    return float(100.0 * np.mean(2.0 * np.abs(yp[valid] - yt[valid]) / denom[valid]))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    """MAPE only where |y_true| > eps (mathematically meaningful)."""
    m = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > eps)
    if not m.any():
        return float("nan")
    return float(100.0 * np.mean(np.abs((y_true[m] - y_pred[m]) / y_true[m])))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not m.any():
        return float("nan")
    yt, yp = y_true[m], y_pred[m]
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "n": int(np.sum(np.isfinite(y_true) & np.isfinite(y_pred))),
    }


def event_binary_metrics(
    y_true_event: np.ndarray,
    y_pred_event: np.ndarray,
) -> dict[str, float]:
    yt = y_true_event.astype(bool)
    yp = y_pred_event.astype(bool)
    tp = int(np.sum(yt & yp))
    fp = int(np.sum(~yt & yp))
    fn = int(np.sum(yt & ~yp))
    tn = int(np.sum(~yt & ~yp))
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = float("nan")
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "missed_event_rate": float(fn / (tp + fn)) if (tp + fn) else float("nan"),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def mean_std_ci(
    values: list[float],
    confidence: float = 0.95,
) -> dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    # Normal approx CI
    from scipy import stats

    if len(arr) > 1:
        half = stats.t.ppf((1 + confidence) / 2, df=len(arr) - 1) * std / np.sqrt(len(arr))
    else:
        half = float("nan")
    return {
        "mean": mean,
        "std": std,
        "ci_low": mean - half if np.isfinite(half) else float("nan"),
        "ci_high": mean + half if np.isfinite(half) else float("nan"),
        "n": int(len(arr)),
    }


def paired_bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap CI for mean(a - b)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) == 0:
        return {"mean_diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    diffs = []
    n = len(a)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs.append(float(np.mean(a[idx] - b[idx])))
    diffs = np.asarray(diffs)
    alpha = (1 - confidence) / 2
    return {
        "mean_diff": float(np.mean(a - b)),
        "ci_low": float(np.quantile(diffs, alpha)),
        "ci_high": float(np.quantile(diffs, 1 - alpha)),
    }
