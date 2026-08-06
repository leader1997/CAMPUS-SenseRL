"""Heteroscedastic Gaussian NLL and regression calibration metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def gaussian_nll(
    mean: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: str = "mean",
    min_log_var: float = -10.0,
    max_log_var: float = 10.0,
) -> torch.Tensor:
    """Negative log-likelihood under diagonal Gaussian predictive distribution."""
    log_var = torch.clamp(log_var, min=min_log_var, max=max_log_var)
    var = torch.exp(log_var)
    nll = 0.5 * (log_var + (target - mean) ** 2 / var + np.log(2 * np.pi))
    if reduction == "none":
        return nll
    if reduction == "sum":
        return nll.sum()
    if reduction == "mean":
        return nll.mean()
    raise ValueError(f"Unknown reduction: {reduction}")


def gaussian_nll_masked(
    mean: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Gaussian NLL averaged only over masked positions (mask=True => include)."""
    mask = mask.to(dtype=torch.bool)
    if not mask.any():
        return mean.new_tensor(0.0)
    nll = gaussian_nll(mean, log_var, target, reduction="none")
    nll = nll[mask]
    if reduction == "sum":
        return nll.sum()
    if reduction == "mean":
        return nll.mean()
    if reduction == "none":
        return nll
    raise ValueError(f"Unknown reduction: {reduction}")


def prediction_interval_coverage(
    y_true: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    z: float = 1.96,
) -> dict[str, float]:
    """Empirical coverage of symmetric Gaussian prediction intervals."""
    y = np.asarray(y_true, dtype=float)
    mu = np.asarray(mean, dtype=float)
    sigma = np.maximum(np.asarray(std, dtype=float), 1e-6)
    m = np.isfinite(y) & np.isfinite(mu) & np.isfinite(sigma)
    if m.sum() == 0:
        return {"coverage": float("nan"), "n": 0, "z": z}
    lower = mu[m] - z * sigma[m]
    upper = mu[m] + z * sigma[m]
    covered = (y[m] >= lower) & (y[m] <= upper)
    return {"coverage": float(covered.mean()), "n": int(m.sum()), "z": z}


def regression_calibration_bins(
    y_true: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    n_bins: int = 10,
) -> dict[str, np.ndarray]:
    """Bin by predicted sigma and compare empirical RMSE to predicted sigma."""
    y = np.asarray(y_true, dtype=float)
    mu = np.asarray(mean, dtype=float)
    sigma = np.maximum(np.asarray(std, dtype=float), 1e-6)
    m = np.isfinite(y) & np.isfinite(mu) & np.isfinite(sigma)
    y, mu, sigma = y[m], mu[m], sigma[m]
    if len(y) == 0:
        return {
            "bin_edges": np.array([]),
            "bin_count": np.array([]),
            "empirical_rmse": np.array([]),
            "mean_sigma": np.array([]),
        }
    edges = np.quantile(sigma, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        edges = np.linspace(sigma.min(), sigma.max() + 1e-6, n_bins + 1)
    counts = []
    rmse = []
    mean_sig = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            sel = (sigma >= lo) & (sigma <= hi)
        else:
            sel = (sigma >= lo) & (sigma < hi)
        if not sel.any():
            counts.append(0)
            rmse.append(np.nan)
            mean_sig.append(np.nan)
            continue
        resid = y[sel] - mu[sel]
        counts.append(int(sel.sum()))
        rmse.append(float(np.sqrt(np.mean(resid**2))))
        mean_sig.append(float(np.mean(sigma[sel])))
    return {
        "bin_edges": edges,
        "bin_count": np.asarray(counts, dtype=int),
        "empirical_rmse": np.asarray(rmse, dtype=float),
        "mean_sigma": np.asarray(mean_sig, dtype=float),
    }


def regression_ece(
    y_true: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    n_bins: int = 10,
) -> float:
    """ECE-like calibration error for regression: |empirical RMSE - mean sigma|."""
    bins = regression_calibration_bins(y_true, mean, std, n_bins=n_bins)
    counts = bins["bin_count"]
    rmse = bins["empirical_rmse"]
    sig = bins["mean_sigma"]
    valid = (counts > 0) & np.isfinite(rmse) & np.isfinite(sig)
    if not valid.any():
        return float("nan")
    w = counts[valid].astype(float)
    w = w / w.sum()
    err = np.abs(rmse[valid] - sig[valid])
    return float(np.sum(w * err))


def summarize_uncertainty_metrics(
    y_true: np.ndarray,
    mean: np.ndarray,
    log_var: np.ndarray | None = None,
    std: np.ndarray | None = None,
    *,
    n_bins: int = 10,
    z_levels: tuple[float, ...] = (1.0, 1.96, 2.58),
) -> dict[str, Any]:
    """Aggregate coverage and calibration metrics for heteroscedastic predictions."""
    if std is None:
        if log_var is None:
            raise ValueError("Provide std or log_var")
        std = np.sqrt(np.exp(np.clip(np.asarray(log_var, dtype=float), -10, 10)))
    out: dict[str, Any] = {"ece": regression_ece(y_true, mean, std, n_bins=n_bins)}
    for z in z_levels:
        out[f"coverage_z{z}"] = prediction_interval_coverage(y_true, mean, std, z=z)
    return out
