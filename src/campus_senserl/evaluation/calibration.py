"""Uncertainty calibration helpers for reconstruction evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

from campus_senserl.models.uncertainty import (
    prediction_interval_coverage,
    regression_calibration_bins,
    regression_ece,
    summarize_uncertainty_metrics,
)


def calibration_summary(
    y_true: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray | None = None,
    log_var: np.ndarray | None = None,
    *,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Return ECE, coverage levels, and bin-wise calibration table."""
    if std is None and log_var is not None:
        std = np.sqrt(np.exp(np.clip(np.asarray(log_var, dtype=float), -10, 10)))
    if std is None:
        raise ValueError("Provide std or log_var")
    metrics = summarize_uncertainty_metrics(y_true, mean, std=std, n_bins=n_bins)
    bins = regression_calibration_bins(y_true, mean, std, n_bins=n_bins)
    metrics["bins"] = bins
    metrics["ece_direct"] = regression_ece(y_true, mean, std, n_bins=n_bins)
    return metrics


def reliability_diagram_data(
    y_true: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    n_bins: int = 10,
) -> dict[str, np.ndarray]:
    """Data for plotting predicted sigma vs empirical RMSE."""
    return regression_calibration_bins(y_true, mean, std, n_bins=n_bins)


def interval_coverage_at_levels(
    y_true: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    z_levels: tuple[float, ...] = (1.0, 1.96, 2.58),
) -> dict[str, dict[str, float]]:
    return {f"z{z}": prediction_interval_coverage(y_true, mean, std, z=z) for z in z_levels}


__all__ = [
    "calibration_summary",
    "interval_coverage_at_levels",
    "prediction_interval_coverage",
    "regression_calibration_bins",
    "regression_ece",
    "reliability_diagram_data",
    "summarize_uncertainty_metrics",
]
