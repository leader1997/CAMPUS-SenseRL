"""Publication figures: Pareto, AoI vs budget, training curves, timeline, ablations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from campus_senserl.utils import ensure_dir


def _save(fig: plt.Figure, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_pareto_frontier(
    df: pd.DataFrame,
    *,
    x_col: str = "transmit_rate",
    y_col: str = "mae",
    hue_col: str = "method",
    out_path: str | Path,
    title: str = "Communication–Accuracy Pareto Frontier",
) -> Path:
    """Scatter Pareto plot of transmission rate vs reconstruction error."""
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, grp in df.groupby(hue_col):
        ax.scatter(grp[x_col], grp[y_col], label=str(name), alpha=0.85, s=60)
    ax.set_xlabel("Transmit rate")
    ax.set_ylabel("Reconstruction MAE (ppm)")
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.25)
    _save(fig, out_path)
    return out_path


def plot_aoi_vs_budget(
    df: pd.DataFrame,
    *,
    budget_col: str = "budget",
    aoi_col: str = "mean_aoi",
    hue_col: str = "method",
    out_path: str | Path,
    title: str = "Mean AoI vs Communication Budget",
) -> Path:
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, grp in df.groupby(hue_col):
        g = grp.sort_values(budget_col)
        ax.plot(g[budget_col], g[aoi_col], marker="o", label=str(name))
    ax.set_xlabel("Communication budget fraction")
    ax.set_ylabel("Mean AoI (intervals)")
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.25)
    _save(fig, out_path)
    return out_path


def plot_training_curves(
    metrics: list[dict[str, Any]] | pd.DataFrame,
    *,
    step_key: str = "step",
    value_key: str = "rollout_reward",
    out_path: str | Path,
    title: str = "Training reward curve",
) -> Path:
    out_path = Path(out_path)
    if isinstance(metrics, list):
        df = pd.DataFrame(metrics)
    else:
        df = metrics.copy()
    fig, ax = plt.subplots(figsize=(7, 4))
    if step_key in df.columns and value_key in df.columns:
        sub = df.dropna(subset=[step_key, value_key])
        ax.plot(sub[step_key], sub[value_key], linewidth=1.5)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(value_key.replace("_", " "))
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    _save(fig, out_path)
    return out_path


def plot_communication_timeline(
    transmit_history: np.ndarray,
    *,
    out_path: str | Path,
    title: str = "Transmission timeline",
    max_sensors: int = 40,
) -> Path:
    """Heatmap of transmit (1) vs skip (0) over time × sensors."""
    out_path = Path(out_path)
    arr = np.asarray(transmit_history, dtype=float)
    if arr.ndim != 2:
        raise ValueError("transmit_history must be 2-D (time, sensors)")
    if arr.shape[1] > max_sensors:
        arr = arr[:, :max_sensors]
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(arr.T, aspect="auto", interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Sensor index")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.02, label="Transmitted")
    _save(fig, out_path)
    return out_path


def plot_ablation_bars(
    df: pd.DataFrame,
    *,
    metric_col: str,
    ablation_col: str = "ablation",
    err_col: str | None = "std",
    out_path: str | Path,
    title: str = "Ablation study",
    higher_is_better: bool = False,
) -> Path:
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    order = df.sort_values(metric_col, ascending=not higher_is_better)[ablation_col]
    plot_df = df.set_index(ablation_col).loc[order]
    yerr = plot_df[err_col].to_numpy() if err_col and err_col in plot_df.columns else None
    ax.bar(range(len(plot_df)), plot_df[metric_col].to_numpy(), yerr=yerr, capsize=3, color="#4C72B0")
    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(plot_df.index, rotation=45, ha="right")
    ax.set_ylabel(metric_col)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, out_path)
    return out_path
