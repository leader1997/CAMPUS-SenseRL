"""Generate manuscript figures for CAMPUS-SenseRL into results/figures/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

from campus_senserl.utils import ensure_dir, repo_root
from campus_senserl.visualization.paper_style import (
    COLORS,
    METHOD_COLORS,
    apply_paper_style,
    save_paper_figure,
)


def _fig_dir() -> Path:
    return ensure_dir(repo_root() / "results" / "figures")


def figure_01_sensor_network(devices: pd.DataFrame, n_co2: int, n_total: int) -> Path:
    """Scientific conclusion: campus deployment is multi-floor, CO2-dominated, spatially clustered."""
    apply_paper_style()
    fig = plt.figure(figsize=(10.5, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.28)

    ax0 = fig.add_subplot(gs[0, 0])
    co2 = devices[devices["device_type"] == "Elsys ERS CO2"]
    other = devices[devices["device_type"] != "Elsys ERS CO2"]
    if len(other):
        ax0.scatter(other["longitude"], other["latitude"], s=18, c="#A0A0A0", alpha=0.7, label="Non-CO$_2$", zorder=2)
    ax0.scatter(co2["longitude"], co2["latitude"], s=22, c=COLORS["co2"], alpha=0.85, label="ERS CO$_2$", zorder=3)
    ax0.set_xlabel("Longitude")
    ax0.set_ylabel("Latitude")
    ax0.legend(frameon=False, loc="best")
    ax0.set_aspect("equal", adjustable="box")

    ax1 = fig.add_subplot(gs[0, 1])
    floor_order = sorted(devices["floor"].dropna().unique(), key=lambda x: float(x))
    ct = (
        devices.groupby(["floor", "device_type"])
        .size()
        .unstack(fill_value=0)
        .reindex(floor_order)
    )
    # Simplify type labels
    ct.columns = [c.replace("Elsys ", "") for c in ct.columns]
    ct.plot(kind="bar", stacked=True, ax=ax1, color=["#1F4E79", "#F58518", "#54A24B"][: len(ct.columns)], width=0.75)
    ax1.set_xlabel("Floor")
    ax1.set_ylabel("Number of sensors")
    ax1.legend(frameon=False, fontsize=7, title=None)
    ax1.set_xticklabels([str(f) for f in floor_order], rotation=0)

    fig.text(
        0.01,
        0.98,
        f"Deployment: {n_total} sensors · {n_co2} CO$_2$ · primary experiments use CO$_2$ subset",
        ha="left",
        va="top",
        fontsize=9,
        color="#333333",
    )
    out = _fig_dir() / "fig01_sensor_network.png"
    save_paper_figure(fig, out)
    return out


def figure_02_co2_example(panel: pd.DataFrame, deveui: str, start: str, end: str) -> Path:
    """Scientific conclusion: stable periods are redundant to transmit frequently; changes are informative."""
    apply_paper_style()
    df = panel[(panel["deveui"] == deveui) & (panel["observed"] == 1)].copy()
    df = df[(df["slot"] >= start) & (df["slot"] <= end)].sort_values("slot")
    df = df.dropna(subset=["co2"])

    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    ax.plot(df["slot"], df["co2"], color=COLORS["co2"], lw=1.2, label="CO$_2$")
    # Mark rapid changes
    delta = df["co2"].diff().abs()
    rapid = df.loc[delta >= 80]
    if len(rapid):
        ax.scatter(rapid["slot"], rapid["co2"], s=18, c=COLORS["event"], zorder=3, label=r"$|\Delta|\geq 80$ ppm")
    ax.axhline(1000, color=COLORS["event"], ls="--", lw=0.9, alpha=0.8, label="1000 ppm threshold")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("CO$_2$ (ppm)")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.autofmt_xdate(rotation=25, ha="right")
    out = _fig_dir() / "fig02_co2_example.png"
    save_paper_figure(fig, out)
    return out


def figure_03_framework() -> Path:
    """Scientific conclusion: system architecture separates local sensing, RL scheduling, and server reconstruction."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#F7F9FC", ec="#1F4E79"):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", linewidth=1.2, edgecolor=ec, facecolor=fc)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, color="#222222", wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#444444", lw=1.2))

    box(0.2, 1.8, 1.8, 1.4, "Campus IoT\nsensors\n(local measure)")
    box(2.4, 1.8, 2.0, 1.4, "Semantic value\nUncertainty · AoI\nEvent cues")
    box(4.8, 1.8, 1.9, 1.4, "RL policy\nTRANSMIT / SKIP")
    box(7.1, 2.55, 1.7, 1.0, "TRANSMIT\n→ server")
    box(7.1, 1.15, 1.7, 1.0, "SKIP\n→ withhold")
    box(9.15, 1.6, 1.7, 1.8, "Spatiotemporal\nreconstruction\n+ monitoring")

    # Safety shield
    box(4.8, 3.6, 1.9, 0.9, "Safety shield\nforce TX if needed", fc="#FFF5F5", ec="#E45756")

    arrow(2.0, 2.5, 2.4, 2.5)
    arrow(4.4, 2.5, 4.8, 2.5)
    arrow(6.7, 2.7, 7.1, 3.0)
    arrow(6.7, 2.3, 7.1, 1.65)
    arrow(8.8, 3.0, 9.15, 2.7)
    arrow(8.8, 1.65, 9.15, 2.2)
    arrow(5.75, 3.6, 5.75, 3.2)

    ax.text(5.5, 0.35, "Trace-driven counterfactual evaluation on historical campus measurements", ha="center", fontsize=8, color="#555555")
    out = _fig_dir() / "fig03_framework.png"
    save_paper_figure(fig, out)
    return out


def figure_04_reconstruction(baselines: pd.DataFrame, neural_mae: float | None) -> Path:
    """Scientific conclusion: which causal reconstructors are competitive on held-out campus CO2."""
    apply_paper_style()
    # Select meaningful methods only
    keep = {
        "locf": "Last observation",
        "linear_extrapolation": "Linear extrapolation",
        "knn_neighbors": "KNN neighbours",
        "extratrees": "ExtraTrees",
        "lightgbm": "LightGBM",
        "masked_st_graph": "Masked ST-GNN",
    }
    rows = []
    for _, r in baselines.iterrows():
        if r["model"] in keep and r.get("causal", True) is not False:
            if str(r["model"]).startswith("noncausal"):
                continue
            rows.append({"method": keep[r["model"]], "mae": r["mae"], "family": "baseline"})
    # neural from separate eval if present and not in baselines
    if neural_mae is not None and "Masked ST-GNN" not in [r["method"] for r in rows]:
        rows.append({"method": "Masked ST-GNN", "mae": neural_mae, "family": "neural"})
    # If masked in baselines file under different name - already handled via eval merge externally

    df = pd.DataFrame(rows)
    # Deduplicate preferring lower mae for same method? keep first
    df = df.drop_duplicates("method", keep="first")
    order = ["Last observation", "Linear extrapolation", "KNN neighbours", "ExtraTrees", "LightGBM", "Masked ST-GNN"]
    df["method"] = pd.Categorical(df["method"], categories=[o for o in order if o in set(df["method"])], ordered=True)
    df = df.sort_values("method")

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    colors = [COLORS["locf"], COLORS["heuristic"], "#9ECBE2", COLORS["tree"], "#86BF6B", COLORS["neural"]]
    colors = colors[: len(df)]
    bars = ax.bar(df["method"].astype(str), df["mae"], color=colors, width=0.7, edgecolor="white")
    ax.set_ylabel("CO$_2$ MAE (ppm)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    # annotate values
    for b, v in zip(bars, df["mae"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(df["mae"].max() * 1.18, 1))
    out = _fig_dir() / "fig04_reconstruction_comparison.png"
    save_paper_figure(fig, out)
    return out


def figure_05_tradeoff_reconstruction(policy_df: pd.DataFrame) -> Path:
    """Scientific conclusion: communication savings vs reconstruction quality trade-off."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for _, r in policy_df.iterrows():
        if not np.isfinite(r["mae_skipped"]):
            continue
        c = METHOD_COLORS.get(r["display_name"], COLORS["heuristic"])
        if r["family"] == "random":
            c = COLORS["random"]
        if r["family"] == "proposed":
            c = COLORS["proposed"]
        ax.scatter(
            r["transmission_reduction_pct"],
            r["mae_skipped"],
            s=70,
            c=c,
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
            label=r["display_name"],
        )
    ax.set_xlabel("Transmission reduction (%)")
    ax.set_ylabel("CO$_2$ reconstruction MAE on skips (ppm)")
    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by = dict(zip(labels, handles))
    ax.legend(by.values(), by.keys(), frameon=False, fontsize=7, loc="best")
    out = _fig_dir() / "fig05_tradeoff_reconstruction.png"
    save_paper_figure(fig, out)
    return out


def figure_06_tradeoff_event_recall(policy_df: pd.DataFrame) -> Path:
    """Scientific conclusion: whether event recall is preserved under communication reduction."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for _, r in policy_df.iterrows():
        if not np.isfinite(r["event_recall"]):
            continue
        c = METHOD_COLORS.get(r["display_name"], COLORS["heuristic"])
        if r["family"] == "random":
            c = COLORS["random"]
        if r["family"] == "proposed":
            c = COLORS["proposed"]
        ax.scatter(
            r["transmission_reduction_pct"],
            100.0 * r["event_recall"],
            s=70,
            c=c,
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
            label=r["display_name"],
        )
    ax.set_xlabel("Transmission reduction (%)")
    ax.set_ylabel("Important CO$_2$ event recall (%)")
    ax.set_ylim(0, 105)
    handles, labels = ax.get_legend_handles_labels()
    by = dict(zip(labels, handles))
    ax.legend(by.values(), by.keys(), frameon=False, fontsize=7, loc="best")
    out = _fig_dir() / "fig06_tradeoff_event_recall.png"
    save_paper_figure(fig, out)
    return out


def figure_07_adaptive(timeline: dict[str, Any]) -> Path:
    """Scientific conclusion: adaptive policy transmits preferentially during changing/high periods."""
    apply_paper_style()
    t = np.asarray(timeline["times"])
    true = np.asarray(timeline["true_co2"], dtype=float)
    recon = np.asarray(timeline["recon_co2"], dtype=float)
    unc = np.asarray(timeline["uncertainty"], dtype=float)
    tx = np.asarray(timeline["transmitted"], dtype=float)
    actions = np.asarray(timeline["actions"], dtype=int)
    thr = float(timeline["event_threshold"])

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 5.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 0.7], "hspace": 0.08})
    ax = axes[0]
    ax.plot(t, true, color=COLORS["co2"], lw=1.3, label="True CO$_2$")
    # reconstructed only where skipped and finite
    skip_mask = actions == 0
    ax.plot(t[skip_mask], recon[skip_mask], color=COLORS["recon"], lw=1.0, ls="--", label="Reconstructed (skip)")
    ax.scatter(t, tx, s=22, c=COLORS["tx"], zorder=4, label="Transmitted")
    if np.isfinite(unc).any():
        # uncertainty band around recon when skipped
        lo = recon - unc
        hi = recon + unc
        ax.fill_between(t, lo, hi, where=skip_mask & np.isfinite(recon), color=COLORS["unc"], alpha=0.2, label="Uncertainty")
    ax.axhline(thr, color=COLORS["event"], ls="--", lw=0.9, alpha=0.85, label=f"{int(thr)} ppm")
    ax.set_ylabel("CO$_2$ (ppm)")
    ax.legend(frameon=False, ncol=3, fontsize=7, loc="upper right")

    ax2 = axes[1]
    ax2.scatter(t[actions == 1], np.ones(np.sum(actions == 1)), c=COLORS["tx"], s=16, label="TX")
    ax2.scatter(t[actions == 0], np.zeros(np.sum(actions == 0)), c=COLORS["skip"], s=12, label="SKIP")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["SKIP", "TX"])
    ax2.set_xlabel("Time step (15 min intervals)")
    ax2.set_ylim(-0.5, 1.5)
    ax2.legend(frameon=False, fontsize=7, loc="upper right", ncol=2)

    out = _fig_dir() / "fig07_adaptive_transmission_example.png"
    save_paper_figure(fig, out)
    return out


def figure_08_aoi(policy_df: pd.DataFrame) -> Path:
    """Scientific conclusion: how AoI scales with communication reduction across policies."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for fam, marker in [("fixed", "o"), ("heuristic", "s"), ("random", "^"), ("proposed", "D")]:
        sub = policy_df[policy_df["family"] == fam]
        if sub.empty:
            continue
        ax.scatter(
            sub["transmission_reduction_pct"],
            sub["mean_aoi"],
            s=65,
            marker=marker,
            label=fam.capitalize(),
            edgecolors="white",
            linewidths=0.5,
        )
    ax.set_xlabel("Transmission reduction (%)")
    ax.set_ylabel("Mean Age of Information (intervals)")
    ax.legend(frameon=False)
    out = _fig_dir() / "fig08_aoi_comparison.png"
    save_paper_figure(fig, out)
    return out


def figure_09_ablation(ablation_df: pd.DataFrame) -> Path | None:
    """Only if ablations differentiate components meaningfully."""
    apply_paper_style()
    # Keep only components that change something relative to full
    focus = ["full", "no_uncertainty", "no_aoi", "no_semantic", "no_safety_shield"]
    df = ablation_df[ablation_df["ablation"].isin(focus)].copy()
    if df.empty:
        return None
    # Metric: mean_reward and transmit_rate — honest limited ablation
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(df))
    labels = {
        "full": "Full",
        "no_uncertainty": "− Uncertainty",
        "no_aoi": "− AoI",
        "no_semantic": "− Semantic",
        "no_safety_shield": "− Safety shield",
    }
    ax.bar(x, -df["mean_reward"], color="#4C78A8", width=0.65)  # plot positive cost magnitude
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(a, a) for a in df["ablation"]], rotation=15, ha="right")
    ax.set_ylabel("Mean episode cost (−reward)")
    ax.text(
        0.01,
        1.02,
        "Preliminary ablation (reward-component removal); not multi-seed matched-budget study",
        transform=ax.transAxes,
        fontsize=7,
        color="#666666",
        va="bottom",
    )
    out = _fig_dir() / "fig09_ablation.png"
    save_paper_figure(fig, out)
    return out


def figure_10_robustness(rob_df: pd.DataFrame) -> Path | None:
    apply_paper_style()
    sub = rob_df[rob_df["condition"].isin(["packet_loss", "missingness"])].copy()
    if sub.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    for cond, g in sub.groupby("condition"):
        g = g.sort_values("level")
        ax.plot(100 * g["level"], -g["mean_reward"], marker="o", label=cond.replace("_", " "))
    ax.set_xlabel("Stress level (%)")
    ax.set_ylabel("Mean episode cost (−reward)")
    ax.legend(frameon=False)
    ax.text(
        0.01,
        1.02,
        "Preliminary synthetic robustness smoke test — not final campus-panel sweep",
        transform=ax.transAxes,
        fontsize=7,
        color="#666666",
        va="bottom",
    )
    out = _fig_dir() / "fig10_robustness.png"
    save_paper_figure(fig, out)
    return out


def figure_11_training(ppo_metrics: list[dict], mappo_metrics: list[dict] | None = None) -> Path | None:
    """Omit if fewer than meaningful multi-seed smoothed curves."""
    if not ppo_metrics or len(ppo_metrics) < 3:
        # User: only include if sufficiently meaningful — 2 points is not enough
        return None
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    steps = [m["step"] for m in ppo_metrics]
    vals = [m["rollout_reward"] for m in ppo_metrics]
    ax.plot(steps, vals, marker="o", color=COLORS["ppo"], label="PPO")
    if mappo_metrics and len(mappo_metrics) >= 3:
        ax.plot(
            [m["step"] for m in mappo_metrics],
            [m["rollout_reward"] for m in mappo_metrics],
            marker="s",
            color=COLORS["marl"],
            label="MAPPO",
        )
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Rollout return")
    ax.legend(frameon=False)
    out = _fig_dir() / "fig11_training_convergence.png"
    save_paper_figure(fig, out)
    return out
