#!/usr/bin/env python
"""Manuscript figure pack (no training).

Regenerates Figures 1–9 from frozen paper_final / scientific_validation artifacts.
Moves obsolete figures into paper_outputs/legacy_provisional/.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from campus_senserl.data.audit import load_devices
from campus_senserl.data.cohort import load_cohort
from campus_senserl.environment.communication_model import TRANSMIT
from campus_senserl.evaluation.rl_policy_eval import load_mappo_policy, make_final_env
from campus_senserl.utils import ensure_dir, load_yaml, repo_root, save_json
from campus_senserl.visualization.paper_style import apply_paper_style, save_paper_figure

DPI = 300
SCI = ROOT / "outputs" / "rl_final" / "scientific_validation"
PAPER = ROOT / "outputs" / "rl_final" / "paper_final"
FIG = ROOT / "paper_outputs" / "figures"
LEGACY = ROOT / "paper_outputs" / "legacy_provisional" / "figures"


def _save(fig: plt.Figure, name: str) -> Path:
    out = FIG / f"{name}.png"
    save_paper_figure(fig, out, dpi=DPI)
    print(f"  wrote {out.name} (+pdf)")
    return out


def archive_obsolete() -> None:
    ensure_dir(LEGACY)
    obsolete = [
        "fig04_reconstruction_comparison",
        "fig05_tradeoff_reconstruction",
        "fig05_pareto_tx_mae",
        "fig06_tradeoff_event_recall",
        "fig07_adaptive_transmission_example",
        "fig08_aoi_comparison",
        "fig09_ablation",
        "fig10_robustness",
        # old versions of 01–03 will be overwritten; archive current copies first
        "fig01_sensor_network",
        "fig02_co2_example",
        "fig03_framework",
    ]
    for stem in obsolete:
        for ext in (".png", ".pdf"):
            src = FIG / f"{stem}{ext}"
            if src.exists():
                dst = LEGACY / f"{stem}{ext}"
                shutil.copy2(src, dst)
                src.unlink()
                print(f"  archived {src.name}")


# ---------------------------------------------------------------------------
# Figure 1 — deployment + cohorts
# ---------------------------------------------------------------------------
def fig01() -> None:
    apply_paper_style()
    cfg = load_yaml(ROOT / "configs" / "data.yaml")
    devices = load_devices(ROOT / cfg["paths"]["raw_release"] / cfg["dataset"]["devices_file"])
    train_ids = {str(s).upper() for s in load_cohort("final")}
    held_ids = {str(s).upper() for s in load_cohort("heldout")}
    devices = devices.copy()
    devices["deveui"] = devices["device_id"].astype(str).str.upper()

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), gridspec_kw={"width_ratios": [1.4, 1.0]})
    ax = axes[0]
    # all sensors grey
    ax.scatter(
        devices["longitude"],
        devices["latitude"],
        s=14,
        c="#C8C8C8",
        alpha=0.55,
        label=f"All sensors (n={len(devices)})",
        zorder=1,
    )
    tr = devices[devices["deveui"].isin(train_ids)]
    ho = devices[devices["deveui"].isin(held_ids)]
    ax.scatter(
        tr["longitude"],
        tr["latitude"],
        s=36,
        c="#4C78A8",
        edgecolors="white",
        linewidths=0.4,
        label=f"RL development cohort (n={len(tr)})",
        zorder=3,
    )
    ax.scatter(
        ho["longitude"],
        ho["latitude"],
        s=36,
        c="#F58518",
        edgecolors="white",
        linewidths=0.4,
        label=f"Held-out transfer cohort (n={len(ho)})",
        zorder=4,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, loc="best", fontsize=7.5)
    ax.set_title("Campus IoT deployment and experimental cohorts")

    ax1 = axes[1]
    # floor counts for CO2 and cohorts
    co2 = devices[devices["device_type"].astype(str).str.contains("CO2", case=False, na=False)].copy()
    floors = sorted(co2["floor"].dropna().unique(), key=lambda x: float(x))
    n_all = [int((co2["floor"] == f).sum()) for f in floors]
    n_tr = [int(((tr["floor"] == f).sum()) if len(tr) else 0) for f in floors]
    n_ho = [int(((ho["floor"] == f).sum()) if len(ho) else 0) for f in floors]
    x = np.arange(len(floors))
    w = 0.28
    ax1.bar(x - w, n_all, width=w, color="#C8C8C8", label="All CO$_2$")
    ax1.bar(x, n_tr, width=w, color="#4C78A8", label="RL cohort")
    ax1.bar(x + w, n_ho, width=w, color="#F58518", label="Held-out")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(int(f)) if float(f) == int(f) else str(f) for f in floors])
    ax1.set_xlabel("Floor")
    ax1.set_ylabel("Number of sensors")
    ax1.set_title("CO$_2$ sensors by floor")
    ax1.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    _save(fig, "fig01_sensor_network")


# ---------------------------------------------------------------------------
# Figure 2 — why adaptive communication
# ---------------------------------------------------------------------------
def _pick_representative_sensor(panel: pd.DataFrame, cohort: list[str]) -> tuple[str, pd.Timestamp, pd.Timestamp]:
    """Pick a final-cohort val sensor with moderate variance (not extreme spikes)."""
    val = panel[(panel["split"] == "val") & (panel["deveui"].isin(cohort)) & (panel["observed"] == 1)]
    best = None
    for deveui, g in val.groupby("deveui"):
        g = g.dropna(subset=["co2"]).sort_values("slot")
        if len(g) < 200:
            continue
        # sliding 3-day windows (~288 slots at 15 min)
        win = 288
        for i in range(0, len(g) - win, 48):
            w = g.iloc[i : i + win]
            std = float(w["co2"].std())
            mx = float(w["co2"].max())
            # representative: clear dynamics but not pathological spikes
            if 40 <= std <= 180 and mx < 1800:
                score = abs(std - 90) + 0.01 * abs(mx - 1100)
                if best is None or score < best[0]:
                    best = (score, str(deveui), w["slot"].iloc[0], w["slot"].iloc[-1])
    if best is None:
        # fallback
        deveui = cohort[0]
        g = val[val["deveui"] == deveui].sort_values("slot").iloc[:288]
        return str(deveui), g["slot"].iloc[0], g["slot"].iloc[-1]
    return best[1], best[2], best[3]


def fig02() -> None:
    apply_paper_style()
    panel = pd.read_parquet(ROOT / "data" / "processed" / "co2_panel_15min.parquet")
    cohort = load_cohort("final")
    deveui, start, end = _pick_representative_sensor(panel, cohort)
    df = panel[(panel["deveui"] == deveui) & (panel["observed"] == 1)].copy()
    df = df[(df["slot"] >= start) & (df["slot"] <= end)].sort_values("slot").dropna(subset=["co2"])

    fig, ax = plt.subplots(figsize=(9.8, 3.6))
    ax.plot(df["slot"], df["co2"], color="#1F4E79", lw=1.25, label="Measured CO$_2$")
    ax.axhline(1000, color="#C44E52", ls="--", lw=0.95, alpha=0.85, label="1000 ppm threshold")

    # Highlight a stable window and a rising window automatically
    co2 = df["co2"].to_numpy()
    slots = df["slot"].to_numpy()
    # stable: lowest rolling std over 20 intervals
    roll = pd.Series(co2).rolling(20, center=True).std()
    if roll.notna().any():
        i_stable = int(roll.idxmin())
        a, b = max(0, i_stable - 10), min(len(df), i_stable + 10)
        ax.axvspan(slots[a], slots[b - 1], color="#9ECBE2", alpha=0.35, label="Stable (redundant TX)")
    # rising: max positive change over 8 steps
    rise = pd.Series(co2).diff(8)
    if rise.notna().any():
        i_rise = int(rise.idxmax())
        a, b = max(0, i_rise - 4), min(len(df), i_rise + 8)
        ax.axvspan(slots[a], slots[b - 1], color="#F58518", alpha=0.28, label="Rising event (valuable TX)")

    # Fixed-15 markers (every point) as faint ticks on top axis idea: show density of fixed schedule
    ax.scatter(df["slot"], df["co2"], s=6, c="#4C78A8", alpha=0.25, zorder=2)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("CO$_2$ (ppm)")
    ax.set_title(f"Why adaptive communication?  (sensor {deveui[-6:]})")
    ax.legend(frameon=False, ncol=2, fontsize=7.5, loc="upper right")
    fig.autofmt_xdate(rotation=25, ha="right")
    fig.tight_layout()
    _save(fig, "fig02_co2_example")
    meta = {"deveui": deveui, "start": str(start), "end": str(end), "source": "final_cohort val"}
    save_json(meta, FIG / "fig02_meta.json")


# ---------------------------------------------------------------------------
# Figure 3 — architecture (shield OFF)
# ---------------------------------------------------------------------------
def fig03() -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#F7F9FC", ec="#1F4E79", fs=8.2):
        p = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.25,
            edgecolor=ec,
            facecolor=fc,
        )
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#222222")

    def arrow(x1, y1, x2, y2):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="#444444", lw=1.25),
        )

    box(0.25, 2.2, 1.7, 1.6, "Campus\nsensor\n(local CO$_2$)", fc="#EEF3F8")
    box(2.2, 2.2, 2.0, 1.6, "Local state\nCO$_2$, $\\Delta$CO$_2$, AoI\nmotion, battery, RSSI", fc="#EEF3F8")
    box(4.5, 3.55, 2.2, 1.35, "(1) Semantic expert\n(demonstration)", fc="#FFF4E8", ec="#F58518")
    box(4.5, 2.0, 2.2, 1.25, "(2) Behavior cloning\n(decentralized actor)", fc="#FFF4E8", ec="#F58518")
    box(4.5, 0.45, 2.2, 1.35, "(3) KL-CMAPPO fine-tune\nconstraints + KL($\\pi\\|\\pi_{BC}$)\nshield OFF", fc="#FDEBEC", ec="#E45756")
    box(7.05, 2.35, 1.7, 1.3, "Action\nTX / SKIP", fc="#E8F5E9", ec="#54A24B")
    box(9.05, 3.35, 2.5, 1.2, "TX → server receives\ntrue measurement", fc="#E8F5E9", ec="#54A24B")
    box(9.05, 1.55, 2.5, 1.2, "SKIP → causal recon\n(LOCF) at server", fc="#E8F5E9", ec="#54A24B")
    box(9.05, 0.25, 2.5, 1.0, "Campus monitoring\nevents · freshness", fc="#F7F9FC")

    arrow(1.95, 3.0, 2.2, 3.0)
    arrow(4.2, 3.0, 4.5, 3.0)
    arrow(5.6, 3.55, 5.6, 3.25)
    arrow(5.6, 2.0, 5.6, 1.8)
    arrow(6.7, 3.0, 7.05, 3.0)
    arrow(8.75, 3.3, 9.05, 3.8)
    arrow(8.75, 2.7, 9.05, 2.3)
    arrow(10.3, 3.35, 10.3, 2.75)
    arrow(10.3, 1.55, 10.3, 1.25)

    ax.text(0.3, 5.5, "CAMPUS-SenseRL method (final)", fontsize=12, fontweight="bold", color="#1F4E79")
    ax.text(
        0.3,
        5.05,
        "Centralized training · decentralized execution  ·  "
        "KL-regularized constrained MAPPO with validation-enforced feasibility",
        fontsize=8.5,
        color="#444444",
    )
    ax.text(4.5, 5.15, "Training pipeline", fontsize=8, color="#F58518", fontweight="bold")
    _save(fig, "fig03_framework")


# ---------------------------------------------------------------------------
# Figure 4 — overall policy Pareto
# ---------------------------------------------------------------------------
def fig04() -> None:
    apply_paper_style()
    s = pd.read_csv(PAPER / "full_val_summary.csv")
    want = {
        "fixed_30": ("Fixed 30", "#9ECBE2", "o"),
        "fixed_60": ("Fixed 60", "#4C78A8", "s"),
        "delta_plus_heartbeat": ("Delta+heartbeat", "#54A24B", "D"),
        "mappo_old_shield": ("Old MAPPO+shield", "#B0B0B0", "x"),
        "semantic_expert": ("Semantic expert", "#2CA02C", "^"),
        "campus_senserl_bc": ("BC", "#B279A2", "o"),
        "cmappo_kl": ("KL-CMAPPO", "#E45756", "*"),
    }
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for method, (label, color, marker) in want.items():
        row = s[s["method"] == method]
        if row.empty:
            continue
        r = row.iloc[0]
        x = float(r["tx_reduction_mean"])
        y = float(r["mae_mean"]) if pd.notna(r["mae_mean"]) else np.nan
        if not np.isfinite(y):
            continue
        xerr = float(r["tx_reduction_std"]) if float(r["n"]) > 1 else None
        yerr = float(r["mae_std"]) if float(r["n"]) > 1 and pd.notna(r["mae_std"]) else None
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt=marker,
            color=color,
            markersize=12 if method == "cmappo_kl" else 8,
            label=label,
            capsize=3,
            elinewidth=1.0,
            zorder=5 if method == "cmappo_kl" else 3,
        )
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=7.5, color=color)
    ax.set_xlabel("Transmission reduction (%)  →")
    ax.set_ylabel("Skipped-slot CO$_2$ MAE (ppm)")
    ax.set_title("Overall policy Pareto (validation mean ± std)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25)
    ax.text(
        0.98,
        0.02,
        "Better → right and down",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#666666",
    )
    fig.tight_layout()
    _save(fig, "fig04_policy_pareto")


# ---------------------------------------------------------------------------
# Figure 5 — matched budget
# ---------------------------------------------------------------------------
def fig05() -> None:
    apply_paper_style()
    raw = pd.read_csv(SCI / "matched_budget_raw.csv")
    summ = pd.read_csv(SCI / "matched_budget_summary.csv")
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    styles = {
        "campus_senserl_bc": ("#4C78A8", "BC", "-"),
        "cmappo_kl": ("#E45756", "KL-CMAPPO", "-"),
    }
    for method, (color, label, ls) in styles.items():
        g = raw[raw["method"] == method]
        curve = (
            g.groupby("prob_threshold")[["transmission_reduction_pct", "mae_skipped"]]
            .mean()
            .reset_index()
            .sort_values("transmission_reduction_pct")
        )
        ax.plot(
            curve["transmission_reduction_pct"],
            curve["mae_skipped"],
            ls=ls,
            color=color,
            marker="o",
            markersize=3.5,
            label=label,
            alpha=0.85,
        )
        sub = summ[summ["method"] == method]
        ax.errorbar(
            sub["target_tx_reduction"],
            sub["mae_mean"],
            yerr=sub["mae_std"],
            fmt="o",
            color=color,
            markersize=9,
            markerfacecolor="white",
            markeredgewidth=2,
            capsize=4,
            label=f"{label} matched 75/78/80%",
            zorder=6,
        )
    # annotate 78%
    for method, color in [("campus_senserl_bc", "#4C78A8"), ("cmappo_kl", "#E45756")]:
        r = summ[(summ["method"] == method) & (summ["target_tx_reduction"] == 78.0)].iloc[0]
        ax.annotate(
            f"{r['mae_mean']:.2f}",
            (78.0, r["mae_mean"]),
            textcoords="offset points",
            xytext=(8, -2 if method.endswith("bc") else 8),
            fontsize=8,
            color=color,
        )
    ax.axvline(78.0, color="#AAAAAA", ls=":", lw=1.0)
    ax.set_xlabel("Transmission reduction (%)")
    ax.set_ylabel("Skipped-slot CO$_2$ MAE (ppm)")
    ax.set_title("Matched-budget comparison (validation, 5 seeds)")
    ax.legend(frameon=False, fontsize=7.5)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save(fig, "fig05_matched_budget")


# ---------------------------------------------------------------------------
# Figure 6 — KL-CMAPPO qualitative timeline
# ---------------------------------------------------------------------------
def fig06() -> None:
    apply_paper_style()
    cfg = load_yaml(ROOT / "configs" / "rl_cmappo.yaml")
    ckpt = ROOT / "outputs" / "rl_final" / "cmappo_kl" / "seed_123" / "best_model.pt"
    act = load_mappo_policy(ckpt, device="cpu", prob_threshold=0.5)
    env = make_final_env(split="val", cfg=cfg, multi_agent=True, shield_enabled=False)
    obs, _ = env.reset(seed=123)

    # Pick sensor with interesting mid-val dynamics
    sensor_ids = list(env.sensor_ids)
    # score sensors by std of GT in a candidate window
    t0, length = 1800, 160
    best_i, best_score = 0, -1.0
    for i in range(env.n_sensors):
        seg = env.ground_truth[t0 : t0 + length, i]
        seg = seg[np.isfinite(seg)]
        if len(seg) < length // 2:
            continue
        std = float(np.std(seg))
        if 30 < std < 200 and std > best_score:
            best_score, best_i = std, i

    # warm-up
    for _ in range(t0):
        local = env.local_available[env._t]
        actions = act(obs, local_available=local)
        obs, _, term, trunc, _ = env.step(actions)
        if term or trunc:
            obs, _ = env.reset(seed=123)
            break

    times, true_co2, recon_co2, tx_pts, actions_h, aoi_h = [], [], [], [], [], []
    for k in range(length):
        t = env._t
        local = env.local_available[t]
        actions = act(obs, local_available=local)
        obs, _, term, trunc, info = env.step(actions)
        i = best_i
        gt = float(env.ground_truth[t, i]) if np.isfinite(env.ground_truth[t, i]) else np.nan
        rc = float(info["reconstruction"][i]) if np.isfinite(info["reconstruction"][i]) else np.nan
        final = int(info["final_actions"][i])
        aoi = float(info.get("aoi_raw", info.get("aoi_state"))[i])
        times.append(k)
        true_co2.append(gt)
        recon_co2.append(rc)
        actions_h.append(final)
        aoi_h.append(aoi)
        tx_pts.append(gt if final == TRANSMIT and local[i] else np.nan)
        if term or trunc:
            break

    times = np.asarray(times)
    true_co2 = np.asarray(true_co2)
    recon_co2 = np.asarray(recon_co2)
    actions_h = np.asarray(actions_h)
    aoi_h = np.asarray(aoi_h)
    tx_pts = np.asarray(tx_pts)
    hours = times * 0.25  # 15-min steps → hours

    fig, axes = plt.subplots(3, 1, figsize=(9.8, 6.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 0.9, 0.9]})
    ax = axes[0]
    ax.plot(hours, true_co2, color="#1F4E79", lw=1.3, label="True CO$_2$")
    # recon on skips
    skip = actions_h == 0
    ax.plot(hours[skip], recon_co2[skip], color="#2CA02C", lw=1.0, alpha=0.85, label="Server recon (SKIP)")
    ax.scatter(hours, tx_pts, s=22, c="#D62728", zorder=4, label="TX")
    ax.axhline(1000, color="#C44E52", ls="--", lw=0.9, alpha=0.8, label="1000 ppm")
    ax.set_ylabel("CO$_2$ (ppm)")
    ax.set_title(f"KL-CMAPPO behaviour (seed 123, sensor {sensor_ids[best_i][-6:]}, shield OFF)")
    ax.legend(frameon=False, ncol=4, fontsize=7.5, loc="upper right")

    ax = axes[1]
    ax.fill_between(hours, 0, actions_h, step="mid", color="#E45756", alpha=0.75)
    ax.set_ylim(-0.05, 1.15)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["SKIP", "TX"])
    ax.set_ylabel("Decision")

    ax = axes[2]
    ax.plot(hours, aoi_h, color="#F58518", lw=1.2)
    ax.set_ylabel("Raw AoI")
    ax.set_xlabel("Time (hours from window start)")
    fig.tight_layout()
    _save(fig, "fig06_kl_cmappo_timeline")
    save_json(
        {
            "checkpoint": str(ckpt),
            "sensor_id": sensor_ids[best_i],
            "sensor_index": best_i,
            "start_step": t0,
            "length": length,
            "shield": False,
        },
        FIG / "fig06_meta.json",
    )


# ---------------------------------------------------------------------------
# Figure 7 — held-out generalization
# ---------------------------------------------------------------------------
def fig07() -> None:
    apply_paper_style()
    h = pd.read_csv(SCI / "heldout_transfer_summary.csv")
    # use held-out TEST for BC and KL-CMAPPO
    rows = h[(h["split"] == "test") & (h["method"].isin(["campus_senserl_bc", "cmappo_kl"]))].copy()
    order = ["campus_senserl_bc", "cmappo_kl"]
    labels = {"campus_senserl_bc": "BC", "cmappo_kl": "KL-CMAPPO"}
    colors = {"campus_senserl_bc": "#4C78A8", "cmappo_kl": "#E45756"}

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
    # Panel A MAE
    ax = axes[0]
    for i, m in enumerate(order):
        r = rows[rows["method"] == m].iloc[0]
        ax.errorbar(
            i,
            r["mae_mean"],
            yerr=r["mae_std"],
            fmt="o",
            color=colors[m],
            markersize=10,
            capsize=5,
            label=f"{labels[m]} (TX↓ {r['tx_reduction_mean']:.1f}%)",
        )
    ax.set_xticks([0, 1])
    ax.set_xticklabels([labels[m] for m in order])
    ax.set_ylabel("CO$_2$ MAE (ppm)")
    ax.set_title("Held-out sensors — reconstruction")
    ax.legend(frameon=False, fontsize=7.5)
    ax.grid(True, axis="y", alpha=0.25)

    # Panel B recall
    ax = axes[1]
    for i, m in enumerate(order):
        r = rows[rows["method"] == m].iloc[0]
        ax.errorbar(
            i,
            100 * r["recall_mean"],
            yerr=100 * r["recall_std"],
            fmt="o",
            color=colors[m],
            markersize=10,
            capsize=5,
        )
        ax.annotate(
            f"{100*r['recall_mean']:.2f}%",
            (i, 100 * r["recall_mean"]),
            textcoords="offset points",
            xytext=(8, 0),
            fontsize=8,
            color=colors[m],
        )
    ax.set_xticks([0, 1])
    ax.set_xticklabels([labels[m] for m in order])
    ax.set_ylabel("Event recall (%)")
    ax.set_title("Held-out sensors — event preservation")
    ax.set_ylim(99.0, 100.0)
    ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Transfer to 40 unseen campus sensors (held-out test, 5 seeds)", fontsize=11, y=1.02)
    fig.tight_layout()
    _save(fig, "fig07_heldout_transfer")


# ---------------------------------------------------------------------------
# Figure 8 — ablation Pareto
# ---------------------------------------------------------------------------
def fig08() -> None:
    apply_paper_style()
    abl = pd.read_csv(SCI / "ablation_kl_cmappo_val.csv")
    keep = {
        "bc_only": ("BC", "#4C78A8"),
        "mappo_no_kl_no_constraints": ("BC+MAPPO", "#9ECBE2"),
        "kl_only": ("BC+KL only", "#54A24B"),
        "constraints_only": ("BC+constraints", "#F58518"),
        "full_kl_cmappo": ("KL-CMAPPO", "#E45756"),
    }
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for key, (label, color) in keep.items():
        r = abl[abl["ablation"] == key]
        if r.empty:
            continue
        r = r.iloc[0]
        x = float(r["transmission_reduction_pct"])
        y = float(r["mae_skipped"])
        ax.scatter(x, y, s=120 if key == "full_kl_cmappo" else 80, c=color, zorder=3, label=label)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8, color=color)
    # footnote about catastrophic MAPPO
    cat = abl[abl["ablation"] == "old_mappo_no_shield"]
    note = ""
    if not cat.empty:
        note = (
            f"Old MAPPO without shield omitted from axes "
            f"(MAE≈{cat.iloc[0]['mae_skipped']:.0f} ppm, degenerate skip policy)."
        )
    ax.set_xlabel("Transmission reduction (%)  →")
    ax.set_ylabel("Skipped-slot CO$_2$ MAE (ppm)")
    ax.set_title("KL-CMAPPO ablation (validation, seed 42)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.25)
    if note:
        ax.text(0.0, -0.18, note, transform=ax.transAxes, fontsize=7.5, color="#666666")
    ax.text(0.98, 0.02, "Better → right and down", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#666666")
    fig.tight_layout()
    _save(fig, "fig08_ablation_pareto")


# ---------------------------------------------------------------------------
# Figure 9 — robustness MAE + recall
# ---------------------------------------------------------------------------
def fig09() -> None:
    apply_paper_style()
    rob = pd.read_csv(SCI / "robustness_summary.csv")
    pl = rob[rob["condition"] == "packet_loss"].copy()
    methods = [
        ("semantic_expert", "Expert", "#54A24B"),
        ("campus_senserl_bc", "BC", "#4C78A8"),
        ("cmappo_kl", "KL-CMAPPO", "#E45756"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    for method, label, color in methods:
        g = pl[pl["method"] == method].sort_values("level")
        if g.empty:
            continue
        axes[0].errorbar(
            g["level"] * 100,
            g["mae_mean"],
            yerr=g["mae_std"].fillna(0),
            fmt="-o",
            color=color,
            label=label,
            capsize=3,
        )
        axes[1].errorbar(
            g["level"] * 100,
            g["recall_mean"],
            yerr=g["recall_std"].fillna(0),
            fmt="-o",
            color=color,
            label=label,
            capsize=3,
        )
    axes[0].set_xlabel("Packet loss (%)")
    axes[0].set_ylabel("Skipped-slot CO$_2$ MAE (ppm)")
    axes[0].set_title("Reconstruction under packet loss")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(True, alpha=0.25)
    axes[1].set_xlabel("Packet loss (%)")
    axes[1].set_ylabel("Event recall")
    axes[1].set_title("Event preservation under packet loss")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(True, alpha=0.25)
    fig.suptitle("Robustness (validation)", fontsize=11)
    fig.tight_layout()
    _save(fig, "fig09_robustness")


def write_docs() -> None:
    index = ROOT / "paper_outputs" / "Figure_Index.md"
    index.write_text(
        """# Figure Index — CAMPUS-SenseRL (final manuscript set)

All figures in `paper_outputs/figures/` are regenerated from frozen
`outputs/rl_final/paper_final/` and `outputs/rl_final/scientific_validation/`.
Obsolete plots are in `paper_outputs/legacy_provisional/figures/`.

## Figure 1 — `fig01_sensor_network`
Campus IoT deployment with **RL development cohort (40)** and **held-out transfer cohort (40)** over the full 429-sensor map; floor composition panel.
**Message:** method developed on a geographic subset, then evaluated on unseen campus sensors.

## Figure 2 — `fig02_co2_example`
Representative real CO₂ trajectory (final cohort, validation) with stable vs rising intervals highlighted.
**Message:** information value varies over time → adaptive communication.

## Figure 3 — `fig03_framework`
Final method: Semantic expert → Behavior cloning → **KL-CMAPPO (shield OFF)** → TX/SKIP → server reconstruction/monitoring. CTDE noted.
**Message:** methodological progression; no safety-shield dependence in the proposed policy.

## Figure 4 — `fig04_policy_pareto`
Validation Pareto: TX reduction vs MAE for Fixed-30/60, delta+heartbeat, old MAPPO+shield, expert, BC, KL-CMAPPO (error bars for multi-seed).
**Message:** adaptive policies dominate periodic scheduling in the communication–information plane.

## Figure 5 — `fig05_matched_budget`
BC vs KL-CMAPPO threshold curves with matched operating points at **75 / 78 / 80%** TX reduction (5-seed mean±std).
**Message:** RL MAE gain is not explained solely by transmitting more.

## Figure 6 — `fig06_kl_cmappo_timeline`
Qualitative rollout of **frozen KL-CMAPPO (seed 123, shield OFF)** on a real sensor: true CO₂, TX markers, skip reconstructions, decisions, AoI.
**Message:** learned policy adapts TX frequency to environmental dynamics.

## Figure 7 — `fig07_heldout_transfer`
Held-out **test** transfer (40 never-trained sensors): MAE and event recall for BC vs KL-CMAPPO (5 seeds).
**Message:** parameter-shared actor generalizes across campus sensors.

## Figure 8 — `fig08_ablation_pareto`
Ablation Pareto (seed 42): BC, BC+MAPPO, BC+KL, BC+constraints, full KL-CMAPPO. Catastrophic no-shield MAPPO omitted from axes (reported in caption/table).
**Message:** constraints preserve quality; KL keeps communication near the BC budget.

## Figure 9 — `fig09_robustness`
Packet loss 0–40%: MAE and event recall for Expert / BC / KL-CMAPPO.
**Message:** operational resilience under communication failures.

## Intentionally not in the main set
- Reconstruction benchmark bars → appendix / table
- Event-recall vs TX curve with tiny 0.989–0.999 range → table/robustness
- AoI-vs-TX curve → table / supplement
- Training reward curves → supplement if requested
""",
        encoding="utf-8",
    )

    main = ROOT / "paper_outputs" / "Main_Results.md"
    main.write_text(
        """# Main Results — CAMPUS-SenseRL (claim-ready)

Sources: `outputs/rl_final/paper_final/`, `outputs/rl_final/scientific_validation/`.
Algorithm wording: **KL-regularized constrained multi-agent policy optimization with validation-enforced feasibility** (shield OFF).

## Dataset / cohorts
- University of Oulu Smart Campus LoRaWAN CO₂ traces.
- **RL development:** frozen 40-sensor cohort (`final_cohort.json`).
- **Transfer:** 40 held-out eligible sensors never used to train/tune RL (`heldout_cohort.json`).

## Validation policy comparison (mean ± std)

| Method | TX↓ % | MAE | Recall | Precision | AoI raw |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed 30 | 50.0 | 9.66 | 0.931 | 0.903 | 0.60 |
| Fixed 60 | 75.3 | 11.34 | 0.887 | 0.856 | 1.89 |
| Delta+heartbeat | 79.6 | 10.60 | 0.966 | 0.902 | 1.99 |
| Semantic expert | 73.9 | **7.22** | **0.999** | 0.988 | **1.75** |
| BC (5 seeds) | 80.0±0.13 | 9.19±0.03 | 0.993±0.001 | 0.975±0.002 | 2.16±0.02 |
| **KL-CMAPPO (5 seeds)** | **77.8±0.78** | **8.66±0.16** | **0.995±0.002** | **0.987±0.006** | **1.94±0.09** |
| Old MAPPO+shield | 78.9 | 13.08 | 0.935 | 0.844 | 3.35 |

**Claim (not domination):** KL-CMAPPO trades ~2.2 pp extra communication vs BC for better MAE, recall, precision, and AoI.

## Matched communication budgets (VAL, 5 seeds)

| Target TX↓ | BC MAE | KL-CMAPPO MAE |
| ---: | ---: | ---: |
| 75% | 7.94±0.02 | **7.90±0.03** |
| 78% | 8.87±0.01 | **8.72±0.07** |
| 80% | 9.20±0.02 | **9.13±0.04** |

## Frozen temporal test (KL-CMAPPO seed 123)
TX↓ 77.5%, MAE 8.37, recall 0.961, precision 0.925 (improves BC; Fixed-60 recall collapses).

## Held-out sensors (test, 5 seeds)
| Method | TX↓ % | MAE | Recall |
| --- | ---: | ---: | ---: |
| BC | 79.95±0.10 | 8.93±0.05 | 0.996 |
| KL-CMAPPO | 77.77±0.70 | **8.43±0.12** | **0.997** |

## Ablation (seed 42)
Constraints alone → best MAE but lower TX↓ (~70.5%). KL alone ≈ BC. Full KL-CMAPPO balances quality and budget (~78.7%, MAE 8.87). Old MAPPO without shield collapses (MAE≈74).

## Reconstruction note
Causal reconstructors were benchmarked; **LOCF** is used online in the RL environment for causal sequential suitability. Detailed recon numbers belong in the appendix — not the main contribution.
""",
        encoding="utf-8",
    )
    print(f"  wrote {index}")
    print(f"  wrote {main}")


def main() -> None:
    ensure_dir(FIG)
    print("[1] archive obsolete figures → legacy_provisional/")
    archive_obsolete()
    print("[2] regenerate manuscript figures 1–9")
    fig01()
    fig02()
    fig03()
    fig04()
    fig05()
    fig06()
    fig07()
    fig08()
    fig09()
    print("[3] update Figure_Index.md + Main_Results.md")
    write_docs()
    save_json(
        {
            "dpi": DPI,
            "sources": ["paper_final", "scientific_validation", "heldout_cohort"],
            "shield": False,
            "no_training": True,
        },
        FIG / "manuscript_figures_meta.json",
    )
    print("[done] manuscript figure pack ready")


if __name__ == "__main__":
    main()
