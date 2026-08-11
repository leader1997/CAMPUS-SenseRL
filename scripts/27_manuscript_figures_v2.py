#!/usr/bin/env python
"""Manuscript figures v2 — visualization only (no training).

ONE scientific question → ONE graph (except method schematic).
Writes paper_outputs/manuscript_figures_v2/{main,supplementary}/
PNG 600 dpi + PDF when possible.
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch, Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from campus_senserl.data.audit import load_devices
from campus_senserl.data.cohort import load_cohort
from campus_senserl.environment.communication_model import TRANSMIT
from campus_senserl.evaluation.rl_policy_eval import load_mappo_policy, make_final_env
from campus_senserl.rl.fixed_policies import FixedIntervalPolicy
from campus_senserl.utils import ensure_dir, load_yaml, save_json
from campus_senserl.visualization.paper_style import apply_paper_style

OUT = ROOT / "paper_outputs" / "manuscript_figures_v2"
MAIN = OUT / "main"
SUPP = OUT / "supplementary"
REPORTS = ROOT / "reports"
PAPER = ROOT / "outputs" / "rl_final" / "paper_final"
SCI = ROOT / "outputs" / "rl_final" / "scientific_validation"
CMAPPO = ROOT / "outputs" / "rl_final" / "cmappo_kl"
RECON = ROOT / "outputs" / "reconstruction_final" / "fair_benchmark" / "fair_results_summary.csv"
SEEDS = [42, 123, 2024, 3407, 9999]
DPI = 600

# Consistent color language
C_FIXED = "#6B7280"
C_FIXED15 = "#9CA3AF"
C_DELTA = "#15803D"
C_EXPERT = "#166534"
C_BC = "#6A51A3"
C_KL = "#B91C1C"
C_RL_DEV = "#1F2937"
C_HELD = "#0369A1"
C_BG = "#D1D5DB"

VALUES: list[dict] = []
INDEX: list[dict] = []
AUDIT: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    AUDIT.append(msg)


def record_value(figure: str, method: str, metric: str, value, extra: str = "", source: str = "") -> None:
    VALUES.append(
        {
            "figure": figure,
            "method": method,
            "metric": metric,
            "value": value,
            "extra": extra,
            "source": source,
        }
    )


def record_index(**kwargs) -> None:
    INDEX.append(kwargs)


def save_fig(fig: plt.Figure, directory: Path, stem: str) -> None:
    ensure_dir(directory)
    png = directory / f"{stem}.png"
    pdf = directory / f"{stem}.pdf"
    fig.savefig(png, dpi=DPI, facecolor="white", bbox_inches="tight")
    try:
        fig.savefig(pdf, facecolor="white", bbox_inches="tight")
        log(f"WROTE {png.relative_to(ROOT)} + PDF")
    except Exception as exc:  # pragma: no cover
        log(f"WROTE {png.relative_to(ROOT)} (PDF failed: {exc})")
    plt.close(fig)


def save_main(fig: plt.Figure, stem: str) -> None:
    save_fig(fig, MAIN, stem)


def save_supp(fig: plt.Figure, stem: str) -> None:
    save_fig(fig, SUPP, stem)


def load_val_summary() -> pd.DataFrame:
    return pd.read_csv(PAPER / "full_val_summary.csv")


def load_test_summary() -> pd.DataFrame:
    return pd.read_csv(PAPER / "full_test_summary.csv")


def load_matched() -> pd.DataFrame:
    return pd.read_csv(SCI / "matched_budget_summary.csv")


def load_heldout() -> pd.DataFrame:
    return pd.read_csv(SCI / "heldout_transfer_summary.csv")


def load_robust() -> pd.DataFrame:
    return pd.read_csv(SCI / "robustness_summary.csv")


def load_ablation() -> pd.DataFrame:
    return pd.read_csv(SCI / "ablation_kl_cmappo_val.csv")


def load_seed_metrics(seed: int) -> dict:
    return json.loads((CMAPPO / f"seed_{seed}" / "metrics.json").read_text(encoding="utf-8"))


def val_rows(seed: int) -> pd.DataFrame:
    m = load_seed_metrics(seed)["metrics"]
    return pd.DataFrame([r for r in m if "val_mae" in r])


def train_rows(seed: int) -> pd.DataFrame:
    m = load_seed_metrics(seed)["metrics"]
    return pd.DataFrame([r for r in m if "mean_tx" in r and "val_mae" not in r])


def aggregate_val(col: str) -> pd.DataFrame:
    frames = []
    for s in SEEDS:
        v = val_rows(s)
        v["seed"] = s
        frames.append(v)
    allv = pd.concat(frames, ignore_index=True)
    g = allv.groupby("step", as_index=False).agg(mean=(col, "mean"), std=(col, "std"))
    feas = allv.groupby("step")["feasible"].sum().reset_index(name="n_feasible")
    return g.merge(feas, on="step")


def _select_window(env, length: int = 192) -> tuple[int, int]:
    gt = env.ground_truth
    n_steps, n_s = gt.shape
    best = None
    for t0 in range(400, min(n_steps - length - 10, 4000), 96):
        for i in range(n_s):
            seg = gt[t0 : t0 + length, i]
            if not np.isfinite(seg).all():
                continue
            std = float(np.std(seg))
            mx = float(np.max(seg))
            rise = float(np.max(seg) - np.min(seg))
            if 35 <= std <= 160 and mx < 1600 and rise >= 120:
                score = abs(std - 80) + 0.002 * abs(mx - 900)
                if best is None or score < best[0]:
                    best = (score, t0, i)
    if best is None:
        return 800, 0
    return best[1], best[2]


def _rollout_actions(env, policy_act, start: int, length: int, seed: int = 42):
    obs, _ = env.reset(seed=seed)
    for _ in range(start):
        local = env.local_available[env._t]
        a = policy_act(obs, local_available=local) if callable(policy_act) else policy_act.act(obs, local_available=local)
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            obs, _ = env.reset(seed=seed)
            break
    hist = {"t": [], "gt": [], "recon": [], "actions": [], "avail": [], "tx_delivered": []}
    for k in range(length):
        t = env._t
        local = env.local_available[t]
        a = policy_act(obs, local_available=local) if callable(policy_act) else policy_act.act(obs, local_available=local)
        obs, _, term, trunc, info = env.step(a)
        hist["t"].append(k)
        hist["gt"].append(env.ground_truth[t].copy())
        hist["recon"].append(np.asarray(info["reconstruction"], dtype=float).copy())
        hist["actions"].append(np.asarray(info["final_actions"], dtype=int).copy())
        hist["avail"].append(local.copy())
        hist["tx_delivered"].append(env.transmit_history[t].copy())
        if term or trunc:
            break
    for k in hist:
        hist[k] = np.asarray(hist[k])
    return hist


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def fig01() -> None:
    apply_paper_style()
    cfg = load_yaml(ROOT / "configs" / "data.yaml")
    devices = load_devices(ROOT / cfg["paths"]["raw_release"] / cfg["dataset"]["devices_file"]).copy()
    devices["deveui"] = devices["device_id"].astype(str).str.upper()
    train = {s.upper() for s in load_cohort("final")}
    held = {s.upper() for s in load_cohort("heldout")}
    co2 = devices["device_type"].astype(str).str.contains("CO2", case=False, na=False)
    tr = devices[devices["deveui"].isin(train)]
    ho = devices[devices["deveui"].isin(held)]
    bg = devices.loc[~devices["deveui"].isin(train | held)]
    bg_co2 = bg.loc[co2]

    fig, ax = plt.subplots(figsize=(6.4, 5.5))
    ax.scatter(bg["longitude"], bg["latitude"], s=8, c=C_BG, alpha=0.45, zorder=1, label=f"Other campus sensors (n={len(bg)})")
    if len(bg_co2):
        ax.scatter(
            bg_co2["longitude"],
            bg_co2["latitude"],
            s=14,
            facecolors="none",
            edgecolors="#9CA3AF",
            linewidths=0.6,
            alpha=0.7,
            zorder=2,
            label=f"Other CO$_2$-capable (n={len(bg_co2)})",
        )
    ax.scatter(
        tr["longitude"],
        tr["latitude"],
        s=42,
        c=C_RL_DEV,
        edgecolors="white",
        linewidths=0.45,
        zorder=3,
        label=f"RL development (n={len(tr)})",
    )
    ax.scatter(
        ho["longitude"],
        ho["latitude"],
        s=48,
        c=C_HELD,
        marker="D",
        edgecolors="white",
        linewidths=0.45,
        zorder=4,
        label=f"Held-out transfer (n={len(ho)})",
    )
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    save_main(fig, "fig01_campus_deployment")
    src = "devices + final_cohort.json + heldout_cohort.json"
    record_value("fig01", "other", "n_sensors", len(bg), "metadata", src)
    record_value("fig01", "rl_development", "n_sensors", len(tr), "metadata", src)
    record_value("fig01", "heldout", "n_sensors", len(ho), "metadata", src)
    record_index(
        file="main/fig01_campus_deployment.png",
        question="Is KL-CMAPPO evaluated on a real distributed smart-campus deployment?",
        source=src,
        split="metadata (spatial)",
        methods="other / RL development / held-out",
        seeds="N/A",
        metric="sensor coordinates",
        values=f"other={len(bg)}, RL={len(tr)}, heldout={len(ho)}",
    )


def fig02() -> None:
    apply_paper_style()
    cfg = load_yaml(ROOT / "configs" / "rl_cmappo.yaml")
    env = make_final_env(split="val", cfg=cfg, multi_agent=True)
    start, sid = _select_window(env, length=192)
    sensor_id = env.sensor_ids[sid]
    length = 192
    fixed = FixedIntervalPolicy(1)

    def fixed_act(obs, *, local_available=None):
        return fixed.act(obs, local_available=local_available)

    env_f = make_final_env(split="val", cfg=cfg, multi_agent=True)
    h_fix = _rollout_actions(env_f, fixed_act, start, length, seed=42)
    ckpt = CMAPPO / "seed_123" / "best_model.pt"
    act = load_mappo_policy(ckpt, device="cpu", prob_threshold=0.5)
    env_k = make_final_env(split="val", cfg=cfg, multi_agent=True)
    h_kl = _rollout_actions(env_k, act, start, length, seed=42)

    hours = np.arange(len(h_kl["t"])) * 0.25
    gt = h_kl["gt"][:, sid]
    avail = h_kl["avail"][:, sid].astype(bool)
    fixed_tx = h_fix["avail"][:, sid].astype(bool)
    kl_tx = (h_kl["actions"][:, sid] == TRANSMIT) & avail
    kl_recon = h_kl["recon"][:, sid]
    n_fix = int(fixed_tx.sum())
    n_kl = int(kl_tx.sum())
    red = 100.0 * (1.0 - n_kl / n_fix) if n_fix else float("nan")

    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    ax.plot(hours, gt, color="#111827", lw=1.5, label="True CO$_2$", zorder=2)
    skip = (~kl_tx) & avail
    ax.plot(hours[skip], kl_recon[skip], color="#A7F3D0", lw=1.0, alpha=0.95, label="Causal recon (SKIP)", zorder=1)
    ax.scatter(hours[kl_tx], gt[kl_tx], s=42, c=C_KL, zorder=4, edgecolors="white", linewidths=0.3, label="KL-CMAPPO TX")
    ax.axhline(1000.0, color="#6B7280", ls="--", lw=1.0, label="1000 ppm threshold")
    ax.set_xlabel("Time (hours from window start)", fontsize=11)
    ax.set_ylabel("CO$_2$ (ppm)", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.text(
        0.98,
        0.02,
        f"Fixed 15-min reference: {n_fix} transmissions\n"
        f"KL-CMAPPO: {n_kl} transmissions\n"
        f"Reduction in this interval: {red:.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#D1D5DB", alpha=0.92),
    )
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    save_main(fig, "fig02_real_adaptive_communication")
    meta = {
        "split": "validation",
        "sensor_id": sensor_id,
        "sensor_index": int(sid),
        "start_step": int(start),
        "length": length,
        "selection": "deterministic: std∈[35,160], max<1600, rise≥120; minimize |std-80|+0.002*|max-900|",
        "fixed15_tx": n_fix,
        "kl_cmappo_tx": n_kl,
        "reduction_vs_fixed15_pct": red,
        "checkpoint": str(ckpt),
        "note": "Frozen checkpoint inference only; no training.",
    }
    save_json(meta, MAIN / "fig02_meta.json")
    src = "frozen KL-CMAPPO seed_123 val rollout + FixedInterval(1)"
    record_value("fig02", "Fixed 15", "interval_tx", n_fix, f"sensor={sensor_id}; start={start}", src)
    record_value("fig02", "KL-CMAPPO", "interval_tx", n_kl, f"sensor={sensor_id}", src)
    record_value("fig02", "KL-CMAPPO", "reduction_vs_fixed15_pct", round(red, 2), "", src)
    record_index(
        file="main/fig02_real_adaptive_communication.png",
        question="What does adaptive TX/SKIP scheduling look like on a real CO₂ series?",
        source=src,
        split="validation (illustrative window)",
        methods="KL-CMAPPO (+ Fixed15 count annotation)",
        seeds="policy seed 123; rollout seed 42",
        metric="CO2 ppm + TX markers",
        values=f"Fixed15 TX={n_fix}, KL TX={n_kl}, red={red:.2f}%",
    )


def fig03() -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(5.2, 9.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16.0)
    ax.axis("off")

    def box(y, text, fc, ec, h=1.0, fs=10, lw=1.4):
        ax.add_patch(
            FancyBboxPatch((0.9, y), 8.2, h, boxstyle="round,pad=0.02,rounding_size=0.08", lw=lw, ec=ec, fc=fc)
        )
        ax.text(5.0, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(y1, y2):
        ax.annotate("", xy=(5.0, y2), xytext=(5.0, y1), arrowprops=dict(arrowstyle="->", color="#374151", lw=1.5))

    box(14.5, "Real campus sensor state\n(local env., freshness, network,\ntemporal & neighbour context)", "#F3F4F6", "#6B7280", h=1.35, fs=9)
    arrow(14.5, 13.7)
    box(12.7, "Semantic expert demonstrations", "#DCFCE7", C_EXPERT, h=0.95, fs=10)
    arrow(12.7, 11.95)
    box(11.0, "BC initialization  →  π_BC", "#EDE9FE", C_BC, h=0.95, fs=10)
    arrow(11.0, 9.85)
    box(
        6.55,
        "KL-CMAPPO  (proposed)\n"
        "• PPO policy improvement\n"
        "• KL anchor to π_BC\n"
        "• event / MAE / AoI constraints",
        "#FEE2E2",
        C_KL,
        h=3.15,
        fs=10.5,
        lw=2.4,
    )
    arrow(6.55, 5.75)
    box(4.8, "Final shared policy\n(shield-free)", "#FEE2E2", C_KL, h=1.1, fs=10)
    arrow(4.8, 3.95)
    box(3.0, "TX / SKIP", "#FEE2E2", C_KL, h=0.9, fs=11)
    arrow(3.0, 2.2)
    box(1.15, "Server monitoring\n(causal reconstruction / LOCF)", "#F3F4F6", "#6B7280", h=1.15, fs=9.5)
    save_main(fig, "fig03_kl_cmappo_framework")
    record_index(
        file="main/fig03_kl_cmappo_framework.png",
        question="How is the proposed KL-CMAPPO policy trained?",
        source="schematic",
        split="N/A",
        methods="expert → BC init → KL-CMAPPO → TX/SKIP",
        seeds="N/A",
        metric="N/A",
        values="BC = initialization; KL-CMAPPO = final shield-free method",
    )


def fig04() -> None:
    apply_paper_style()
    s = load_val_summary().set_index("method")
    methods = ["fixed_60", "delta_plus_heartbeat", "cmappo_kl"]
    labels = ["Fixed 60", "Delta +\nheartbeat", "KL-CMAPPO"]
    colors = [C_FIXED, C_DELTA, C_KL]
    mae = [float(s.loc[m, "mae_mean"]) for m in methods]
    mae_std = [float(s.loc[m, "mae_std"]) if int(s.loc[m, "n"]) > 1 else 0.0 for m in methods]
    tx = [float(s.loc[m, "tx_reduction_mean"]) for m in methods]
    n = [int(s.loc[m, "n"]) for m in methods]
    src = "outputs/rl_final/paper_final/full_val_summary.csv"

    fig, ax = plt.subplots(figsize=(5.8, 4.3))
    x = np.arange(3)
    bars = ax.bar(x, mae, color=colors, width=0.62, edgecolor="white", linewidth=0.5)
    for i, (std, nn) in enumerate(zip(mae_std, n)):
        if nn > 1 and std > 0:
            ax.errorbar(x[i], mae[i], yerr=std, fmt="none", ecolor="#374151", capsize=4, lw=1.0)
    for i, (b, t) in enumerate(zip(bars, tx)):
        top = mae[i] + (mae_std[i] if n[i] > 1 else 0.0)
        ax.text(b.get_x() + b.get_width() / 2, top + 0.18, f"{t:.1f}% TX↓", ha="center", va="bottom", fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Skipped-observation CO$_2$ MAE (ppm)", fontsize=11)
    ax.set_ylim(0, max(mae) * 1.32)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    save_main(fig, "fig04_main_mae_comparison")
    for m, lab, a, t, nn, std in zip(methods, labels, mae, tx, n, mae_std):
        record_value("fig04", lab.replace("\n", " "), "mae_ppm", round(a, 4), f"tx↓={t:.3f}; n={nn}; std={std:.4f}", src)
    record_index(
        file="main/fig04_main_mae_comparison.png",
        question="At similar high communication savings, which strategy preserves CO₂ best?",
        source=src,
        split="validation",
        methods="Fixed 60, Delta+heartbeat, KL-CMAPPO",
        seeds="Fixed60/Delta n=1; KL-CMAPPO n=5 (mean±std)",
        metric="skipped-observation CO2 MAE (ppm)",
        values="; ".join(f"{lab.replace(chr(10),' ')}={a:.3f} ({t:.1f}% TX↓)" for lab, a, t in zip(labels, mae, tx)),
    )


def fig05() -> None:
    """Frozen TEST important-event recall — Fixed 60 / Delta / KL-CMAPPO."""
    apply_paper_style()
    s = load_test_summary().set_index("method")
    methods = ["fixed_60", "delta_plus_heartbeat", "cmappo_kl"]
    labels = ["Fixed 60", "Delta +\nheartbeat", "KL-CMAPPO"]
    colors = [C_FIXED, C_DELTA, C_KL]
    rec = [100.0 * float(s.loc[m, "recall_mean"]) for m in methods]
    n = [int(s.loc[m, "n"]) for m in methods]
    tx = [float(s.loc[m, "tx_reduction_mean"]) for m in methods]
    src = "outputs/rl_final/paper_final/full_test_summary.csv"
    freeze = json.loads((PAPER / "test_freeze_meta.json").read_text(encoding="utf-8"))

    fig, ax = plt.subplots(figsize=(5.8, 4.3))
    x = np.arange(3)
    bars = ax.bar(x, rec, color=colors, width=0.62, edgecolor="white", linewidth=0.5)
    for b, t in zip(bars, tx):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{t:.1f}% TX↓", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Important-event recall (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    save_main(fig, "fig05_event_recall_test")
    for lab, r, t, nn in zip(labels, rec, tx, n):
        record_value(
            "fig05",
            lab.replace("\n", " "),
            "event_recall_pct",
            round(r, 4),
            f"tx↓={t:.3f}; n={nn}; frozen_seed={freeze.get('frozen_cmappo_seed')}",
            src,
        )
    record_index(
        file="main/fig05_event_recall_test.png",
        question="Does communication reduction cause missed important CO₂ events?",
        source=src,
        split="frozen TEST (development cohort)",
        methods="Fixed 60, Delta+heartbeat, KL-CMAPPO",
        seeds="all n=1 on this frozen test report (KL = seed 123 checkpoint)",
        metric="important-event recall (%)",
        values="; ".join(f"{lab.replace(chr(10),' ')}={r:.2f}%" for lab, r in zip(labels, rec)),
    )


def fig06() -> None:
    apply_paper_style()
    g = aggregate_val("val_mae")
    src = "outputs/rl_final/cmappo_kl/seed_*/metrics.json (val rows)"
    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    # light individual seeds
    for s in SEEDS:
        v = val_rows(s)
        ax.plot(v["step"], v["val_mae"], color="#E5E7EB", lw=0.8, zorder=1)
    ax.fill_between(
        g["step"],
        g["mean"] - g["std"].fillna(0),
        g["mean"] + g["std"].fillna(0),
        color=C_KL,
        alpha=0.20,
        label="±1 std",
        zorder=2,
    )
    ax.plot(g["step"], g["mean"], color=C_KL, lw=2.2, label="5-seed mean", zorder=3)
    ax.axhline(9.0, color="#047857", ls="--", lw=1.3, label="MAE = 9 ppm", zorder=4)
    ax.axhspan(0, 9.0, color="#D1FAE5", alpha=0.28, zorder=0)
    ax.text(g["step"].iloc[0], 8.55, "feasible reconstruction region", fontsize=8, color="#047857", va="top")
    ax.set_xlabel("Training steps", fontsize=11)
    ax.set_ylabel("Validation CO$_2$ MAE (ppm)", fontsize=11)
    ax.set_ylim(max(0, float(g["mean"].min()) - 1.2), max(10.2, float((g["mean"] + g["std"].fillna(0)).max()) + 0.5))
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.18)
    fig.tight_layout()
    save_main(fig, "fig06_rl_training_validation_mae")
    for _, r in g.iterrows():
        record_value("fig06", "KL-CMAPPO", "val_mae_mean", round(float(r["mean"]), 4), f"step={int(r['step'])}; std={float(r['std']):.4f}", src)
    record_index(
        file="main/fig06_rl_training_validation_mae.png",
        question="Did KL-CMAPPO learning enter the predefined feasible reconstruction-quality region?",
        source=src,
        split="validation checkpoints during training",
        methods="KL-CMAPPO",
        seeds=5,
        metric="validation CO2 MAE (ppm)",
        values="mean±std; constraint MAE=9 ppm",
    )


def fig07() -> None:
    apply_paper_style()
    abl = load_ablation().set_index("ablation")
    order = [
        ("bc_only", "BC initialization\nonly"),
        ("mappo_no_kl_no_constraints", "RL fine-tune\nno KL / no constr."),
        ("kl_only", "+ KL only"),
        ("constraints_only", "+ constraints\nonly"),
        ("full_kl_cmappo", "Full\nKL-CMAPPO"),
    ]
    colors = [C_BC, "#93C5FD", "#34D399", "#F97316", C_KL]
    mae = [float(abl.loc[k, "mae_skipped"]) for k, _ in order]
    tx = [float(abl.loc[k, "transmission_reduction_pct"]) for k, _ in order]
    labels = [lab for _, lab in order]
    src = "outputs/rl_final/scientific_validation/ablation_kl_cmappo_val.csv"

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    x = np.arange(len(order))
    bars = ax.bar(x, mae, color=colors, width=0.7, edgecolor="white", linewidth=0.4)
    for b, t in zip(bars, tx):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, f"{t:.1f}% TX↓", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("CO$_2$ MAE (ppm)", fontsize=11)
    ax.set_ylim(0, max(mae) * 1.25)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    save_main(fig, "fig07_kl_constraint_ablation")
    for (k, lab), a, t in zip(order, mae, tx):
        record_value("fig07", lab.replace("\n", " "), "mae_ppm", round(a, 4), f"tx↓={t:.3f}; seed=42", src)
    record_index(
        file="main/fig07_kl_constraint_ablation.png",
        question="What do KL anchoring and explicit constraints contribute?",
        source=src,
        split="validation; single-seed mechanistic ablation (seed 42)",
        methods="BC init / no KL-no constr. / KL only / constr. only / full KL-CMAPPO",
        seeds="1 (seed 42) — not multi-seed statistics",
        metric="CO2 MAE (ppm) + TX↓ annotation",
        values="; ".join(f"{lab.replace(chr(10),' ')}={a:.3f}" for lab, a in zip(labels, mae)),
    )


def fig08() -> None:
    """KL-CMAPPO only: development-cohort frozen TEST vs held-out TEST."""
    apply_paper_style()
    test = load_test_summary().set_index("method")
    held = load_heldout()
    kl_dev = test.loc["cmappo_kl"]
    kl_ho = held[(held["cohort"] == "heldout") & (held["split"] == "test") & (held["method"] == "cmappo_kl")].iloc[0]
    freeze = json.loads((PAPER / "test_freeze_meta.json").read_text(encoding="utf-8"))

    labels = ["Development\nsensors", "Held-out\nsensors"]
    mae = [float(kl_dev["mae_mean"]), float(kl_ho["mae_mean"])]
    std = [0.0, float(kl_ho["mae_std"])]  # frozen test report is n=1
    n = [int(kl_dev["n"]), int(kl_ho["n"])]
    src_dev = "paper_final/full_test_summary.csv"
    src_ho = "scientific_validation/heldout_transfer_summary.csv"

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    x = np.arange(2)
    ax.bar(x[0], mae[0], color=C_KL, width=0.55, alpha=0.85, edgecolor="white")
    ax.bar(x[1], mae[1], yerr=std[1], color=C_KL, width=0.55, capsize=4, ecolor="#374151", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("CO$_2$ MAE (ppm)", fontsize=11)
    ax.set_ylim(0, max(mae[0], mae[1] + std[1]) * 1.28)
    for i, (a, nn) in enumerate(zip(mae, n)):
        ax.text(i, a + (std[i] if nn > 1 else 0) + 0.12, f"{a:.2f}", ha="center", va="bottom", fontsize=10)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    save_main(fig, "fig08_unseen_sensor_generalization")
    record_value(
        "fig08",
        "KL-CMAPPO development",
        "mae_ppm",
        round(mae[0], 4),
        f"frozen TEST; n={n[0]}; seed={freeze.get('frozen_cmappo_seed')}",
        src_dev,
    )
    record_value(
        "fig08",
        "KL-CMAPPO heldout",
        "mae_ppm",
        round(mae[1], 4),
        f"held-out TEST; n={n[1]}; std={std[1]:.4f}",
        src_ho,
    )
    record_index(
        file="main/fig08_unseen_sensor_generalization.png",
        question="Does KL-CMAPPO work on sensors unused during policy development?",
        source=f"{src_dev}; {src_ho}",
        split="TEST vs TEST (development frozen report vs held-out transfer)",
        methods="KL-CMAPPO only",
        seeds=f"dev n={n[0]} (frozen seed {freeze.get('frozen_cmappo_seed')}); heldout n={n[1]} mean±std",
        metric="CO2 MAE (ppm)",
        values=f"dev={mae[0]:.3f}; heldout={mae[1]:.3f}±{std[1]:.3f}",
    )


def fig09() -> None:
    apply_paper_style()
    r = load_robust()
    sub = r[(r["method"] == "cmappo_kl") & (r["condition"] == "packet_loss")].sort_values("level")
    x = (sub["level"].astype(float) * 100).values
    y = sub["mae_mean"].astype(float).values
    e = sub["mae_std"].astype(float).values
    src = "outputs/rl_final/scientific_validation/robustness_summary.csv"

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.errorbar(x, y, yerr=e, color=C_KL, marker="o", lw=2.2, ms=8, capsize=3, label="KL-CMAPPO")
    ax.set_xlabel("Packet loss (%)", fontsize=11)
    ax.set_ylabel("CO$_2$ MAE (ppm)", fontsize=11)
    ax.set_xticks(x)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.18)
    fig.tight_layout()
    save_main(fig, "fig09_packet_loss_robustness")
    for xx, yy, ee in zip(x, y, e):
        record_value("fig09", "KL-CMAPPO", f"mae_loss_{int(xx)}", round(float(yy), 4), f"std={ee:.4f}", src)
    record_index(
        file="main/fig09_packet_loss_robustness.png",
        question="How does KL-CMAPPO behave as wireless packet delivery deteriorates?",
        source=src,
        split="validation stress (packet loss)",
        methods="KL-CMAPPO",
        seeds=5,
        metric="CO2 MAE (ppm)",
        values="; ".join(f"{int(xx)}%→{yy:.3f}" for xx, yy in zip(x, y)),
    )


def fig10() -> None:
    apply_paper_style()
    cfg = load_yaml(ROOT / "configs" / "rl_cmappo.yaml")
    length = 672
    start = 3552
    fixed15 = FixedIntervalPolicy(1)
    fixed60 = FixedIntervalPolicy(4)

    def wrap(pol):
        def act(obs, *, local_available=None):
            return pol.act(obs, local_available=local_available)

        return act

    env = make_final_env(split="val", cfg=cfg, multi_agent=True)
    h15 = _rollout_actions(env, wrap(fixed15), start, length, seed=42)
    env = make_final_env(split="val", cfg=cfg, multi_agent=True)
    h60 = _rollout_actions(env, wrap(fixed60), start, length, seed=42)
    ckpt = CMAPPO / "seed_123" / "best_model.pt"
    act = load_mappo_policy(ckpt, device="cpu", prob_threshold=0.5)
    env = make_final_env(split="val", cfg=cfg, multi_agent=True)
    hkl = _rollout_actions(env, act, start, length, seed=42)

    def cum_tx(h):
        tx = ((h["actions"] == TRANSMIT) & h["avail"]).sum(axis=1)
        return np.cumsum(tx)

    c15, c60, ckl = cum_tx(h15), cum_tx(h60), cum_tx(hkl)
    hours = np.arange(len(c15)) * 0.25
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    ax.plot(hours, c15, color=C_FIXED15, lw=2.0)
    ax.plot(hours, c60, color=C_FIXED, lw=2.0)
    ax.plot(hours, ckl, color=C_KL, lw=2.3)
    ax.text(hours[-1] + 1.5, c15[-1], f"Fixed 15  ({int(c15[-1])})", color=C_FIXED15, va="center", fontsize=9)
    ax.text(hours[-1] + 1.5, c60[-1], f"Fixed 60  ({int(c60[-1])})", color=C_FIXED, va="center", fontsize=9)
    ax.text(hours[-1] + 1.5, ckl[-1], f"KL-CMAPPO  ({int(ckl[-1])})", color=C_KL, va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Time (hours)", fontsize=11)
    ax.set_ylabel("Cumulative transmissions", fontsize=11)
    ax.set_xlim(0, hours[-1] + 42)
    ax.grid(True, alpha=0.18)
    fig.tight_layout()
    save_main(fig, "fig10_cumulative_transmissions")
    src = f"val rollouts start={start} length={length}; KL seed_123"
    record_value("fig10", "Fixed 15", "cum_tx_end", int(c15[-1]), src, src)
    record_value("fig10", "Fixed 60", "cum_tx_end", int(c60[-1]), src, src)
    record_value("fig10", "KL-CMAPPO", "cum_tx_end", int(ckl[-1]), src, src)
    save_json({"start": start, "length": length, "split": "val", "checkpoint": str(ckpt)}, MAIN / "fig10_meta.json")
    record_index(
        file="main/fig10_cumulative_transmissions.png",
        question="What does ~78% TX reduction mean operationally over time?",
        source=src,
        split="validation",
        methods="Fixed 15, Fixed 60, KL-CMAPPO",
        seeds="illustrative (KL seed 123)",
        metric="cumulative transmissions",
        values=f"end Fixed15={int(c15[-1])}, Fixed60={int(c60[-1])}, KL={int(ckl[-1])}",
    )


# ---------------------------------------------------------------------------
# SUPPLEMENTARY
# ---------------------------------------------------------------------------
def figS01() -> None:
    apply_paper_style()
    m = load_matched()
    targets = [75.0, 78.0, 80.0]
    src = "matched_budget_summary.csv"
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    for method, label, color, marker in [
        ("campus_senserl_bc", "BC initialization", C_BC, "o"),
        ("cmappo_kl", "KL-CMAPPO", C_KL, "s"),
    ]:
        sub = m[m["method"] == method].set_index("target_tx_reduction").loc[targets]
        y = sub["mae_mean"].astype(float).values
        e = sub["mae_std"].astype(float).values
        ax.errorbar(targets, y, yerr=e, marker=marker, color=color, lw=2, ms=7, capsize=3, label=label)
        for t, yy, ee in zip(targets, y, e):
            record_value("figS01", label, f"mae_at_{int(t)}", round(float(yy), 4), f"std={ee:.4f}", src)
    ax.set_xlabel("Target transmission reduction (%)", fontsize=11)
    ax.set_ylabel("CO$_2$ MAE (ppm)", fontsize=11)
    ax.set_xticks(targets)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.18)
    fig.tight_layout()
    save_supp(fig, "figS01_bc_initialization_vs_kl_matched_budget")
    record_index(
        file="supplementary/figS01_bc_initialization_vs_kl_matched_budget.png",
        question="Effect of KL-CMAPPO fine-tuning relative to BC initialization at matched budgets?",
        source=src,
        split="validation",
        methods="BC initialization, KL-CMAPPO",
        seeds=5,
        metric="CO2 MAE",
        values="targets 75/78/80%",
    )


def figS02() -> None:
    apply_paper_style()
    g = aggregate_val("val_recall")
    src = "cmappo_kl metrics.json val_recall"
    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    y = 100 * g["mean"]
    e = 100 * g["std"].fillna(0)
    ax.fill_between(g["step"], y - e, y + e, color=C_KL, alpha=0.18)
    ax.plot(g["step"], y, color=C_KL, lw=2, label="5-seed mean")
    ax.axhline(98.5, color="#047857", ls="--", lw=1.2, label="Recall constraint (98.5%)")
    ax.set_xlabel("Training steps", fontsize=11)
    ax.set_ylabel("Validation event recall (%)", fontsize=11)
    ax.set_ylim(90, 100.5)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.18)
    fig.tight_layout()
    save_supp(fig, "figS02_training_event_recall")
    for _, r in g.iterrows():
        record_value("figS02", "KL-CMAPPO", "val_recall_pct_mean", round(100 * float(r["mean"]), 4), f"step={int(r['step'])}", src)
    record_index(
        file="supplementary/figS02_training_event_recall.png",
        question="Does validation event recall stay near the constraint during training?",
        source=src,
        split="validation checkpoints",
        methods="KL-CMAPPO",
        seeds=5,
        metric="event recall (%)",
        values="mean±std; threshold 98.5%; axis 90–100%",
    )


def figS03() -> None:
    apply_paper_style()
    g = aggregate_val("val_tx_reduction")
    src = "cmappo_kl metrics.json val_tx_reduction"
    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    ax.fill_between(g["step"], g["mean"] - g["std"].fillna(0), g["mean"] + g["std"].fillna(0), color=C_KL, alpha=0.18)
    ax.plot(g["step"], g["mean"], color=C_KL, lw=2, label="5-seed mean")
    ax.set_xlabel("Training steps", fontsize=11)
    ax.set_ylabel("Validation transmission reduction (%)", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.18)
    fig.tight_layout()
    save_supp(fig, "figS03_training_tx_reduction")
    for _, r in g.iterrows():
        record_value("figS03", "KL-CMAPPO", "val_tx_reduction_mean", round(float(r["mean"]), 4), f"step={int(r['step'])}", src)
    record_index(
        file="supplementary/figS03_training_tx_reduction.png",
        question="Did reaching the MAE constraint simply force always-transmit?",
        source=src,
        split="validation checkpoints",
        methods="KL-CMAPPO",
        seeds=5,
        metric="TX reduction (%)",
        values="mean±std",
    )


def figS04() -> None:
    apply_paper_style()
    frames = [val_rows(s).assign(seed=s) for s in SEEDS]
    allv = pd.concat(frames, ignore_index=True)
    g = allv.groupby("step")["feasible"].sum().reset_index(name="n_feasible")
    src = "metrics.json feasible"
    fig, ax = plt.subplots(figsize=(6.0, 3.7))
    ax.step(g["step"], g["n_feasible"], where="mid", color=C_KL, lw=2)
    ax.scatter(g["step"], g["n_feasible"], color=C_KL, s=30, zorder=3)
    ax.set_xlabel("Training steps", fontsize=11)
    ax.set_ylabel("Number of feasible seeds", fontsize=11)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_ylim(-0.2, 5.3)
    ax.grid(True, alpha=0.18)
    fig.tight_layout()
    save_supp(fig, "figS04_feasible_seeds")
    for _, r in g.iterrows():
        record_value("figS04", "KL-CMAPPO", "n_feasible", int(r["n_feasible"]), f"step={int(r['step'])}", src)
    record_index(
        file="supplementary/figS04_feasible_seeds.png",
        question="How many seeds satisfy constraints at each checkpoint?",
        source=src,
        split="validation checkpoints",
        methods="KL-CMAPPO",
        seeds=5,
        metric="feasible count 0–5",
        values="Boolean feasible field sum",
    )


def figS05() -> None:
    apply_paper_style()
    frames = []
    for s in SEEDS:
        t = train_rows(s)
        frames.append(t[["step", "kl_beta"]].assign(seed=s))
    allv = pd.concat(frames, ignore_index=True)
    g = allv.groupby("step", as_index=False).agg(mean=("kl_beta", "mean"), std=("kl_beta", "std"))
    src = "metrics.json kl_beta (coefficient, NOT D_KL)"
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.fill_between(g["step"], g["mean"] - g["std"].fillna(0), g["mean"] + g["std"].fillna(0), color=C_KL, alpha=0.15)
    ax.plot(g["step"], g["mean"], color=C_KL, lw=2, label="KL coefficient β")
    ax.text(0.02, 0.98, "larger β = stronger anchoring to BC initialization\nsmaller β = greater RL freedom", transform=ax.transAxes, ha="left", va="top", fontsize=8, color="#374151")
    ax.set_xlabel("Training steps", fontsize=11)
    ax.set_ylabel("KL regularization coefficient β", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.18)
    fig.tight_layout()
    save_supp(fig, "figS05_kl_beta_schedule")
    record_value("figS05", "KL-CMAPPO", "beta_start", round(float(g.iloc[0]["mean"]), 4), "", src)
    record_value("figS05", "KL-CMAPPO", "beta_end", round(float(g.iloc[-1]["mean"]), 4), "", src)
    record_index(
        file="supplementary/figS05_kl_beta_schedule.png",
        question="How does the KL regularization coefficient β anneal?",
        source=src,
        split="training logs",
        methods="KL-CMAPPO",
        seeds=5,
        metric="kl_beta coefficient",
        values=f"β≈{g.iloc[0]['mean']:.3f}→{g.iloc[-1]['mean']:.3f}",
    )


def figS06() -> None:
    apply_paper_style()
    frames = [train_rows(s).assign(seed=s) for s in SEEDS]
    allv = pd.concat(frames, ignore_index=True)
    src = "metrics.json lam_e/lam_m/lam_a"
    fig, ax = plt.subplots(figsize=(6.3, 3.9))
    mapping = [("lam_e", "λ_event", "#2563EB"), ("lam_m", "λ_MAE", C_KL), ("lam_a", "λ_AoI", "#059669")]
    for col, lab, color in mapping:
        g = allv.groupby("step")[col].mean()
        ax.plot(g.index, g.values, color=color, lw=1.9)
        ax.text(g.index[-1] + max(g.index) * 0.01, g.values[-1], lab, color=color, fontsize=9, va="center")
        record_value("figS06", lab, "final_mean", round(float(g.values[-1]), 4), "", src)
    ax.set_xlabel("Training steps", fontsize=11)
    ax.set_ylabel("Lagrange multiplier", fontsize=11)
    ax.grid(True, alpha=0.18)
    fig.tight_layout()
    save_supp(fig, "figS06_adaptive_lagrange")
    record_index(
        file="supplementary/figS06_adaptive_lagrange.png",
        question="How do adaptive constraint multipliers evolve?",
        source=src,
        split="training logs",
        methods="KL-CMAPPO (λ_event, λ_MAE, λ_AoI)",
        seeds=5,
        metric="Lagrange multipliers",
        values="lam_e→λ_event, lam_m→λ_MAE, lam_a→λ_AoI",
    )


def figS07() -> None:
    apply_paper_style()
    s = load_val_summary().set_index("method")
    methods = [
        ("fixed_30", "Fixed 30", C_FIXED15),
        ("fixed_60", "Fixed 60", C_FIXED),
        ("delta_plus_heartbeat", "Delta +\nheartbeat", C_DELTA),
        ("semantic_expert", "Semantic\nexpert", C_EXPERT),
        ("campus_senserl_bc", "BC\ninitialization", C_BC),
        ("cmappo_kl", "KL-CMAPPO", C_KL),
    ]
    src = "full_val_summary.csv"
    mae, err, labs, cols = [], [], [], []
    for key, lab, col in methods:
        mae.append(float(s.loc[key, "mae_mean"]))
        nn = int(s.loc[key, "n"])
        err.append(float(s.loc[key, "mae_std"]) if nn > 1 else 0.0)
        labs.append(lab)
        cols.append(col)
        record_value("figS07", lab.replace("\n", " "), "mae", round(float(s.loc[key, "mae_mean"]), 4), f"n={nn}", src)
    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    x = np.arange(len(methods))
    ax.bar(x, mae, yerr=err, color=cols, capsize=3, ecolor="#374151", width=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel("Skipped-observation CO$_2$ MAE (ppm)", fontsize=11)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    save_supp(fig, "figS07_complete_policy_benchmark")
    record_index(
        file="supplementary/figS07_complete_policy_benchmark.png",
        question="Complete validation MAE benchmark across reported policies?",
        source=src,
        split="validation",
        methods="Fixed30/60, Delta, Expert, BC init, KL-CMAPPO",
        seeds="mixed (1 or 5)",
        metric="MAE ppm",
        values="prefer main TABLE for completeness",
    )


def figS08a() -> None:
    apply_paper_style()
    r = load_robust()
    src = "robustness_summary.csv"
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    for method, label, color, marker in [
        ("semantic_expert", "Semantic expert", C_EXPERT, "^"),
        ("campus_senserl_bc", "BC initialization", C_BC, "o"),
        ("cmappo_kl", "KL-CMAPPO", C_KL, "s"),
    ]:
        sub = r[(r["method"] == method) & (r["condition"] == "packet_loss")].sort_values("level")
        x = 100 * sub["level"].astype(float).values
        y = sub["mae_mean"].astype(float).values
        e = sub["mae_std"].astype(float).values
        if method == "semantic_expert":
            ax.plot(x, y, color=color, marker=marker, lw=2, ms=7, label=label)
        else:
            ax.errorbar(x, y, yerr=e, color=color, marker=marker, lw=2, ms=7, capsize=3, label=label)
        for xx, yy in zip(x, y):
            record_value("figS08a", label, f"mae_loss_{int(xx)}", round(float(yy), 4), "", src)
    ax.set_xlabel("Packet loss (%)", fontsize=11)
    ax.set_ylabel("CO$_2$ MAE (ppm)", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.18)
    fig.tight_layout()
    save_supp(fig, "figS08a_packet_loss_mae_all_methods")
    record_index(
        file="supplementary/figS08a_packet_loss_mae_all_methods.png",
        question="Full packet-loss MAE comparison (Expert / BC init / KL)?",
        source=src,
        split="validation stress",
        methods="Semantic expert, BC initialization, KL-CMAPPO",
        seeds="Expert n=1; BC/KL n=5",
        metric="MAE",
        values="MAE only",
    )


def figS08b() -> None:
    apply_paper_style()
    r = load_robust()
    src = "robustness_summary.csv"
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    for method, label, color, marker in [
        ("semantic_expert", "Semantic expert", C_EXPERT, "^"),
        ("campus_senserl_bc", "BC initialization", C_BC, "o"),
        ("cmappo_kl", "KL-CMAPPO", C_KL, "s"),
    ]:
        sub = r[(r["method"] == method) & (r["condition"] == "packet_loss")].sort_values("level")
        x = 100 * sub["level"].astype(float).values
        y = 100 * sub["recall_mean"].astype(float).values
        e = 100 * sub["recall_std"].astype(float).values
        if method == "semantic_expert":
            ax.plot(x, y, color=color, marker=marker, lw=2, ms=7, label=label)
        else:
            ax.errorbar(x, y, yerr=e, color=color, marker=marker, lw=2, ms=7, capsize=3, label=label)
        for xx, yy in zip(x, y):
            record_value("figS08b", label, f"recall_pct_loss_{int(xx)}", round(float(yy), 4), "", src)
    ax.set_xlabel("Packet loss (%)", fontsize=11)
    ax.set_ylabel("Event recall (%)", fontsize=11)
    ax.set_ylim(90, 100.5)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.18)
    fig.tight_layout()
    save_supp(fig, "figS08b_packet_loss_recall_all_methods")
    record_index(
        file="supplementary/figS08b_packet_loss_recall_all_methods.png",
        question="Full packet-loss recall comparison?",
        source=src,
        split="validation stress",
        methods="Semantic expert, BC initialization, KL-CMAPPO",
        seeds="Expert n=1; BC/KL n=5",
        metric="event recall (%)",
        values="recall only (separate from MAE)",
    )


def figS09() -> None:
    apply_paper_style()
    cfg = load_yaml(ROOT / "configs" / "rl_cmappo.yaml")
    length = 96
    start = 3552
    ckpt = CMAPPO / "seed_123" / "best_model.pt"
    act = load_mappo_policy(ckpt, device="cpu", prob_threshold=0.5)
    env = make_final_env(split="val", cfg=cfg, multi_agent=True)
    h = _rollout_actions(env, act, start, length, seed=42)
    A = np.zeros((h["avail"].shape[1], h["avail"].shape[0]), dtype=float)
    for t in range(h["avail"].shape[0]):
        for i in range(h["avail"].shape[1]):
            if not h["avail"][t, i]:
                A[i, t] = 0.0
            elif h["actions"][t, i] == TRANSMIT:
                A[i, t] = 2.0
            else:
                A[i, t] = 1.0
    cmap = ListedColormap(["#D1D5DB", "#FFFFFF", "#1D4ED8"])
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.imshow(A, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=2)
    ax.set_xlabel("Time step (15 min) within validation window", fontsize=11)
    ax.set_ylabel("Sensor index (40)", fontsize=11)
    legend = [
        Patch(facecolor="#1D4ED8", edgecolor="#1D4ED8", label="TX"),
        Patch(facecolor="#FFFFFF", edgecolor="#9CA3AF", label="SKIP"),
        Patch(facecolor="#D1D5DB", edgecolor="#9CA3AF", label="Natural missing"),
    ]
    ax.legend(handles=legend, frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 1.12), ncol=3)
    fig.tight_layout()
    save_supp(fig, "figS09_transmission_heatmap")
    save_json({"start": start, "length": length, "checkpoint": str(ckpt), "split": "val"}, SUPP / "figS09_meta.json")
    record_value("figS09", "KL-CMAPPO", "heatmap_start", start, f"length={length}", "val rollout seed_123")
    record_index(
        file="supplementary/figS09_transmission_heatmap.png",
        question="Do TX/SKIP decisions vary across sensors and time?",
        source="KL-CMAPPO seed_123 val rollout",
        split="validation",
        methods="KL-CMAPPO only",
        seeds="illustrative",
        metric="TX / SKIP / missing map",
        values=f"40×{length}",
    )


def figS10() -> None:
    apply_paper_style()
    if not RECON.exists():
        log("SKIP figS10 — fair_results_summary.csv missing")
        return
    df = pd.read_csv(RECON)
    mae_df = df[df["metric"] == "mae"].set_index("method")
    order = ["locf", "extratrees", "lightgbm", "linear_extrapolation", "stgnn", "historical_mean"]
    labels = {
        "locf": "LOCF",
        "extratrees": "ExtraTrees",
        "lightgbm": "LightGBM",
        "linear_extrapolation": "Linear\nextrapolation",
        "stgnn": "ST-GNN",
        "historical_mean": "Historical\nmean",
    }
    keep = [m for m in order if m in mae_df.index]
    mae = mae_df.loc[keep]
    y = mae["mean"].astype(float).values
    e = mae["std"].astype(float).values
    labs = [labels[m] for m in mae.index]
    colors = ["#374151", "#6B7280", "#9CA3AF", "#60A5FA", "#93C5FD", "#D1D5DB"][: len(labs)]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    x = np.arange(len(labs))
    ax.bar(x, y, yerr=e, color=colors, capsize=3, ecolor="#374151")
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel("Masked reconstruction MAE (ppm)", fontsize=11)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    save_supp(fig, "figS10_reconstruction_benchmark")
    for m, yy, ee in zip(mae.index, y, e):
        record_value("figS10", labels[m].replace("\n", " "), "mae", round(float(yy), 4), f"std={ee:.4f}", str(RECON))
    record_index(
        file="supplementary/figS10_reconstruction_benchmark.png",
        question="Fair causal reconstruction benchmark (not central contribution)?",
        source=str(RECON.relative_to(ROOT)),
        split="fair reconstruction benchmark",
        methods=", ".join(labs).replace("\n", " "),
        seeds=5,
        metric="masked MAE",
        values="; ".join(f"{labels[m].replace(chr(10),' ')}={yy:.2f}" for m, yy in zip(mae.index, y)),
    )


def write_docs() -> None:
    ensure_dir(OUT)
    ensure_dir(REPORTS)

    with (OUT / "figure_values.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["figure", "method", "metric", "value", "extra", "source"])
        w.writeheader()
        for row in VALUES:
            w.writerow(row)

    audit_lines = [
        "# Figure source audit — manuscript_figures_v2",
        "",
        "No training. All quantitative points read from frozen CSV/JSON under `outputs/rl_final/`.",
        "Illustrative rollouts use frozen checkpoint `cmappo_kl/seed_123/best_model.pt` only.",
        "",
    ]
    for r in INDEX:
        audit_lines += [
            f"## `{r['file']}`",
            f"- **Scientific question:** {r['question']}",
            f"- **Source:** `{r['source']}`",
            f"- **Split:** {r['split']}",
            f"- **Methods:** {r['methods']}",
            f"- **Seeds:** {r['seeds']}",
            f"- **Metric:** {r['metric']}",
            f"- **Exact plotted numbers:** {r['values']}",
            "",
        ]
    (OUT / "figure_source_audit.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    # Caption stubs
    captions = [
        "# Suggested captions (manuscript_figures_v2)",
        "",
        "**Fig. 1** Spatial layout of the smart-campus deployment (metadata). Light markers: remaining campus sensors; dark markers: 40 RL development sensors; diamonds: 40 held-out transfer sensors. Counts are for caption use only.",
        "",
        "**Fig. 2** Illustrative validation window: true CO₂, KL-CMAPPO TX markers, causal reconstruction during SKIP, and 1000 ppm threshold. Fixed-15 TX count and reduction verified from this frozen rollout (see `fig02_meta.json`).",
        "",
        "**Fig. 3** KL-CMAPPO training workflow. BC is an initialization stage; the final policy is shield-free KL-CMAPPO.",
        "",
        "**Fig. 4** Validation skipped-observation CO₂ MAE for Fixed 60, Delta+heartbeat, and KL-CMAPPO. TX↓ annotations from the same validation summary. KL-CMAPPO: mean ± std over five seeds; baselines deterministic (n=1).",
        "",
        "**Fig. 5** Frozen TEST important-event recall (development cohort). Methods: Fixed 60, Delta+heartbeat, KL-CMAPPO (frozen seed-123 checkpoint). Y-axis 0–100%.",
        "",
        "**Fig. 6** Validation reconstruction quality during KL-CMAPPO training (mean ± std over five seeds). Dashed line: predefined MAE = 9 ppm feasible region. Not a raw-reward plot.",
        "",
        "**Fig. 7** Mechanistic ablation, seed 42 only. Bars: CO₂ MAE; annotations: TX reduction. BC framed as initialization. Do not interpret as multi-seed significance.",
        "",
        "**Fig. 8** KL-CMAPPO generalization: development-cohort frozen TEST MAE vs held-out TEST MAE (five seeds mean ± std on held-out). BC omitted.",
        "",
        "**Fig. 9** Packet-loss robustness of KL-CMAPPO alone (validation stress; five-seed mean ± std).",
        "",
        "**Fig. 10** Cumulative transmissions over a fixed validation interval for Fixed 15, Fixed 60, and KL-CMAPPO (illustrative frozen rollout).",
        "",
    ]
    (OUT / "suggested_captions.md").write_text("\n".join(captions), encoding="utf-8")

    review = [
        "# Manuscript figure review v2",
        "",
        "Generated without training from frozen artifacts. Main pack: `paper_outputs/manuscript_figures_v2/main/`.",
        "Prior simplified pack archived at `paper_outputs/legacy_complex_figures/results_figures_prior/`.",
        "",
    ]
    checks = [
        (
            "fig01",
            "Is this a real campus deployment with held-out sensors?",
            "yes",
            "no (spatial only)",
            "no — RL development not labeled BC",
            "no",
            "metadata",
            "N/A",
            "yes — dark vs diamond cohorts",
            "no",
        ),
        (
            "fig02",
            "What does adaptive TX look like?",
            "yes",
            "Fixed15 only as count annotation",
            "no",
            "no (one axis CO2)",
            "validation illustrative",
            "illustrative only",
            "yes — red TX markers",
            "no",
        ),
        (
            "fig03",
            "How is KL-CMAPPO trained?",
            "yes",
            "BC as init stage only",
            "no — init not competitor",
            "schematic",
            "N/A",
            "N/A",
            "yes — large red box",
            "no",
        ),
        (
            "fig04",
            "At similar savings, who preserves CO2 best?",
            "yes",
            "Fixed60/Delta/KL only",
            "no BC",
            "MAE only (+ TX↓ tags)",
            "validation",
            "KL yes; baselines n=1",
            "yes",
            "no",
        ),
        (
            "fig05",
            "Does reduction miss important events?",
            "yes — Fixed60 collapses",
            "same three methods",
            "no BC",
            "recall only",
            "frozen TEST",
            "n=1 frozen report (honest)",
            "yes",
            "no",
        ),
        (
            "fig06",
            "Did training enter MAE≤9 feasible region?",
            "yes",
            "KL only",
            "N/A",
            "MAE only",
            "val training dynamics",
            "yes 5-seed",
            "yes",
            "no",
        ),
        (
            "fig07",
            "What do KL + constraints contribute?",
            "yes",
            "ablation variants",
            "BC as initialization only",
            "MAE only (+ TX↓)",
            "val seed 42",
            "no multi-seed — caption required",
            "yes — full KL red",
            "no",
        ),
        (
            "fig08",
            "Does policy transfer to unseen sensors?",
            "yes",
            "KL only (dev vs heldout)",
            "no BC",
            "MAE only",
            "TEST vs TEST",
            "dev n=1 frozen; heldout n=5",
            "yes",
            "no",
        ),
        (
            "fig09",
            "Graceful degradation under packet loss?",
            "yes",
            "KL only",
            "N/A",
            "MAE only",
            "val stress",
            "yes 5-seed",
            "yes",
            "no",
        ),
        (
            "fig10",
            "What do savings look like over time?",
            "yes",
            "Fixed15/60/KL",
            "no BC",
            "cumulative TX only",
            "validation",
            "illustrative",
            "yes — direct labels",
            "no",
        ),
    ]
    for name, q, q2, q3, q4, q5, q6, q7, q8, q9 in checks:
        review += [
            f"## {name}",
            f"1. Single question? **{q}**",
            f"2. Understandable in ~10s? **{q2}**",
            f"3. Unnecessary methods? **{q3}**",
            f"4. BC as second proposed method? **{q4}**",
            f"5. More than one metric? **{q5}**",
            f"6. Split obvious? **{q6}**",
            f"7. Error bars legitimate? **{q7}**",
            f"8. KL-CMAPPO conclusion visible? **{q8}**",
            f"9. Could be simpler? **{q9}**",
            "",
        ]
    (REPORTS / "manuscript_figure_review_v2.md").write_text("\n".join(review) + "\n", encoding="utf-8")
    (OUT / "generation_audit.log").write_text("\n".join(AUDIT) + "\n", encoding="utf-8")
    log(f"WROTE {OUT / 'figure_source_audit.md'}")
    log(f"WROTE {OUT / 'figure_values.csv'}")
    log(f"WROTE {REPORTS / 'manuscript_figure_review_v2.md'}")


def main() -> None:
    ensure_dir(MAIN)
    ensure_dir(SUPP)
    apply_paper_style()
    log("=== manuscript_figures_v2 (no training) ===")
    fig01()
    fig02()
    fig03()
    fig04()
    fig05()
    fig06()
    fig07()
    fig08()
    fig09()
    fig10()
    figS01()
    figS02()
    figS03()
    figS04()
    figS05()
    figS06()
    figS07()
    figS08a()
    figS08b()
    figS09()
    figS10()
    write_docs()
    log("DONE")


if __name__ == "__main__":
    main()
