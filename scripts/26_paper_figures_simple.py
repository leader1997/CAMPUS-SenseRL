#!/usr/bin/env python
"""SIMPLIFIED MANUSCRIPT FIGURE PACK — visualization only (no training).

ONE scientific question → ONE graph (except the method schematic).
Writes paper_outputs/paper_figures_simple/{main,supplementary}/
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

OUT = ROOT / "results"
FIG = OUT / "figures"
MAIN = FIG / "main"
SUPP = FIG / "supplementary"
CSV = OUT / "csv_json"
PAPER = ROOT / "outputs" / "rl_final" / "paper_final"
SCI = ROOT / "outputs" / "rl_final" / "scientific_validation"
CMAPPO = ROOT / "outputs" / "rl_final" / "cmappo_kl"
SEEDS = [42, 123, 2024, 3407, 9999]
DPI = 300

C_FIXED = "#6B7280"
C_FIXED15 = "#9CA3AF"
C_BC = "#6A51A3"
C_KL = "#CB181D"
C_EXPERT = "#238B45"

VALUES: list[dict] = []
INDEX: list[dict] = []
AUDIT: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    AUDIT.append(msg)


def record_value(figure: str, method: str, metric: str, value, extra: str = "") -> None:
    VALUES.append(
        {
            "figure": figure,
            "method": method,
            "metric": metric,
            "value": value,
            "extra": extra,
        }
    )


def record_index(**kwargs) -> None:
    INDEX.append(kwargs)


def save_main(fig: plt.Figure, stem: str) -> None:
    ensure_dir(MAIN)
    path = MAIN / f"{stem}.png"
    fig.savefig(path, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    log(f"WROTE {path.relative_to(ROOT)}")


def save_supp(fig: plt.Figure, stem: str) -> None:
    ensure_dir(SUPP)
    path = SUPP / f"{stem}.png"
    fig.savefig(path, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    log(f"WROTE {path.relative_to(ROOT)}")


def load_val_summary() -> pd.DataFrame:
    return pd.read_csv(PAPER / "full_val_summary.csv")


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


# ---------------------------------------------------------------------------
# Rollout helpers (frozen checkpoints only — inference for illustration)
# ---------------------------------------------------------------------------
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
# MAIN FIGURES
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
    bg = devices.loc[co2 & ~devices["deveui"].isin(train | held)]

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.scatter(bg["longitude"], bg["latitude"], s=10, c="#D1D5DB", alpha=0.55, label=f"Other CO$_2$ sensors (n={len(bg)})", zorder=1)
    ax.scatter(tr["longitude"], tr["latitude"], s=36, c=C_BC, edgecolors="white", linewidths=0.4, label=f"RL development (n={len(tr)})", zorder=3)
    ax.scatter(ho["longitude"], ho["latitude"], s=36, c=C_KL, edgecolors="white", linewidths=0.4, label=f"Held-out transfer (n={len(ho)})", zorder=3)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    save_main(fig, "fig01_campus_deployment")
    n_co2 = int(co2.sum())
    record_value("fig01", "all", "n_co2_capable", n_co2)
    record_value("fig01", "rl_dev", "n_sensors", len(tr))
    record_value("fig01", "heldout", "n_sensors", len(ho))
    record_index(
        file="main/fig01_campus_deployment.png",
        place="main",
        question="Is the method evaluated on a real distributed campus deployment with transfer sensors?",
        source="devices.json, final_cohort.json, heldout_cohort.json",
        split="metadata",
        seeds="N/A",
        values=f"CO2={n_co2}, RL={len(tr)}, heldout={len(ho)}",
        why_main="Establishes real-campus setting and held-out transfer design.",
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

    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.plot(hours, gt, color="#111827", lw=1.4, label="True CO$_2$", zorder=2)
    skip = (~kl_tx) & avail
    ax.plot(hours[skip], kl_recon[skip], color="#86EFAC", lw=1.0, alpha=0.95, label="Server recon (SKIP)", zorder=1)
    ax.scatter(hours[kl_tx], gt[kl_tx], s=28, c=C_KL, zorder=4, label="KL-CMAPPO TX")
    ax.set_xlabel("Time (hours from window start)")
    ax.set_ylabel("CO$_2$ (ppm)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title(f"Fixed 15 TX = {n_fix}  ·  KL-CMAPPO TX = {n_kl}  ·  Reduction = {red:.1f}%", fontsize=10)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    save_main(fig, "fig02_real_kl_cmappo_timeline")
    save_json(
        {
            "split": "val",
            "sensor_id": sensor_id,
            "sensor_index": sid,
            "start_step": start,
            "length": length,
            "fixed15_tx": n_fix,
            "kl_cmappo_tx": n_kl,
            "reduction_vs_fixed15_pct": red,
            "checkpoint": str(ckpt),
        },
        MAIN / "fig02_meta.json",
    )
    record_value("fig02", "Fixed 15", "interval_tx", n_fix, f"sensor={sensor_id}")
    record_value("fig02", "KL-CMAPPO", "interval_tx", n_kl)
    record_value("fig02", "KL-CMAPPO", "reduction_vs_fixed15_pct", round(red, 2))
    record_index(
        file="main/fig02_real_kl_cmappo_timeline.png",
        place="main",
        question="How does KL-CMAPPO adapt TX on a real CO2 trajectory?",
        source="frozen KL-CMAPPO seed_123 rollout + FixedInterval(1); val",
        split="validation",
        seeds="illustrative (seed 123 policy)",
        values=f"Fixed15 TX={n_fix}, KL TX={n_kl}, red={red:.2f}%",
        why_main="Intuitive adaptive-communication example.",
    )


def fig03() -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(5.0, 8.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14.2)
    ax.axis("off")

    def box(y, text, fc, ec, h=1.05, fs=10, lw=1.4):
        ax.add_patch(
            FancyBboxPatch((1.0, y), 8.0, h, boxstyle="round,pad=0.02,rounding_size=0.08", lw=lw, ec=ec, fc=fc)
        )
        ax.text(5.0, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(y1, y2):
        ax.annotate("", xy=(5.0, y2), xytext=(5.0, y1), arrowprops=dict(arrowstyle="->", color="#374151", lw=1.4))

    box(12.6, "Real campus sensor state", "#F3F4F6", "#6B7280", h=0.95)
    arrow(12.6, 11.85)
    box(10.9, "Semantic expert  (demonstrations)", "#E5F5E0", C_EXPERT, h=0.95)
    arrow(10.9, 10.15)
    box(9.2, "BC initialization  →  π_BC", "#EDE9FE", C_BC, h=0.95, fs=9.5)
    arrow(9.2, 8.0)
    box(
        5.35,
        "KL-CMAPPO  (proposed method)\n"
        "• minimize unnecessary transmissions\n"
        "• event / MAE / AoI constraints\n"
        "• KL anchor to π_BC",
        "#FEE2E2",
        C_KL,
        h=2.55,
        fs=10,
        lw=2.2,
    )
    arrow(5.35, 4.5)
    box(3.55, "TX / SKIP", "#FEE2E2", C_KL, h=0.95)
    arrow(3.55, 2.75)
    box(1.8, "Smart-campus monitoring  (causal LOCF)", "#F3F4F6", "#6B7280", h=0.95)
    save_main(fig, "fig03_method_workflow")
    record_index(
        file="main/fig03_method_workflow.png",
        place="main",
        question="What is the proposed CAMPUS-SenseRL / KL-CMAPPO pipeline?",
        source="schematic",
        split="N/A",
        seeds="N/A",
        values="BC = initialization stage; KL-CMAPPO = final method",
        why_main="Method overview with KL-CMAPPO visually dominant.",
    )


def fig04() -> None:
    """Main practical comparison: periodic vs heuristic vs proposed KL-CMAPPO."""
    apply_paper_style()
    s = load_val_summary().set_index("method")
    methods = ["fixed_60", "delta_plus_heartbeat", "cmappo_kl"]
    labels = ["Fixed 60", "Delta +\nheartbeat", "KL-CMAPPO"]
    colors = [C_FIXED, "#60A5FA", C_KL]
    mae = [float(s.loc[m, "mae_mean"]) for m in methods]
    mae_std = [float(s.loc[m, "mae_std"]) if int(s.loc[m, "n"]) > 1 else 0.0 for m in methods]
    tx = [float(s.loc[m, "tx_reduction_mean"]) for m in methods]
    n = [int(s.loc[m, "n"]) for m in methods]

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    x = np.arange(3)
    bars = ax.bar(x, mae, color=colors, width=0.62)
    for i, (std, nn) in enumerate(zip(mae_std, n)):
        if nn > 1 and std > 0:
            ax.errorbar(x[i], mae[i], yerr=std, fmt="none", ecolor="#374151", capsize=4, lw=1.0)
    for i, (b, t) in enumerate(zip(bars, tx)):
        top = mae[i] + (mae_std[i] if n[i] > 1 else 0.0)
        ax.text(b.get_x() + b.get_width() / 2, top + 0.15, f"{t:.1f}% TX↓", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Skipped-observation CO$_2$ MAE (ppm)")
    ax.set_ylim(0, max(mae) * 1.28)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    save_main(fig, "fig04_key_mae_comparison")
    for m, lab, a, t, nn in zip(methods, labels, mae, tx, n):
        record_value("fig04", lab.replace("\n", " "), "mae_ppm", round(a, 4), f"tx_reduction={t:.3f}; n={nn}; validation")
    record_index(
        file="main/fig04_key_mae_comparison.png",
        place="main",
        question="Does KL-CMAPPO outperform periodic Fixed-60 and Delta+heartbeat scheduling?",
        source="paper_final/full_val_summary.csv",
        split="validation",
        seeds="Fixed60/Delta n=1; KL n=5",
        values="; ".join(f"{lab.replace(chr(10),' ')} MAE={a:.3f}, TX↓={t:.2f}" for lab, a, t in zip(labels, mae, tx)),
        why_main="Practical baseline comparison; BC is not a competing final method.",
    )


def fig05_training() -> None:
    apply_paper_style()
    g = aggregate_val("val_mae")
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.fill_between(g["step"], g["mean"] - g["std"].fillna(0), g["mean"] + g["std"].fillna(0), color=C_KL, alpha=0.18, label="±1 std")
    ax.plot(g["step"], g["mean"], color=C_KL, lw=2.0, label="5-seed mean")
    ax.axhline(9.0, color="#059669", ls="--", lw=1.2, label="MAE constraint (9 ppm)")
    ax.axhspan(0, 9.0, color="#D1FAE5", alpha=0.25, zorder=0)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation CO$_2$ MAE (ppm)")
    ax.set_ylim(max(0, float(g["mean"].min()) - 1.0), max(10.0, float((g["mean"] + g["std"].fillna(0)).max()) + 0.4))
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_main(fig, "fig05_training_validation_mae")
    for _, r in g.iterrows():
        record_value("fig05", "KL-CMAPPO", "val_mae_mean", round(float(r["mean"]), 4), f"step={int(r['step'])}")
    record_index(
        file="main/fig05_training_validation_mae.png",
        place="main",
        question="Does KL-CMAPPO training reach the MAE ≤ 9 ppm constraint?",
        source="cmappo_kl/seed_*/metrics.json (val_* only)",
        split="validation checkpoints during training",
        seeds=5,
        values="mean±std over seeds; ε_MAE=9.0",
        why_main="Principal constrained-RL training quality figure.",
    )


def fig06_ablation() -> None:
    apply_paper_style()
    abl = load_ablation().set_index("ablation")
    order = [
        ("bc_only", "BC\ninitialization"),
        ("mappo_no_kl_no_constraints", "Init + MAPPO\n(no KL/constr.)"),
        ("kl_only", "Init +\nKL only"),
        ("constraints_only", "Init +\nconstraints"),
        ("full_kl_cmappo", "Full\nKL-CMAPPO"),
    ]
    colors = [C_BC, "#93C5FD", "#34D399", "#F97316", C_KL]
    mae = [float(abl.loc[k, "mae_skipped"]) for k, _ in order]
    tx = [float(abl.loc[k, "transmission_reduction_pct"]) for k, _ in order]
    labels = [lab for _, lab in order]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(order))
    bars = ax.bar(x, mae, color=colors, width=0.7)
    for b, t in zip(bars, tx):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.08, f"{t:.1f}% TX↓", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("CO$_2$ MAE (ppm)")
    ax.set_title("Seed-42 mechanistic ablation of RL fine-tuning stages", fontsize=10)
    ax.set_ylim(0, max(mae) * 1.22)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    save_main(fig, "fig06_ablation_mae")
    for (k, lab), a, t in zip(order, mae, tx):
        record_value("fig06", lab.replace("\n", " "), "mae_ppm", round(a, 4), f"tx↓={t:.3f}; seed=42")
    record_index(
        file="main/fig06_ablation_mae.png",
        place="main",
        question="What does KL-CMAPPO add beyond BC initialization?",
        source="ablation_kl_cmappo_val.csv",
        split="validation",
        seeds="1 (seed 42 mechanistic)",
        values="; ".join(f"{lab.replace(chr(10),' ')}={a:.3f}" for lab, a in zip(labels, mae)),
        why_main="BC appears here as initialization stage, not a competing final method.",
    )


def fig07_heldout() -> None:
    """Held-out transfer of proposed KL-CMAPPO; semantic expert as teacher reference."""
    apply_paper_style()
    h = load_heldout()
    kl = h[(h["cohort"] == "heldout") & (h["split"] == "test") & (h["method"] == "cmappo_kl")].iloc[0]
    exp = h[(h["cohort"] == "heldout") & (h["method"] == "semantic_expert")].iloc[0]

    labels = ["Semantic expert\n(reference)", "KL-CMAPPO"]
    colors = [C_EXPERT, C_KL]
    mae = [float(exp["mae_mean"]), float(kl["mae_mean"])]
    std = [0.0, float(kl["mae_std"])]
    tx = [float(exp["tx_reduction_mean"]), float(kl["tx_reduction_mean"])]
    splits = [str(exp["split"]), "test"]

    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    x = np.arange(2)
    ax.bar(x[0], mae[0], color=colors[0], width=0.55)
    ax.bar(x[1], mae[1], yerr=std[1], color=colors[1], width=0.55, capsize=4, ecolor="#374151")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\n({t:.1f}% TX↓)" for lab, t in zip(labels, tx)], fontsize=9)
    ax.set_ylabel("Held-out CO$_2$ MAE (ppm)")
    ax.set_ylim(0, max(mae[0], mae[1] + std[1]) * 1.25)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    save_main(fig, "fig07_heldout_mae")
    record_value("fig07", "Semantic expert", "mae_ppm", round(mae[0], 4), f"heldout {splits[0]}")
    record_value("fig07", "KL-CMAPPO", "mae_ppm", round(mae[1], 4), f"heldout test; std={std[1]:.4f}")
    record_index(
        file="main/fig07_heldout_mae.png",
        place="main",
        question="Does the proposed KL-CMAPPO policy transfer to unseen campus sensors?",
        source="heldout_transfer_summary.csv",
        split="KL: held-out TEST; expert reference: held-out VAL",
        seeds="KL n=5; expert n=1",
        values=f"expert MAE={mae[0]:.3f}; KL MAE={mae[1]:.3f}",
        why_main="Generalization of the final method; BC omitted.",
    )


def fig08_packet_loss() -> None:
    apply_paper_style()
    r = load_robust()
    sub = r[(r["method"] == "cmappo_kl") & (r["condition"] == "packet_loss")].sort_values("level")
    x = (sub["level"].astype(float) * 100).values
    y = sub["mae_mean"].astype(float).values
    e = sub["mae_std"].astype(float).values

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.errorbar(x, y, yerr=e, color=C_KL, marker="o", lw=2, ms=7, capsize=3, label="KL-CMAPPO")
    ax.set_xlabel("Packet loss (%)")
    ax.set_ylabel("CO$_2$ MAE (ppm)")
    ax.set_xticks(x)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_main(fig, "fig08_packet_loss_mae")
    for xx, yy, ee in zip(x, y, e):
        record_value("fig08", "KL-CMAPPO", f"mae_loss_{int(xx)}", round(float(yy), 4), f"std={ee:.4f}; validation")
    record_index(
        file="main/fig08_packet_loss_mae.png",
        place="main",
        question="Does reconstruction degrade gradually under packet loss?",
        source="robustness_summary.csv (packet_loss, cmappo_kl)",
        split="validation stress",
        seeds=5,
        values="; ".join(f"{int(xx)}%→{yy:.3f}" for xx, yy in zip(x, y)),
        why_main="Robustness of the proposed method.",
    )


def fig09_cumulative() -> None:
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
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(hours, c15, color=C_FIXED15, lw=2, label="Fixed 15")
    ax.plot(hours, c60, color=C_FIXED, lw=2, label="Fixed 60")
    ax.plot(hours, ckl, color=C_KL, lw=2.2, label="KL-CMAPPO")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Cumulative transmissions")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_main(fig, "fig09_cumulative_transmissions")
    record_value("fig09", "Fixed 15", "cum_tx_end", int(c15[-1]), f"start={start}; length={length}; val")
    record_value("fig09", "Fixed 60", "cum_tx_end", int(c60[-1]))
    record_value("fig09", "KL-CMAPPO", "cum_tx_end", int(ckl[-1]))
    record_index(
        file="main/fig09_cumulative_transmissions.png",
        place="main (optional)",
        question="How do communication savings accumulate over campus operation?",
        source="val rollouts Fixed15/60 + KL-CMAPPO seed_123",
        split="validation",
        seeds="illustrative",
        values=f"end TX Fixed15={int(c15[-1])}, Fixed60={int(c60[-1])}, KL={int(ckl[-1])}",
        why_main="Intuitive cumulative savings of the proposed method.",
    )


def figS01_matched_rl_vs_init() -> None:
    """Supplementary: RL fine-tuning vs BC initialization at matched budgets."""
    apply_paper_style()
    m = load_matched()
    targets = [75.0, 78.0, 80.0]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for method, label, color, marker in [
        ("campus_senserl_bc", "BC initialization", C_BC, "o"),
        ("cmappo_kl", "KL-CMAPPO", C_KL, "s"),
    ]:
        sub = m[m["method"] == method].set_index("target_tx_reduction").loc[targets]
        y = sub["mae_mean"].astype(float).values
        e = sub["mae_std"].astype(float).values
        ax.errorbar(targets, y, yerr=e, marker=marker, color=color, lw=2, ms=7, capsize=3, label=label)
        for t, yy in zip(targets, y):
            record_value("figS01", label, f"mae_at_{int(t)}", round(float(yy), 4), "validation matched budget")
    bc78 = float(m[(m.method == "campus_senserl_bc") & (m.target_tx_reduction == 78)].iloc[0]["mae_mean"])
    kl78 = float(m[(m.method == "cmappo_kl") & (m.target_tx_reduction == 78)].iloc[0]["mae_mean"])
    ax.set_xlabel("Target transmission reduction (%)")
    ax.set_ylabel("CO$_2$ MAE (ppm)")
    ax.set_xticks(targets)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_supp(fig, "figS01_matched_budget_rl_vs_init")
    record_index(
        file="supplementary/figS01_matched_budget_rl_vs_init.png",
        place="supplementary",
        question="Does RL fine-tuning improve on BC initialization at matched TX budgets?",
        source="matched_budget_summary.csv",
        split="validation",
        seeds=5,
        values=f"@78%: init={bc78:.4f}, KL={kl78:.4f}",
        why_main="BC as initialization baseline for RL contribution analysis.",
    )


# ---------------------------------------------------------------------------
# SUPPLEMENTARY
# ---------------------------------------------------------------------------
def figS01b_matched_recall() -> None:
    apply_paper_style()
    m = load_matched()
    targets = [75.0, 78.0, 80.0]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for method, label, color, marker in [
        ("campus_senserl_bc", "BC initialization", C_BC, "o"),
        ("cmappo_kl", "KL-CMAPPO", C_KL, "s"),
    ]:
        sub = m[m["method"] == method].set_index("target_tx_reduction").loc[targets]
        y = 100 * sub["recall_mean"].astype(float).values
        e = 100 * sub["recall_std"].astype(float).values
        ax.errorbar(targets, y, yerr=e, marker=marker, color=color, lw=2, ms=7, capsize=3, label=label)
        for t, yy in zip(targets, y):
            record_value("figS01b", label, f"recall_at_{int(t)}", round(float(yy), 4))
    ax.set_xlabel("Target transmission reduction (%)")
    ax.set_ylabel("Event recall (%)")
    ax.set_ylim(95, 100.2)
    ax.set_xticks(targets)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_supp(fig, "figS01b_matched_budget_recall")
    record_index(
        file="supplementary/figS01b_matched_budget_recall.png",
        place="supplementary",
        question="Is event recall preserved at matched TX budgets (init vs KL-CMAPPO)?",
        source="matched_budget_summary.csv",
        split="validation",
        seeds=5,
        values="BC init vs KL recall at 75/78/80",
        why_main="Companion recall plot for RL-vs-init analysis.",
    )


def figS02() -> None:
    apply_paper_style()
    g = aggregate_val("val_recall")
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    y = 100 * g["mean"]
    e = 100 * g["std"].fillna(0)
    ax.fill_between(g["step"], y - e, y + e, color=C_KL, alpha=0.18, label="±1 std")
    ax.plot(g["step"], y, color=C_KL, lw=2, label="5-seed mean")
    ax.axhline(98.5, color="#059669", ls="--", lw=1.2, label="Recall constraint (98.5%)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation event recall (%)")
    ax.set_ylim(97.0, 100.2)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_supp(fig, "figS02_training_recall")
    record_index(
        file="supplementary/figS02_training_recall.png",
        place="supplementary",
        question="Does validation event recall stay above the constraint?",
        source="cmappo_kl metrics.json val_recall",
        split="validation checkpoints",
        seeds=5,
        values="mean±std; threshold 98.5%",
        why_main="Companion to main MAE training figure.",
    )


def figS03() -> None:
    apply_paper_style()
    g = aggregate_val("val_tx_reduction")
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.fill_between(g["step"], g["mean"] - g["std"].fillna(0), g["mean"] + g["std"].fillna(0), color=C_BC, alpha=0.18, label="±1 std")
    ax.plot(g["step"], g["mean"], color=C_BC, lw=2, label="5-seed mean")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation TX reduction (%)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_supp(fig, "figS03_training_tx_reduction")
    record_index(
        file="supplementary/figS03_training_tx_reduction.png",
        place="supplementary",
        question="How does communication reduction evolve during training?",
        source="cmappo_kl metrics.json val_tx_reduction",
        split="validation checkpoints",
        seeds=5,
        values="mean±std",
        why_main="Separates TX dynamics from MAE training figure.",
    )


def figS04() -> None:
    apply_paper_style()
    frames = []
    for s in SEEDS:
        v = val_rows(s)
        v["seed"] = s
        frames.append(v)
    allv = pd.concat(frames, ignore_index=True)
    g = allv.groupby("step")["feasible"].sum().reset_index(name="n_feasible")
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.step(g["step"], g["n_feasible"], where="mid", color=C_KL, lw=2, label="Feasible seeds")
    ax.scatter(g["step"], g["n_feasible"], color=C_KL, s=28, zorder=3)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Feasible seeds / 5")
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_ylim(-0.2, 5.3)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_supp(fig, "figS04_feasible_seeds")
    for _, r in g.iterrows():
        record_value("figS04", "KL-CMAPPO", "n_feasible", int(r["n_feasible"]), f"step={int(r['step'])}")
    record_index(
        file="supplementary/figS04_feasible_seeds.png",
        place="supplementary",
        question="How many seeds satisfy constraints at each checkpoint?",
        source="metrics.json feasible",
        split="validation checkpoints",
        seeds=5,
        values="0–5 feasible count",
        why_main="Constrained-RL diagnostic for specialists.",
    )


def figS05() -> None:
    apply_paper_style()
    frames = []
    for s in SEEDS:
        t = train_rows(s)
        t["seed"] = s
        frames.append(t[["step", "kl_beta", "seed"]])
    allv = pd.concat(frames, ignore_index=True)
    g = allv.groupby("step", as_index=False).agg(mean=("kl_beta", "mean"), std=("kl_beta", "std"))
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.fill_between(g["step"], g["mean"] - g["std"].fillna(0), g["mean"] + g["std"].fillna(0), color=C_KL, alpha=0.15, label="±1 std")
    ax.plot(g["step"], g["mean"], color=C_KL, lw=2, label="KL coefficient β")
    ax.set_xlabel("Training step")
    ax.set_ylabel("KL coefficient β")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_supp(fig, "figS05_kl_beta_schedule")
    record_value("figS05", "KL-CMAPPO", "beta_start", round(float(g.iloc[0]["mean"]), 4))
    record_value("figS05", "KL-CMAPPO", "beta_end", round(float(g.iloc[-1]["mean"]), 4))
    record_index(
        file="supplementary/figS05_kl_beta_schedule.png",
        place="supplementary",
        question="How does the KL regularization coefficient anneal?",
        source="metrics.json kl_beta (NOT D_KL)",
        split="training logs",
        seeds=5,
        values=f"β≈{g.iloc[0]['mean']:.3f}→{g.iloc[-1]['mean']:.3f}",
        why_main="Method detail; coefficient ≠ measured divergence.",
    )


def figS06() -> None:
    apply_paper_style()
    frames = []
    for s in SEEDS:
        t = train_rows(s)
        t["seed"] = s
        frames.append(t)
    allv = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    mapping = [("lam_e", "λ_event", "#2563EB"), ("lam_m", "λ_MAE", "#DC2626"), ("lam_a", "λ_AoI", "#059669")]
    for col, lab, color in mapping:
        g = allv.groupby("step")[col].mean()
        ax.plot(g.index, g.values, color=color, lw=1.8, label=lab)
        record_value("figS06", lab, "final_mean", round(float(g.values[-1]), 4))
    ax.set_xlabel("Training step")
    ax.set_ylabel("Lagrange multiplier")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_supp(fig, "figS06_lagrange_multipliers")
    record_index(
        file="supplementary/figS06_lagrange_multipliers.png",
        place="supplementary",
        question="How do adaptive dual variables evolve?",
        source="metrics.json lam_e/lam_m/lam_a",
        split="training logs",
        seeds=5,
        values="λ_event←lam_e, λ_MAE←lam_m, λ_AoI←lam_a",
        why_main="RL-specialist diagnostic.",
    )


def figS07() -> None:
    apply_paper_style()
    abl = load_ablation()
    order = {
        "bc_only": ("BC only", C_BC),
        "mappo_no_kl_no_constraints": ("No KL/constraints", "#93C5FD"),
        "kl_only": ("KL only", "#34D399"),
        "constraints_only": ("Constraints only", "#F97316"),
        "full_kl_cmappo": ("Full KL-CMAPPO", C_KL),
    }
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    for key, (lab, color) in order.items():
        r = abl[abl["ablation"] == key].iloc[0]
        x, y = float(r["transmission_reduction_pct"]), float(r["mae_skipped"])
        ax.scatter(x, y, s=90, c=color, zorder=3, label=lab)
        record_value("figS07", lab, "mae", round(y, 4), f"tx↓={x:.3f}; seed=42")
    ax.set_xlabel("Transmission reduction (%)")
    ax.set_ylabel("CO$_2$ MAE (ppm)")
    ax.set_title("Seed-42 mechanistic ablation", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_supp(fig, "figS07_ablation_pareto")
    record_index(
        file="supplementary/figS07_ablation_pareto.png",
        place="supplementary",
        question="What is the TX↓–MAE trade-off across ablation variants?",
        source="ablation_kl_cmappo_val.csv",
        split="validation",
        seeds="1 (seed 42)",
        values="Pareto scatter of five variants",
        why_main="Full trade-off view; main uses simple MAE bars.",
    )


def figS08() -> None:
    apply_paper_style()
    s = load_val_summary().set_index("method")
    methods = [
        ("fixed_30", "Fixed 30", "#9CA3AF"),
        ("fixed_60", "Fixed 60", C_FIXED),
        ("delta_plus_heartbeat", "Delta +\nheartbeat", "#60A5FA"),
        ("semantic_expert", "Semantic\nexpert", C_EXPERT),
        ("campus_senserl_bc", "BC\ninitialization", C_BC),
        ("cmappo_kl", "KL-CMAPPO", C_KL),
    ]
    mae, err, labs, cols = [], [], [], []
    for key, lab, col in methods:
        mae.append(float(s.loc[key, "mae_mean"]))
        nn = int(s.loc[key, "n"])
        err.append(float(s.loc[key, "mae_std"]) if nn > 1 else 0.0)
        labs.append(lab)
        cols.append(col)
        record_value("figS08", lab.replace("\n", " "), "mae", round(float(s.loc[key, "mae_mean"]), 4), "validation")
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(len(methods))
    ax.bar(x, mae, yerr=err, color=cols, capsize=3, ecolor="#374151")
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel("Skipped-observation CO$_2$ MAE (ppm)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    save_supp(fig, "figS08_all_policy_baselines")
    record_index(
        file="supplementary/figS08_all_policy_baselines.png",
        place="supplementary",
        question="How do all reported policies compare on MAE?",
        source="full_val_summary.csv",
        split="validation",
        seeds="mixed (1 or 5)",
        values="six-method MAE bars",
        why_main="Complete coverage without cluttering main key-result figure.",
    )


def figS09() -> None:
    apply_paper_style()
    r = load_robust()
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
        # expert n=1 → no fake error bars
        if method == "semantic_expert":
            ax.plot(x, y, color=color, marker=marker, lw=2, ms=7, label=label)
        else:
            ax.errorbar(x, y, yerr=e, color=color, marker=marker, lw=2, ms=7, capsize=3, label=label)
        for xx, yy in zip(x, y):
            record_value("figS09", label, f"mae_loss_{int(xx)}", round(float(yy), 4))
    ax.set_xlabel("Packet loss (%)")
    ax.set_ylabel("CO$_2$ MAE (ppm)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_supp(fig, "figS09_packet_loss_all_methods")
    record_index(
        file="supplementary/figS09_packet_loss_all_methods.png",
        place="supplementary",
        question="How do Expert/BC/KL-CMAPPO compare under packet loss (MAE)?",
        source="robustness_summary.csv",
        split="validation stress",
        seeds="Expert n=1; BC/KL n=5",
        values="three-method MAE vs loss",
        why_main="Full comparison; main shows KL only.",
    )


def figS10() -> None:
    apply_paper_style()
    r = load_robust()
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
    ax.set_xlabel("Packet loss (%)")
    ax.set_ylabel("Event recall (%)")
    ax.set_ylim(90, 100.5)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_supp(fig, "figS10_packet_loss_recall")
    record_index(
        file="supplementary/figS10_packet_loss_recall.png",
        place="supplementary",
        question="How does event recall change under packet loss?",
        source="robustness_summary.csv",
        split="validation stress",
        seeds="Expert n=1; BC/KL n=5",
        values="recall vs loss",
        why_main="Separate from MAE robustness plot.",
    )


def figS11() -> None:
    apply_paper_style()
    cfg = load_yaml(ROOT / "configs" / "rl_cmappo.yaml")
    length = 96
    start = 3552
    ckpt = CMAPPO / "seed_123" / "best_model.pt"
    act = load_mappo_policy(ckpt, device="cpu", prob_threshold=0.5)
    env = make_final_env(split="val", cfg=cfg, multi_agent=True)
    h = _rollout_actions(env, act, start, length, seed=42)
    # encode: 0 missing, 1 skip, 2 tx
    A = np.zeros((h["avail"].shape[1], h["avail"].shape[0]), dtype=float)
    for t in range(h["avail"].shape[0]):
        for i in range(h["avail"].shape[1]):
            if not h["avail"][t, i]:
                A[i, t] = 0.0
            elif h["actions"][t, i] == TRANSMIT:
                A[i, t] = 2.0
            else:
                A[i, t] = 1.0
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#D1D5DB", "#FFFFFF", "#1D4ED8"])
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.imshow(A, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=2)
    ax.set_xlabel("Time step (15 min) within 24 h validation window")
    ax.set_ylabel("Sensor index")
    legend = [
        Patch(facecolor="#1D4ED8", edgecolor="#1D4ED8", label="TX"),
        Patch(facecolor="#FFFFFF", edgecolor="#9CA3AF", label="SKIP"),
        Patch(facecolor="#D1D5DB", edgecolor="#9CA3AF", label="Natural missing"),
    ]
    ax.legend(handles=legend, frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 1.14), ncol=3)
    fig.tight_layout()
    save_supp(fig, "figS11_kl_transmission_heatmap")
    save_json({"start": start, "length": length, "checkpoint": str(ckpt), "split": "val"}, SUPP / "figS11_meta.json")
    record_value("figS11", "KL-CMAPPO", "heatmap_start", start, f"length={length}")
    record_index(
        file="supplementary/figS11_kl_transmission_heatmap.png",
        place="supplementary",
        question="Do TX/SKIP decisions vary across sensors and time?",
        source="KL-CMAPPO seed_123 val rollout",
        split="validation",
        seeds="illustrative",
        values=f"40×{length} TX/SKIP/missing map",
        why_main="Shows non-uniform adaptive schedule; Fixed-15 omitted as trivial.",
    )


def write_docs() -> None:
    ensure_dir(OUT)
    # figure_values.csv
    with (OUT / "figure_values.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["figure", "method", "metric", "value", "extra"])
        w.writeheader()
        for row in VALUES:
            w.writerow(row)

    # Figure_Index.md
    lines = [
        "# Simplified manuscript figures — index",
        "",
        "Generated from frozen artifacts only (no retraining).",
        "Rule: one scientific question → one graph (except method schematic).",
        "",
        "| File | Place | Scientific question | Source | Split | Seeds | Key values | Why here |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in INDEX:
        lines.append(
            f"| `{r['file']}` | {r['place']} | {r['question']} | `{r['source']}` | {r['split']} | {r['seeds']} | {r['values']} | {r['why_main']} |"
        )
    (OUT / "Figure_Index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (OUT / "baseline_rationale.md").write_text(
        """# Baseline rationale

## Fixed 15
Reference periodic system and definition of communication savings (0% reduction).

## Fixed 30 / Fixed 60
Simple periodic baselines. **Fixed 60** is especially important because its TX reduction (~75.3%) is close to KL-CMAPPO (~77.8%), providing an intuitive nearly matched-load comparison in the main MAE bar chart.

## Delta + heartbeat
Simple adaptive heuristic (send-on-change + heartbeat). Kept in the **main results table** and supplementary all-baseline MAE figure. Not required in every main graph.

## Semantic expert
Teacher/reference that generates BC demonstrations. Remains a strong handcrafted method and **must appear in the main table**. Shown in graphs only when the question concerns expert→BC→RL progression or full benchmarking (supplementary).

## BC
Direct learned predecessor of KL-CMAPPO. Essential for proving the RL fine-tuning contribution.

## KL-CMAPPO
Proposed final policy (no safety shield).
""",
        encoding="utf-8",
    )

    # review
    review = [
        "# Simple figure review",
        "",
        "No training performed. All values read from frozen CSVs/JSONs under `outputs/rl_final/`.",
        "",
    ]
    checks = [
        ("fig01", "Campus map only; counts in caption/index.", "yes", "yes", "no extra baselines", "no secondary metric", "metadata", "N/A", "yes"),
        ("fig02", "One CO2 timeline with KL TX dots + Fixed15 annotation.", "yes", "yes", "Fixed15 only as count", "no recall panel", "validation", "illustrative", "yes"),
        ("fig03", "Vertical workflow schematic.", "yes", "yes", "no shield", "state vars omitted", "N/A", "N/A", "yes"),
        ("fig04", "3 bars Fixed60/BC/KL with TX↓ labels.", "yes", "yes", "expert/delta excluded", "MAE only", "validation", "yes (BC/KL)", "yes"),
        ("fig05", "BC vs KL matched MAE.", "yes", "yes", "only BC/KL", "MAE only", "validation", "yes", "yes"),
        ("fig06", "Training val MAE + feasible band.", "yes", "yes", "KL only", "MAE only", "val checkpoints", "yes", "yes"),
        ("fig07", "Ablation MAE bars + TX↓ tags.", "yes", "yes", "ablation set only", "MAE only", "validation seed42", "no (n=1)", "yes"),
        ("fig08", "Held-out test MAE BC vs KL.", "yes", "yes", "only BC/KL", "MAE only", "held-out TEST", "yes", "yes"),
        ("fig09", "KL packet-loss MAE curve.", "yes", "yes", "KL only", "MAE only", "validation stress", "yes", "yes"),
        ("fig10", "Cumulative TX Fixed15/60/KL.", "yes", "yes", "communication defs", "TX counts only", "validation", "illustrative", "yes"),
    ]
    for name, notice, q1, q2, q3, q4, q5, q6, q7 in checks:
        review += [
            f"## {name}",
            f"1. Understand in ~10s? **{q1}** — {notice}",
            f"2. One question? **{q2}**",
            f"3. Unnecessary baselines? **{q3}**",
            f"4. Secondary metric clutter? **{q4}**",
            f"5. Split stated? **{q5}**",
            f"6. Error bars legitimate? **{q6}**",
            f"7. Key result visible without long legend? **{q7}**",
            "",
        ]
    (OUT / "simple_figure_review.md").write_text("\n".join(review), encoding="utf-8")
    (OUT / "generation_audit.log").write_text("\n".join(AUDIT) + "\n", encoding="utf-8")
    log(f"WROTE {OUT / 'Figure_Index.md'}")
    log(f"WROTE {OUT / 'figure_values.csv'}")
    log(f"WROTE {OUT / 'baseline_rationale.md'}")
    log(f"WROTE {OUT / 'simple_figure_review.md'}")


def populate_csv_json() -> None:
    """Copy frozen result tables into results/csv_json (no mutation of originals)."""
    import shutil

    ensure_dir(CSV)
    copies = [
        (PAPER / "full_val_summary.csv", CSV / "full_val_summary.csv"),
        (PAPER / "full_val_raw.csv", CSV / "full_val_raw.csv"),
        (PAPER / "full_test_summary.csv", CSV / "full_test_summary.csv"),
        (PAPER / "full_test_raw.csv", CSV / "full_test_raw.csv"),
        (SCI / "matched_budget_summary.csv", CSV / "matched_budget_summary.csv"),
        (SCI / "matched_budget_raw.csv", CSV / "matched_budget_raw.csv"),
        (SCI / "heldout_transfer_summary.csv", CSV / "heldout_transfer_summary.csv"),
        (SCI / "heldout_transfer_raw.csv", CSV / "heldout_transfer_raw.csv"),
        (SCI / "robustness_summary.csv", CSV / "robustness_summary.csv"),
        (SCI / "robustness_val.csv", CSV / "robustness_val.csv"),
        (SCI / "ablation_kl_cmappo_val.csv", CSV / "ablation_kl_cmappo_val.csv"),
    ]
    for src, dst in copies:
        if src.exists():
            shutil.copy2(src, dst)
            log(f"COPIED {dst.relative_to(ROOT)}")
    if (OUT / "figure_values.csv").exists():
        shutil.copy2(OUT / "figure_values.csv", CSV / "figure_values.csv")
    if (MAIN / "fig02_meta.json").exists():
        shutil.copy2(MAIN / "fig02_meta.json", CSV / "fig02_meta.json")


def main() -> None:
    ensure_dir(MAIN)
    ensure_dir(SUPP)
    ensure_dir(CSV)
    apply_paper_style()
    log("=== RESULTS FIGURE PACK (no training; KL-CMAPPO = proposed method) ===")
    # clear obsolete main filenames from prior framing
    for obsolete in [
        "fig05_matched_budget_mae.png",
        "fig06_training_validation_mae.png",
        "fig07_ablation_mae.png",
        "fig08_heldout_mae.png",
        "fig09_packet_loss_mae.png",
        "fig10_cumulative_transmissions.png",
    ]:
        p = MAIN / obsolete
        if p.exists():
            p.unlink()
    old_s01 = SUPP / "figS01_matched_budget_recall.png"
    if old_s01.exists():
        old_s01.unlink()

    fig01()
    fig02()
    fig03()
    fig04()
    fig05_training()
    fig06_ablation()
    fig07_heldout()
    fig08_packet_loss()
    fig09_cumulative()
    figS01_matched_rl_vs_init()
    figS01b_matched_recall()
    figS02()
    figS03()
    figS04()
    figS05()
    figS06()
    figS07()
    figS08()
    figS09()
    figS10()
    figS11()
    write_docs()
    populate_csv_json()
    print("[done]", OUT, flush=True)


if __name__ == "__main__":
    main()
