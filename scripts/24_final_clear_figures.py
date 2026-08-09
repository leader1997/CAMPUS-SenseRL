#!/usr/bin/env python
"""FINAL CLEAR FIGURE PACK — visualization only (no training).

Reads frozen CSVs/JSONs under outputs/rl_final/{paper_final,scientific_validation,cmappo_kl}
and writes paper_outputs/final_clear_figures/.
"""

from __future__ import annotations

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
from campus_senserl.environment.communication_model import SKIP, TRANSMIT
from campus_senserl.evaluation.rl_policy_eval import load_mappo_policy, make_final_env
from campus_senserl.rl.fixed_policies import FixedIntervalPolicy
from campus_senserl.utils import ensure_dir, load_yaml, repo_root, save_json
from campus_senserl.visualization.paper_style import apply_paper_style

OUT = ROOT / "paper_outputs" / "final_clear_figures"
SUPP = OUT / "supplementary"
PAPER = ROOT / "outputs" / "rl_final" / "paper_final"
SCI = ROOT / "outputs" / "rl_final" / "scientific_validation"
CMAPPO = ROOT / "outputs" / "rl_final" / "cmappo_kl"
SEEDS = [42, 123, 2024, 3407, 9999]
DPI = 300
AUDIT: list[str] = []

DISPLAY = {
    "fixed_15": "Fixed 15 min",
    "fixed_30": "Fixed 30 min",
    "fixed_60": "Fixed 60 min",
    "delta_plus_heartbeat": "Delta + heartbeat",
    "semantic_expert": "Semantic expert",
    "campus_senserl_bc": "BC policy",
    "cmappo_kl": "KL-CMAPPO",
}
COLORS = {
    "Fixed 15 min": "#9ECAE1",
    "Fixed 30 min": "#6BAED6",
    "Fixed 60 min": "#2171B5",
    "Delta + heartbeat": "#41AB5D",
    "Semantic expert": "#006D2C",
    "BC policy": "#6A51A3",
    "KL-CMAPPO": "#CB181D",
}


def log(msg: str) -> None:
    print(msg)
    AUDIT.append(msg)


def save(fig: plt.Figure, stem: str, sub: Path | None = None) -> None:
    d = sub or OUT
    ensure_dir(d)
    png = d / f"{stem}.png"
    fig.savefig(png, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    log(f"WROTE {png.relative_to(ROOT)}")


def panel_label(ax, text: str) -> None:
    ax.text(-0.12, 1.05, text, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
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


def train_rows(seed: int) -> pd.DataFrame:
    m = load_seed_metrics(seed)["metrics"]
    return pd.DataFrame([r for r in m if "mean_tx" in r and "val_mae" not in r])


def val_rows(seed: int) -> pd.DataFrame:
    m = load_seed_metrics(seed)["metrics"]
    return pd.DataFrame([r for r in m if "val_mae" in r])


def aggregate_val_curves() -> pd.DataFrame:
    frames = []
    for s in SEEDS:
        v = val_rows(s)
        v["seed"] = s
        frames.append(v)
    allv = pd.concat(frames, ignore_index=True)
    g = allv.groupby("step", as_index=False).agg(
        val_mae_mean=("val_mae", "mean"),
        val_mae_std=("val_mae", "std"),
        val_recall_mean=("val_recall", "mean"),
        val_recall_std=("val_recall", "std"),
        val_tx_mean=("val_tx_reduction", "mean"),
        val_tx_std=("val_tx_reduction", "std"),
        n_feasible=("feasible", "sum"),
        n=("feasible", "count"),
    )
    return g


def aggregate_train_curves() -> pd.DataFrame:
    frames = []
    for s in SEEDS:
        t = train_rows(s)
        t["seed"] = s
        frames.append(t)
    allv = pd.concat(frames, ignore_index=True)
    # align approximate steps by round to nearest 1024
    allv["step_bin"] = (allv["step"] // 1024) * 1024
    g = allv.groupby("step_bin", as_index=False).agg(
        lam_e_mean=("lam_e", "mean"),
        lam_e_std=("lam_e", "std"),
        lam_m_mean=("lam_m", "mean"),
        lam_m_std=("lam_m", "std"),
        lam_a_mean=("lam_a", "mean"),
        lam_a_std=("lam_a", "std"),
        beta_mean=("kl_beta", "mean"),
        beta_std=("kl_beta", "std"),
        n=("seed", "count"),
    )
    return g.rename(columns={"step_bin": "step"})


# ---------------------------------------------------------------------------
# Fig 01 deployment
# ---------------------------------------------------------------------------
def fig01() -> None:
    apply_paper_style()
    cfg = load_yaml(ROOT / "configs" / "data.yaml")
    devices = load_devices(ROOT / cfg["paths"]["raw_release"] / cfg["dataset"]["devices_file"])
    devices = devices.copy()
    devices["deveui"] = devices["device_id"].astype(str).str.upper()
    train = {s.upper() for s in load_cohort("final")}
    held = {s.upper() for s in load_cohort("heldout")}
    co2 = devices["device_type"].astype(str).str.contains("CO2", case=False, na=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.45, 1.0]})
    ax = axes[0]
    ax.scatter(devices["longitude"], devices["latitude"], s=12, c="#D9D9D9", label=f"All sensors (n={len(devices)})", zorder=1)
    ax.scatter(
        devices.loc[co2, "longitude"],
        devices.loc[co2, "latitude"],
        s=14,
        c="#9ECAE1",
        label=f"CO$_2$-capable (n={int(co2.sum())})",
        zorder=2,
    )
    tr = devices[devices["deveui"].isin(train)]
    ho = devices[devices["deveui"].isin(held)]
    ax.scatter(tr["longitude"], tr["latitude"], s=34, c=COLORS["BC policy"], edgecolors="white", linewidths=0.3, label=f"RL development (n={len(tr)})", zorder=3)
    ax.scatter(ho["longitude"], ho["latitude"], s=34, c=COLORS["KL-CMAPPO"], edgecolors="white", linewidths=0.3, label=f"Held-out transfer (n={len(ho)})", zorder=4)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, fontsize=7.5, loc="best")
    panel_label(ax, "(a)")

    ax = axes[1]
    labels = ["All sensors", "CO$_2$-capable", "RL development", "Held-out RL"]
    vals = [len(devices), int(co2.sum()), len(tr), len(ho)]
    cols = ["#D9D9D9", "#9ECAE1", COLORS["BC policy"], COLORS["KL-CMAPPO"]]
    ax.barh(labels[::-1], vals[::-1], color=cols[::-1])
    for y, v in enumerate(vals[::-1]):
        ax.text(v + 5, y, str(v), va="center", fontsize=9)
    ax.set_xlabel("Number of sensors")
    ax.set_xlim(0, max(vals) * 1.18)
    panel_label(ax, "(b)")
    fig.tight_layout()
    save(fig, "fig01_campus_deployment")
    log("FIG01 source: devices.json + final_cohort.json + heldout_cohort.json; split=N/A (metadata)")


# ---------------------------------------------------------------------------
# Shared window selection + rollouts for figs 02, 09, 12
# ---------------------------------------------------------------------------
def _select_window(env, length: int = 192) -> tuple[int, int]:
    """Deterministic window: first val segment with moderate std and a rise, not extreme."""
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
    hist = {
        "t": [],
        "gt": [],
        "recon": [],
        "actions": [],
        "avail": [],
        "tx_delivered": [],
        "true_event": [],
    }
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
        te = info.get("true_events")
        hist["true_event"].append(np.asarray(te, dtype=bool).copy() if te is not None else np.zeros(env.n_sensors, bool))
        if term or trunc:
            break
    for k in hist:
        hist[k] = np.asarray(hist[k])
    return hist


def fig02() -> None:
    apply_paper_style()
    cfg = load_yaml(ROOT / "configs" / "rl_cmappo.yaml")
    env = make_final_env(split="val", cfg=cfg, multi_agent=True)
    start, sid = _select_window(env, length=192)
    sensor_id = env.sensor_ids[sid]
    length = 192

    # Fixed-15: interval=1
    fixed = FixedIntervalPolicy(1)

    def fixed_act(obs, *, local_available=None):
        return fixed.act(obs, local_available=local_available)

    env_f = make_final_env(split="val", cfg=cfg, multi_agent=True)
    h_fix = _rollout_actions(env_f, fixed_act, start, length, seed=42)

    ckpt = CMAPPO / "seed_123" / "best_model.pt"
    act = load_mappo_policy(ckpt, device="cpu", prob_threshold=0.5)
    env_k = make_final_env(split="val", cfg=cfg, multi_agent=True)
    h_kl = _rollout_actions(env_k, act, start, length, seed=42)

    hours = np.arange(length) * 0.25
    gt = h_fix["gt"][:, sid]
    avail = h_fix["avail"][:, sid]
    # Fixed TX whenever available
    fixed_tx = avail.astype(bool)
    kl_tx = (h_kl["actions"][:, sid] == TRANSMIT) & h_kl["avail"][:, sid]
    kl_recon = h_kl["recon"][:, sid]
    n_fix = int(fixed_tx.sum())
    n_kl = int(kl_tx.sum())
    red = 100.0 * (1.0 - n_kl / n_fix) if n_fix else float("nan")

    fig, axes = plt.subplots(2, 1, figsize=(10.0, 5.6), sharex=True)
    ax = axes[0]
    ax.plot(hours, gt, color="#222222", lw=1.2, label="True CO$_2$")
    ax.scatter(hours[fixed_tx], gt[fixed_tx], s=10, c=COLORS["Fixed 15 min"], label="TX (Fixed 15)", zorder=3)
    ax.axhline(1000, color="#CB181D", ls="--", lw=0.9, alpha=0.7, label="1000 ppm")
    ax.set_ylabel("CO$_2$ (ppm)")
    ax.set_title(f"(a) Fixed 15 min  —  TX count = {n_fix}")
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper right")

    ax = axes[1]
    ax.plot(hours, gt, color="#222222", lw=1.2, label="True CO$_2$")
    skip = (~kl_tx) & h_kl["avail"][:, sid]
    ax.plot(hours[skip], kl_recon[skip], color="#41AB5D", lw=1.0, alpha=0.9, label="Server recon (SKIP)")
    ax.scatter(hours[kl_tx], gt[kl_tx], s=18, c=COLORS["KL-CMAPPO"], label="TX (KL-CMAPPO)", zorder=3)
    ax.axhline(1000, color="#CB181D", ls="--", lw=0.9, alpha=0.7)
    ax.set_ylabel("CO$_2$ (ppm)")
    ax.set_xlabel("Time (hours from window start)")
    ax.set_title(f"(b) KL-CMAPPO (seed 123)  —  TX = {n_kl}  ·  reduction vs Fixed-15 = {red:.1f}%")
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper right")
    fig.suptitle(f"Validation · sensor {sensor_id[-6:]} · start step {start}", fontsize=10, y=1.01)
    fig.tight_layout()
    save(fig, "fig02_fixed_vs_adaptive_timeline")
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
        OUT / "fig02_meta.json",
    )
    log(f"FIG02 interval TX Fixed15={n_fix} KL={n_kl} red={red:.2f}% sensor={sensor_id} start={start}")


# ---------------------------------------------------------------------------
# Fig 03 workflow
# ---------------------------------------------------------------------------
def fig03() -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(11.2, 5.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#F7FBFF", ec="#2171B5", fs=8):
        ax.add_patch(
            FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", lw=1.2, ec=ec, fc=fc)
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#444", lw=1.2))

    box(0.2, 2.3, 1.7, 1.5, "Real campus\nsensor", fc="#EFF3FF")
    box(2.1, 2.15, 2.2, 1.8, "Local sensing state\nCO$_2$, $\\Delta$CO$_2$, AoI\nmotion, battery,\nlink / neighbour cues", fc="#EFF3FF")
    box(4.55, 3.7, 2.3, 1.1, "Semantic expert\n(demonstrations)", fc="#E5F5E0", ec="#41AB5D")
    box(4.55, 2.35, 2.3, 1.1, "Behavior cloning\ninitial actor $\\pi_{BC}$", fc="#E5F5E0", ec="#41AB5D")
    box(
        4.55,
        0.35,
        2.3,
        1.8,
        "KL-CMAPPO fine-tuning\n(1) reduce TX\n(2) constraints:\nevent · MAE · AoI\n(3) KL($\\pi\\|\\pi_{BC}$)",
        fc="#FEE0D2",
        ec="#CB181D",
        fs=7.5,
    )
    box(7.15, 2.4, 1.6, 1.2, "TX / SKIP", fc="#FFF5F0", ec="#CB181D")
    box(9.0, 3.2, 2.6, 1.2, "TX → true measurement\nto server", fc="#E5F5E0", ec="#41AB5D")
    box(9.0, 1.6, 2.6, 1.2, "SKIP → causal recon\n(LOCF) at server", fc="#E5F5E0", ec="#41AB5D")
    box(9.0, 0.3, 2.6, 1.0, "Campus monitoring\nevents · freshness", fc="#F7FBFF")

    arrow(1.9, 3.0, 2.1, 3.0)
    arrow(4.3, 3.0, 4.55, 3.0)
    arrow(5.7, 3.7, 5.7, 3.45)
    arrow(5.7, 2.35, 5.7, 2.15)
    arrow(6.85, 3.0, 7.15, 3.0)
    arrow(8.75, 3.2, 9.0, 3.7)
    arrow(8.75, 2.7, 9.0, 2.3)
    arrow(10.3, 3.2, 10.3, 2.8)
    arrow(10.3, 1.6, 10.3, 1.3)

    ax.text(0.25, 5.55, "CAMPUS-SenseRL / KL-CMAPPO workflow", fontsize=12, fontweight="bold")
    ax.text(0.25, 5.15, "Centralized training · decentralized execution", fontsize=9, color="#555")
    ax.text(7.1, 4.55, r"$\pi_{BC}$  ←── KL regularization ($\beta$) ──→  $\pi_{RL}$", fontsize=9, color="#CB181D")
    ax.text(7.1, 5.05, "Final method: KL-CMAPPO", fontsize=9, fontweight="bold", color="#CB181D")
    save(fig, "fig03_method_workflow")
    log("FIG03 schematic (no numerical CSV); KL-CMAPPO method schematic")


# ---------------------------------------------------------------------------
# Fig 04 overall results
# ---------------------------------------------------------------------------
def fig04() -> None:
    apply_paper_style()
    s = load_val_summary()
    order = ["fixed_30", "fixed_60", "delta_plus_heartbeat", "semantic_expert", "campus_senserl_bc", "cmappo_kl"]
    labels = [DISPLAY[m] for m in order]
    x = np.arange(len(order))
    tx, txe, mae, maee = [], [], [], []
    for m in order:
        r = s[s["method"] == m].iloc[0]
        tx.append(float(r["tx_reduction_mean"]))
        txe.append(float(r["tx_reduction_std"]) if float(r["n"]) > 1 else 0.0)
        mae.append(float(r["mae_mean"]))
        maee.append(float(r["mae_std"]) if float(r["n"]) > 1 and pd.notna(r["mae_std"]) else 0.0)
        log(f"FIG04 {m}: TX↓={tx[-1]:.3f}±{txe[-1]:.3f} MAE={mae[-1]:.3f}±{maee[-1]:.3f} n={int(r['n'])}")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    cols = [COLORS[l] for l in labels]
    ax.bar(x, tx, yerr=[0 if e == 0 else e for e in txe], color=cols, capsize=3, ecolor="#333")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Transmission reduction (%)")
    ax.set_title("Communication savings")
    panel_label(ax, "(a)")

    ax = axes[1]
    ax.bar(x, mae, yerr=[0 if e == 0 else e for e in maee], color=cols, capsize=3, ecolor="#333")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Skipped-slot CO$_2$ MAE (ppm)")
    ax.set_title("Reconstruction quality")
    panel_label(ax, "(b)")
    fig.suptitle("Validation results (mean ± std over seeds when n>1)", fontsize=10)
    fig.tight_layout()
    save(fig, "fig04_overall_results")


# ---------------------------------------------------------------------------
# Fig 05 matched budget
# ---------------------------------------------------------------------------
def fig05() -> None:
    apply_paper_style()
    m = load_matched()
    targets = [75.0, 78.0, 80.0]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    for method, label, marker in [
        ("campus_senserl_bc", "BC policy", "o"),
        ("cmappo_kl", "KL-CMAPPO", "s"),
    ]:
        sub = m[m["method"] == method].set_index("target_tx_reduction").loc[targets]
        axes[0].errorbar(
            targets,
            sub["mae_mean"],
            yerr=sub["mae_std"],
            fmt=f"-{marker}",
            color=COLORS[label],
            label=label,
            capsize=4,
            markersize=8,
        )
        axes[1].errorbar(
            targets,
            100 * sub["recall_mean"],
            yerr=100 * sub["recall_std"],
            fmt=f"-{marker}",
            color=COLORS[label],
            label=label,
            capsize=4,
            markersize=8,
        )
        for t in targets:
            log(f"FIG05 {method} @{t}: MAE={sub.loc[t,'mae_mean']:.4f}±{sub.loc[t,'mae_std']:.4f} rec={sub.loc[t,'recall_mean']:.4f}")

    # annotate 78%
    for method, label, dy in [("campus_senserl_bc", "BC policy", -10), ("cmappo_kl", "KL-CMAPPO", 8)]:
        r = m[(m["method"] == method) & (m["target_tx_reduction"] == 78.0)].iloc[0]
        axes[0].annotate(f"{r['mae_mean']:.2f}", (78, r["mae_mean"]), textcoords="offset points", xytext=(6, dy), color=COLORS[label], fontsize=8)

    axes[0].set_xlabel("Target transmission reduction (%)")
    axes[0].set_ylabel("CO$_2$ MAE (ppm)")
    axes[0].set_title("Reconstruction at matched budgets")
    axes[0].legend(frameon=False)
    axes[0].grid(True, alpha=0.25)
    panel_label(axes[0], "(a)")

    axes[1].set_xlabel("Target transmission reduction (%)")
    axes[1].set_ylabel("Event recall (%)")
    axes[1].set_ylim(95, 100)
    axes[1].set_title("Event recall at matched budgets")
    axes[1].legend(frameon=False)
    axes[1].grid(True, alpha=0.25)
    panel_label(axes[1], "(b)")
    fig.suptitle("Validation · 5 seeds · threshold-matched TX budgets", fontsize=10)
    fig.tight_layout()
    save(fig, "fig05_matched_budget")


# ---------------------------------------------------------------------------
# Fig 06 RL training dynamics
# ---------------------------------------------------------------------------
def fig06() -> None:
    apply_paper_style()
    g = aggregate_val_curves()
    best_steps = []
    for s in SEEDS:
        bv = load_seed_metrics(s).get("best_val") or {}
        if bv:
            best_steps.append((s, bv.get("step")))
            log(f"FIG06 best seed={s} step={bv.get('step')} mae={bv.get('val_mae')} feasible={bv.get('feasible')}")

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), sharex=True)
    steps = g["step"].to_numpy()

    ax = axes[0, 0]
    ax.fill_between(steps, 0, 9.0, color="#E5F5E0", alpha=0.5, label="Feasible MAE ≤ 9")
    ax.errorbar(steps, g["val_mae_mean"], yerr=g["val_mae_std"].fillna(0), fmt="-o", color=COLORS["KL-CMAPPO"], capsize=3, markersize=4)
    ax.axhline(9.0, color="#238B45", ls="--", lw=1.0, label=r"$\varepsilon_{MAE}=9.0$")
    ax.set_ylabel("Validation MAE (ppm)")
    ax.set_title("Skipped-observation MAE")
    ax.legend(frameon=False, fontsize=7)
    panel_label(ax, "(a)")

    ax = axes[0, 1]
    ax.fill_between(steps, 98.5, 100.5, color="#E5F5E0", alpha=0.5, label="Feasible recall ≥ 98.5%")
    ax.errorbar(steps, 100 * g["val_recall_mean"], yerr=100 * g["val_recall_std"].fillna(0), fmt="-o", color=COLORS["KL-CMAPPO"], capsize=3, markersize=4)
    ax.axhline(98.5, color="#238B45", ls="--", lw=1.0, label="Recall ≥ 98.5%")
    ax.set_ylabel("Validation event recall (%)")
    ax.set_ylim(97.5, 100.2)
    ax.set_title("Event preservation")
    ax.legend(frameon=False, fontsize=7)
    panel_label(ax, "(b)")

    ax = axes[1, 0]
    ax.errorbar(steps, g["val_tx_mean"], yerr=g["val_tx_std"].fillna(0), fmt="-o", color=COLORS["BC policy"], capsize=3, markersize=4)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation TX reduction (%)")
    ax.set_title("Communication savings")
    panel_label(ax, "(c)")

    ax = axes[1, 1]
    ax.step(steps, g["n_feasible"], where="mid", color=COLORS["KL-CMAPPO"], lw=1.8)
    ax.scatter(steps, g["n_feasible"], c=COLORS["KL-CMAPPO"], s=28, zorder=3)
    for s, st in best_steps:
        if st is not None:
            ax.axvline(st, color="#CCCCCC", lw=0.8, alpha=0.7)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Feasible seeds / 5")
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_title("Constraint feasibility across seeds")
    panel_label(ax, "(d)")

    fig.suptitle("KL-CMAPPO training dynamics (5 seeds · validation checkpoints)", fontsize=10)
    fig.tight_layout()
    save(fig, "fig06_rl_training_dynamics")


# ---------------------------------------------------------------------------
# Fig 07 KL + constraint dynamics
# ---------------------------------------------------------------------------
def fig07() -> None:
    apply_paper_style()
    g = aggregate_train_curves()
    steps = g["step"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))

    ax = axes[0]
    for key, label, color in [
        ("lam_e", r"$\lambda_{event}$", "#E6550D"),
        ("lam_m", r"$\lambda_{MAE}$", "#31A354"),
        ("lam_a", r"$\lambda_{AoI}$", "#3182BD"),
    ]:
        ax.errorbar(steps, g[f"{key}_mean"], yerr=g[f"{key}_std"].fillna(0), fmt="-o", markersize=3, color=color, label=label, capsize=2)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Dual variable value")
    ax.set_title("Adaptive Lagrange multipliers")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    panel_label(ax, "(a)")

    ax = axes[1]
    ax.errorbar(steps, g["beta_mean"], yerr=g["beta_std"].fillna(0), fmt="-o", markersize=3, color=COLORS["KL-CMAPPO"], capsize=2)
    ax.set_xlabel("Training step")
    ax.set_ylabel(r"KL coefficient $\beta$")
    ax.set_title("KL regularization schedule (not KL divergence)")
    ax.annotate("stronger BC anchoring", xy=(0.05, 0.92), xycoords="axes fraction", fontsize=8, color="#555")
    ax.annotate("→ more RL freedom", xy=(0.55, 0.15), xycoords="axes fraction", fontsize=8, color="#555")
    ax.grid(True, alpha=0.25)
    panel_label(ax, "(b)")
    fig.suptitle("Constraint duals and KL coefficient (5-seed mean ± std · training logs)", fontsize=10)
    fig.tight_layout()
    save(fig, "fig07_kl_constraint_dynamics")
    log("FIG07 uses train-time lam_* and kl_beta only; NOT D_KL values")


# ---------------------------------------------------------------------------
# Fig 08 ablation
# ---------------------------------------------------------------------------
def fig08() -> None:
    apply_paper_style()
    abl = load_ablation()
    # Marker + color identity (legend) avoids ambiguous overlapping callouts in the 78–82% cluster.
    specs = [
        ("constraints_only", "Constraints only", "#E6550D", "D"),
        ("full_kl_cmappo", "Full KL-CMAPPO", COLORS["KL-CMAPPO"], "s"),
        ("bc_only", "BC only", COLORS["BC policy"], "o"),
        ("kl_only", "KL only", "#31A354", "^"),
        ("mappo_no_kl_no_constraints", "MAPPO w/o KL/constraints", "#6BAED6", "v"),
    ]
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    for key, label, color, marker in specs:
        r = abl[abl["ablation"] == key]
        if r.empty:
            continue
        r = r.iloc[0]
        x = float(r["transmission_reduction_pct"])
        y = float(r["mae_skipped"])
        rec = 100.0 * float(r["event_recall"])
        ax.scatter(
            x,
            y,
            s=140,
            c=color,
            marker=marker,
            zorder=3,
            edgecolors="white",
            linewidths=0.7,
            label=f"{label} (rec={rec:.1f}%)",
        )
        if key in {"constraints_only", "full_kl_cmappo"}:
            ax.annotate(
                f"{x:.1f}%, {y:.2f} ppm",
                (x, y),
                textcoords="offset points",
                xytext=(8, 8) if key != "full_kl_cmappo" else (-8, -18),
                fontsize=7.5,
                ha="left" if key != "full_kl_cmappo" else "right",
            )
        log(f"FIG08 {key}: TX↓={x:.2f} MAE={y:.2f} rec={r['event_recall']:.4f} seed={r['seed']}")

    ax.set_xlabel("Transmission reduction (%) →")
    ax.set_ylabel("Skipped-slot CO$_2$ MAE (ppm)")
    ax.set_title("KL / constraint ablation (mechanistic · seed 42 · validation)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=7.5, frameon=True, framealpha=0.95)
    ax.text(0.98, 0.02, "Better → right and down", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#666")
    fig.tight_layout()
    save(fig, "fig08_kl_constraint_ablation")


# ---------------------------------------------------------------------------
# Fig 09 heatmap
# ---------------------------------------------------------------------------
def fig09() -> None:
    apply_paper_style()
    cfg = load_yaml(ROOT / "configs" / "rl_cmappo.yaml")
    length = 96  # 24 hours
    env0 = make_final_env(split="val", cfg=cfg, multi_agent=True)
    start, _ = _select_window(env0, length=length)
    # use a fixed start near selected window but aligned
    start = (start // 96) * 96

    fixed = FixedIntervalPolicy(1)

    def fixed_act(obs, *, local_available=None):
        return fixed.act(obs, local_available=local_available)

    env_f = make_final_env(split="val", cfg=cfg, multi_agent=True)
    h_fix = _rollout_actions(env_f, fixed_act, start, length, seed=42)
    act = load_mappo_policy(CMAPPO / "seed_123" / "best_model.pt", device="cpu")
    env_k = make_final_env(split="val", cfg=cfg, multi_agent=True)
    h_kl = _rollout_actions(env_k, act, start, length, seed=42)

    def encode(h):
        # 0 missing, 1 skip, 2 tx
        a = np.zeros_like(h["actions"], dtype=float)
        avail = h["avail"].astype(bool)
        tx = (h["actions"] == TRANSMIT) & avail
        skip = (~tx) & avail
        a[~avail] = np.nan
        a[skip] = 0.0
        a[tx] = 1.0
        return a.T  # sensors x time

    A_fix = encode(h_fix)
    A_kl = encode(h_kl)
    event_strip = h_kl["true_event"].any(axis=1)

    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#F0F0F0", "#CB181D"])  # skip light, tx dark — missing as masked

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.2), sharex=True, gridspec_kw={"height_ratios": [1, 1], "hspace": 0.18})
    for ax, A, title in [
        (axes[0], A_fix, "(a) Fixed 15 min"),
        (axes[1], A_kl, "(b) KL-CMAPPO"),
    ]:
        data = np.ma.array(A, mask=np.isnan(A))
        im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1, origin="lower")
        # event markers
        for t in np.where(event_strip)[0]:
            ax.axvline(t, color="#FDAE6B", alpha=0.25, lw=0.8)
        ax.set_ylabel("Sensor index")
        ax.set_title(title)
    axes[1].set_xlabel("Time step (15 min) within 24 h validation window")
    # legend
    legend_elems = [
        Patch(facecolor="#CB181D", label="TX"),
        Patch(facecolor="#F0F0F0", edgecolor="#AAAAAA", label="SKIP (available)"),
        Patch(facecolor="white", edgecolor="#333333", label="Natural missing (masked)"),
        Patch(facecolor="#FDAE6B", label="Interval with any true high-CO₂ event"),
    ]
    axes[0].legend(handles=legend_elems, frameon=False, fontsize=7.5, loc="upper right", bbox_to_anchor=(1.0, 1.35), ncol=2)
    fig.suptitle(f"Campus-wide TX/SKIP map · validation · start step {start}", fontsize=10, y=1.02)
    save(fig, "fig09_transmission_heatmap")
    save_json({"split": "val", "start_step": start, "length": length, "n_sensors": int(A_kl.shape[0])}, OUT / "fig09_meta.json")
    log(f"FIG09 heatmap start={start} length={length} sensors={A_kl.shape[0]}")


# ---------------------------------------------------------------------------
# Fig 10 held-out
# ---------------------------------------------------------------------------
def fig10() -> None:
    apply_paper_style()
    h = load_heldout()
    rows = h[(h["split"] == "test") & (h["method"].isin(["campus_senserl_bc", "cmappo_kl"]))].copy()
    order = ["campus_senserl_bc", "cmappo_kl"]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.7))
    for i, m in enumerate(order):
        r = rows[rows["method"] == m].iloc[0]
        label = DISPLAY[m]
        axes[0].errorbar(i, r["mae_mean"], yerr=r["mae_std"], fmt="o", color=COLORS[label], markersize=10, capsize=5)
        axes[1].errorbar(i, 100 * r["recall_mean"], yerr=100 * r["recall_std"], fmt="o", color=COLORS[label], markersize=10, capsize=5)
        axes[0].annotate(f"TX↓ {r['tx_reduction_mean']:.1f}%", (i, r["mae_mean"]), textcoords="offset points", xytext=(8, 0), fontsize=8)
        log(f"FIG10 {m} test heldout: MAE={r['mae_mean']:.4f}±{r['mae_std']:.4f} rec={r['recall_mean']:.4f} TX↓={r['tx_reduction_mean']:.2f}")

    for ax, ylab, title, ylim in [
        (axes[0], "CO$_2$ MAE (ppm)", "Reconstruction", None),
        (axes[1], "Event recall (%)", "Event preservation", (95, 100)),
    ]:
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["BC policy", "KL-CMAPPO"])
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        if ylim:
            ax.set_ylim(*ylim)
    fig.suptitle("Held-out sensors (never used in RL development) · test · 5 seeds", fontsize=10)
    fig.tight_layout()
    save(fig, "fig10_heldout_generalization")


# ---------------------------------------------------------------------------
# Fig 11 packet loss
# ---------------------------------------------------------------------------
def fig11() -> None:
    apply_paper_style()
    rob = load_robust()
    pl = rob[rob["condition"] == "packet_loss"].copy()
    methods = [
        ("semantic_expert", "Semantic expert"),
        ("campus_senserl_bc", "BC policy"),
        ("cmappo_kl", "KL-CMAPPO"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9))
    for m, label in methods:
        g = pl[pl["method"] == m].sort_values("level")
        yerr_mae = g["mae_std"].fillna(0) if int(g["n"].iloc[0]) > 1 else None
        yerr_rec = g["recall_std"].fillna(0) if int(g["n"].iloc[0]) > 1 else None
        axes[0].errorbar(g["level"] * 100, g["mae_mean"], yerr=yerr_mae, fmt="-o", color=COLORS[label], label=label, capsize=3)
        axes[1].errorbar(g["level"] * 100, 100 * g["recall_mean"], yerr=(100 * yerr_rec) if yerr_rec is not None else None, fmt="-o", color=COLORS[label], label=label, capsize=3)
        for _, r in g.iterrows():
            log(f"FIG11 {m} loss={r['level']}: MAE={r['mae_mean']:.3f} rec={r['recall_mean']:.4f} n={int(r['n'])}")

    axes[0].set_xlabel("Packet loss (%)")
    axes[0].set_ylabel("CO$_2$ MAE (ppm)")
    axes[0].set_title("Reconstruction")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(True, alpha=0.25)
    panel_label(axes[0], "(a)")
    axes[1].set_xlabel("Packet loss (%)")
    axes[1].set_ylabel("Event recall (%)")
    axes[1].set_ylim(90, 100)
    axes[1].set_title("Event preservation")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(True, alpha=0.25)
    panel_label(axes[1], "(b)")
    fig.suptitle("Robustness to packet loss · validation", fontsize=10)
    fig.tight_layout()
    save(fig, "fig11_packet_loss_robustness")


# ---------------------------------------------------------------------------
# Fig 12 cumulative TX
# ---------------------------------------------------------------------------
def fig12() -> None:
    apply_paper_style()
    cfg = load_yaml(ROOT / "configs" / "rl_cmappo.yaml")
    length = 96 * 7  # one week
    start = 960
    policies = []
    for name, maker in [
        ("Fixed 15 min", lambda: FixedIntervalPolicy(1)),
        ("Fixed 60 min", lambda: FixedIntervalPolicy(4)),
        ("KL-CMAPPO", None),
    ]:
        env = make_final_env(split="val", cfg=cfg, multi_agent=True)
        if name == "KL-CMAPPO":
            act = load_mappo_policy(CMAPPO / "seed_123" / "best_model.pt", device="cpu")
        else:
            pol = maker()

            def act(obs, *, local_available=None, _p=pol):
                return _p.act(obs, local_available=local_available)

        h = _rollout_actions(env, act, start, length, seed=42)
        # count TX decisions on available
        tx = ((h["actions"] == TRANSMIT) & h["avail"]).sum(axis=1)
        cum = np.cumsum(tx)
        policies.append((name, cum, int(cum[-1])))
        log(f"FIG12 {name}: cumulative TX end={int(cum[-1])}")

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    hours = np.arange(length) * 0.25
    for name, cum, total in policies:
        ax.plot(hours, cum, color=COLORS[name], lw=1.6, label=f"{name} (total={total})")
    # annotate reduction vs fixed15
    tot15 = policies[0][2]
    totkl = policies[2][2]
    red = 100 * (1 - totkl / tot15) if tot15 else float("nan")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Cumulative transmissions (all 40 sensors)")
    ax.set_title(f"Validation week · KL-CMAPPO reduces {red:.1f}% of Fixed-15 uplinks")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save(fig, "fig12_cumulative_transmissions")


# ---------------------------------------------------------------------------
# Supplementary
# ---------------------------------------------------------------------------
def figS01() -> None:
    apply_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), sharex=True)
    for s in SEEDS:
        v = val_rows(s)
        axes[0].plot(v["step"], v["val_mae"], alpha=0.35, color=COLORS["KL-CMAPPO"])
        axes[1].plot(v["step"], 100 * v["val_recall"], alpha=0.35, color=COLORS["KL-CMAPPO"])
        axes[2].plot(v["step"], v["val_tx_reduction"], alpha=0.35, color=COLORS["KL-CMAPPO"])
    g = aggregate_val_curves()
    axes[0].plot(g["step"], g["val_mae_mean"], color=COLORS["KL-CMAPPO"], lw=2.2, label="5-seed mean")
    axes[1].plot(g["step"], 100 * g["val_recall_mean"], color=COLORS["KL-CMAPPO"], lw=2.2)
    axes[2].plot(g["step"], g["val_tx_mean"], color=COLORS["KL-CMAPPO"], lw=2.2)
    axes[0].axhline(9.0, ls="--", color="#238B45", lw=1)
    axes[1].axhline(98.5, ls="--", color="#238B45", lw=1)
    axes[0].set_ylabel("Val MAE (ppm)")
    axes[1].set_ylabel("Val recall (%)")
    axes[2].set_ylabel("Val TX↓ (%)")
    for ax, lab in zip(axes, ["(a)", "(b)", "(c)"]):
        ax.set_xlabel("Training step")
        panel_label(ax, lab)
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Supplementary: individual seed validation trajectories", fontsize=10)
    fig.tight_layout()
    save(fig, "figS01_seed_training_dynamics", SUPP)


def figS02() -> None:
    apply_paper_style()
    rob = load_robust()
    methods = [("semantic_expert", "Semantic expert"), ("campus_senserl_bc", "BC policy"), ("cmappo_kl", "KL-CMAPPO")]
    conds = ["packet_loss", "sensor_outage", "temp_block_outage", "neighbour_edge_drop"]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))
    for ax, cond in zip(axes.ravel(), conds):
        sub = rob[rob["condition"] == cond]
        for m, label in methods:
            g = sub[sub["method"] == m].sort_values("level")
            if g.empty:
                continue
            yerr = g["mae_std"].fillna(0) if int(g["n"].iloc[0]) > 1 else None
            ax.errorbar(g["level"], g["mae_mean"], yerr=yerr, fmt="-o", color=COLORS[label], label=label, capsize=3, markersize=5)
        ax.set_title(cond.replace("_", " "))
        ax.set_xlabel("Stress level")
        ax.set_ylabel("MAE (ppm)")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Supplementary: full robustness conditions · validation", fontsize=10)
    fig.tight_layout()
    save(fig, "figS02_full_robustness", SUPP)


def figS04() -> None:
    apply_paper_style()
    fair = pd.read_csv(ROOT / "outputs" / "reconstruction_final" / "fair_benchmark" / "fair_results_summary.csv")
    mae = fair[fair["metric"] == "mae"].sort_values("mean")
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.barh(mae["method"], mae["mean"], xerr=mae["std"], color="#6BAED6", ecolor="#333", capsize=3)
    ax.set_xlabel("MAE (ppm)")
    ax.set_title("Supplementary: fair reconstruction benchmark (causal mask)")
    fig.tight_layout()
    save(fig, "figS04_reconstruction_benchmark", SUPP)
    log("FIGS04 source fair_results_summary.csv metric=mae")


# ---------------------------------------------------------------------------
# Docs
# ---------------------------------------------------------------------------
def write_docs() -> None:
    index = OUT / "Figure_Index.md"
    index.write_text(
        """# Final clear figures — index

All figures use frozen artifacts only (no retraining). Method names are manuscript display names.

| File | MS # | Scientific question | Sources | Metrics | Main conclusion |
| --- | --- | --- | --- | --- | --- |
| `fig01_campus_deployment` | 1 | Real distributed campus evaluation? | `devices.json`, `final_cohort.json`, `heldout_cohort.json` | sensor counts / coordinates | Developed on 40 sensors; transferred to 40 held-out sensors. |
| `fig02_fixed_vs_adaptive_timeline` | 2 | How does adaptive TX differ from Fixed-15 on real CO₂? | frozen KL-CMAPPO seed 123 rollout + FixedInterval(1), val | interval TX counts, CO₂, recon | KL-CMAPPO suppresses stable uplinks and transmits during change. |
| `fig03_method_workflow` | 3 | What is the method? | schematic | — | Expert→BC→KL-CMAPPO (KL-CMAPPO) with constraints + KL anchor. |
| `fig04_overall_results` | 4 | How do policies compare? | `paper_final/full_val_summary.csv` | TX↓, MAE (±std) | KL-CMAPPO ~78% TX↓ with much lower MAE than Fixed-60. |
| `fig05_matched_budget` | 5 | Is RL gain only from sending more? | `matched_budget_summary.csv` | MAE, recall at 75/78/80% | KL-CMAPPO still improves BC at matched budgets. |
| `fig06_rl_training_dynamics` | 6 | Do constraints become feasible during training? | `cmappo_kl/seed_*/metrics.json` val rows | val MAE/recall/TX↓, #feasible seeds | Constraints are reached across seeds while keeping large TX savings. |
| `fig07_kl_constraint_dynamics` | 7 | How do KL β and duals evolve? | same metrics train rows | λ_event, λ_MAE, λ_AoI, β | Duals adapt; β anneals from strong BC anchor to more RL freedom. |
| `fig08_kl_constraint_ablation` | 8 | Are KL and constraints complementary? | `ablation_kl_cmappo_val.csv` (**seed 42**) | TX↓, MAE, recall | Mechanistic: constraints protect quality; KL preserves BC-like TX. |
| `fig09_transmission_heatmap` | 9 | Is communication sensor- and time-specific? | val rollouts Fixed-15 vs KL-CMAPPO | TX/SKIP/missing map | KL-CMAPPO varies decisions across sensors/time (not a new fixed period). |
| `fig10_heldout_generalization` | 10 | Does the policy transfer? | `heldout_transfer_summary.csv` **test** | MAE, recall (±std) | Parameter-shared actor transfers to unseen campus sensors. |
| `fig11_packet_loss_robustness` | 11 | Is degradation graceful under loss? | `robustness_summary.csv` packet_loss | MAE, recall | Gradual degradation; KL-CMAPPO remains competitive. |
| `fig12_cumulative_transmissions` | 12 (opt.) | How do savings accumulate? | val week rollouts | cumulative TX | Continuous accumulation of uplink savings vs Fixed-15/60. |

## Supplementary
- `supplementary/figS01_seed_training_dynamics` — individual seeds + mean
- `supplementary/figS02_full_robustness` — outage / edge-drop conditions
- `supplementary/figS04_reconstruction_benchmark` — fair recon MAE (appendix)

## Draft captions (short)
1. University of Oulu campus IoT deployment with RL development and held-out transfer cohorts.
2. Same real CO₂ window under Fixed 15 min vs KL-CMAPPO KL-CMAPPO (validation).
3. CAMPUS-SenseRL workflow: semantic expert, BC initialization, KL-CMAPPO fine-tuning (KL-CMAPPO).
4. Validation transmission reduction and skipped-slot MAE across policies (mean±std).
5. Matched-budget comparison of BC and KL-CMAPPO at 75/78/80% TX reduction (validation, 5 seeds).
6. Five-seed KL-CMAPPO validation dynamics and constraint feasibility during training.
7. Adaptive dual variables and KL coefficient schedule (regularization coefficient β, not measured KL).
8. Mechanistic KL/constraint ablation on validation (seed 42).
9. 24 h campus-wide TX/SKIP heatmap: Fixed 15 vs KL-CMAPPO (validation).
10. Held-out sensor test transfer: MAE and event recall (5 seeds).
11. Packet-loss robustness of Expert, BC, and KL-CMAPPO (validation).
12. Cumulative transmissions over one validation week.
""",
        encoding="utf-8",
    )

    audit = OUT / "figure_data_audit.md"
    audit.write_text(
        "# Figure data audit\n\n"
        + "Generated by `scripts/24_final_clear_figures.py`.\n\n"
        + "## Logged source values\n\n```\n"
        + "\n".join(AUDIT)
        + "\n```\n",
        encoding="utf-8",
    )

    review = ROOT / "reports" / "final_figure_review.md"
    review.write_text(
        """# Final figure review

No training was performed. All numerical figures read frozen CSVs/JSONs under `outputs/rl_final/`.

## fig01_campus_deployment — MAIN
1. Real distributed campus evaluation?
2. Grey campus map + colored RL/held-out cohorts; bar counts.
3. `devices.json`, cohort JSONs.
4. Main.
5. Spatial view is coordinates, not a building floorplan.

## fig02_fixed_vs_adaptive_timeline — MAIN
1. How adaptive TX differs from Fixed-15 on real CO₂.
2. Dense Fixed-15 markers vs sparse adaptive TX with recon on skips; annotated counts.
3. Frozen KL-CMAPPO seed 123 rollout on validation (see `fig02_meta.json`).
4. Main.
5. One representative window (deterministic selection); not a statistical summary.

## fig03_method_workflow — MAIN
1. What is the method?
2. Expert→BC→KL-CMAPPO; KL-CMAPPO; KL as regularization.
3. Schematic only.
4. Main.
5. Not an empirical result plot.

## fig04_overall_results — MAIN
1. Overall policy comparison.
2. KL-CMAPPO ~78% TX↓ with lower MAE than Fixed-60; expert best MAE.
3. `full_val_summary.csv` (validation).
4. Main.
5. Fixed-15 omitted from MAE panel (no skips). Event recall left to tables for clarity.

## fig05_matched_budget — MAIN
1. Is the RL gain only from more TX?
2. At 75/78/80%, KL-CMAPPO MAE ≤ BC.
3. `matched_budget_summary.csv` (validation, 5 seeds).
4. Main.
5. Threshold matching approximates budgets; not hard rate constraints.

## fig06_rl_training_dynamics — MAIN
1. Do constraints become feasible while keeping savings?
2. MAE/recall enter feasible bands; feasible-seed count rises; TX↓ remains high.
3. `cmappo_kl/seed_*/metrics.json` validation rows only.
4. Main.
5. Validation every ~8k steps; best checkpoints may occur mid-training.

## fig07_kl_constraint_dynamics — MAIN/METHOD
1. How duals and β evolve?
2. λ's adapt; β decreases (stronger early BC anchoring).
3. Training log fields `lam_*`, `kl_beta`.
4. Main or method/supplement if space-limited.
5. β is a coefficient — **not** measured KL divergence.

## fig08_kl_constraint_ablation — MAIN
1. Are KL and constraints complementary?
2. Constraints-only: best MAE, lower TX↓; full KL-CMAPPO balances.
3. `ablation_kl_cmappo_val.csv` (**seed 42 only**).
4. Main (label as mechanistic single-seed).
5. Not multi-seed significance; Old MAPPO baselines removed.

## fig09_transmission_heatmap — MAIN
1. Is communication sensor-/time-specific?
2. Dense Fixed-15 grid vs structured KL-CMAPPO TX/SKIP pattern.
3. Validation rollouts (meta JSON).
4. Main (or supplement if page limit).
5. 24 h window; natural missing masked separately from SKIP.

## fig10_heldout_generalization — MAIN
1. Transfer to unseen sensors?
2. KL-CMAPPO improves MAE vs BC on held-out **test**.
3. `heldout_transfer_summary.csv`.
4. Main.
5. Temporal test for held-out sensors; development sensors excluded by construction.

## fig11_packet_loss_robustness — MAIN
1. Graceful degradation under loss?
2. MAE rises / recall falls smoothly; ordering preserved.
3. `robustness_summary.csv` packet_loss.
4. Main.
5. Expert has n=1 (no fake error bars). Other stress tests → S2.

## fig12_cumulative_transmissions — OPTIONAL MAIN
1. How savings accumulate?
2. Diverging cumulative TX curves.
3. Validation week rollouts.
4. Optional.
5. Transmission counts only — **not** battery lifetime.

## Supplementary
- S1: reproducibility of training trajectories.
- S2: broader robustness conditions.
- S3: skipped (no existing per-sensor error raw table without new inference).
- S4: reconstruction appendix (fair benchmark).
""",
        encoding="utf-8",
    )
    log(f"WROTE {index}")
    log(f"WROTE {audit}")
    log(f"WROTE {review}")


def main() -> None:
    ensure_dir(OUT)
    ensure_dir(SUPP)
    apply_paper_style()
    log("=== FINAL CLEAR FIGURE GENERATION (no training) ===")
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
    fig11()
    fig12()
    figS01()
    figS02()
    figS04()
    write_docs()
    save_json({"dpi": DPI, "no_training": True, "out": str(OUT)}, OUT / "meta.json")
    print("[done]", OUT)


if __name__ == "__main__":
    main()
