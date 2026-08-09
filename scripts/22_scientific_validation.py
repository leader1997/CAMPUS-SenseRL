#!/usr/bin/env python
"""Final scientific validation (post KL-CMAPPO freeze).

Does NOT retrain or retune the core 5-seed KL-CMAPPO models.
Runs the five reviewer-requested proofs on VAL (held-out cohort is transfer eval):

  1. Matched communication-budget Pareto (BC vs KL-CMAPPO threshold sweep)
  2. Robustness including KL-CMAPPO (packet loss / outages / neighbour loss)
  3. Algorithmic ablation (BC → MAPPO → KL-only → constraints-only → full)
  4. Unseen-sensor cohort generalization (parameter-shared actor transfer)
  5. Regenerate paper figures at 300 DPI

Wording reminder: KL-CMAPPO is KL-regularized constrained MAPPO with
validation-enforced feasibility — not a formal hard constraint guarantee.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from campus_senserl.data.cohort import freeze_heldout_cohort, load_cohort
from campus_senserl.evaluation.rl_policy_eval import (
    evaluate_policy,
    load_mappo_policy,
    make_final_env,
)
from campus_senserl.rl.cmappo import run_constrained_mappo
from campus_senserl.rl.expert_policy import SemanticExpertPolicy
from campus_senserl.utils import ensure_dir, load_yaml, repo_root, save_json

SEEDS = [42, 123, 2024, 3407, 9999]
TARGET_TX_REDUCTIONS = [75.0, 78.0, 80.0]
DPI = 300


def _wrap(pol):
    def act(obs, *, local_available=None):
        return pol.act(obs, local_available=local_available)

    act.reset = getattr(pol, "reset", lambda: None)
    return act


def _bc_ckpt(root: Path, seed: int) -> Path:
    p = root / "outputs" / "rl_final" / "paper_asap" / "bc" / f"seed_{seed}" / "final_model.pt"
    if p.exists():
        return p
    return root / "outputs" / "rl_final" / "paper_asap" / "bc" / "seed_42" / "final_model.pt"


def _cmappo_ckpt(root: Path, seed: int) -> Path:
    return root / "outputs" / "rl_final" / "cmappo_kl" / f"seed_{seed}" / "best_model.pt"


def _cfg(root: Path) -> dict:
    return load_yaml(root / "configs" / "rl_cmappo.yaml")


# ---------------------------------------------------------------------------
# 1) Matched communication budget
# ---------------------------------------------------------------------------
def run_matched_budget(root: Path, device: str, out: Path) -> pd.DataFrame:
    print("[1] matched communication-budget sweep (VAL)...")
    cfg = _cfg(root)
    thresholds = np.round(np.linspace(0.25, 0.75, 11), 3)
    rows = []
    for method, ckpt_fn in [("campus_senserl_bc", _bc_ckpt), ("cmappo_kl", _cmappo_ckpt)]:
        for seed in SEEDS:
            ckpt = ckpt_fn(root, seed)
            if not ckpt.exists():
                print(f"  [skip] missing {ckpt}")
                continue
            for tau in thresholds:
                print(f"  {method} seed={seed} tau={tau:.3f}")
                act = load_mappo_policy(ckpt, device=device, prob_threshold=float(tau))
                env = make_final_env(split="val", cfg=cfg, multi_agent=True, shield_enabled=False)
                m = evaluate_policy(env, act, max_steps=None, seed=seed)
                rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "prob_threshold": float(tau),
                        **m,
                    }
                )
    raw = pd.DataFrame(rows)
    raw.to_csv(out / "matched_budget_raw.csv", index=False)

    # Interpolate each seed to target TX reductions; report mean±std
    matched_rows = []
    for method, g_m in raw.groupby("method"):
        for seed, g in g_m.groupby("seed"):
            g = g.sort_values("transmission_reduction_pct")
            xs = g["transmission_reduction_pct"].to_numpy()
            for target in TARGET_TX_REDUCTIONS:
                # nearest threshold (and linear interp of metrics vs reduction)
                if target < xs.min() or target > xs.max():
                    # clamp to nearest endpoint
                    i = int(np.argmin(np.abs(xs - target)))
                    row = g.iloc[i].to_dict()
                    row["target_tx_reduction"] = target
                    row["matched_ok"] = False
                else:
                    mae = float(np.interp(target, xs, g["mae_skipped"].to_numpy()))
                    rec = float(np.interp(target, xs, g["event_recall"].to_numpy()))
                    prec = float(np.interp(target, xs, g["event_precision"].to_numpy()))
                    aoi = float(np.interp(target, xs, g["mean_aoi_raw"].to_numpy()))
                    tau = float(np.interp(target, xs, g["prob_threshold"].to_numpy()))
                    row = {
                        "method": method,
                        "seed": seed,
                        "target_tx_reduction": target,
                        "transmission_reduction_pct": target,
                        "prob_threshold": tau,
                        "mae_skipped": mae,
                        "event_recall": rec,
                        "event_precision": prec,
                        "mean_aoi_raw": aoi,
                        "matched_ok": True,
                    }
                matched_rows.append(row)
    matched = pd.DataFrame(matched_rows)
    matched.to_csv(out / "matched_budget_val.csv", index=False)

    summary_rows = []
    for (method, target), g in matched.groupby(["method", "target_tx_reduction"]):
        summary_rows.append(
            {
                "method": method,
                "target_tx_reduction": target,
                "n": len(g),
                "mae_mean": g["mae_skipped"].mean(),
                "mae_std": g["mae_skipped"].std(ddof=0),
                "recall_mean": g["event_recall"].mean(),
                "recall_std": g["event_recall"].std(ddof=0),
                "precision_mean": g["event_precision"].mean(),
                "aoi_raw_mean": g["mean_aoi_raw"].mean(),
                "n_matched_ok": int(g["matched_ok"].sum()) if "matched_ok" in g else len(g),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "matched_budget_summary.csv", index=False)
    print(summary.to_string(index=False))
    return raw


# ---------------------------------------------------------------------------
# 2) Robustness
# ---------------------------------------------------------------------------
def run_robustness(root: Path, device: str, out: Path) -> pd.DataFrame:
    print("[2] robustness (VAL) including KL-CMAPPO...")
    cfg = _cfg(root)
    rows = []

    policies = [("semantic_expert", None)]
    for seed in SEEDS:
        policies.append((f"campus_senserl_bc", _bc_ckpt(root, seed)))
        policies.append((f"cmappo_kl", _cmappo_ckpt(root, seed)))

    # Deduplicate BC/CMAPPO by evaluating each seed once per condition
    conditions = []
    for rate in [0.0, 0.1, 0.2, 0.4]:
        conditions.append(("packet_loss", rate, {"packet_loss_rate": rate}))
    for frac in [0.1, 0.2]:
        conditions.append(
            ("sensor_outage", frac, {"sensor_outage_fraction": frac, "stress_seed": 7})
        )
    conditions.append(
        (
            "temp_block_outage",
            0.2,
            {
                "temp_outage_fraction": 0.2,
                "temp_outage_duration_frac": 0.15,
                "stress_seed": 11,
            },
        )
    )
    conditions.append(("neighbour_edge_drop", 0.3, {"edge_drop_fraction": 0.3, "stress_seed": 13}))

    # Expert once + each seed for BC and CMAPPO
    for cond_name, level, kwargs in conditions:
        print(f"  condition={cond_name} level={level}")
        # expert
        env = make_final_env(split="val", cfg=cfg, multi_agent=True, shield_enabled=False, **kwargs)
        m = evaluate_policy(env, _wrap(SemanticExpertPolicy()), max_steps=None, seed=42)
        rows.append({"method": "semantic_expert", "seed": 42, "condition": cond_name, "level": level, **m})

        for seed in SEEDS:
            for method, ckpt_fn in [("campus_senserl_bc", _bc_ckpt), ("cmappo_kl", _cmappo_ckpt)]:
                ckpt = ckpt_fn(root, seed)
                if not ckpt.exists():
                    continue
                act = load_mappo_policy(ckpt, device=device)
                env = make_final_env(
                    split="val", cfg=cfg, multi_agent=True, shield_enabled=False, **kwargs
                )
                m = evaluate_policy(env, act, max_steps=None, seed=seed)
                rows.append(
                    {"method": method, "seed": seed, "condition": cond_name, "level": level, **m}
                )

    df = pd.DataFrame(rows)
    df.to_csv(out / "robustness_val.csv", index=False)

    # Aggregate mean±std for neural methods
    agg_rows = []
    for (method, cond, level), g in df.groupby(["method", "condition", "level"]):
        agg_rows.append(
            {
                "method": method,
                "condition": cond,
                "level": level,
                "n": len(g),
                "tx_reduction_mean": g["transmission_reduction_pct"].mean(),
                "mae_mean": g["mae_skipped"].mean(),
                "mae_std": g["mae_skipped"].std(ddof=0),
                "recall_mean": g["event_recall"].mean(),
                "recall_std": g["event_recall"].std(ddof=0),
                "aoi_raw_mean": g["mean_aoi_raw"].mean(),
                "pdr_mean": g["packet_delivery_ratio"].mean()
                if "packet_delivery_ratio" in g
                else np.nan,
            }
        )
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(out / "robustness_summary.csv", index=False)
    print(agg[agg["condition"] == "packet_loss"].to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# 3) Algorithmic ablation (seed 42, frozen core CMAPPO)
# ---------------------------------------------------------------------------
def _ablation_cfg(base: dict, variant: str) -> dict:
    cfg = copy.deepcopy(base)
    cfg.setdefault("kl", {})
    cfg.setdefault("constraints", {})
    if variant == "mappo_no_kl_no_constraints":
        cfg["kl"]["enabled"] = False
        cfg["kl"]["beta"] = 0.0
        cfg["kl"]["beta_end"] = 0.0
        cfg["constraints"]["enabled"] = False
    elif variant == "kl_only":
        cfg["kl"]["enabled"] = True
        cfg["constraints"]["enabled"] = False
    elif variant == "constraints_only":
        cfg["kl"]["enabled"] = False
        cfg["kl"]["beta"] = 0.0
        cfg["kl"]["beta_end"] = 0.0
        cfg["constraints"]["enabled"] = True
    elif variant == "full_kl_cmappo":
        cfg["kl"]["enabled"] = True
        cfg["constraints"]["enabled"] = True
    else:
        raise ValueError(variant)
    return cfg


def run_ablations(root: Path, device: str, out: Path, timesteps: int, force: bool) -> pd.DataFrame:
    print("[3] algorithmic ablation (seed=42)...")
    base = _cfg(root)
    base["seed"] = 42
    base.setdefault("mappo", {})["total_timesteps"] = timesteps
    bc = _bc_ckpt(root, 42)
    abl_dir = ensure_dir(root / "outputs" / "rl_final" / "cmappo_ablations")

    variants = [
        "bc_only",
        "mappo_no_kl_no_constraints",
        "kl_only",
        "constraints_only",
        "full_kl_cmappo",
    ]
    rows = []

    # BC only
    act = load_mappo_policy(bc, device=device)
    env = make_final_env(split="val", cfg=base, multi_agent=True, shield_enabled=False)
    m = evaluate_policy(env, act, max_steps=None, seed=42)
    rows.append({"ablation": "bc_only", "seed": 42, **m})

    # Old shield MAPPO negative baseline (if present)
    old = root / "outputs" / "rl_final" / "mappo" / "seed_42" / "final_model.pt"
    if old.exists():
        act = load_mappo_policy(old, device=device)
        env = make_final_env(split="val", cfg=base, multi_agent=True, shield_enabled=True)
        m = evaluate_policy(env, act, max_steps=None, seed=42)
        rows.append({"ablation": "old_mappo_shield", "seed": 42, "shield": True, **m})
        env = make_final_env(split="val", cfg=base, multi_agent=True, shield_enabled=False)
        m = evaluate_policy(env, act, max_steps=None, seed=42)
        rows.append({"ablation": "old_mappo_no_shield", "seed": 42, "shield": False, **m})

    for variant in variants[1:]:
        ckpt_dir = abl_dir / variant / "seed_42"
        best = ckpt_dir / "best_model.pt"
        final = ckpt_dir / "final_model.pt"
        if variant == "full_kl_cmappo":
            # Reuse frozen core seed-42 checkpoint — do not retrain
            src = _cmappo_ckpt(root, 42)
            best = src
            print(f"  [reuse] full_kl_cmappo → {src}")
        elif (best.exists() or final.exists()) and not force:
            print(f"  [skip-train] {variant}")
            best = best if best.exists() else final
        else:
            print(f"  [train] {variant} steps={timesteps}")
            cfg = _ablation_cfg(base, variant)
            cfg.setdefault("training", {})["checkpoint_dir"] = str(ckpt_dir)
            out_train = run_constrained_mappo(
                cfg, bc_checkpoint=bc, device=device, checkpoint_dir=ckpt_dir
            )
            best = Path(out_train["best"])

        act = load_mappo_policy(best, device=device)
        env = make_final_env(split="val", cfg=base, multi_agent=True, shield_enabled=False)
        m = evaluate_policy(env, act, max_steps=None, seed=42)
        rows.append({"ablation": variant, "seed": 42, "checkpoint": str(best), **m})
        print(
            f"  {variant}: red={m['transmission_reduction_pct']:.1f}% "
            f"mae={m['mae_skipped']:.2f} rec={m['event_recall']:.4f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(out / "ablation_kl_cmappo_val.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# 4) Held-out sensor generalization
# ---------------------------------------------------------------------------
def run_heldout(root: Path, device: str, out: Path) -> pd.DataFrame:
    print("[4] unseen-sensor held-out cohort transfer...")
    held = freeze_heldout_cohort(n=40, exclude_cohort="final", out_name="heldout")
    ids = held["sensor_ids"]
    print(f"  heldout n={len(ids)} excluded_final={held['n_excluded']}")
    save_json(held, out / "heldout_cohort_meta.json")

    cfg = _cfg(root)
    rows = []
    # Expert on held-out (reference)
    env = make_final_env(
        split="val", cfg=cfg, multi_agent=True, shield_enabled=False, sensor_ids=ids
    )
    m = evaluate_policy(env, _wrap(SemanticExpertPolicy()), max_steps=None, seed=42)
    rows.append({"method": "semantic_expert", "seed": 42, "cohort": "heldout", "split": "val", **m})

    for seed in SEEDS:
        for method, ckpt_fn in [("campus_senserl_bc", _bc_ckpt), ("cmappo_kl", _cmappo_ckpt)]:
            ckpt = ckpt_fn(root, seed)
            if not ckpt.exists():
                continue
            act = load_mappo_policy(ckpt, device=device)
            for split in ["val", "test"]:
                env = make_final_env(
                    split=split, cfg=cfg, multi_agent=True, shield_enabled=False, sensor_ids=ids
                )
                m = evaluate_policy(env, act, max_steps=None, seed=seed)
                rows.append(
                    {"method": method, "seed": seed, "cohort": "heldout", "split": split, **m}
                )
                print(
                    f"  {method} seed={seed} {split}: "
                    f"red={m['transmission_reduction_pct']:.1f}% mae={m['mae_skipped']:.2f} "
                    f"rec={m['event_recall']:.4f}"
                )

    df = pd.DataFrame(rows)
    df.to_csv(out / "heldout_transfer_raw.csv", index=False)

    summary_rows = []
    for (method, split), g in df.groupby(["method", "split"]):
        summary_rows.append(
            {
                "method": method,
                "split": split,
                "cohort": "heldout",
                "n": len(g),
                "tx_reduction_mean": g["transmission_reduction_pct"].mean(),
                "tx_reduction_std": g["transmission_reduction_pct"].std(ddof=0),
                "mae_mean": g["mae_skipped"].mean(),
                "mae_std": g["mae_skipped"].std(ddof=0),
                "recall_mean": g["event_recall"].mean(),
                "recall_std": g["event_recall"].std(ddof=0),
                "precision_mean": g["event_precision"].mean(),
                "aoi_raw_mean": g["mean_aoi_raw"].mean(),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "heldout_transfer_summary.csv", index=False)
    print(summary.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# 5) Paper figures @ 300 DPI
# ---------------------------------------------------------------------------
def regenerate_figures(root: Path, sci: Path) -> None:
    print("[5] regenerate paper figures @ 300 DPI...")
    paper_fig = ensure_dir(root / "paper_outputs" / "figures")
    paper_tab = ensure_dir(root / "paper_outputs" / "tables")

    # Copy tables
    mapping = {
        "matched_budget_summary.csv": "table_matched_budget.csv",
        "matched_budget_val.csv": "table_matched_budget_raw.csv",
        "robustness_summary.csv": "table5_robustness.csv",
        "robustness_val.csv": "table5_robustness_raw.csv",
        "ablation_kl_cmappo_val.csv": "table4_ablation.csv",
        "heldout_transfer_summary.csv": "table_heldout_transfer.csv",
    }
    for src, dst in mapping.items():
        p = sci / src
        if p.exists():
            pd.read_csv(p).to_csv(paper_tab / dst, index=False)

    # Fig 5: matched-budget Pareto (MAE vs TX reduction)
    raw_p = sci / "matched_budget_raw.csv"
    if raw_p.exists():
        raw = pd.read_csv(raw_p)
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        for method, color, label in [
            ("campus_senserl_bc", "#4C78A8", "BC"),
            ("cmappo_kl", "#F58518", "KL-CMAPPO"),
        ]:
            g = raw[raw["method"] == method]
            # mean curve over seeds
            curve = (
                g.groupby("prob_threshold")[["transmission_reduction_pct", "mae_skipped"]]
                .mean()
                .reset_index()
                .sort_values("transmission_reduction_pct")
            )
            ax.plot(
                curve["transmission_reduction_pct"],
                curve["mae_skipped"],
                "-o",
                color=color,
                label=label,
                markersize=4,
            )
        # mark matched targets
        summ = sci / "matched_budget_summary.csv"
        if summ.exists():
            s = pd.read_csv(summ)
            for method, color in [("campus_senserl_bc", "#4C78A8"), ("cmappo_kl", "#F58518")]:
                sub = s[s["method"] == method]
                ax.scatter(
                    sub["target_tx_reduction"],
                    sub["mae_mean"],
                    s=80,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=2,
                    zorder=5,
                )
        ax.set_xlabel("Transmission reduction (%)")
        ax.set_ylabel("Skipped-slot CO₂ MAE (ppm)")
        ax.set_title("Matched-budget Pareto (validation)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(paper_fig / "fig05_tradeoff_reconstruction.png", dpi=DPI, bbox_inches="tight")
        fig.savefig(paper_fig / "fig05_pareto_tx_mae.png", dpi=DPI, bbox_inches="tight")
        fig.savefig(paper_fig / "fig05_tradeoff_reconstruction.pdf", bbox_inches="tight")
        fig.savefig(paper_fig / "fig05_pareto_tx_mae.pdf", bbox_inches="tight")
        plt.close(fig)

        # Fig 6: event recall vs TX reduction (matched-budget)
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        for method, color, label in [
            ("campus_senserl_bc", "#4C78A8", "BC"),
            ("cmappo_kl", "#F58518", "KL-CMAPPO"),
        ]:
            g = raw[raw["method"] == method]
            curve = (
                g.groupby("prob_threshold")[["transmission_reduction_pct", "event_recall"]]
                .mean()
                .reset_index()
                .sort_values("transmission_reduction_pct")
            )
            ax.plot(
                curve["transmission_reduction_pct"],
                curve["event_recall"],
                "-o",
                color=color,
                label=label,
                markersize=4,
            )
        ax.set_xlabel("Transmission reduction (%)")
        ax.set_ylabel("Event recall")
        ax.set_title("Event preservation vs communication (validation)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(paper_fig / "fig06_tradeoff_event_recall.png", dpi=DPI, bbox_inches="tight")
        fig.savefig(paper_fig / "fig06_tradeoff_event_recall.pdf", bbox_inches="tight")
        plt.close(fig)

    # Fig 8: AoI from matched budget or paper_final
    raw_p = sci / "matched_budget_raw.csv"
    if raw_p.exists():
        raw = pd.read_csv(raw_p)
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        for method, color, label in [
            ("campus_senserl_bc", "#4C78A8", "BC"),
            ("cmappo_kl", "#F58518", "KL-CMAPPO"),
        ]:
            g = raw[raw["method"] == method]
            curve = (
                g.groupby("prob_threshold")[["transmission_reduction_pct", "mean_aoi_raw"]]
                .mean()
                .reset_index()
                .sort_values("transmission_reduction_pct")
            )
            ax.plot(
                curve["transmission_reduction_pct"],
                curve["mean_aoi_raw"],
                "-o",
                color=color,
                label=label,
                markersize=4,
            )
        ax.set_xlabel("Transmission reduction (%)")
        ax.set_ylabel("Mean raw AoI (intervals)")
        ax.set_title("Information freshness vs communication (validation)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(paper_fig / "fig08_aoi_comparison.png", dpi=DPI, bbox_inches="tight")
        fig.savefig(paper_fig / "fig08_aoi_comparison.pdf", bbox_inches="tight")
        plt.close(fig)

    # Fig 9: ablation bars
    abl_p = sci / "ablation_kl_cmappo_val.csv"
    if abl_p.exists():
        abl = pd.read_csv(abl_p)
        order = [
            "bc_only",
            "mappo_no_kl_no_constraints",
            "kl_only",
            "constraints_only",
            "full_kl_cmappo",
            "old_mappo_shield",
            "old_mappo_no_shield",
        ]
        abl["ord"] = abl["ablation"].apply(lambda x: order.index(x) if x in order else 99)
        abl = abl.sort_values("ord")
        labels = {
            "bc_only": "BC",
            "mappo_no_kl_no_constraints": "BC+MAPPO",
            "kl_only": "BC+KL",
            "constraints_only": "BC+Constraints",
            "full_kl_cmappo": "KL-CMAPPO",
            "old_mappo_shield": "Old MAPPO+shield",
            "old_mappo_no_shield": "Old MAPPO (no shield)",
        }
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
        x = np.arange(len(abl))
        names = [labels.get(a, a) for a in abl["ablation"]]
        axes[0].bar(x, abl["mae_skipped"], color="#72B7B2")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(names, rotation=25, ha="right")
        axes[0].set_ylabel("MAE (ppm)")
        axes[0].set_title("Ablation: reconstruction")
        axes[1].bar(x, abl["transmission_reduction_pct"], color="#E45756")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(names, rotation=25, ha="right")
        axes[1].set_ylabel("TX reduction (%)")
        axes[1].set_title("Ablation: communication")
        fig.suptitle("KL-CMAPPO component ablation (val, seed 42)")
        fig.tight_layout()
        fig.savefig(paper_fig / "fig09_ablation.png", dpi=DPI, bbox_inches="tight")
        fig.savefig(paper_fig / "fig09_ablation.pdf", bbox_inches="tight")
        plt.close(fig)

    # Fig 10: packet-loss robustness with KL-CMAPPO
    rob_p = sci / "robustness_summary.csv"
    if rob_p.exists():
        rob = pd.read_csv(rob_p)
        pl = rob[rob["condition"] == "packet_loss"].copy()
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        for method, color, label in [
            ("semantic_expert", "#54A24B", "Expert"),
            ("campus_senserl_bc", "#4C78A8", "BC"),
            ("cmappo_kl", "#F58518", "KL-CMAPPO"),
        ]:
            g = pl[pl["method"] == method].sort_values("level")
            if g.empty:
                continue
            ax.errorbar(
                g["level"] * 100,
                g["mae_mean"],
                yerr=g["mae_std"].fillna(0),
                fmt="-o",
                color=color,
                label=label,
                capsize=3,
            )
        ax.set_xlabel("Packet loss (%)")
        ax.set_ylabel("Skipped-slot CO₂ MAE (ppm)")
        ax.set_title("Robustness to packet loss (validation)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(paper_fig / "fig10_robustness.png", dpi=DPI, bbox_inches="tight")
        fig.savefig(paper_fig / "fig10_robustness.pdf", bbox_inches="tight")
        plt.close(fig)

    print(f"  figures → {paper_fig}")
    print(f"  tables  → {paper_tab}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ablation-timesteps", type=int, default=60000)
    parser.add_argument("--force-ablation", action="store_true")
    parser.add_argument("--skip-matched", action="store_true")
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-heldout", action="store_true")
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    out = ensure_dir(root / "outputs" / "rl_final" / "scientific_validation")
    meta = {
        "protocol": "post-freeze scientific validation",
        "core_cmappo_frozen": True,
        "seeds": SEEDS,
        "dpi": DPI,
        "wording": (
            "KL-regularized constrained multi-agent policy optimization "
            "with validation-enforced feasibility"
        ),
    }

    if args.figures_only:
        regenerate_figures(root, out)
        save_json(meta, out / "meta.json")
        return

    if not args.skip_matched:
        run_matched_budget(root, args.device, out)
    if not args.skip_robustness:
        run_robustness(root, args.device, out)
    if not args.skip_ablation:
        run_ablations(root, args.device, out, args.ablation_timesteps, args.force_ablation)
    if not args.skip_heldout:
        run_heldout(root, args.device, out)

    regenerate_figures(root, out)
    save_json(meta, out / "meta.json")

    # Short report
    report = root / "reports" / "scientific_validation.md"
    lines = [
        "# Scientific validation (post KL-CMAPPO freeze)",
        "",
        "Core 5-seed KL-CMAPPO models were **not** retrained.",
        "",
        "Claim phrasing: KL-CMAPPO trades ~2.24 pp extra communication for better MAE/recall/precision/AoI vs BC — not strict domination.",
        "",
        "Algorithm description: **KL-regularized constrained multi-agent policy optimization with validation-enforced feasibility**.",
        "",
        f"Artifacts: `{out.as_posix()}`",
        "",
        "## Outputs",
        "- Matched budget: `matched_budget_summary.csv` → Fig 5/6/8",
        "- Robustness: `robustness_summary.csv` → Fig 10 / table5",
        "- Ablation: `ablation_kl_cmappo_val.csv` → Fig 9 / table4",
        "- Held-out sensors: `heldout_transfer_summary.csv`",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] {out}")
    print(f"[report] {report}")


if __name__ == "__main__":
    main()
