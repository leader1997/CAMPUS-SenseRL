#!/usr/bin/env python
"""Final paper results: 5-seed KL-CMAPPO, val mean±std, frozen test, figures.

Protocol (VAL selection only; single test pass after freeze):
  - Seeds: 42, 123, 2024, 3407, 9999
  - Timesteps: 60000 (KL fine-tune from matching BC seed when available)
  - Feasible-only checkpointing (recall/MAE/AoI constraints)
  - Shield OFF for proposed methods
"""

from __future__ import annotations

import argparse
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

from campus_senserl.evaluation.rl_policy_eval import (
    evaluate_policy,
    load_mappo_policy,
    make_final_env,
)
from campus_senserl.rl.cmappo import run_constrained_mappo
from campus_senserl.rl.expert_policy import (
    DeltaPlusHeartbeatPolicy,
    SemanticExpertPolicy,
)
from campus_senserl.rl.fixed_policies import FixedIntervalPolicy
from campus_senserl.utils import ensure_dir, load_yaml, repo_root, save_json
from campus_senserl.visualization.figures import plot_pareto_frontier

SEEDS = [42, 123, 2024, 3407, 9999]


def _wrap(pol):
    def act(obs, *, local_available=None):
        return pol.act(obs, local_available=local_available)

    act.reset = getattr(pol, "reset", lambda: None)
    return act


def _bc_path(root: Path, seed: int) -> Path:
    p = root / "outputs" / "rl_final" / "paper_asap" / "bc" / f"seed_{seed}" / "final_model.pt"
    if p.exists():
        return p
    # fallback
    return root / "outputs" / "rl_final" / "paper_asap" / "bc" / "seed_42" / "final_model.pt"


def train_seed(seed: int, timesteps: int, device: str, force: bool) -> Path:
    root = repo_root()
    cfg = load_yaml(root / "configs" / "rl_cmappo.yaml")
    cfg["seed"] = seed
    cfg.setdefault("mappo", {})["total_timesteps"] = timesteps
    ckpt_dir = root / "outputs" / "rl_final" / "cmappo_kl" / f"seed_{seed}"
    best = ckpt_dir / "best_model.pt"
    if best.exists() and not force:
        print(f"[skip-train] seed={seed} existing {best}")
        return best
    bc = _bc_path(root, seed)
    print(f"[train] seed={seed} bc={bc} steps={timesteps}")
    out = run_constrained_mappo(cfg, bc_checkpoint=bc, device=device, checkpoint_dir=ckpt_dir)
    return Path(out["best"])


def eval_named(name: str, act, *, split: str, cfg: dict, seed: int, shield: bool = False) -> dict:
    env = make_final_env(split=split, cfg=cfg, multi_agent=True, shield_enabled=shield)
    m = evaluate_policy(env, act, max_steps=None, seed=seed)
    m["method"] = name
    m["seed"] = seed
    m["shield"] = shield
    m["split"] = split
    return m


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, g in df.groupby("method"):
        row = {"method": method, "n": int(len(g))}
        for col, short in [
            ("transmission_reduction_pct", "tx_reduction"),
            ("mae_skipped", "mae"),
            ("event_recall", "recall"),
            ("event_precision", "precision"),
            ("event_f1", "f1"),
            ("mean_aoi_raw", "aoi_raw"),
            ("recall_rapid_rise", "recall_rapid"),
        ]:
            if col not in g.columns:
                continue
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            if len(vals) == 0:
                continue
            row[f"{short}_mean"] = float(vals.mean())
            row[f"{short}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def make_figures(df_val: pd.DataFrame, paper_fig: Path, rob_path: Path | None) -> None:
    # Bar tradeoff
    g = summarize(df_val)
    keep = [
        "fixed_30",
        "fixed_60",
        "delta_plus_heartbeat",
        "semantic_expert",
        "campus_senserl_bc",
        "cmappo_kl",
    ]
    g = g[g["method"].isin(keep)].copy()
    order = {m: i for i, m in enumerate(keep)}
    g["ord"] = g["method"].map(order)
    g = g.sort_values("ord")

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    x = np.arange(len(g))
    labels = g["method"].tolist()
    axes[0].bar(x, g["tx_reduction_mean"], yerr=g.get("tx_reduction_std", 0), color="#2c7fb8", capsize=3)
    axes[0].set_ylabel("TX reduction (%)")
    axes[1].bar(x, g["mae_mean"], yerr=g.get("mae_std", 0), color="#d95f0e", capsize=3)
    axes[1].set_ylabel("MAE skipped (ppm)")
    axes[2].bar(x, g["recall_mean"] * 100, yerr=g.get("recall_std", 0) * 100, color="#31a354", capsize=3)
    axes[2].set_ylabel("Event recall (%)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Validation comparison (mean ± std over seeds where applicable)")
    fig.tight_layout()
    fig.savefig(paper_fig / "fig05_tradeoff_reconstruction.png", dpi=200, bbox_inches="tight")
    fig.savefig(paper_fig / "fig05_tradeoff_reconstruction.pdf", bbox_inches="tight")
    plt.close(fig)

    # Pareto
    plot_df = df_val[df_val["method"].isin(keep)].copy()
    plot_df["transmit_rate"] = 1.0 - plot_df["transmission_reduction_pct"] / 100.0
    plot_df["mae"] = plot_df["mae_skipped"]
    # one point per method (mean)
    pts = plot_df.groupby("method", as_index=False).agg({"transmit_rate": "mean", "mae": "mean"})
    plot_pareto_frontier(
        pts,
        x_col="transmit_rate",
        y_col="mae",
        hue_col="method",
        out_path=paper_fig / "fig05_pareto_tx_mae.png",
        title="TX rate vs skip MAE (validation means)",
    )

    # Recall categories
    fig, ax = plt.subplots(figsize=(7.2, 4))
    pg = df_val[df_val["method"].isin(keep)].groupby("method", as_index=False)[
        ["event_recall", "recall_rapid_rise"]
    ].mean()
    pg["ord"] = pg["method"].map(order)
    pg = pg.sort_values("ord")
    x = np.arange(len(pg))
    w = 0.35
    ax.bar(x - w / 2, pg["event_recall"] * 100, w, label="Primary events", color="#3182bd")
    ax.bar(x + w / 2, pg["recall_rapid_rise"] * 100, w, label="Rapid-rise", color="#e6550d")
    ax.set_xticks(x)
    ax.set_xticklabels(pg["method"], rotation=25, ha="right")
    ax.set_ylabel("Recall (%)")
    ax.set_title("Event recall by category (validation)")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(paper_fig / "fig06_tradeoff_event_recall.png", dpi=200, bbox_inches="tight")
    fig.savefig(paper_fig / "fig06_tradeoff_event_recall.pdf", bbox_inches="tight")
    plt.close(fig)

    # AoI
    fig, ax = plt.subplots(figsize=(7, 4))
    ag = g.sort_values("ord")
    ax.bar(ag["method"], ag["aoi_raw_mean"], yerr=ag.get("aoi_raw_std", 0), color="#756bb1", capsize=3)
    ax.set_ylabel("Mean raw AoI (intervals)")
    ax.set_title("Unbounded Age of Information (validation)")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(paper_fig / "fig08_aoi_comparison.png", dpi=200, bbox_inches="tight")
    fig.savefig(paper_fig / "fig08_aoi_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    if rob_path and rob_path.exists():
        rdf = pd.read_csv(rob_path)
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for name, gg in rdf.groupby("method"):
            gg = gg.sort_values("level")
            ax.plot(gg["level"] * 100, gg["event_recall"] * 100, marker="o", label=name)
        ax.set_xlabel("Packet loss (%)")
        ax.set_ylabel("Event recall (%)")
        ax.set_title("Robustness to packet loss (validation)")
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(paper_fig / "fig10_robustness.png", dpi=200, bbox_inches="tight")
        fig.savefig(paper_fig / "fig10_robustness.pdf", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=60000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    cfg = load_yaml(root / "configs" / "rl_cmappo.yaml")
    cfg.setdefault("safety_shield", {})["enabled"] = False
    out = ensure_dir(root / "outputs" / "rl_final" / "paper_final")
    paper_fig = ensure_dir(root / "paper_outputs" / "figures")
    paper_tab = ensure_dir(root / "paper_outputs" / "tables")

    # ---- 1) Train 5 seeds ----
    ckpts = {}
    if not args.skip_train:
        for seed in SEEDS:
            ckpts[seed] = train_seed(seed, args.timesteps, args.device, args.force_retrain)
    else:
        for seed in SEEDS:
            p = root / "outputs" / "rl_final" / "cmappo_kl" / f"seed_{seed}" / "best_model.pt"
            if not p.exists():
                p = root / "outputs" / "rl_final" / "cmappo_kl" / f"seed_{seed}" / "final_model.pt"
            if p.exists():
                ckpts[seed] = p

    # ---- 2) Full VAL evaluation ----
    rows = []
    print("[eval] baselines val...")
    for name, pol in [
        ("fixed_15", FixedIntervalPolicy(1)),
        ("fixed_30", FixedIntervalPolicy(2)),
        ("fixed_60", FixedIntervalPolicy(4)),
        ("delta_plus_heartbeat", DeltaPlusHeartbeatPolicy(50.0, 4.0)),
        ("semantic_expert", SemanticExpertPolicy()),
    ]:
        rows.append(eval_named(name, _wrap(pol), split="val", cfg=cfg, seed=42))

    print("[eval] BC seeds val...")
    for seed in SEEDS:
        bc = _bc_path(root, seed)
        if not bc.exists():
            continue
        act = load_mappo_policy(bc, device=args.device)
        rows.append(eval_named("campus_senserl_bc", act, split="val", cfg=cfg, seed=seed))

    print("[eval] KL-CMAPPO seeds val...")
    for seed, ckpt in ckpts.items():
        act = load_mappo_policy(ckpt, device=args.device)
        rows.append(eval_named("cmappo_kl", act, split="val", cfg=cfg, seed=seed))

    # old mappo shield reference (seed 42)
    old = root / "outputs" / "rl_final" / "mappo" / "seed_42" / "final_model.pt"
    if old.exists():
        act = load_mappo_policy(old, device=args.device)
        rows.append(eval_named("mappo_old_shield", act, split="val", cfg=cfg, seed=42, shield=True))

    df_val = pd.DataFrame(rows)
    df_val.to_csv(out / "full_val_raw.csv", index=False)
    df_val.to_csv(paper_tab / "table3_policy_comparison.csv", index=False)
    summ = summarize(df_val)
    summ.to_csv(out / "full_val_summary.csv", index=False)
    summ.to_csv(paper_tab / "table3_policy_summary.csv", index=False)
    print(summ.to_string(index=False))

    # ---- 3) Frozen TEST (after val freeze) ----
    if not args.skip_test:
        # Pick KL-CMAPPO seed with best feasible score on val: lowest MAE among recall>=0.985
        cm = df_val[df_val["method"] == "cmappo_kl"].copy()
        cm = cm[cm["event_recall"] >= 0.985]
        if len(cm) == 0:
            cm = df_val[df_val["method"] == "cmappo_kl"]
        best_seed = int(cm.sort_values(["mae_skipped", "transmission_reduction_pct"]).iloc[0]["seed"])
        print(f"[test] frozen KL-CMAPPO seed={best_seed}")
        test_rows = []
        for name, pol in [
            ("fixed_30", FixedIntervalPolicy(2)),
            ("fixed_60", FixedIntervalPolicy(4)),
            ("delta_plus_heartbeat", DeltaPlusHeartbeatPolicy(50.0, 4.0)),
            ("semantic_expert", SemanticExpertPolicy()),
        ]:
            test_rows.append(eval_named(name, _wrap(pol), split="test", cfg=cfg, seed=42))
        # BC: report mean over seeds on test for stability, but also frozen seed
        for seed in SEEDS:
            bc = _bc_path(root, seed)
            if bc.exists():
                act = load_mappo_policy(bc, device=args.device)
                test_rows.append(eval_named("campus_senserl_bc", act, split="test", cfg=cfg, seed=seed))
        act = load_mappo_policy(ckpts[best_seed], device=args.device)
        test_rows.append(eval_named("cmappo_kl", act, split="test", cfg=cfg, seed=best_seed))
        df_test = pd.DataFrame(test_rows)
        df_test.to_csv(out / "full_test_raw.csv", index=False)
        df_test.to_csv(paper_tab / "table3_policy_comparison_test.csv", index=False)
        tsum = summarize(df_test)
        tsum.to_csv(out / "full_test_summary.csv", index=False)
        tsum.to_csv(paper_tab / "table3_policy_summary_test.csv", index=False)
        print("[test summary]")
        print(tsum.to_string(index=False))
        save_json(
            {
                "frozen_cmappo_seed": best_seed,
                "cmappo_checkpoint": str(ckpts[best_seed]),
                "note": "Method/hyperparams frozen on VAL; this is the single test report.",
            },
            out / "test_freeze_meta.json",
        )

    # ---- 4) Figures ----
    rob = root / "outputs" / "rl_final" / "corrected_eval" / "robustness_val.csv"
    make_figures(df_val, paper_fig, rob if rob.exists() else None)
    if rob.exists():
        pd.read_csv(rob).to_csv(paper_tab / "table5_robustness.csv", index=False)

    # Ablation table if present
    abl = root / "outputs" / "rl_final" / "paper_asap" / "table4_ablation.csv"
    if abl.exists():
        pd.read_csv(abl).to_csv(paper_tab / "table4_ablation.csv", index=False)

    save_json(
        {
            "seeds": SEEDS,
            "timesteps": args.timesteps,
            "n_cmappo_ckpts": len(ckpts),
            "outputs": str(out),
        },
        out / "meta.json",
    )
    print(f"[done] paper tables={paper_tab} figures={paper_fig} raw={out}")


if __name__ == "__main__":
    main()
