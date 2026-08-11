#!/usr/bin/env python
"""ASAP paper pipeline: tune semantic expert, 5-seed BC, ablations, test freeze.

Does NOT run unconstrained RL (known to collapse). Produces paper-ready tables under
results/rl_final/paper_asap/ and results/tables/.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from campus_senserl.data.cohort import load_cohort
from campus_senserl.environment.trace_environment import TraceDrivenCampusEnv
from campus_senserl.evaluation.rl_policy_eval import (
    evaluate_policy,
    load_mappo_policy,
    make_final_env,
)
from campus_senserl.rl.expert_policy import SemanticExpertPolicy
from campus_senserl.rl.fixed_policies import FixedIntervalPolicy
from campus_senserl.rl.heuristic_policies import ChangeThresholdPolicy
from campus_senserl.rl.mappo_boosted import MeanPoolCritic, ResidualSharedActor
from campus_senserl.utils import ensure_dir, load_yaml, repo_root, save_json, set_seed

SEEDS = [42, 123, 2024, 3407, 9999]


def _wrap(pol):
    def act(obs, *, local_available=None):
        return pol.act(obs, local_available=local_available)

    act.reset = getattr(pol, "reset", lambda: None)
    return act


def _score(m: dict) -> float:
    """Higher is better: prioritize recall + low MAE, then TX reduction."""
    recall = float(m["event_recall"]) if np.isfinite(m["event_recall"]) else 0.0
    mae = float(m["mae_skipped"]) if np.isfinite(m.get("mae_skipped", np.nan)) else 80.0
    red = float(m["transmission_reduction_pct"])
    # Require meaningful savings; heavy penalty if too chatty (<50% reduction)
    if red < 50:
        return -1000.0 + red
    return 100.0 * recall - 1.5 * mae + 0.25 * red


def eval_policy_named(name: str, act, *, split: str, cfg: dict, max_steps: int | None, seed: int) -> dict:
    env = make_final_env(split=split, cfg=cfg, multi_agent=True)
    m = evaluate_policy(env, act, max_steps=max_steps, seed=seed)
    m["method"] = name
    m["seed"] = seed
    m["shield"] = False
    return m


def tune_expert(cfg: dict, *, search_steps: int = 1200) -> SemanticExpertPolicy:
    """Grid-search expert thresholds on a val prefix, then verify winner briefly."""
    # Compact grid (16 configs) — full 3^4 is too slow for ASAP
    grid = {
        "delta_ppm": [35.0, 45.0],
        "aoi_threshold": [2.5, 3.5],
        "co2_ppm": [900.0, 1000.0],
        "disagreement_ppm": [20.0, 35.0],
    }
    keys = list(grid.keys())
    rows = []
    best_pol = SemanticExpertPolicy()
    best_s = -1e18
    print(f"[tune] grid size={int(np.prod([len(v) for v in grid.values()]))} steps={search_steps}")
    for vals in itertools.product(*[grid[k] for k in keys]):
        kwargs = dict(zip(keys, vals))
        pol = SemanticExpertPolicy(**kwargs)
        m = eval_policy_named(
            "expert_cand",
            _wrap(pol),
            split="val",
            cfg=cfg,
            max_steps=search_steps,
            seed=42,
        )
        s = _score(m)
        row = {**kwargs, **{k: m[k] for k in ("transmission_reduction_pct", "mae_skipped", "event_recall", "n_true_events")}, "score": s}
        rows.append(row)
        print(
            f"  {kwargs} red={m['transmission_reduction_pct']:.1f} mae={m['mae_skipped']} "
            f"rec={m['event_recall']} score={s:.2f}"
        )
        if s > best_s:
            best_s = s
            best_pol = pol
    return best_pol, pd.DataFrame(rows)


def distill_seed(
    *,
    expert: SemanticExpertPolicy,
    seed: int,
    steps: int,
    device: str,
    cfg: dict,
    out_dir: Path,
) -> Path:
    set_seed(seed)
    cfg = copy.deepcopy(cfg)
    cfg["seed"] = seed
    sensors = load_cohort("final")
    env = TraceDrivenCampusEnv(cfg=cfg, split="train", sensor_ids=sensors, multi_agent=True)
    obs_dim = int(env.observation_space.shape[-1])
    n_agents = env.n_sensors
    actor = ResidualSharedActor(obs_dim, residual_init=1.5).to(device)
    critic = MeanPoolCritic(obs_dim).to(device)
    opt = optim.Adam(actor.parameters(), lr=1e-3)
    obs, _ = env.reset()
    buf_o, buf_a = [], []
    for step in range(1, steps + 1):
        local = env.local_available[env._t]
        a = expert.act(obs, local_available=local)
        buf_o.append(obs.copy())
        buf_a.append(a.copy())
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            obs, _ = env.reset()
        if len(buf_o) >= 256 or step == steps:
            x = torch.as_tensor(np.asarray(buf_o), dtype=torch.float32, device=device)
            y = torch.as_tensor(np.asarray(buf_a), dtype=torch.float32, device=device)
            t, n, d = x.shape
            logits = actor(x.reshape(t * n, d))
            loss = F.binary_cross_entropy_with_logits(logits, y.reshape(-1))
            neural = actor.net(x.reshape(t * n, d)).squeeze(-1)
            loss = loss + 0.1 * (neural**2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            buf_o, buf_a = [], []
            if step % 5000 == 0 or step == steps:
                with torch.no_grad():
                    acc = float(((torch.sigmoid(logits) > 0.5).float() == y.reshape(-1)).float().mean())
                print(f"[distill seed={seed}] step={step} loss={float(loss):.4f} acc={acc:.4f}")

    seed_dir = ensure_dir(out_dir / f"seed_{seed}")
    ckpt = {
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "cfg": cfg,
        "critic_type": "mean_pool",
        "actor_type": "residual_heuristic",
        "obs_dim": obs_dim,
        "n_agents_train": n_agents,
        "method": "campus_senserl_bc",
        "expert": {
            "delta_ppm": expert.delta_ppm,
            "aoi_threshold": expert.aoi_threshold,
            "co2_ppm": expert.co2_ppm,
            "disagreement_ppm": expert.disagreement_ppm,
        },
    }
    path = seed_dir / "final_model.pt"
    torch.save(ckpt, path)
    torch.save(ckpt, seed_dir / "best_model.pt")
    return path


def expert_ablations(base: SemanticExpertPolicy) -> dict[str, SemanticExpertPolicy]:
    """Disable one rule at a time (set threshold unreachable)."""
    return {
        "expert_full": SemanticExpertPolicy(
            delta_ppm=base.delta_ppm,
            aoi_threshold=base.aoi_threshold,
            co2_ppm=base.co2_ppm,
            disagreement_ppm=base.disagreement_ppm,
        ),
        "expert_no_delta": SemanticExpertPolicy(
            delta_ppm=1e9,
            aoi_threshold=base.aoi_threshold,
            co2_ppm=base.co2_ppm,
            disagreement_ppm=base.disagreement_ppm,
        ),
        "expert_no_aoi": SemanticExpertPolicy(
            delta_ppm=base.delta_ppm,
            aoi_threshold=1e9,
            co2_ppm=base.co2_ppm,
            disagreement_ppm=base.disagreement_ppm,
        ),
        "expert_no_co2": SemanticExpertPolicy(
            delta_ppm=base.delta_ppm,
            aoi_threshold=base.aoi_threshold,
            co2_ppm=1e9,
            disagreement_ppm=base.disagreement_ppm,
        ),
        "expert_no_disagree": SemanticExpertPolicy(
            delta_ppm=base.delta_ppm,
            aoi_threshold=base.aoi_threshold,
            co2_ppm=base.co2_ppm,
            disagreement_ppm=1e9,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bc-steps", type=int, default=40000)
    parser.add_argument("--tune-steps", type=int, default=1500)
    parser.add_argument("--skip-tune", action="store_true")
    parser.add_argument("--skip-distill", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    args = parser.parse_args()

    root = repo_root()
    cfg = load_yaml(root / "configs" / "rl_learn.yaml")
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    out = ensure_dir(root / "results" / "rl_final" / "paper_asap")
    paper_tables = ensure_dir(root / "results" / "tables")

    # ---- 1) Tune expert ----
    if args.skip_tune:
        expert = SemanticExpertPolicy()
        print("[tune] skipped; using defaults", expert)
    else:
        expert, tune_df = tune_expert(cfg, search_steps=args.tune_steps)
        tune_df.to_csv(out / "expert_threshold_search.csv", index=False)
        save_json(
            {
                "delta_ppm": expert.delta_ppm,
                "aoi_threshold": expert.aoi_threshold,
                "co2_ppm": expert.co2_ppm,
                "disagreement_ppm": expert.disagreement_ppm,
            },
            out / "expert_best.json",
        )
        print("[tune] best", expert)

    # Full-val expert
    print("[eval] full-val semantic expert...")
    expert_m = eval_policy_named("semantic_expert", _wrap(expert), split="val", cfg=cfg, max_steps=None, seed=42)
    print(
        f"  red={expert_m['transmission_reduction_pct']:.2f}% mae={expert_m['mae_skipped']:.3f} "
        f"rec={expert_m['event_recall']:.4f}"
    )

    # ---- 2) Multi-seed BC ----
    bc_rows = []
    if not args.skip_distill:
        for seed in args.seeds:
            print(f"[distill] seed={seed}")
            ckpt = distill_seed(
                expert=expert,
                seed=seed,
                steps=args.bc_steps,
                device=device,
                cfg=cfg,
                out_dir=out / "bc",
            )
            act = load_mappo_policy(ckpt, device=device)
            m = eval_policy_named("campus_senserl_bc", act, split="val", cfg=cfg, max_steps=None, seed=seed)
            bc_rows.append(m)
            print(
                f"  seed={seed} red={m['transmission_reduction_pct']:.2f}% mae={m['mae_skipped']:.3f} "
                f"rec={m['event_recall']:.4f}"
            )
        pd.DataFrame(bc_rows).to_csv(out / "bc_seeds_val.csv", index=False)
    else:
        # load existing if present
        p = out / "bc_seeds_val.csv"
        if p.exists():
            bc_rows = pd.read_csv(p).to_dict("records")

    # ---- 3) Baselines + ablations (full val) ----
    print("[eval] baselines + ablations...")
    rows = [expert_m] + list(bc_rows)
    baselines = [
        ("fixed_15", FixedIntervalPolicy(1)),
        ("fixed_30", FixedIntervalPolicy(2)),
        ("fixed_45", FixedIntervalPolicy(3)),
        ("fixed_60", FixedIntervalPolicy(4)),
        ("change_threshold", ChangeThresholdPolicy(delta_ppm=50.0)),
    ]
    for name, pol in baselines:
        m = eval_policy_named(name, _wrap(pol), split="val", cfg=cfg, max_steps=None, seed=42)
        rows.append(m)
        print(f"  {name}: red={m['transmission_reduction_pct']:.1f}% mae={m['mae_skipped']} rec={m['event_recall']}")

    for name, pol in expert_ablations(expert).items():
        if name == "expert_full":
            continue
        m = eval_policy_named(name, _wrap(pol), split="val", cfg=cfg, max_steps=None, seed=42)
        rows.append(m)
        print(f"  {name}: red={m['transmission_reduction_pct']:.1f}% mae={m['mae_skipped']} rec={m['event_recall']}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "full_val_comparison.csv", index=False)

    # Aggregate BC seeds
    if bc_rows:
        bcdf = pd.DataFrame(bc_rows)
        agg = {
            "method": "campus_senserl_bc",
            "n_seeds": int(len(bcdf)),
            "tx_reduction_mean": float(bcdf["transmission_reduction_pct"].mean()),
            "tx_reduction_std": float(bcdf["transmission_reduction_pct"].std(ddof=1)) if len(bcdf) > 1 else 0.0,
            "mae_mean": float(bcdf["mae_skipped"].mean()),
            "mae_std": float(bcdf["mae_skipped"].std(ddof=1)) if len(bcdf) > 1 else 0.0,
            "recall_mean": float(bcdf["event_recall"].mean()),
            "recall_std": float(bcdf["event_recall"].std(ddof=1)) if len(bcdf) > 1 else 0.0,
            "f1_mean": float(bcdf["event_f1"].mean()),
            "aoi_mean": float(bcdf["mean_aoi"].mean()),
        }
        save_json(agg, out / "bc_seed_summary.json")
        print("[summary]", agg)

    # Paper table (primary methods)
    keep = [
        "fixed_15",
        "fixed_30",
        "fixed_45",
        "fixed_60",
        "change_threshold",
        "semantic_expert",
        "campus_senserl_bc",
    ]
    paper = df[df["method"].isin(keep)].copy()
    # For BC keep mean row if multi-seed
    if bc_rows and Path(out / "bc_seed_summary.json").exists():
        import json

        agg = json.loads((out / "bc_seed_summary.json").read_text())
        paper = paper[paper["method"] != "campus_senserl_bc"]
        paper = pd.concat(
            [
                paper,
                pd.DataFrame(
                    [
                        {
                            "method": "campus_senserl_bc",
                            "transmission_reduction_pct": agg["tx_reduction_mean"],
                            "mae_skipped": agg["mae_mean"],
                            "event_recall": agg["recall_mean"],
                            "event_f1": agg["f1_mean"],
                            "mean_aoi": agg["aoi_mean"],
                            "seed": "mean",
                            "note": f"mean±std over {agg['n_seeds']} seeds; "
                            f"red {agg['tx_reduction_mean']:.2f}±{agg['tx_reduction_std']:.2f}; "
                            f"mae {agg['mae_mean']:.2f}±{agg['mae_std']:.2f}; "
                            f"rec {agg['recall_mean']:.3f}±{agg['recall_std']:.3f}",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    cols = [
        "method",
        "transmission_reduction_pct",
        "mae_skipped",
        "event_recall",
        "event_f1",
        "mean_aoi",
        "n_true_events",
        "seed",
    ]
    paper_out = paper[[c for c in cols if c in paper.columns]]
    paper_out.to_csv(paper_tables / "table3_policy_comparison.csv", index=False)
    paper_out.to_csv(out / "table3_policy_comparison.csv", index=False)

    # Ablation table
    abl = df[df["method"].str.startswith("expert_")].copy()
    abl.to_csv(paper_tables / "table4_ablation.csv", index=False)
    abl.to_csv(out / "table4_ablation.csv", index=False)

    # ---- 4) Test freeze (proposed = mean-best seed 42 checkpoint or best MAE seed) ----
    if not args.skip_test and not args.skip_distill:
        # Pick seed with best score on val among BC
        bcdf = pd.DataFrame(bc_rows)
        bcdf["score"] = bcdf.apply(_score, axis=1)
        best_seed = int(bcdf.loc[bcdf["score"].idxmax(), "seed"])
        ckpt = out / "bc" / f"seed_{best_seed}" / "final_model.pt"
        print(f"[test] frozen checkpoint seed={best_seed} → {ckpt}")
        act = load_mappo_policy(ckpt, device=device)
        test_rows = []
        for name, pol in [("campus_senserl_bc", None)] + baselines + [("semantic_expert", expert)]:
            if name == "campus_senserl_bc":
                m = eval_policy_named(name, act, split="test", cfg=cfg, max_steps=None, seed=best_seed)
            elif name == "semantic_expert":
                m = eval_policy_named(name, _wrap(expert), split="test", cfg=cfg, max_steps=None, seed=42)
            else:
                m = eval_policy_named(name, _wrap(pol), split="test", cfg=cfg, max_steps=None, seed=42)
            test_rows.append(m)
            print(
                f"  TEST {name}: red={m['transmission_reduction_pct']:.1f}% mae={m['mae_skipped']} "
                f"rec={m['event_recall']} events={m['n_true_events']}"
            )
        tdf = pd.DataFrame(test_rows)
        tdf.to_csv(out / "full_test_comparison.csv", index=False)
        tdf.to_csv(paper_tables / "table3_policy_comparison_test.csv", index=False)
        save_json(
            {
                "frozen_seed": best_seed,
                "checkpoint": str(ckpt),
                "expert": {
                    "delta_ppm": expert.delta_ppm,
                    "aoi_threshold": expert.aoi_threshold,
                    "co2_ppm": expert.co2_ppm,
                    "disagreement_ppm": expert.disagreement_ppm,
                },
                "split": "test",
                "note": "Method selected on val only; single frozen test pass.",
            },
            out / "test_freeze_meta.json",
        )

    print(f"[done] artifacts in {out}")


if __name__ == "__main__":
    main()
