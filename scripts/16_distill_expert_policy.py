#!/usr/bin/env python
"""Distill SemanticExpert into ResidualSharedActor via pure BC (no destructive RL).

This is the reliable path: the expert already dominates baselines on val
(MAE ~7.9, recall ~0.999, ~72% TX reduction). BC copies it into a neural policy
compatible with the MAPPO checkpoint format.
"""

from __future__ import annotations

import argparse
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
from campus_senserl.evaluation.rl_policy_eval import evaluate_policy, load_mappo_policy, make_final_env
from campus_senserl.rl.expert_policy import SemanticExpertPolicy
from campus_senserl.rl.mappo_boosted import MeanPoolCritic, ResidualSharedActor
from campus_senserl.utils import ensure_dir, load_yaml, repo_root, save_json, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    root = repo_root()
    cfg = load_yaml(root / "configs" / "rl_learn.yaml")
    cfg["seed"] = args.seed
    cfg.setdefault("safety_shield", {})["enabled"] = False
    set_seed(args.seed)
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    sensors = load_cohort("final")
    env = TraceDrivenCampusEnv(cfg=cfg, split="train", sensor_ids=sensors, multi_agent=True)
    obs_dim = int(env.observation_space.shape[-1])
    n_agents = env.n_sensors
    actor = ResidualSharedActor(obs_dim, residual_init=1.5).to(device)
    critic = MeanPoolCritic(obs_dim).to(device)  # unused but saved for format compat
    opt = optim.Adam(actor.parameters(), lr=args.lr)
    expert = SemanticExpertPolicy()

    obs, _ = env.reset()
    buf_o, buf_a = [], []
    for step in range(1, args.steps + 1):
        local = env.local_available[env._t]
        a = expert.act(obs, local_available=local)
        buf_o.append(obs.copy())
        buf_a.append(a.copy())
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            obs, _ = env.reset()
        if len(buf_o) >= 256 or step == args.steps:
            x = torch.as_tensor(np.asarray(buf_o), dtype=torch.float32, device=device)
            y = torch.as_tensor(np.asarray(buf_a), dtype=torch.float32, device=device)
            t, n, d = x.shape
            logits = actor(x.reshape(t * n, d))
            loss = F.binary_cross_entropy_with_logits(logits, y.reshape(-1))
            # Keep neural residual small so expert bias dominates
            neural = actor.net(x.reshape(t * n, d)).squeeze(-1)
            loss = loss + 0.1 * (neural ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 2000 == 0 or step == args.steps:
                with torch.no_grad():
                    pred = (torch.sigmoid(logits) > 0.5).float()
                    acc = float((pred == y.reshape(-1)).float().mean())
                print(f"[distill] step={step} loss={loss.item():.4f} acc={acc:.4f} alpha={float(actor.residual_scale):.3f}")
            buf_o, buf_a = [], []

    out_dir = ensure_dir(root / "outputs" / "rl_final" / "mappo_boosted" / f"seed_{args.seed}")
    ckpt = {
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "cfg": cfg,
        "critic_type": "mean_pool",
        "actor_type": "residual_heuristic",
        "obs_dim": obs_dim,
        "n_agents_train": n_agents,
        "method": "bc_distill_semantic_expert",
    }
    path = out_dir / "final_model.pt"
    torch.save(ckpt, path)
    torch.save(ckpt, out_dir / "best_model.pt")
    print(f"[distill] saved {path}")

    # Full val eval (no shield)
    act = load_mappo_policy(path, device=device)
    val_env = make_final_env(split="val", cfg=cfg, multi_agent=True, shield_enabled=False)
    m = evaluate_policy(val_env, act, max_steps=None, seed=args.seed)
    m["method"] = "mappo_boosted_bc"
    m["seed"] = args.seed
    m["shield"] = False
    print(
        f"[distill] VAL red={m['transmission_reduction_pct']:.2f}% mae={m['mae_skipped']:.3f} "
        f"recall={m['event_recall']:.4f} f1={m['event_f1']:.4f} aoi={m['mean_aoi']:.3f} events={m['n_true_events']}"
    )
    pd.DataFrame([m]).to_csv(out_dir / "bc_full_val.csv", index=False)
    save_json(m, out_dir / "bc_full_val.json")

    # Also compare vs baselines quickly on same env settings (reuse script metrics)
    from campus_senserl.rl.fixed_policies import FixedIntervalPolicy
    from campus_senserl.rl.heuristic_policies import ChangeThresholdPolicy, AoIThresholdPolicy, InfoValuePolicy

    rows = [m]
    baselines = [
        ("fixed_15", FixedIntervalPolicy(1)),
        ("fixed_30", FixedIntervalPolicy(2)),
        ("fixed_60", FixedIntervalPolicy(4)),
        ("change_threshold", ChangeThresholdPolicy(delta_ppm=50.0)),
        ("aoi_threshold", AoIThresholdPolicy(threshold=4.0)),
        ("semantic_heuristic", InfoValuePolicy(threshold=0.35)),
        ("semantic_expert", SemanticExpertPolicy()),
    ]
    for name, pol in baselines:
        print(f"[distill] baseline {name}...")
        env_b = make_final_env(split="val", cfg=cfg, multi_agent=True, shield_enabled=False)

        def _act(obs, *, local_available=None, _p=pol):
            return _p.act(obs, local_available=local_available)

        bm = evaluate_policy(env_b, _act, max_steps=None, seed=args.seed)
        bm["method"] = name
        bm["seed"] = args.seed
        bm["shield"] = False
        rows.append(bm)
        print(f"  red={bm['transmission_reduction_pct']:.1f}% mae={bm['mae_skipped']} recall={bm['event_recall']}")

    df = pd.DataFrame(rows)
    cmp_path = ensure_dir(root / "outputs" / "rl_final" / "policy_eval") / "boosted_vs_baselines_val.csv"
    df.to_csv(cmp_path, index=False)
    print(f"[done] wrote {cmp_path}")
    show = df[["method", "transmission_reduction_pct", "mae_skipped", "event_recall", "event_f1", "mean_aoi"]]
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
