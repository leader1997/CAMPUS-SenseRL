#!/usr/bin/env python
"""Step 8: Run fixed and heuristic baseline policies."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from campus_senserl.environment.trace_environment import TraceDrivenCampusEnv
from campus_senserl.rl.fixed_policies import make_fixed_policies
from campus_senserl.rl.heuristic_policies import make_heuristic_policies
from campus_senserl.utils import ensure_dir, load_yaml, repo_root, save_json


def run_policy(env: TraceDrivenCampusEnv, policy, max_steps: int | None = None) -> dict:
    obs, _ = env.reset()
    total_reward = 0.0
    tx = 0
    steps = 0
    policy.reset()
    max_steps = max_steps or env.n_steps - 1
    while steps < max_steps:
        local_available = env.local_available[env._t]
        actions = policy.act(obs, local_available=local_available)
        obs, reward, terminated, truncated, info = env.step(actions)
        total_reward += reward
        tx += int(info.get("transmit_count", 0))
        steps += 1
        if terminated or truncated:
            break
    metrics = env.get_episode_metrics()
    metrics.update(
        {
            "total_reward": float(total_reward),
            "mean_reward": float(total_reward / max(steps, 1)),
            "transmit_events": tx,
            "steps": steps,
        }
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline policies")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--max-sensors", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args()

    cfg = load_yaml(repo_root() / "configs" / "rl.yaml")
    env = TraceDrivenCampusEnv(cfg=cfg, split=args.split, max_sensors=args.max_sensors)

    policies = {}
    policies.update(make_fixed_policies(cfg.get("baselines", {}).get("fixed_intervals_min")))
    if cfg.get("baselines", {}).get("threshold_policies", True):
        policies.update(make_heuristic_policies(cfg))

    results = {}
    for name, policy in policies.items():
        print(f"[baseline] Running {name}…")
        results[name] = run_policy(env, policy, max_steps=args.max_steps)

    out_dir = ensure_dir(repo_root() / "outputs" / "baselines")
    save_json(results, out_dir / f"results_{args.split}.json")
    print(f"[done] wrote {out_dir / f'results_{args.split}.json'}")
    for name, m in results.items():
        print(
            f"  {name}: reward={m['mean_reward']:.4f} tx_rate={m['transmit_rate']:.3f} "
            f"rl_skip={m['rl_skip_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
