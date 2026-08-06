#!/usr/bin/env python
"""Step 9: Run ablation configurations and write CSV summary table."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from campus_senserl.environment.trace_environment import (
    TraceDrivenCampusEnv,
    build_synthetic_trace,
)
from campus_senserl.rl.fixed_policies import FixedIntervalPolicy
from campus_senserl.utils import ensure_dir, load_yaml, repo_root, set_seed


def _make_env(cfg: dict, split: str, max_sensors: int | None, use_synthetic: bool):
    trace = None
    if use_synthetic:
        trace = build_synthetic_trace(n_steps=48, n_sensors=max_sensors or 6, seed=int(cfg.get("seed", 42)))
    return TraceDrivenCampusEnv(cfg=cfg, split=split, max_sensors=max_sensors, trace=trace, multi_agent=True)


def _apply_ablation(cfg: dict, name: str) -> dict:
    c = copy.deepcopy(cfg)
    if name == "no_graph":
        c.setdefault("model", {})["use_graph"] = False
    elif name == "no_motion":
        c.setdefault("obs", {})["include_motion"] = False
    elif name == "no_network":
        c.setdefault("obs", {})["include_network"] = False
    elif name == "no_uncertainty":
        c.setdefault("reward", {})["w_unc"] = 0.0
    elif name == "no_aoi":
        c.setdefault("reward", {})["w_aoi"] = 0.0
    elif name == "no_semantic":
        c.setdefault("reward", {})["w_event"] = 0.0
        c.setdefault("reward", {})["w_miss"] = 0.0
    elif name == "no_safety_shield":
        c["safety_shield"] = {"enabled": False}
    elif name == "full":
        pass
    else:
        c["ablation_tag"] = name
    return c


def _eval_policy(env: TraceDrivenCampusEnv, policy, max_steps: int) -> dict[str, float]:
    obs, _ = env.reset()
    policy.reset()
    total_reward = 0.0
    steps = 0
    while steps < max_steps and env._t < env.n_steps - 1:
        actions = policy.act(obs, local_available=env.local_available[env._t])
        obs, reward, terminated, truncated, info = env.step(actions)
        total_reward += float(reward)
        steps += 1
        if terminated or truncated:
            break
    metrics = env.get_episode_metrics()
    metrics["mean_reward"] = total_reward / max(steps, 1)
    metrics["shield_overrides"] = float(info.get("shield_overrides", 0))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--max-sensors", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--synthetic", action="store_true", help="Use in-memory trace when parquet missing")
    args = parser.parse_args()

    root = repo_root()
    exp_cfg = load_yaml(root / "configs" / "experiments.yaml")
    rl_cfg = load_yaml(root / "configs" / "rl.yaml")
    set_seed(int(exp_cfg.get("seed", 42)))

    ablations = ["full"] + list(exp_cfg.get("ablations", []))
    use_synthetic = args.synthetic
    panel = root / "data" / "processed" / "co2_panel_15min.parquet"
    if not panel.exists():
        use_synthetic = True
        print("[ablation] Processed panel missing — using synthetic trace.")

    rows = []
    for name in ablations:
        cfg = _apply_ablation(rl_cfg, name)
        try:
            env = _make_env(cfg, args.split, args.max_sensors, use_synthetic)
        except FileNotFoundError:
            env = _make_env(cfg, args.split, args.max_sensors, True)
        policy = FixedIntervalPolicy(interval_steps=4)
        print(f"[ablation] {name} …")
        m = _eval_policy(env, policy, args.max_steps)
        rows.append(
            {
                "ablation": name,
                "split": args.split,
                "mean_reward": m["mean_reward"],
                "transmit_rate": m["transmit_rate"],
                "rl_skip_rate": m["rl_skip_rate"],
                "natural_missing_rate": m["natural_missing_rate"],
                "shield_overrides": m.get("shield_overrides", 0.0),
            }
        )

    df = pd.DataFrame(rows)
    out_dir = ensure_dir(root / exp_cfg.get("outputs", {}).get("tables_dir", "outputs/tables"))
    out_csv = out_dir / "ablations.csv"
    df.to_csv(out_csv, index=False)
    print(f"[done] wrote {out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
