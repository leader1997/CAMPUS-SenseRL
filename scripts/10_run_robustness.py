#!/usr/bin/env python
"""Step 10: Robustness sweeps — packet loss, missingness, event thresholds."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from campus_senserl.environment.communication_model import TRANSMIT
from campus_senserl.environment.trace_environment import (
    TraceDrivenCampusEnv,
    build_synthetic_trace,
)
from campus_senserl.rl.fixed_policies import FixedIntervalPolicy
from campus_senserl.utils import ensure_dir, load_yaml, repo_root, set_seed


def _inject_packet_loss(trace: dict, rate: float, seed: int) -> dict:
    """Simulate dropped uplinks by marking slots as naturally missing."""
    out = copy.deepcopy(trace)
    rng = np.random.default_rng(seed)
    n_t, n_s = out["ground_truth"].shape
    drop = rng.random((n_t, n_s)) < rate
    out["natural_missing"] = out["natural_missing"] | drop
    out["local_available"] = ~out["natural_missing"] & np.isfinite(out["ground_truth"])
    return out


def _inject_extra_missingness(trace: dict, rate: float, seed: int) -> dict:
    out = copy.deepcopy(trace)
    rng = np.random.default_rng(seed + 1)
    avail = out["local_available"].copy()
    mask = rng.random(avail.shape) < rate
    out["local_available"] = avail & ~mask
    return out


def _eval(env: TraceDrivenCampusEnv, max_steps: int) -> dict[str, float]:
    policy = FixedIntervalPolicy(interval_steps=3)
    obs, _ = env.reset()
    policy.reset()
    total = 0.0
    steps = 0
    tx_attempts = 0
    tx_success = 0
    while steps < max_steps and env._t < env.n_steps - 1:
        actions = policy.act(obs, local_available=env.local_available[env._t])
        tx_attempts += int(np.sum(actions == TRANSMIT))
        obs, reward, terminated, truncated, info = env.step(actions)
        tx_success += int(info.get("transmit_count", 0))
        total += float(reward)
        steps += 1
        if terminated or truncated:
            break
    m = env.get_episode_metrics()
    m["mean_reward"] = total / max(steps, 1)
    m["packet_delivery_ratio"] = tx_success / max(tx_attempts, 1)
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description="Robustness experiments")
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-sensors", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=150)
    args = parser.parse_args()

    root = repo_root()
    exp_cfg = load_yaml(root / "configs" / "experiments.yaml")
    rl_cfg = load_yaml(root / "configs" / "rl.yaml")
    seed = int(exp_cfg.get("seed", 42))
    set_seed(seed)

    base_trace = build_synthetic_trace(n_steps=64, n_sensors=args.max_sensors, seed=seed)
    rows = []

    for rate in exp_cfg.get("robustness", {}).get("packet_loss_rates", [0.0, 0.1, 0.2]):
        trace = _inject_packet_loss(base_trace, rate, seed)
        env = TraceDrivenCampusEnv(cfg=rl_cfg, trace=trace, multi_agent=True)
        m = _eval(env, args.max_steps)
        rows.append({"condition": "packet_loss", "level": rate, **m})

    for rate in exp_cfg.get("robustness", {}).get("missingness_rates", [0.1, 0.2, 0.4]):
        trace = _inject_extra_missingness(base_trace, rate, seed)
        env = TraceDrivenCampusEnv(cfg=rl_cfg, trace=trace, multi_agent=True)
        m = _eval(env, args.max_steps)
        rows.append({"condition": "missingness", "level": rate, **m})

    for thr in exp_cfg.get("robustness", {}).get("event_thresholds_ppm", [800, 1000, 1200]):
        cfg = copy.deepcopy(rl_cfg)
        cfg.setdefault("events", {})["primary_threshold_ppm"] = thr
        env = TraceDrivenCampusEnv(cfg=cfg, trace=base_trace, multi_agent=True)
        m = _eval(env, args.max_steps)
        rows.append({"condition": "event_threshold", "level": thr, **m})

    df = pd.DataFrame(rows)
    out_dir = ensure_dir(root / "outputs" / "robustness")
    out_csv = out_dir / "robustness.csv"
    df.to_csv(out_csv, index=False)
    print(f"[done] wrote {out_csv}")
    print(df[["condition", "level", "mean_reward", "transmit_rate", "packet_delivery_ratio"]].to_string(index=False))


if __name__ == "__main__":
    main()
