#!/usr/bin/env python
"""Step 6: Train centralized PPO communication policy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from campus_senserl.rl.ppo import run_ppo_training
from campus_senserl.utils import load_yaml, repo_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO policy")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config override")
    parser.add_argument("--max-sensors", type=int, default=None, help="Limit sensors for debugging")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total timesteps")
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config) if args.config else load_yaml(repo_root() / "configs" / "rl.yaml")
    if args.timesteps is not None:
        cfg.setdefault("ppo", {})["total_timesteps"] = args.timesteps

    device = None if args.device == "auto" else args.device
    summary = run_ppo_training(
        cfg,
        max_sensors=args.max_sensors,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
    )
    print(f"[done] checkpoint={summary['checkpoint']} device={summary.get('device')}")


if __name__ == "__main__":
    main()
