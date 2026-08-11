#!/usr/bin/env python
"""Train BC-initialized Constrained MAPPO (VAL selection only — no test peeking)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from campus_senserl.rl.cmappo import run_constrained_mappo
from campus_senserl.utils import load_yaml, repo_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--bc-checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-dir", default=None)
    args = parser.parse_args()

    root = repo_root()
    cfg = load_yaml(args.config) if args.config else load_yaml(root / "configs" / "rl_cmappo.yaml")
    cfg["seed"] = args.seed
    if args.timesteps is not None:
        cfg.setdefault("mappo", {})["total_timesteps"] = args.timesteps
    ckpt_dir = args.checkpoint_dir or str(
        root / "results" / "rl_final" / "cmappo_kl" / f"seed_{args.seed}"
    )
    out = run_constrained_mappo(
        cfg,
        bc_checkpoint=args.bc_checkpoint,
        device=None if args.device == "auto" else args.device,
        checkpoint_dir=ckpt_dir,
    )
    print(f"[done] {out['checkpoint']} best={out.get('best')}")


if __name__ == "__main__":
    main()
