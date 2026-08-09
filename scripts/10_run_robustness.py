#!/usr/bin/env python
"""Step 10: Robustness on REAL validation traces with trained MAPPO.

Packet loss is applied post-shield via env.packet_loss_rate (requested TX vs delivered),
not by clearing local_available (which would simulate measurement failure).
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from campus_senserl.evaluation.rl_policy_eval import (
    evaluate_policy,
    load_mappo_policy,
    make_final_env,
)
from campus_senserl.utils import ensure_dir, load_yaml, repo_root, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Robustness experiments (real val + MAPPO)")
    parser.add_argument("--split", default="val")
    parser.add_argument("--seed", type=int, default=42, help="MAPPO checkpoint seed")
    parser.add_argument("--max-steps", type=int, default=None, help="None = full split")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    root = repo_root()
    exp_cfg = load_yaml(root / "configs" / "experiments.yaml")
    rl_cfg = load_yaml(root / "configs" / "rl.yaml")
    set_seed(int(exp_cfg.get("seed", 42)))

    ckpt = root / "outputs" / "rl_final" / "mappo" / f"seed_{args.seed}" / "final_model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"MAPPO checkpoint required: {ckpt}")

    act = load_mappo_policy(ckpt, device=args.device)
    rows = []

    for rate in exp_cfg.get("robustness", {}).get("packet_loss_rates", [0.0, 0.1, 0.2]):
        print(f"[robust] packet_loss={rate}")
        env = make_final_env(
            split=args.split,
            cfg=rl_cfg,
            multi_agent=True,
            shield_enabled=True,
            packet_loss_rate=float(rate),
        )
        m = evaluate_policy(env, act, max_steps=args.max_steps, seed=args.seed)
        rows.append({"condition": "packet_loss", "level": rate, "policy": "mappo", **m})

    for thr in exp_cfg.get("robustness", {}).get("event_thresholds_ppm", [800, 1000, 1200, 1500]):
        print(f"[robust] event_threshold={thr}")
        cfg = copy.deepcopy(rl_cfg)
        cfg.setdefault("events", {})["primary_threshold_ppm"] = thr
        env = make_final_env(
            split=args.split,
            cfg=cfg,
            multi_agent=True,
            shield_enabled=True,
            packet_loss_rate=0.0,
        )
        m = evaluate_policy(env, act, max_steps=args.max_steps, seed=args.seed)
        rows.append({"condition": "event_threshold", "level": thr, "policy": "mappo", **m})

    df = pd.DataFrame(rows)
    out_dir = ensure_dir(root / "outputs" / "robustness_final")
    out_csv = out_dir / "robustness_mappo_val.csv"
    df.to_csv(out_csv, index=False)
    # Also refresh live outputs/robustness with a clear filename
    legacy = ensure_dir(root / "outputs" / "robustness")
    df.to_csv(legacy / "robustness_real_val_mappo.csv", index=False)
    print(f"[done] wrote {out_csv}")
    cols = [
        "condition",
        "level",
        "transmission_reduction_pct",
        "event_recall",
        "n_true_events",
        "packet_delivery_ratio",
        "mae_skipped",
    ]
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
