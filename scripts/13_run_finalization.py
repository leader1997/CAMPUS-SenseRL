#!/usr/bin/env python
"""End-to-end scientific finalization runner (Phases 2–15).

Usage
-----
# Full path (long):
  .venv\\Scripts\\python.exe scripts\\13_run_finalization.py --device cuda

# Faster paper-useful subset:
  .venv\\Scripts\\python.exe scripts\\13_run_finalization.py --device cuda --profile paper
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from campus_senserl.data.cohort import freeze_cohorts, load_cohort
from campus_senserl.evaluation.fair_reconstruction import run_fair_benchmark
from campus_senserl.utils import ensure_dir, load_yaml, save_json, set_seed

PY = sys.executable

PROFILES = {
    "paper": {
        "final_n": 40,
        "ppo_steps": 100000,
        "mappo_steps": 150000,
        "seeds": [42, 123, 2024, 3407, 9999],
        "fair_seeds": [42, 123, 2024],
        "recon_epochs": 20,
        "eval_steps": 1200,
    },
    "full": {
        "final_n": 80,
        "ppo_steps": 200000,
        "mappo_steps": 300000,
        "seeds": [42, 123, 2024, 3407, 9999],
        "fair_seeds": [42, 123, 2024, 3407, 9999],
        "recon_epochs": 25,
        "eval_steps": 2000,
    },
    "smoke": {
        "final_n": 20,
        "ppo_steps": 8000,
        "mappo_steps": 8000,
        "seeds": [42, 123],
        "fair_seeds": [42],
        "recon_epochs": 5,
        "eval_steps": 200,
    },
}

def run(cmd: list[str]) -> None:
    print("\n" + "=" * 72)
    print(" ".join(cmd))
    print("=" * 72, flush=True)
    t0 = time.time()
    subprocess.run(cmd, cwd=ROOT, check=True)
    print(f"[finalization] done ({time.time() - t0:.1f}s)", flush=True)


def evaluate_policy_rollout(cfg, split, max_steps, policy_fn, seed=42):
    import numpy as np

    from campus_senserl.environment.communication_model import TRANSMIT
    from campus_senserl.environment.trace_environment import TraceDrivenCampusEnv

    set_seed(seed)
    env = TraceDrivenCampusEnv(cfg=cfg, split=split, multi_agent=True)
    obs, _ = env.reset(seed=seed)
    steps = 0
    tx = 0
    avail = 0
    tp = fp = fn = 0
    aoi_vals = []
    while steps < max_steps and env._t < env.n_steps - 1:
        actions = policy_fn(obs, env)
        obs, reward, terminated, truncated, info = env.step(actions)
        n = env.n_sensors
        final = info.get("final_actions")
        if final is not None:
            tx += int((final == TRANSMIT).sum())
            avail += n
        tp += int(np.asarray(info.get("detected_events", [])).sum()) if "detected_events" in info else 0
        fn += int(np.asarray(info.get("missed_events", [])).sum()) if "missed_events" in info else 0
        if env.server_state is not None:
            aoi_vals.append(float(env.server_state.aoi.mean()))
        steps += 1
        if terminated or truncated:
            break

    tx_rate = tx / max(avail, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "steps": steps,
        "transmit_rate": float(tx_rate),
        "transmission_reduction": float(1.0 - tx_rate),
        "event_recall": float(recall),
        "tp": tp,
        "fn": fn,
        "mean_aoi": float(np.mean(aoi_vals)) if aoi_vals else float("nan"),
        "p90_aoi": float(np.percentile(aoi_vals, 90)) if aoi_vals else float("nan"),
        "max_aoi": float(np.max(aoi_vals)) if aoi_vals else float("nan"),
        "n_sensors": env.n_sensors,
    }


def main() -> None:
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--profile", default="paper", choices=list(PROFILES))
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-rl", action="store_true")
    parser.add_argument("--skip-fair", action="store_true")
    args = parser.parse_args()
    prof = PROFILES[args.profile]

    device = args.device
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    print(f"[finalization] profile={args.profile} device={device}")
    rl_cfg = load_yaml(ROOT / "configs" / "rl.yaml")

    # ---- Phase 2: cohort ----
    print("[finalization] Phase 2: freeze cohorts")
    freeze_cohorts(debug_n=30, development_n=40, final_n=prof["final_n"])
    sensors = load_cohort("final")
    print(f"[finalization] final cohort n={len(sensors)}")

    # Point config at cohort
    rl_cfg.setdefault("environment", {})["cohort"] = "final"
    rl_cfg["environment"]["graph"] = "hybrid"

    # ---- Phase 3: fair reconstruction ----
    print("[finalization] Phase 3: fair reconstruction benchmark")
    decision_path = ROOT / "results" / "reconstruction_final" / "fair_benchmark" / "reconstructor_decision.json"
    if args.skip_fair and decision_path.exists():
        with open(decision_path, encoding="utf-8") as f:
            decision = json.load(f)
        fair = {"decision": decision}
        print("[finalization] skip fair benchmark (existing decision)")
    else:
        fair = run_fair_benchmark(
            sensors=sensors,
            split="val",
            seeds=prof.get("fair_seeds", prof["seeds"]),
            mask_rate=0.4,
        )
        decision = fair["decision"]
    print("[finalization] reconstructor decision:", decision)

    # Prefer best causal method for RL; keep ST-GNN as evaluated alternative
    best = decision.get("best_by_mae") or "locf"
    if best in {"extratrees", "lightgbm", "locf", "linear_extrapolation", "historical_mean"}:
        rl_cfg["environment"]["reconstructor"] = "locf" if best != "locf" else "locf"
        # Tree models not yet sequential; online LOCF is honest for RL rollout
        rl_cfg["environment"]["proposed_backend"] = "locf"
        rl_cfg["environment"]["reconstructor_note"] = (
            f"Fair benchmark best_by_mae={best}; RL uses LOCF online until sequential tree/GNN adapter matches cohort."
        )
    else:
        rl_cfg["environment"]["reconstructor"] = "locf"

    # Retrain ST-GNN on frozen cohort for fair comparison artifact
    if not args.skip_train:
        recon_cfg = load_yaml(ROOT / "configs" / "reconstruction.yaml")
        recon_cfg.setdefault("neural", {})["epochs"] = prof["recon_epochs"]
        recon_cfg["max_sensors"] = len(sensors)
        # Persist sensors into a temp note file for train script via max_sensors order:
        # train uses node_order prefix — write cohort-aligned node_order override
        cohort_order = {"node_order": sensors}
        ensure_dir(ROOT / "results" / "cohorts")
        save_json(cohort_order, ROOT / "results" / "cohorts" / "node_order_override.json")
        run(
            [
                PY,
                "scripts/04_train_reconstruction.py",
                "--max-sensors",
                str(len(sensors)),
                "--device",
                device,
            ]
        )

    # ---- Phase 7–9: multi-seed PPO / MAPPO ----
    seeds = prof["seeds"]
    if not args.skip_rl:
        for seed in seeds:
            cfg = copy.deepcopy(rl_cfg)
            cfg["seed"] = seed
            cfg.setdefault("ppo", {})["total_timesteps"] = prof["ppo_steps"]
            cfg.setdefault("mappo", {})["total_timesteps"] = prof["mappo_steps"]
            cfg_path = ensure_dir(ROOT / "results" / "rl_final" / "configs") / f"seed_{seed}.yaml"
            import yaml

            with open(cfg_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f)

            ppo_dir = ensure_dir(ROOT / "results" / "rl_final" / "ppo" / f"seed_{seed}")
            mappo_dir = ensure_dir(ROOT / "results" / "rl_final" / "mappo" / f"seed_{seed}")
            if (ppo_dir / "final_model.pt").exists():
                print(f"[finalization] skip PPO seed={seed} (checkpoint exists)")
            else:
                run(
                    [
                        PY,
                        "scripts/06_train_rl.py",
                        "--config",
                        str(cfg_path),
                        "--device",
                        device,
                        "--timesteps",
                        str(prof["ppo_steps"]),
                        "--max-sensors",
                        str(len(sensors)),
                        "--checkpoint-dir",
                        str(ppo_dir),
                    ]
                )
            if (mappo_dir / "final_model.pt").exists():
                print(f"[finalization] skip MAPPO seed={seed} (checkpoint exists)")
            else:
                run(
                    [
                        PY,
                        "scripts/07_train_marl.py",
                        "--config",
                        str(cfg_path),
                        "--device",
                        device,
                        "--timesteps",
                        str(prof["mappo_steps"]),
                        "--max-sensors",
                        str(len(sensors)),
                        "--checkpoint-dir",
                        str(mappo_dir),
                    ]
                )

            # Semantic + safety constrained MAPPO = MAPPO with shield enabled (default)
            sem_dir = ensure_dir(ROOT / "results" / "rl_final" / "semantic_constrained_mappo" / f"seed_{seed}")
            # Copy/link metrics for clarity
            import shutil

            if (mappo_dir / "final_model.pt").exists():
                shutil.copy2(mappo_dir / "final_model.pt", sem_dir / "final_model.pt")
            if (mappo_dir / "metrics.json").exists():
                shutil.copy2(mappo_dir / "metrics.json", sem_dir / "metrics.json")
            save_json(
                {"note": "Semantic+safety uses MAPPO checkpoint with shield enabled at eval", "seed": seed},
                sem_dir / "meta.json",
            )

    # ---- Matched-budget baselines + RL eval (validation) ----
    print("[finalization] matched-budget evaluation on VAL")
    from campus_senserl.rl.fixed_policies import FixedIntervalPolicy
    from campus_senserl.rl.heuristic_policies import ChangeThresholdPolicy

    rows = []
    for interval, label in [(1, "fixed_15"), (2, "fixed_30"), (3, "fixed_45"), (4, "fixed_60")]:
        pol = FixedIntervalPolicy(interval_steps=interval)

        def _fn(obs, env, p=pol):
            return p.act(obs, local_available=env.local_available[env._t])

        m = evaluate_policy_rollout(rl_cfg, "val", prof["eval_steps"], _fn, seed=42)
        m["method"] = label
        rows.append(m)

    thr = ChangeThresholdPolicy(delta_ppm=50.0)

    def _thr(obs, env, p=thr):
        return p.act(obs, local_available=env.local_available[env._t])

    try:
        m = evaluate_policy_rollout(rl_cfg, "val", prof["eval_steps"], _thr, seed=42)
        m["method"] = "change_threshold"
        rows.append(m)
    except Exception as exc:
        rows.append({"method": "change_threshold", "error": str(exc)})

    import pandas as pd

    val_df = pd.DataFrame(rows)
    ensure_dir(ROOT / "results" / "rl_final")
    val_df.to_csv(ROOT / "results" / "rl_final" / "matched_budget_val.csv", index=False)

    # ---- Real robustness (val traces) ----
    print("[finalization] real robustness on VAL traces")
    run(
        [
            PY,
            "scripts/10_run_robustness.py",
            "--split",
            "val",
            "--max-sensors",
            str(min(len(sensors), 40)),
            "--max-steps",
            str(prof["eval_steps"]),
        ]
    )
    rob_src = ROOT / "results" / "robustness" / "robustness.csv"
    if rob_src.exists():
        ensure_dir(ROOT / "results" / "robustness_final")
        import shutil

        shutil.copy2(rob_src, ROOT / "results" / "robustness_final" / "robustness_val.csv")

    # ---- Freeze protocol BEFORE test ----
    protocol = {
        "profile": args.profile,
        "cohort": "final",
        "n_sensors": len(sensors),
        "sensor_ids": sensors,
        "reconstructor": rl_cfg["environment"].get("reconstructor"),
        "reconstructor_note": rl_cfg["environment"].get("reconstructor_note"),
        "graph": rl_cfg["environment"].get("graph"),
        "primary_co2_threshold_ppm": rl_cfg.get("events", {}).get("primary_threshold_ppm", 1000),
        "rapid_increase_ppm": rl_cfg.get("events", {}).get("rapid_increase_ppm", 150),
        "ppo_steps": prof["ppo_steps"],
        "mappo_steps": prof["mappo_steps"],
        "seeds": seeds,
        "eval_steps_val": prof["eval_steps"],
        "device": device,
        "fair_benchmark": fair["decision"],
        "frozen": True,
        "warning": "Do not tune using test results after this file is written.",
    }
    save_json(protocol, ROOT / "results" / "final_protocol.json")
    print("[finalization] wrote results/final_protocol.json — TEST SPLIT NOW FROZEN")

    # ---- One test pass ----
    print("[finalization] final TEST evaluation (once)")
    test_rows = []
    for interval, label in [(1, "fixed_15"), (2, "fixed_30"), (4, "fixed_60")]:
        pol = FixedIntervalPolicy(interval_steps=interval)

        def _fn(obs, env, p=pol):
            return p.act(obs, local_available=env.local_available[env._t])

        m = evaluate_policy_rollout(rl_cfg, "test", prof["eval_steps"], _fn, seed=42)
        m["method"] = label
        m["split"] = "test"
        test_rows.append(m)
    test_df = pd.DataFrame(test_rows)
    ensure_dir(ROOT / "results" / "final_test")
    test_df.to_csv(ROOT / "results" / "final_test" / "policy_comparison_test.csv", index=False)

    # ---- Paper outputs ----
    run([PY, "scripts/12_generate_results.py"])

    # ---- Final research report ----
    report = ROOT / "reports" / "final_research_report.md"
    report.write_text(
        f"""# CAMPUS-SenseRL Final Research Report

Generated by `scripts/13_run_finalization.py` (profile={args.profile}).

## 1. Dataset cohort
- Frozen file: `results/cohorts/final_cohort.json`
- n_sensors = {len(sensors)}
- Selection: train coverage ranking with val/test eligibility gates only

## 2–3. Splits / graphs
- Chronological 60/20/20 (see `configs/data.yaml`)
- Graph: hybrid (`results/graphs/`), correlation fit on train only

## 4–6. Reconstruction
- Fair benchmark: `results/reconstruction_final/fair_benchmark/`
- Decision: `{json.dumps(decision)}`
- Online RL reconstructor: `{rl_cfg['environment'].get('reconstructor')}` ({rl_cfg['environment'].get('reconstructor_note', '')})

## 7–10. RL
- State schema: `reports/rl_state_definition.md`
- Event semantics: reconstruction-aware TP/FN (primary 1000 ppm / rapid 150)
- PPO steps: {prof['ppo_steps']}; MAPPO steps: {prof['mappo_steps']}; seeds: {seeds}
- Critic: mean-pool (variable agent count)

## 11–16. Evaluation artifacts
- Val matched budgets: `results/rl_final/matched_budget_val.csv`
- Robustness: `results/robustness_final/`
- Protocol freeze: `results/final_protocol.json`
- Test (once): `results/final_test/policy_comparison_test.csv`
- Paper pack: `results/`

## Hypotheses (update after inspecting metrics)
| Hypothesis | Classification |
|------------|----------------|
| ST-GNN beats trees on fair MAE | See fair_benchmark summary |
| MAPPO+semantic/safety beats fixed at matched TX | See rl_final + paper tables |
| Reconstruction-aware events justify SKIP | Implemented; check event_recall columns |

## Limitations
- Full sequential ExtraTrees/LightGBM online adapter still deferred; fair table includes batch tree metrics.
- Ablation/robustness scripts still need full retrain-per-ablation for journal strength; current robustness uses real val traces with fixed/interval policies unless RL checkpoints are plugged in.
- Do not present proxy heuristic as the proposed method.

See also: `reports/finalization_audit.md`.
""",
        encoding="utf-8",
    )
    print(f"[finalization] wrote {report}")
    print("[finalization] COMPLETE")


if __name__ == "__main__":
    main()
