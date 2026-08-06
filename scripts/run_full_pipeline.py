#!/usr/bin/env python
"""Run the full CAMPUS-SenseRL pipeline with one command.

Examples
--------
~2 hour paper-useful run (recommended), GPU if available:

    .venv\\Scripts\\python.exe scripts\\run_full_pipeline.py --force --profile 2h

Full longer run:

    .venv\\Scripts\\python.exe scripts\\run_full_pipeline.py --force

Paper pack only:

    .venv\\Scripts\\python.exe scripts\\run_full_pipeline.py --paper-only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


PROFILES = {
    # Quick smoke — not for paper claims
    "fast": {
        "max_sensors": 20,
        "recon_sensors": 20,
        "baseline_sensors": 30,
        "baseline_steps": 400,
        "rl_steps": 5000,
        "marl_steps": 5000,
        "recon_epochs": 10,
        "recon_batch": 32,
        "policy_max_steps": 400,
    },
    # ~2h useful training on RTX-class GPU / still OK on CPU
    "2h": {
        "max_sensors": 30,
        "recon_sensors": 40,
        "baseline_sensors": 40,
        "baseline_steps": 800,
        "rl_steps": 60000,
        "marl_steps": 50000,
        "recon_epochs": 20,
        "recon_batch": 64,
        "policy_max_steps": 800,
    },
    # Longer default quality
    "full": {
        "max_sensors": 40,
        "recon_sensors": 60,
        "baseline_sensors": 40,
        "baseline_steps": 1000,
        "rl_steps": 100000,
        "marl_steps": 150000,
        "recon_epochs": 25,
        "recon_batch": 64,
        "policy_max_steps": 1000,
    },
}


def detect_device(prefer: str = "auto") -> str:
    if prefer in {"cpu", "cuda"}:
        if prefer == "cuda":
            try:
                import torch

                if not torch.cuda.is_available():
                    print("[pipeline] WARNING: --device cuda requested but CUDA unavailable; using cpu")
                    return "cpu"
            except ImportError:
                return "cpu"
        return prefer
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def print_device_banner(device: str) -> None:
    print(f"[pipeline] compute device = {device}")
    try:
        import torch

        print(f"[pipeline] torch {torch.__version__}")
        if device == "cuda" and torch.cuda.is_available():
            print(f"[pipeline] GPU: {torch.cuda.get_device_name(0)}")
            mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[pipeline] GPU memory: {mem:.1f} GiB")
        elif device == "cpu":
            print("[pipeline] CUDA not active (CPU PyTorch build or no GPU).")
    except Exception as e:
        print(f"[pipeline] device probe failed: {e}")


def run_step(name: str, args: list[str]) -> None:
    cmd = [PY, *args]
    print("\n" + "=" * 72)
    print(f"[pipeline] {name}")
    print(" ".join(cmd))
    print("=" * 72, flush=True)
    t0 = time.time()
    subprocess.run(cmd, cwd=ROOT, check=True)
    print(f"[pipeline] done: {name} ({time.time() - t0:.1f}s)", flush=True)


def exists(*parts: str) -> bool:
    return (ROOT.joinpath(*parts)).exists()


def main() -> None:
    parser = argparse.ArgumentParser(description="CAMPUS-SenseRL full pipeline")
    parser.add_argument("--force", action="store_true", help="Re-run all steps even if outputs exist")
    parser.add_argument("--resume", action="store_true", default=True, help="Skip finished steps (default)")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip existing steps")
    parser.add_argument("--paper-only", action="store_true", help="Only regenerate paper_outputs")
    parser.add_argument(
        "--profile",
        choices=["fast", "2h", "full"],
        default=None,
        help="Runtime profile: fast (~45m), 2h (~2h useful), full (longer)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Alias for --profile fast",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Torch device for neural/RL training (default: auto)",
    )
    parser.add_argument("--skip-rl", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true")
    args = parser.parse_args()

    if args.fast and args.profile is None:
        args.profile = "fast"
    if args.profile is None:
        args.profile = "2h"  # sensible default for practical paper runs

    profile = PROFILES[args.profile]
    resume = args.resume and not args.force and not args.no_resume
    device = detect_device(args.device)

    # Larger recon batch on GPU
    recon_batch = profile["recon_batch"]
    if device == "cuda":
        recon_batch = max(recon_batch, 64)

    if args.paper_only:
        print_device_banner(device)
        run_step("12 paper outputs", ["scripts/12_generate_paper_outputs.py"])
        print("\n[pipeline] COMPLETE - see paper_outputs/")
        return

    steps: list[tuple[str, list[str], tuple[str, ...]]] = [
        ("01 audit", ["scripts/01_audit_data.py"], ("outputs/data_summary.json",)),
        ("02 preprocess", ["scripts/02_preprocess.py"], ("data/processed/co2_panel_15min.parquet",)),
        ("03 graphs", ["scripts/03_build_graph.py"], ("outputs/graphs/graph_summary.json",)),
        (
            "05b reconstruction baselines",
            [
                "scripts/05b_run_reconstruction_baselines.py",
                "--split",
                "val",
                "--mask-rate",
                "0.4",
                "--max-sensors",
                str(profile["baseline_sensors"]),
            ],
            ("outputs/reconstruction_baselines/baselines_val_mask0.4.csv",),
        ),
        (
            "04 train reconstruction",
            [
                "scripts/04_train_reconstruction.py",
                "--max-sensors",
                str(profile["recon_sensors"]),
                "--epochs",
                str(profile["recon_epochs"]),
                "--batch-size",
                str(recon_batch),
                "--device",
                device,
            ],
            ("outputs/models/reconstruction/best_model.pt",),
        ),
        (
            "05 evaluate reconstruction",
            [
                "scripts/05_evaluate_reconstruction.py",
                "--split",
                "val",
                "--mask-rate",
                "0.4",
                "--max-sensors",
                str(profile["recon_sensors"]),
                "--device",
                device,
            ],
            ("outputs/models/reconstruction/eval/metrics_val.json",),
        ),
        (
            "08 policy baselines",
            [
                "scripts/08_run_baselines.py",
                "--split",
                "val",
                "--max-sensors",
                str(profile["max_sensors"]),
                "--max-steps",
                str(profile["policy_max_steps"]),
            ],
            ("outputs/baselines/results_val.json",),
        ),
    ]

    if not args.skip_rl:
        steps += [
            (
                "06 train PPO",
                [
                    "scripts/06_train_rl.py",
                    "--max-sensors",
                    str(profile["max_sensors"]),
                    "--timesteps",
                    str(profile["rl_steps"]),
                    "--device",
                    device,
                ],
                ("outputs/experiments/ppo/final_model.pt",),
            ),
            (
                "07 train MAPPO",
                [
                    "scripts/07_train_marl.py",
                    "--max-sensors",
                    str(profile["max_sensors"]),
                    "--timesteps",
                    str(profile["marl_steps"]),
                    "--device",
                    device,
                ],
                ("outputs/experiments/mappo/final_model.pt",),
            ),
        ]

    if not args.skip_ablations:
        steps += [
            ("09 ablations", ["scripts/09_run_ablations.py"], ("outputs/tables/ablations.csv",)),
            ("10 robustness", ["scripts/10_run_robustness.py"], ("outputs/robustness/robustness.csv",)),
        ]

    steps.append(
        (
            "12 paper outputs",
            ["scripts/12_generate_paper_outputs.py"],
            ("paper_outputs/Figure_Index.md",),
        )
    )

    print(f"[pipeline] root={ROOT}")
    print(f"[pipeline] python={PY}")
    print(f"[pipeline] profile={args.profile} resume={resume} skip_rl={args.skip_rl}")
    print_device_banner(device)
    print(
        f"[pipeline] sensors RL={profile['max_sensors']} recon={profile['recon_sensors']} "
        f"PPO={profile['rl_steps']} MAPPO={profile['marl_steps']}"
    )

    for name, cmd, markers in steps:
        if resume and all(exists(m) for m in markers):
            print(f"[pipeline] SKIP (exists): {name} -> {markers[0]}")
            continue
        run_step(name, cmd)

    print("\n" + "=" * 72)
    print("[pipeline] COMPLETE")
    print("Paper figures/tables: paper_outputs/")
    print("Experiment artifacts: outputs/")
    print("=" * 72)


if __name__ == "__main__":
    main()
