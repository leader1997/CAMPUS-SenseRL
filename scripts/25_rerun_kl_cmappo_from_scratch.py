#!/usr/bin/env python
"""From-scratch KL-CMAPPO paper pipeline (no shield, no old MAPPO).

Order: BC distill (17) → KL-CMAPPO train+eval (21) → scientific validation (22)
→ clear figures PNG-only (24).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(cmd: list[str]) -> None:
    print("\n===", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    device = "cuda"
    run([PY, "scripts/17_paper_asap_pipeline.py", "--device", device])
    run(
        [
            PY,
            "scripts/21_final_paper_results.py",
            "--device",
            device,
            "--timesteps",
            "60000",
            "--force-retrain",
        ]
    )
    run(
        [
            PY,
            "scripts/22_scientific_validation.py",
            "--device",
            device,
            "--ablation-timesteps",
            "60000",
            "--force-ablation",
        ]
    )
    run([PY, "scripts/24_final_clear_figures.py"])
    print("\n[done] full KL-CMAPPO pipeline", flush=True)


if __name__ == "__main__":
    main()
