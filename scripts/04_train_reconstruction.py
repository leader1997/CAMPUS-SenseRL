#!/usr/bin/env python
"""Step 4: Train masked spatio-temporal graph reconstruction model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from campus_senserl.models.graph_reconstruction import train_reconstruction_model
from campus_senserl.utils import load_yaml, repo_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Train reconstruction model")
    parser.add_argument("--max-sensors", type=int, default=None, help="Subset sensors for debugging")
    parser.add_argument("--config", type=str, default="reconstruction.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "auto"])
    args = parser.parse_args()

    cfg = load_yaml(repo_root() / "configs" / args.config)
    if args.max_sensors is not None:
        cfg["max_sensors"] = args.max_sensors
    if args.epochs is not None:
        cfg.setdefault("neural", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg.setdefault("neural", {})["batch_size"] = args.batch_size

    device = None if args.device in (None, "auto") else args.device
    summary = train_reconstruction_model(cfg, max_sensors=args.max_sensors, device=device)
    print(f"[done] best_val_loss={summary['best_val_loss']:.4f} checkpoint={summary['checkpoint']} device={summary.get('device')}")


if __name__ == "__main__":
    main()
