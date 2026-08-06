#!/usr/bin/env python
"""Step 3: Build spatial/correlation/hybrid sensor graphs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from campus_senserl.graph import run_build_graphs
from campus_senserl.utils import load_yaml, repo_root


if __name__ == "__main__":
    cfg = load_yaml(repo_root() / "configs" / "model.yaml")
    run_build_graphs(cfg)
