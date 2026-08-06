#!/usr/bin/env python
"""Step 2: Preprocess raw traces into interim/processed parquet datasets."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from campus_senserl.data.preprocess import run_preprocess


if __name__ == "__main__":
    run_preprocess()
