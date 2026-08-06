#!/usr/bin/env python
"""Step 1: Dataset audit."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from campus_senserl.data.audit import run_audit


if __name__ == "__main__":
    run_audit()
