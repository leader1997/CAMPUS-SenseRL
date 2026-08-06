"""Shared utilities: paths, config loading, seeding, I/O."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def repo_root() -> Path:
    """Return repository root (contains configs/, src/, data/)."""
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / "configs" / "data.yaml").exists():
            return parent
    return Path.cwd()


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = repo_root() / path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(*names: str) -> dict[str, Any]:
    """Load and shallow-merge YAML configs from configs/ by stem name."""
    merged: dict[str, Any] = {}
    root = repo_root() / "configs"
    for name in names:
        path = root / (name if name.endswith(".yaml") else f"{name}.yaml")
        cfg = load_yaml(path)
        merged = deep_merge(merged, cfg)
    return merged


def deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    """Deterministic seeding for Python, NumPy, and PyTorch if available."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def normalize_deveui(value: str) -> str:
    """Normalize device EUI to uppercase hex without separators."""
    if value is None:
        return ""
    s = str(value).strip().upper().replace("-", "").replace(":", "")
    return s


def save_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, default=_json_default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def git_commit_hash() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return None


def environment_fingerprint() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version,
        "platform": sys.platform,
        "git_commit": git_commit_hash(),
        "packages": {},
    }
    for pkg in [
        "numpy",
        "pandas",
        "torch",
        "sklearn",
        "gymnasium",
        "lightgbm",
        "xgboost",
        "networkx",
    ]:
        try:
            mod = __import__(pkg if pkg != "sklearn" else "sklearn")
            info["packages"][pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info["packages"][pkg] = None
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda
    except Exception:
        info["cuda_available"] = False
        info["cuda_version"] = None
    return info


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres (WGS84)."""
    r = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))
