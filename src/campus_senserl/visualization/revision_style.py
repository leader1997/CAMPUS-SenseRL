"""Shared identity for second-revision paper figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from campus_senserl.utils import ensure_dir
from campus_senserl.visualization.paper_style import apply_paper_style

# Wong/IBM colorblind-safe palette
C_PERIODIC = "#7F7F7F"
C_DELTA = "#E69F00"
C_EXPERT = "#009E73"
C_BC = "#0072B2"
C_KL = "#D55E00"
C_PARETO = "#000000"
C_GRID = "#C8C8C8"

MARKERS = {
    "fixed": "s",
    "delta_plus_heartbeat": "P",
    "semantic_expert": "D",
    "campus_senserl_bc": "^",
    "cmappo_kl": "o",
}

DISPLAY_ORDER = [
    "fixed_15",
    "fixed_30",
    "fixed_45",
    "fixed_60",
    "fixed_75",
    "fixed_90",
    "delta_plus_heartbeat",
    "semantic_expert",
    "campus_senserl_bc",
    "cmappo_kl",
]


def apply_revision_style() -> None:
    apply_paper_style()
    mpl.rcParams.update(
        {
            "savefig.dpi": 600,
            "figure.dpi": 150,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "axes.grid": False,
        }
    )


def method_color(method: str) -> str:
    if str(method).startswith("fixed_"):
        return C_PERIODIC
    return {
        "delta_plus_heartbeat": C_DELTA,
        "semantic_expert": C_EXPERT,
        "campus_senserl_bc": C_BC,
        "cmappo_kl": C_KL,
    }.get(str(method), "#111827")


def method_marker(method: str) -> str:
    if str(method).startswith("fixed_"):
        return MARKERS["fixed"]
    return MARKERS.get(str(method), "o")


def method_zorder(method: str) -> int:
    if method == "cmappo_kl":
        return 6
    if method in {"semantic_expert", "campus_senserl_bc", "delta_plus_heartbeat"}:
        return 5
    return 3


def save_revision_figure(fig: plt.Figure, path: Path, dpi: int = 600) -> None:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, facecolor="white", edgecolor="white", bbox_inches="tight")
    plt.close(fig)
