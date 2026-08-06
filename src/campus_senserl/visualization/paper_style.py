"""Publication visual style for CAMPUS-SenseRL manuscript figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from campus_senserl.utils import ensure_dir

# Consistent palette (colourblind-friendly, non-purple default)
COLORS = {
    "fixed": "#4C78A8",
    "random": "#9ECBE2",
    "heuristic": "#F58518",
    "ppo": "#54A24B",
    "marl": "#E45756",
    "proposed": "#B279A2",
    "locf": "#4C78A8",
    "tree": "#54A24B",
    "neural": "#E45756",
    "upper": "#B0B0B0",
    "co2": "#1F4E79",
    "tx": "#D62728",
    "skip": "#7F7F7F",
    "recon": "#2CA02C",
    "unc": "#FF7F0E",
    "event": "#C44E52",
}

METHOD_COLORS = {
    "Fixed 15 min": COLORS["fixed"],
    "Fixed 30 min": COLORS["fixed"],
    "Fixed 45 min": COLORS["fixed"],
    "Fixed 60 min": COLORS["fixed"],
    "Random": COLORS["random"],
    "Change threshold": COLORS["heuristic"],
    "Uncertainty heuristic": COLORS["heuristic"],
    "AoI threshold": COLORS["heuristic"],
    "CO2 threshold": COLORS["heuristic"],
    "Info-value heuristic": COLORS["heuristic"],
    "PPO": COLORS["ppo"],
    "MAPPO": COLORS["marl"],
    "Proposed (semantic + shield)": COLORS["proposed"],
}


def apply_paper_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def save_paper_figure(fig: plt.Figure, path: Path, dpi: int = 300) -> None:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, facecolor="white", edgecolor="white", bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white", edgecolor="white", bbox_inches="tight")
    plt.close(fig)
