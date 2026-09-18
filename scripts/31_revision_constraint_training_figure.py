#!/usr/bin/env python
"""Training-constraint evolution figure from frozen KL-CMAPPO logs.

Reads existing results/rl_final/cmappo_kl/seed_*/metrics.json only.
Does not train, evaluate, or modify frozen numerical result tables.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from campus_senserl.evaluation.revision_metrics import EPS_AOI, EPS_MAE, EPS_MISS
from campus_senserl.utils import ensure_dir, repo_root
from campus_senserl.visualization.revision_style import (
    C_BC,
    C_EXPERT,
    C_KL,
    apply_revision_style,
)

SEEDS = [42, 123, 2024, 3407, 9999]
LOG_REL = "results/rl_final/cmappo_kl/seed_{seed}/metrics.json"
NOMINAL_ROLLOUT_STEPS = 1024

C_MAE = C_KL
C_MISS = C_BC
C_AOI = C_EXPERT
C_BOUNDARY = "#111827"

METRIC_SPEC = [
    {
        "metric": "mae",
        "raw_col": "mean_mae",
        "ratio_col": "mae_ratio",
        "constraint_value": EPS_MAE,
        "label": r"Reconstruction MAE / 9 ppm",
        "color": C_MAE,
    },
    {
        "metric": "event_miss",
        "raw_col": "mean_miss",
        "ratio_col": "miss_ratio",
        "constraint_value": EPS_MISS,
        "label": r"Event miss rate / 0.015",
        "color": C_MISS,
    },
    {
        "metric": "aoi",
        "raw_col": "mean_aoi",
        "ratio_col": "aoi_ratio",
        "constraint_value": EPS_AOI,
        "label": r"AoI / 3.5",
        "color": C_AOI,
    },
]


def is_train_row(row: dict) -> bool:
    return "mean_mae" in row and "mean_miss" in row and "mean_aoi" in row


def load_training_logs(root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for seed in SEEDS:
        path = root / LOG_REL.format(seed=seed)
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen training log: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload["metrics"] if isinstance(payload, dict) else payload
        n_train = 0
        for rec in metrics:
            if not is_train_row(rec):
                continue
            n_train += 1
            rows.append(
                {
                    "step": int(rec["step"]),
                    "seed": int(seed),
                    "mean_mae": float(rec["mean_mae"]),
                    "mean_miss": float(rec["mean_miss"]),
                    "mean_aoi": float(rec["mean_aoi"]),
                    "mean_tx": float(rec.get("mean_tx", np.nan)),
                    "lam_e": float(rec.get("lam_e", np.nan)),
                    "lam_m": float(rec.get("lam_m", np.nan)),
                    "lam_a": float(rec.get("lam_a", np.nan)),
                    "kl_beta": float(rec.get("kl_beta", np.nan)),
                    "source_log": LOG_REL.format(seed=seed).replace("\\", "/"),
                }
            )
        if n_train == 0:
            raise ValueError(f"No training-rollout rows in {path}")
    df = pd.DataFrame(rows).sort_values(["seed", "step"], kind="mergesort").reset_index(drop=True)
    df["mae_ratio"] = df["mean_mae"] / EPS_MAE
    df["miss_ratio"] = df["mean_miss"] / EPS_MISS
    df["aoi_ratio"] = df["mean_aoi"] / EPS_AOI
    return df


def confirm_alignment(df: pd.DataFrame) -> dict:
    step_map = {int(seed): set(g["step"].astype(int)) for seed, g in df.groupby("seed")}
    common = set.intersection(*step_map.values()) if step_map else set()
    union = set.union(*step_map.values()) if step_map else set()
    aligned = common == union and all(len(s) == len(common) for s in step_map.values())
    if not aligned:
        raise RuntimeError("Training steps do not align exactly across the five frozen runs.")
    steps = sorted(common)
    diffs = np.diff(steps)
    return {
        "aligned_exactly": True,
        "n_steps": len(steps),
        "step_min": int(steps[0]),
        "step_max": int(steps[-1]),
        "step_diffs_unique": [int(x) for x in sorted(set(diffs.tolist()))],
        "final_interval_steps": int(diffs[-1]) if len(diffs) else None,
        "nominal_rollout_steps": NOMINAL_ROLLOUT_STEPS,
        "n_runs": len(SEEDS),
        "n_logged_steps_per_run": {str(s): int(len(step_map[s])) for s in SEEDS},
    }


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for spec in METRIC_SPEC:
        for step, g in df.groupby("step"):
            raw = pd.to_numeric(g[spec["raw_col"]], errors="coerce")
            ratio = pd.to_numeric(g[spec["ratio_col"]], errors="coerce")
            finite_raw = raw[np.isfinite(raw.to_numpy(dtype=float))]
            finite_ratio = ratio[np.isfinite(ratio.to_numpy(dtype=float))]
            n = int(len(finite_ratio))
            if n == 0:
                mean = sd = nmean = nsd = float("nan")
            elif n == 1:
                mean = float(finite_raw.iloc[0])
                nmean = float(finite_ratio.iloc[0])
                sd = nsd = float("nan")
            else:
                mean = float(finite_raw.mean())
                sd = float(finite_raw.std(ddof=1))
                nmean = float(finite_ratio.mean())
                nsd = float(finite_ratio.std(ddof=1))
            rows.append(
                {
                    "step": int(step),
                    "metric": spec["metric"],
                    "mean": mean,
                    "sd": sd,
                    "n_runs": n,
                    "constraint_value": spec["constraint_value"],
                    "normalized_mean": nmean,
                    "normalized_sd": nsd,
                    "plot_mean": nmean,
                    "plot_sd": nsd,
                    "smoothing": "none",
                }
            )
    return pd.DataFrame(rows).sort_values(["metric", "step"], kind="mergesort").reset_index(drop=True)


def miss_diagnostics(df: pd.DataFrame) -> dict:
    miss = pd.to_numeric(df["mean_miss"], errors="coerce")
    n_nan = int(miss.isna().sum())
    n_inf = int(np.isinf(miss.to_numpy(dtype=float)).sum())
    n_zero = int((miss == 0).sum())
    return {
        "n_logged_train_points": int(len(df)),
        "n_nan": n_nan,
        "n_inf": n_inf,
        "n_zero": n_zero,
        "zeros_per_seed": {str(s): int((g["mean_miss"] == 0).sum()) for s, g in df.groupby("seed")},
        "min": float(miss.min()),
        "max": float(miss.max()),
        "handling": (
            "No NaN/Inf event-miss values exist in the frozen logs. "
            "Zeros are kept as logged. They were not converted to NaN, zero-filled, "
            "or forward-filled at figure time. CMAPPOTrainer._step_costs already stores "
            "j_miss=0.0 when a step has no locally available true events (n_te==0), so "
            "logged zeros mix genuine zero-miss rollouts with no-event placeholders. "
            "Those two cases cannot be separated from metrics.json."
        ),
    }


def save_png_pdf(fig: plt.Figure, png_path: Path, dpi: int = 600) -> None:
    ensure_dir(png_path.parent)
    fig.savefig(png_path, dpi=dpi, facecolor="white", edgecolor="white", bbox_inches="tight")
    fig.savefig(png_path.with_suffix(".pdf"), facecolor="white", edgecolor="white", bbox_inches="tight")
    plt.close(fig)


def fig_constraint_training(summary: pd.DataFrame, figdir: Path) -> Path:
    apply_revision_style()
    fig, ax = plt.subplots(figsize=(6.8, 4.7))
    for spec in METRIC_SPEC:
        g = summary[summary["metric"] == spec["metric"]].sort_values("step")
        x = g["step"].to_numpy(dtype=float)
        y = g["plot_mean"].to_numpy(dtype=float)
        sd = g["plot_sd"].to_numpy(dtype=float)
        ax.fill_between(x, y - sd, y + sd, color=spec["color"], alpha=0.18, linewidth=0, zorder=2)
        ax.plot(x, y, color=spec["color"], lw=2.0, zorder=3, label=spec["label"])
    ax.axhline(1.0, color=C_BOUNDARY, ls="--", lw=1.2, zorder=1, label="Constraint boundary")
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Normalized constraint value")
    ax.set_title(
        "How do monitoring-constraint estimates evolve during KL-CMAPPO training?\n"
        "On-policy rollout estimates — not validation or test constraint checks",
        fontsize=11,
    )
    ax.set_xlim(float(summary["step"].min()), float(summary["step"].max()))
    ymax = float(np.nanmax(summary["plot_mean"] + summary["plot_sd"]))
    ax.set_ylim(0.0, max(1.55, ymax + 0.12))
    ax.legend(
        frameon=False,
        fontsize=8,
        loc="upper right",
        title="Five-run mean ± sample SD",
        title_fontsize=8,
    )
    ax.grid(True, alpha=0.18)
    fig.tight_layout()
    out = figdir / "fig_revision_constraint_training.png"
    save_png_pdf(fig, out)
    return out


def fig_lagrange(df: pd.DataFrame, figdir: Path, valuedir: Path) -> Path:
    apply_revision_style()
    specs = [
        ("lam_m", r"$\lambda_{\mathrm{MAE}}$", C_MAE),
        ("lam_e", r"$\lambda_{\mathrm{event}}$", C_MISS),
        ("lam_a", r"$\lambda_{\mathrm{AoI}}$", C_AOI),
    ]
    rows = []
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for col, label, color in specs:
        g = (
            df.groupby("step")[col]
            .agg(mean="mean", sd=lambda s: float(s.std(ddof=1)), n_runs="size")
            .reset_index()
        )
        ax.fill_between(
            g["step"],
            g["mean"] - g["sd"],
            g["mean"] + g["sd"],
            color=color,
            alpha=0.18,
            linewidth=0,
            zorder=2,
        )
        ax.plot(g["step"], g["mean"], color=color, lw=2.0, zorder=3, label=label)
        for _, r in g.iterrows():
            rows.append(
                {
                    "step": int(r["step"]),
                    "metric": col,
                    "mean": float(r["mean"]),
                    "sd": float(r["sd"]),
                    "n_runs": int(r["n_runs"]),
                }
            )
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Lagrange multiplier (rollout dual variable)")
    ax.set_title(
        "Lagrange multipliers during KL-CMAPPO training (supplementary)\n"
        "Dual variables on rollout estimates — not validation/test feasibility",
        fontsize=11,
    )
    ax.legend(frameon=False, fontsize=8, loc="upper right", title="Five-run mean ± sample SD", title_fontsize=8)
    ax.grid(True, alpha=0.18)
    fig.tight_layout()
    out = figdir / "supp_fig_lagrange_evolution.png"
    save_png_pdf(fig, out)
    pd.DataFrame(rows).to_csv(valuedir / "lagrange_evolution_summary.csv", index=False)
    df[["step", "seed", "lam_e", "lam_m", "lam_a"]].to_csv(valuedir / "lagrange_evolution.csv", index=False)
    return out


def write_caption(path: Path) -> None:
    path.write_text(
        (
            "Evolution of normalized monitoring-constraint estimates during KL-CMAPPO "
            "training across five independent runs. Reconstruction MAE, event-miss rate, "
            "and Age of Information are divided by their corresponding constraint limits "
            "(9 ppm, 0.015, and 3.5 decision intervals). The horizontal line at 1 denotes "
            "the constraint boundary. Curves show the five-run mean and shaded regions "
            "indicate sample standard deviation (ddof = 1). These curves represent "
            "rollout-level training estimates from on-policy windows (1024 environment "
            "steps, except the final 608-step remainder to 60 000 total steps). They are "
            "not validation or test constraint checks. Final constraint satisfaction is "
            "evaluated separately on the validation and test data. Logged event-miss "
            "values of 0 are retained as stored by the trainer: windows with no locally "
            "available true events were recorded as 0 rather than as undefined, so zeros "
            "mix genuine zero-miss rollouts with no-event placeholders.\n"
        ),
        encoding="utf-8",
    )


def write_protocol(path: Path, *, alignment: dict, miss: dict, logs: list[str]) -> None:
    payload = {
        "purpose": "Document construction of fig_revision_constraint_training.png from frozen KL-CMAPPO training logs.",
        "no_training": True,
        "no_evaluation_rerun": True,
        "seeds": SEEDS,
        "source_logs": logs,
        "row_filter": "Keep metrics.json entries that contain mean_mae, mean_miss, and mean_aoi. Exclude periodic validation rows (val_mae/val_recall).",
        "logged_fields_used": ["step", "mean_mae", "mean_miss", "mean_aoi"],
        "logged_fields_not_plotted_on_main_figure": ["mean_tx", "lam_e", "lam_m", "lam_a", "kl_beta"],
        "fields_never_invented": ["policy_loss", "entropy", "gradient_norm", "actual_kl_divergence"],
        "normalization": {
            "mae_ratio": "mean_mae / 9.0",
            "miss_ratio": "mean_miss / 0.015",
            "aoi_ratio": "mean_aoi / 3.5",
            "interpretation": {
                "<1": "rollout estimate below the predefined constraint limit",
                "=1": "constraint boundary",
                ">1": "rollout estimate exceeds the predefined limit",
            },
            "not_final_constraint_satisfaction": True,
        },
        "aggregation": {
            "independent_units": "five frozen training runs (seeds)",
            "aligned_exactly": alignment["aligned_exactly"],
            "n_steps": alignment["n_steps"],
            "n_logged_steps_per_run": alignment["n_logged_steps_per_run"],
            "step_min": alignment["step_min"],
            "step_max": alignment["step_max"],
            "step_diffs_unique": alignment["step_diffs_unique"],
            "final_interval_steps": alignment["final_interval_steps"],
            "interpolation": "none; steps matched exactly across seeds",
            "mean": "arithmetic mean across the five runs at each common step",
            "sd": "sample standard deviation, ddof=1, across the five runs at each common step",
            "do_not_treat_log_rows_as_independent_experiments": True,
        },
        "event_miss": miss,
        "smoothing": {
            "applied": False,
            "method": "none",
            "reason": "Five-run mean curves are plotted from the logged values. Seed-to-seed SD is small; remaining variation is temporal structure of on-policy windows, not plotting noise.",
            "plot_columns_equal_raw_normalized_columns": True,
        },
        "trainer_event_miss_definition": (
            "campus_senserl.rl.cmappo.CMAPPOTrainer._step_costs: "
            "j_miss = missed[local].sum() / n_te if n_te > 0 else 0.0, "
            "then mean_miss = mean(j_miss) over the rollout window."
        ),
        "scientific_caveat": (
            "Training-rollout event-miss estimates stay below 0.015 in these logs, "
            "including zeros that may reflect no-event windows. This must not be read "
            "as frozen temporal TEST event-constraint satisfaction (0/5 on the frozen test)."
        ),
        "reproduction_command": "python scripts/31_revision_constraint_training_figure.py",
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_manifest(path: Path) -> None:
    rows = [
        {
            "figure_role": "Figure 1",
            "filename": "fig04_kl_cmappo_framework.png",
            "paper_section": "Introduction / system",
            "scientific_question": "How do campus agents decide to transmit or skip, and how is the server monitor formed?",
            "main_or_supplementary": "main",
            "source_data": "scripts/28_manuscript_figures.py fig04",
            "notes": "Existing system/decision-process flowchart in this repository (local sensing, TX/SKIP, server reconstruction). If the manuscript already has a separate system schematic, keep that as Figure 1 and use this file as Figure 2 only. Not regenerated here.",
        },
        {
            "figure_role": "Figure 2",
            "filename": "fig04_kl_cmappo_framework.png",
            "paper_section": "Methods",
            "scientific_question": "How is CAMPUS-SenseRL (KL-CMAPPO) trained from expert demonstrations and BC initialization?",
            "main_or_supplementary": "main",
            "source_data": "scripts/28_manuscript_figures.py fig04",
            "notes": "Existing learning/training framework figure. Same PNG currently also depicts the system/decision loop. Not regenerated here.",
        },
        {
            "figure_role": "Figure 3",
            "filename": "fig01_campus_deployment.png",
            "paper_section": "Dataset",
            "scientific_question": "Where are the development and held-out CO2 sensors located?",
            "main_or_supplementary": "main",
            "source_data": "results/cohorts plus device metadata; scripts/28_manuscript_figures.py fig01",
            "notes": "Existing spatial distribution of development vs held-out sensor locations. Not regenerated here.",
        },
        {
            "figure_role": "Figure 4",
            "filename": "fig_revision_constraint_training.png",
            "paper_section": "Results — training dynamics",
            "scientific_question": "How do monitoring-constraint estimates evolve during KL-CMAPPO training?",
            "main_or_supplementary": "main",
            "source_data": "results/figures/revision_values/constraint_training.csv; constraint_training_summary.csv",
            "notes": "New. Five frozen training logs. Rollout estimates only; not VAL/TEST constraint satisfaction. Mean ± sample SD (ddof=1). No smoothing.",
        },
        {
            "figure_role": "Figure 5",
            "filename": "fig_revision_tradeoff_mae.png",
            "paper_section": "Results — reconstruction trade-off",
            "scientific_question": "Who keeps low reconstruction MAE at high communication reduction?",
            "main_or_supplementary": "main",
            "source_data": "results/figures/revision_values/fig_revision_tradeoff_mae.csv",
            "notes": "Existing revision figure. MAE in ppm. CAMPUS-SenseRL (KL-CMAPPO) named consistently. Not redesigned here.",
        },
        {
            "figure_role": "Figure 6",
            "filename": "fig_revision_event_miss.png",
            "paper_section": "Results — event preservation",
            "scientific_question": "Who preserves high-CO2 or rapid-rise events relative to the 1.5% miss requirement?",
            "main_or_supplementary": "main",
            "source_data": "results/figures/revision_values/fig_revision_event_miss.csv",
            "notes": "Existing revision figure. Event miss as percent. 1.5% constraint line. Not redesigned here.",
        },
        {
            "figure_role": "Figure 7",
            "filename": "fig_revision_generalization_dumbbell.png",
            "paper_section": "Results — generalization",
            "scientific_question": "Does reconstruction MAE change from development TEST to held-out TEST on the same chronological interval?",
            "main_or_supplementary": "main",
            "source_data": "results/figures/revision_values/fig_revision_generalization_dumbbell.csv",
            "notes": "Existing revision figure. Same-split TEST vs held-out TEST. Not VAL vs held-out. Not redesigned here.",
        },
        {
            "figure_role": "Figure 8",
            "filename": "fig_revision_packet_loss_mae.png",
            "paper_section": "Results — robustness",
            "scientific_question": "How does reconstruction MAE degrade under increasing packet loss?",
            "main_or_supplementary": "main",
            "source_data": "results/figures/revision_values/fig_revision_packet_loss_mae.csv",
            "notes": "Existing revision figure. Packet loss in percent; MAE in ppm. Not redesigned here.",
        },
        {
            "figure_role": "Supplementary",
            "filename": "fig_revision_contextual_relations.png",
            "paper_section": "Supplementary — context graph",
            "scientific_question": "What spatial/statistical contextual relations exist among development sensors?",
            "main_or_supplementary": "supplementary",
            "source_data": "results/figures/revision_values/fig_revision_contextual_relations.csv",
            "notes": "Not a communication network. Do not include in the main-paper figure set.",
        },
        {
            "figure_role": "Supplementary",
            "filename": "fig05_adaptive_communication.png",
            "paper_section": "Supplementary — illustration",
            "scientific_question": "What does an illustrative event window look like under KL-CMAPPO?",
            "main_or_supplementary": "supplementary",
            "source_data": "results/figures/fig05_window.json",
            "notes": "Old illustrative event-window figure. Not primary quantitative evidence.",
        },
        {
            "figure_role": "Supplementary",
            "filename": "fig07_agent_budget_vs_information.png",
            "paper_section": "Supplementary — behaviour",
            "scientific_question": "Do agents spend transmissions where the CO2 series is more variable?",
            "main_or_supplementary": "supplementary",
            "source_data": "results/figures/figure_values.csv (fig07)",
            "notes": "Variability-versus-transmission-rate figure from the original manuscript pack.",
        },
        {
            "figure_role": "Supplementary",
            "filename": "fig08_diurnal_transmission_profile.png",
            "paper_section": "Supplementary — behaviour",
            "scientific_question": "Does transmission follow campus occupancy over the day?",
            "main_or_supplementary": "supplementary",
            "source_data": "results/figures/figure_values.csv (fig08)",
            "notes": "Diurnal profile from the original manuscript pack.",
        },
        {
            "figure_role": "Supplementary",
            "filename": "fig_revision_constraint_matrix.png",
            "paper_section": "Supplementary — constraint table",
            "scientific_question": "Which methods pass MAE, event, and AoI constraints on validation?",
            "main_or_supplementary": "supplementary",
            "source_data": "results/figures/revision_values/fig_revision_constraint_matrix.csv",
            "notes": "VAL pass/fail matrix. Keep supplementary; numeric k/n belong in tables.",
        },
        {
            "figure_role": "Supplementary",
            "filename": "fig15_kl_gain_over_bc_anchor.png",
            "paper_section": "Supplementary — matched budget",
            "scientific_question": "What does KL-CMAPPO add over BC initialization at matched communication budgets?",
            "main_or_supplementary": "supplementary",
            "source_data": "results/revision/table_matched_budget.csv",
            "notes": "BC/KL matched-budget figure. KL points are direct VAL tau measurements; BC remains interpolated_legacy. Do not present mixed sources as primary evidence.",
        },
        {
            "figure_role": "Supplementary",
            "filename": "results/revision/ablation.csv",
            "paper_section": "Supplementary — ablation",
            "scientific_question": "What do KL regularization and constraints contribute relative to BC initialization?",
            "main_or_supplementary": "supplementary",
            "source_data": "results/revision/ablation.csv",
            "notes": "Existing seed-42 ablation table. No ablation PNG is currently in results/figures/. Single-run; no significance claims. Not regenerated.",
        },
        {
            "figure_role": "Supplementary",
            "filename": "supp_fig_lagrange_evolution.png",
            "paper_section": "Supplementary — dual variables",
            "scientific_question": "How do MAE, event, and AoI Lagrange multipliers evolve during training?",
            "main_or_supplementary": "supplementary",
            "source_data": "results/figures/revision_values/lagrange_evolution.csv",
            "notes": "Optional diagnostic. Dual variables on rollout estimates, with floors. Do not plot kl_beta as KL divergence. Not a main-paper figure.",
        },
        {
            "figure_role": "Internal review only",
            "filename": "revision_main_figures_preview.png",
            "paper_section": "none",
            "scientific_question": "Contact sheet of the five main result figures for visual QC.",
            "main_or_supplementary": "internal_review",
            "source_data": "the five main result PNGs listed above",
            "notes": "NOT for the manuscript.",
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def contact_sheet(figdir: Path) -> Path:
    panels = [
        ("fig_revision_constraint_training.png", "Fig. 4  Training-constraint estimates"),
        ("fig_revision_tradeoff_mae.png", "Fig. 5  Reconstruction trade-off"),
        ("fig_revision_event_miss.png", "Fig. 6  Event-miss rate"),
        ("fig_revision_generalization_dumbbell.png", "Fig. 7  Same-split generalization"),
        ("fig_revision_packet_loss_mae.png", "Fig. 8  Packet-loss MAE"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.2), facecolor="white")
    axes = axes.ravel()
    for ax, (name, title) in zip(axes, panels):
        img = np.asarray(Image.open(figdir / name).convert("RGB"))
        ax.imshow(img)
        ax.set_title(title, fontsize=10, pad=6)
        ax.axis("off")
    axes[5].axis("off")
    axes[5].text(
        0.5,
        0.5,
        "Internal review only\nNot for the manuscript",
        ha="center",
        va="center",
        fontsize=11,
        color="#4B5563",
        transform=axes[5].transAxes,
    )
    fig.suptitle("CAMPUS-SenseRL main result figures (preview)", fontsize=13, y=0.98)
    fig.tight_layout()
    out = figdir / "revision_main_figures_preview.png"
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    root = repo_root()
    figdir = ensure_dir(root / "results" / "figures")
    valuedir = ensure_dir(figdir / "revision_values")
    out = ensure_dir(root / "results" / "revision")

    df = load_training_logs(root)
    alignment = confirm_alignment(df)
    miss = miss_diagnostics(df)
    summary = summarize(df)

    individual = df[["step", "seed", "mean_mae", "mean_miss", "mean_aoi", "mae_ratio", "miss_ratio", "aoi_ratio"]].copy()
    individual.to_csv(valuedir / "constraint_training.csv", index=False)
    summary.to_csv(valuedir / "constraint_training_summary.csv", index=False)

    fig_path = fig_constraint_training(summary, figdir)
    lam_path = fig_lagrange(df, figdir, valuedir)
    write_caption(out / "fig_constraint_training_caption.txt")
    logs = [LOG_REL.format(seed=s).replace("\\", "/") for s in SEEDS]
    write_protocol(out / "training_constraint_figure_protocol.json", alignment=alignment, miss=miss, logs=logs)
    write_manifest(out / "figure_manifest.csv")
    preview = contact_sheet(figdir)

    print(f"[done] {fig_path}")
    print(f"[done] {lam_path}")
    print(f"[done] {preview}")
    print(f"[done] aligned_exactly={alignment['aligned_exactly']} n_steps={alignment['n_steps']}")
    print(f"[done] mean_miss nan={miss['n_nan']} inf={miss['n_inf']} zeros={miss['n_zero']}")


if __name__ == "__main__":
    main()
