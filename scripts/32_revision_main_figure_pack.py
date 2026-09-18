#!/usr/bin/env python
"""Clean main-paper figures from frozen revision tables. No training or re-evaluation.

Regenerates:
  - fig01 campus map legend
  - fig_revision_methodology (Figure 2)
  - four main result figures (PNG + PDF)
  - figure_manifest.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from campus_senserl.data.audit import load_devices
from campus_senserl.data.cohort import load_cohort
from campus_senserl.evaluation.revision_export import selected_delta_cfg
from campus_senserl.utils import ensure_dir, load_yaml, repo_root
from campus_senserl.visualization.revision_style import (
    C_BC,
    C_EXPERT,
    C_KL,
    apply_revision_style,
    save_revision_figure,
)

# Import figure functions from 30 after path setup.
sys.path.insert(0, str(ROOT / "scripts"))
from importlib.machinery import SourceFileLoader

_fig30 = SourceFileLoader("revision_figures_30", str(ROOT / "scripts" / "30_revision_figures.py")).load_module()


def load_frozen_revision(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rev = root / "results" / "revision"
    master = pd.read_csv(rev / "all_method_evaluations.csv")
    summary = pd.read_csv(rev / "all_method_summary.csv")
    return master, summary


def fig_campus_deployment(root: Path, figdir: Path) -> Path:
    apply_revision_style()
    cfg = load_yaml(root / "configs" / "data.yaml")
    d = load_devices(root / cfg["paths"]["raw_release"] / cfg["dataset"]["devices_file"]).copy()
    d["deveui"] = d["device_id"].astype(str).str.upper()
    d["is_co2"] = d["device_type"].astype(str).str.contains("CO2", case=False, na=False)
    tr_ids = {s.upper() for s in load_cohort("final")}
    ho_ids = {s.upper() for s in load_cohort("heldout")}
    co2 = d[d["is_co2"]]
    tr, ho = co2[co2["deveui"].isin(tr_ids)], co2[co2["deveui"].isin(ho_ids)]
    bg = co2[~co2["deveui"].isin(tr_ids | ho_ids)]
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    ax.scatter(bg["longitude"], bg["latitude"], s=14, c="#D9DCE1", alpha=0.75, label=f"Other CO$_2$ sensors (n={len(bg)})", zorder=1)
    ax.scatter(tr["longitude"], tr["latitude"], s=52, c="#1F2937", edgecolors="white", linewidths=0.5, label=f"Development cohort (n={len(tr)})", zorder=3)
    ax.scatter(ho["longitude"], ho["latitude"], s=58, c="#0369A1", marker="D", edgecolors="white", linewidths=0.5, label=f"Held-out cohort (n={len(ho)})", zorder=4)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_aspect("equal", adjustable="box")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    out = figdir / "fig01_campus_deployment.png"
    save_revision_figure(fig, out, also_pdf=True)
    return out


def _box(ax, x, y, w, h, text, fc, ec, fs=8.6, lw=1.3, weight="normal"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            lw=lw,
            ec=ec,
            fc=fc,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight=weight, color="#111827")


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=11,
            lw=1.35,
            color="#374151",
        )
    )


def fig_methodology(figdir: Path) -> Path:
    """Corrected Figure 2: training path and residual actor at deployment."""
    apply_revision_style()
    fig, ax = plt.subplots(figsize=(7.6, 5.5))
    ax.set_xlim(0, 10.25)
    ax.set_ylim(0, 7.05)
    ax.axis("off")

    ax.text(0.18, 6.72, "Training", fontsize=11, fontweight="bold", color="#111827")
    ax.text(0.18, 2.78, "Deployment", fontsize=11, fontweight="bold", color="#111827")

    y_t, h_t = 4.15, 1.70
    w_small, w_last = 1.50, 2.58
    xs = [0.16, 1.88, 3.60, 5.32, 7.48]
    train = [
        (xs[0], w_small, "Historical\ncampus traces", "#F3F4F6", "#6B7280", 1.3),
        (xs[1], w_small, "Semantic expert\ndemonstrations", "#DCFCE7", C_EXPERT, 1.3),
        (xs[2], w_small, "Behaviour-cloning\ninitialization", "#DBEAFE", C_BC, 1.3),
        (xs[3], w_small, "KL-CMAPPO\nrefinement", "#FCE7F3", C_KL, 1.35),
        (
            xs[4],
            w_last,
            "Final shared actor\nlearned neural component\n+ fixed expert residual",
            "#FEE2E2",
            C_KL,
            1.85,
        ),
    ]
    for x, w, text, fc, ec, lw in train:
        _box(ax, x, y_t, w, h_t, text, fc, ec, fs=8.0, lw=lw)
    for (a, wa, *_), (b, *_rest) in zip(train, train[1:]):
        _arrow(ax, a + wa + 0.02, y_t + h_t / 2, b - 0.03, y_t + h_t / 2)

    y_d, h_d, w_d = 0.55, 1.48, 2.12
    xd = [0.22, 2.72, 5.22, 7.72]
    deploy = [
        (xd[0], "Observation", "#F3F4F6", "#6B7280", 1.3),
        (xd[1], "Final shared actor\nneural + residual", "#FEE2E2", C_KL, 1.85),
        (xd[2], "Transmit / skip", "#FEE2E2", C_KL, 1.35),
        (xd[3], "Server monitoring\nstate", "#F3F4F6", "#6B7280", 1.3),
    ]
    for x, text, fc, ec, lw in deploy:
        _box(ax, x, y_d, w_d, h_d, text, fc, ec, fs=8.3, lw=lw)
    for a, b in zip(xd, xd[1:]):
        _arrow(ax, a + w_d + 0.02, y_d + h_d / 2, b - 0.03, y_d + h_d / 2)

    ax.text(
        xs[4] + w_last / 2,
        y_t - 0.28,
        "same actor is used at runtime",
        fontsize=8,
        color=C_KL,
        ha="center",
        va="top",
        style="italic",
    )
    out = figdir / "fig_revision_methodology.png"
    save_revision_figure(fig, out, also_pdf=True)
    return out


def write_captions(out: Path) -> None:
    (out / "fig_methodology_caption.txt").write_text(
        (
            "CAMPUS-SenseRL training and deployment. Historical campus traces are used to "
            "generate semantic-expert demonstrations, which initialize a behaviour-cloned "
            "policy. KL-CMAPPO then refines that policy. The deployed actor is a shared "
            "ResidualSharedActor whose logit is the sum of a learned neural component and a "
            "fixed expert-informed residual; the residual is evaluated at every forward pass "
            "and does not disappear after training. At runtime each agent maps an observation "
            "to transmit or skip, and the server updates its monitoring state from delivered "
            "samples and causal reconstruction of skipped slots. SemanticExpertPolicy is not "
            "called at deployment.\n"
        ),
        encoding="utf-8",
    )
    (out / "fig_tradeoff_mae_caption.txt").write_text(
        (
            "Reconstruction MAE versus transmission reduction on the development-cohort "
            "validation split (zero packet loss). Periodic Fixed-30…90 sensors form the "
            "communication–accuracy frontier; Fixed-60 and Fixed-75 are labelled. Light "
            "markers show the Delta+heartbeat grid, outlined markers its Pareto set, and the "
            "filled marker the VAL-selected configuration (Δ = 25 ppm, heartbeat = 3). The "
            "semantic expert and CAMPUS-SenseRL (KL-CMAPPO) are shown at their native "
            "operating points. CAMPUS-SenseRL markers are the five-run mean ± sample SD "
            "(ddof = 1). The dashed line is the 9 ppm MAE constraint. Behaviour-cloning "
            "initialization is omitted because those rows have legacy provenance; they remain "
            "in supplementary matched-budget analysis.\n"
        ),
        encoding="utf-8",
    )
    (out / "fig_event_miss_caption.txt").write_text(
        (
            "Union-event miss rate on the development-cohort validation split. Miss rate is "
            "100×(1 − recall) for the union event (CO₂ ≥ 1000 ppm or rise ≥ 150 ppm). The "
            "dashed line is the 1.5% event-miss constraint. CAMPUS-SenseRL shows the five-run "
            "mean with sample-SD error bar (ddof = 1). Behaviour-cloning initialization is "
            "omitted from this main figure because of legacy provenance.\n"
        ),
        encoding="utf-8",
    )
    (out / "fig_generalization_caption.txt").write_text(
        (
            "Reconstruction MAE on the same chronological test interval for the development "
            "cohort (open circles) and the held-out cohort (filled markers). The dashed line "
            "is the 9 ppm MAE constraint. CAMPUS-SenseRL values are five-run means. "
            "Behaviour-cloning initialization is omitted because those rows have legacy "
            "provenance.\n"
        ),
        encoding="utf-8",
    )
    (out / "fig_packet_loss_mae_caption.txt").write_text(
        (
            "Reconstruction MAE under independent-slot packet loss. The same five loss masks "
            "(mask seed 20260317 + mask id) are shared across methods. For CAMPUS-SenseRL, "
            "each policy seed is first averaged over the five masks, then the plotted point "
            "and error bar are the mean and sample SD (ddof = 1) across the five policy seeds. "
            "For deterministic baselines the error bar is the sample SD across the five masks. "
            "Zero loss uses a single evaluation with no mask. The dashed line is the 9 ppm "
            "MAE constraint.\n"
        ),
        encoding="utf-8",
    )


def write_manifest(path: Path) -> None:
    rows = [
        {
            "figure_role": "Figure 1",
            "filename": "(manuscript system schematic)",
            "paper_section": "Introduction / system",
            "scientific_question": "How do campus agents decide to transmit or skip, and how is the server monitor formed?",
            "main_or_supplementary": "main",
            "source_data": "manuscript figure; not fig04_kl_cmappo_framework.png",
            "notes": "Keep the current manuscript system/decision-process concept. Edit labels: replace neighbor context with contextual relation feature. Do not draw a relation graph or inter-sensor arrows. Do not use fig04. Not regenerated in this pack.",
        },
        {
            "figure_role": "Figure 2",
            "filename": "fig_revision_methodology.png",
            "paper_section": "Methods",
            "scientific_question": "How is CAMPUS-SenseRL trained, and what actor is deployed at runtime?",
            "main_or_supplementary": "main",
            "source_data": "verified ResidualSharedActor: logit = f_theta(obs) + residual_scale * heuristic_logits_torch(obs)",
            "notes": "Corrected methodology figure. Shows traces → expert demonstrations → BC initialization → KL-CMAPPO → final actor with learned neural component + fixed expert-informed residual, then observation → actor → transmit/skip → server monitoring. No shield-free wording, no relation graph. fig04_kl_cmappo_framework.png is outdated and must not be used.",
        },
        {
            "figure_role": "Figure 3",
            "filename": "fig01_campus_deployment.png",
            "paper_section": "Dataset",
            "scientific_question": "Where are the development and held-out CO2 sensors located?",
            "main_or_supplementary": "main",
            "source_data": "results/cohorts plus device metadata",
            "notes": "Legend: Development cohort / Held-out cohort. PDF exported.",
        },
        {
            "figure_role": "Figure 4",
            "filename": "fig_revision_tradeoff_mae.png",
            "paper_section": "Results — reconstruction trade-off",
            "scientific_question": "How does reconstruction MAE trade off against transmission reduction?",
            "main_or_supplementary": "main",
            "source_data": "results/figures/revision_values/fig_revision_tradeoff_mae.csv",
            "notes": "Main result figure. BC initialization omitted (legacy provenance). Fixed-60 and Fixed-75 labelled. PDF exported.",
        },
        {
            "figure_role": "Figure 5",
            "filename": "fig_revision_event_miss.png",
            "paper_section": "Results — event preservation",
            "scientific_question": "What is the union-event miss rate relative to the 1.5% constraint?",
            "main_or_supplementary": "main",
            "source_data": "results/figures/revision_values/fig_revision_event_miss.csv",
            "notes": "Main result figure. BC omitted. Numeric miss percentages on bars. PDF exported.",
        },
        {
            "figure_role": "Figure 6",
            "filename": "fig_revision_generalization_dumbbell.png",
            "paper_section": "Results — generalization",
            "scientific_question": "Does reconstruction MAE change from development TEST to held-out TEST on the same chronological interval?",
            "main_or_supplementary": "main",
            "source_data": "results/figures/revision_values/fig_revision_generalization_dumbbell.csv",
            "notes": "Main result figure. BC omitted. MAE limit = 9 ppm labelled. PDF exported.",
        },
        {
            "figure_role": "Figure 7",
            "filename": "fig_revision_packet_loss_mae.png",
            "paper_section": "Results — robustness",
            "scientific_question": "How does reconstruction MAE change under increasing packet loss?",
            "main_or_supplementary": "main",
            "source_data": "results/figures/revision_values/fig_revision_packet_loss_mae.csv",
            "notes": "Main result figure. Uncertainty from shared independent-slot masks and, for CAMPUS-SenseRL, five policy seeds. PDF exported.",
        },
        {
            "figure_role": "Supplementary",
            "filename": "fig_revision_constraint_training.png",
            "paper_section": "Supplementary — training dynamics",
            "scientific_question": "How do rollout constraint estimates evolve during KL-CMAPPO training?",
            "main_or_supplementary": "supplementary",
            "source_data": "results/figures/revision_values/constraint_training.csv",
            "notes": "Not main-paper. Logged mean_miss=0 mixes genuine zero-miss windows with no-event placeholders. Rollout MAE oscillating around the boundary is not a test-feasibility claim.",
        },
        {
            "figure_role": "Supplementary",
            "filename": "supp_fig_lagrange_evolution.png",
            "paper_section": "Supplementary — dual variables",
            "scientific_question": "How do MAE, event, and AoI Lagrange multipliers evolve during training?",
            "main_or_supplementary": "supplementary",
            "source_data": "results/figures/revision_values/lagrange_evolution.csv",
            "notes": "Not main-paper. Dual variables on rollout estimates. Do not plot kl_beta as KL divergence.",
        },
        {
            "figure_role": "Supplementary",
            "filename": "fig_revision_contextual_relations.png",
            "paper_section": "Supplementary — context graph",
            "scientific_question": "What spatial/statistical contextual relations exist among development sensors?",
            "main_or_supplementary": "supplementary",
            "source_data": "results/figures/revision_values/fig_revision_contextual_relations.csv",
            "notes": "Not a communication network. Keep out of the main article.",
        },
        {
            "figure_role": "Supplementary",
            "filename": "fig_revision_constraint_matrix.png",
            "paper_section": "Supplementary — constraint table",
            "scientific_question": "Which methods pass MAE, event, and AoI constraints on validation?",
            "main_or_supplementary": "supplementary",
            "source_data": "results/figures/revision_values/fig_revision_constraint_matrix.csv",
            "notes": "VAL pass/fail matrix. Numeric k/n belong in tables.",
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
            "notes": "Variability-versus-transmission-rate figure.",
        },
        {
            "figure_role": "Supplementary",
            "filename": "fig08_diurnal_transmission_profile.png",
            "paper_section": "Supplementary — behaviour",
            "scientific_question": "Does transmission follow campus occupancy over the day?",
            "main_or_supplementary": "supplementary",
            "source_data": "results/figures/figure_values.csv (fig08)",
            "notes": "Diurnal profile.",
        },
        {
            "figure_role": "Supplementary",
            "filename": "fig15_kl_gain_over_bc_anchor.png",
            "paper_section": "Supplementary — matched budget",
            "scientific_question": "What does KL-CMAPPO add over BC initialization at matched communication budgets?",
            "main_or_supplementary": "supplementary",
            "source_data": "results/revision/table_matched_budget.csv",
            "notes": "BC/KL matched-budget figure. BC remains interpolated_legacy. Not a main-paper figure.",
        },
        {
            "figure_role": "Supplementary",
            "filename": "results/revision/ablation.csv",
            "paper_section": "Supplementary — ablation",
            "scientific_question": "What do KL regularization and constraints contribute relative to BC initialization?",
            "main_or_supplementary": "supplementary",
            "source_data": "results/revision/ablation.csv",
            "notes": "Existing seed-42 ablation table. Single-run; no significance claims.",
        },
        {
            "figure_role": "Legacy / do not use",
            "filename": "fig04_kl_cmappo_framework.png",
            "paper_section": "none",
            "scientific_question": "n/a",
            "main_or_supplementary": "do_not_use",
            "source_data": "scripts/28_manuscript_figures.py fig04",
            "notes": "Outdated. Contains shield-free final policy and neighbour-context wording; does not show the deployed residual component. Replaced by fig_revision_methodology.png. Do not assign to Figure 1 or Figure 2.",
        },
        {
            "figure_role": "Internal review only",
            "filename": "revision_main_figures_preview.png",
            "paper_section": "none",
            "scientific_question": "Contact sheet of the main figures for visual QC.",
            "main_or_supplementary": "internal_review",
            "source_data": "Figures 2–7 PNGs",
            "notes": "NOT for the manuscript.",
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def contact_sheet(figdir: Path) -> Path:
    panels = [
        ("fig_revision_methodology.png", "Fig. 2  Methodology"),
        ("fig01_campus_deployment.png", "Fig. 3  Deployment cohorts"),
        ("fig_revision_tradeoff_mae.png", "Fig. 4  Reconstruction trade-off"),
        ("fig_revision_event_miss.png", "Fig. 5  Event-miss rate"),
        ("fig_revision_generalization_dumbbell.png", "Fig. 6  Generalization"),
        ("fig_revision_packet_loss_mae.png", "Fig. 7  Packet-loss MAE"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 8.4), facecolor="white")
    axes = axes.ravel()
    for ax, (name, title) in zip(axes, panels):
        img = np.asarray(Image.open(figdir / name).convert("RGB"))
        ax.imshow(img)
        ax.set_title(title, fontsize=10, pad=6)
        ax.axis("off")
    fig.suptitle("CAMPUS-SenseRL main figures (preview; not for the manuscript)", fontsize=13, y=0.98)
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
    master, summary = load_frozen_revision(root)
    sel_cfg = selected_delta_cfg(out, master)

    fig_campus_deployment(root, figdir)
    fig_methodology(figdir)
    _fig30.fig_tradeoff_mae(master, summary, figdir, valuedir, sel_cfg)
    _fig30.fig_event_miss(master, summary, figdir, valuedir, sel_cfg)
    _fig30.fig_generalization_dumbbell(summary, figdir, valuedir, sel_cfg)
    _fig30.fig_packet_loss_mae(summary, figdir, valuedir)
    write_captions(out)
    write_manifest(out / "figure_manifest.csv")
    preview = contact_sheet(figdir)
    print(f"[done] main figure pack in {figdir}")
    print(f"[done] manifest {out / 'figure_manifest.csv'}")
    print(f"[done] preview {preview}")
    print("[note] frozen evaluation tables were read, not rewritten")


if __name__ == "__main__":
    main()
