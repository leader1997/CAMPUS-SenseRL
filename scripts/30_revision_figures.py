#!/usr/bin/env python
"""Revision paper figures + effect sizes from results/revision CSVs. No training."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from campus_senserl.data.audit import load_devices
from campus_senserl.data.cohort import load_cohort
from campus_senserl.evaluation.revision_checks import validate_figure_values, validate_master
from campus_senserl.evaluation.revision_metrics import DISPLAY_NAMES, EPS_MAE, RECALL_MIN, pareto_mask
from campus_senserl.utils import ensure_dir, load_yaml, repo_root
from campus_senserl.visualization.paper_style import apply_paper_style

C_KL = "#B91C1C"
C_FIXED = "#6B7280"
C_DELTA = "#D97706"
C_EXPERT = "#166534"
C_BC = "#6A51A3"
C_PARETO = "#92400E"

VALUES: list[dict] = []


def rec(fig: str, key: str, value, note: str = "", source: str = "") -> None:
    VALUES.append({"figure": fig, "quantity": key, "value": value, "note": note, "source": source})


def method_color(method: str) -> str:
    if method.startswith("fixed_"):
        return C_FIXED
    if method == "delta_plus_heartbeat":
        return C_DELTA
    if method == "semantic_expert":
        return C_EXPERT
    if method == "campus_senserl_bc":
        return C_BC
    if method == "cmappo_kl":
        return C_KL
    return "#111827"


def load_revision(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rev = root / "results" / "revision"
    master = pd.read_csv(rev / "all_method_evaluations.csv")
    summary = pd.read_csv(rev / "all_method_summary.csv")
    return master, summary


def val_summary(summary: pd.DataFrame) -> pd.DataFrame:
    return summary[
        (summary["scenario"] == "validation_cohort")
        & (summary["split"] == "val")
        & np.isclose(summary["packet_loss"].astype(float), 0.0)
    ].copy()


def fig_mae_vs_reduction(master: pd.DataFrame, summary: pd.DataFrame, figdir: Path) -> None:
    apply_paper_style()
    val = master[
        (master["scenario"] == "validation_cohort")
        & np.isclose(master["packet_loss"].astype(float), 0.0)
        & (master["source"] == "revision_eval")
    ].copy()
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    # Periodic
    per = val[val["method"].str.startswith("fixed_")]
    ax.plot(
        100 * per["transmission_reduction"],
        per["mae"],
        "s-",
        color=C_FIXED,
        lw=1.4,
        ms=7,
        label="Periodic (Fixed-15…90)",
        zorder=3,
    )
    for _, r in per.iterrows():
        rec("fig_revision_mae_vs_reduction", r["method"], float(r["mae"]), f"tx↓={100*r['transmission_reduction']:.3f}%")
    # Delta grid + Pareto
    dlt = val[val["method"] == "delta_plus_heartbeat"].copy()
    dlt["pareto"] = pareto_mask(dlt)
    ax.scatter(
        100 * dlt["transmission_reduction"],
        dlt["mae"],
        s=28,
        c=C_DELTA,
        alpha=0.35,
        label="Delta+heartbeat grid",
        zorder=2,
    )
    p = dlt[dlt["pareto"]]
    ax.scatter(
        100 * p["transmission_reduction"],
        p["mae"],
        s=70,
        facecolors="none",
        edgecolors=C_PARETO,
        linewidths=1.6,
        label="Delta+heartbeat Pareto",
        zorder=4,
    )
    # Expert
    ex = val[val["method"] == "semantic_expert"]
    if len(ex):
        r = ex.iloc[0]
        ax.scatter(100 * r["transmission_reduction"], r["mae"], marker="D", s=90, c=C_EXPERT, zorder=5, label="Semantic expert")
        rec("fig_revision_mae_vs_reduction", "semantic_expert", float(r["mae"]), f"tx↓={100*r['transmission_reduction']:.3f}%")
    # BC mean if present
    vs = val_summary(summary)
    bc = vs[vs["method"] == "campus_senserl_bc"]
    if len(bc):
        r = bc.iloc[0]
        ax.errorbar(
            100 * r["transmission_reduction_mean"],
            r["mae_mean"],
            xerr=100 * r["transmission_reduction_std"] if r["n_runs"] > 1 else None,
            yerr=r["mae_std"] if r["n_runs"] > 1 else None,
            fmt="^",
            color=C_BC,
            ms=9,
            capsize=3,
            label="BC initialization",
            zorder=5,
        )
        rec("fig_revision_mae_vs_reduction", "campus_senserl_bc", float(r["mae_mean"]), f"tx↓={100*r['transmission_reduction_mean']:.3f}% n={int(r['n_runs'])}")
    kl = vs[vs["method"] == "cmappo_kl"]
    if len(kl):
        r = kl.iloc[0]
        ax.errorbar(
            100 * r["transmission_reduction_mean"],
            r["mae_mean"],
            xerr=100 * r["transmission_reduction_std"] if r["n_runs"] > 1 else None,
            yerr=r["mae_std"] if r["n_runs"] > 1 else None,
            fmt="o",
            color=C_KL,
            ms=10,
            capsize=3,
            label="CAMPUS-SenseRL (KL-CMAPPO)",
            zorder=6,
        )
        rec("fig_revision_mae_vs_reduction", "cmappo_kl", float(r["mae_mean"]), f"tx↓={100*r['transmission_reduction_mean']:.3f}% n={int(r['n_runs'])}")
    ax.axhline(EPS_MAE, color="#9CA3AF", ls="--", lw=1.1, label=f"MAE constraint ({EPS_MAE:.0f} ppm)")
    ax.set_xlabel("Transmission reduction (%)")
    ax.set_ylabel("CO$_2$ reconstruction MAE (ppm)")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.grid(True, alpha=0.18)
    fig.tight_layout()
    fig.savefig(figdir / "fig_revision_mae_vs_reduction.png", dpi=300, facecolor="white")
    plt.close(fig)


def fig_recall_vs_reduction(master: pd.DataFrame, summary: pd.DataFrame, figdir: Path) -> None:
    apply_paper_style()
    val = master[
        (master["scenario"] == "validation_cohort")
        & np.isclose(master["packet_loss"].astype(float), 0.0)
        & (master["source"] == "revision_eval")
    ].copy()
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    per = val[val["method"].str.startswith("fixed_")]
    ax.plot(100 * per["transmission_reduction"], per["recall"], "s-", color=C_FIXED, lw=1.4, ms=7, label="Periodic (Fixed-15…90)")
    dlt = val[val["method"] == "delta_plus_heartbeat"].copy()
    dlt["pareto"] = pareto_mask(dlt)
    ax.scatter(100 * dlt["transmission_reduction"], dlt["recall"], s=28, c=C_DELTA, alpha=0.35, label="Delta+heartbeat grid")
    p = dlt[dlt["pareto"]]
    ax.scatter(100 * p["transmission_reduction"], p["recall"], s=70, facecolors="none", edgecolors=C_PARETO, linewidths=1.6, label="Delta+heartbeat Pareto")
    ex = val[val["method"] == "semantic_expert"]
    if len(ex):
        r = ex.iloc[0]
        ax.scatter(100 * r["transmission_reduction"], r["recall"], marker="D", s=90, c=C_EXPERT, label="Semantic expert")
        rec("fig_revision_recall_vs_reduction", "semantic_expert", float(r["recall"]))
    vs = val_summary(summary)
    bc = vs[vs["method"] == "campus_senserl_bc"]
    if len(bc):
        r = bc.iloc[0]
        ax.errorbar(100 * r["transmission_reduction_mean"], r["recall_mean"], yerr=r["recall_std"] if r["n_runs"] > 1 else None, fmt="^", color=C_BC, ms=9, capsize=3, label="BC initialization")
        rec("fig_revision_recall_vs_reduction", "campus_senserl_bc", float(r["recall_mean"]))
    kl = vs[vs["method"] == "cmappo_kl"]
    if len(kl):
        r = kl.iloc[0]
        ax.errorbar(100 * r["transmission_reduction_mean"], r["recall_mean"], yerr=r["recall_std"] if r["n_runs"] > 1 else None, fmt="o", color=C_KL, ms=10, capsize=3, label="CAMPUS-SenseRL (KL-CMAPPO)")
        rec("fig_revision_recall_vs_reduction", "cmappo_kl", float(r["recall_mean"]))
    ax.axhline(RECALL_MIN, color="#9CA3AF", ls="--", lw=1.1, label=f"Recall constraint ({RECALL_MIN:.3f})")
    ax.set_xlabel("Transmission reduction (%)")
    ax.set_ylabel("Event recall")
    ax.set_ylim(0.75, 1.01)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.18)
    fig.tight_layout()
    fig.savefig(figdir / "fig_revision_recall_vs_reduction.png", dpi=300, facecolor="white")
    plt.close(fig)


def fig_unseen(summary: pd.DataFrame, figdir: Path) -> None:
    apply_paper_style()
    sub = summary[
        (summary["scenario"] == "heldout_sensors")
        & (summary["split"] == "test")
        & np.isclose(summary["packet_loss"].astype(float), 0.0)
    ].copy()
    order = ["fixed_15", "fixed_30", "fixed_45", "fixed_60", "fixed_75", "fixed_90", "delta_plus_heartbeat", "semantic_expert", "campus_senserl_bc", "cmappo_kl"]
    sub["ord"] = sub["method"].map({m: i for i, m in enumerate(order)})
    sub = sub.dropna(subset=["ord"]).sort_values("ord")
    labels = [DISPLAY_NAMES.get(m, m) for m in sub["method"]]
    x = np.arange(len(sub))
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8))
    axes[0].bar(x, 100 * sub["transmission_reduction_mean"], color=[method_color(m) for m in sub["method"]])
    axes[0].set_ylabel("Transmission reduction (%)")
    axes[1].bar(x, sub["mae_mean"], yerr=sub["mae_std"], color=[method_color(m) for m in sub["method"]], capsize=3)
    axes[1].axhline(EPS_MAE, color="#9CA3AF", ls="--", lw=1.0)
    axes[1].set_ylabel("MAE (ppm)")
    axes[2].bar(x, sub["recall_mean"], yerr=sub["recall_std"], color=[method_color(m) for m in sub["method"]], capsize=3)
    axes[2].axhline(RECALL_MIN, color="#9CA3AF", ls="--", lw=1.0)
    axes[2].set_ylabel("Event recall")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7.5)
        ax.grid(True, axis="y", alpha=0.18)
    fig.suptitle("Held-out 40-sensor cohort (temporal test split)", fontsize=11)
    fig.tight_layout()
    fig.savefig(figdir / "fig_revision_unseen_comparison.png", dpi=300, facecolor="white")
    plt.close(fig)
    for _, r in sub.iterrows():
        rec("fig_revision_unseen_comparison", f"{r['method']}_mae", float(r["mae_mean"]), f"n={int(r['n_runs'])}")
        rec("fig_revision_unseen_comparison", f"{r['method']}_recall", float(r["recall_mean"]))
        rec("fig_revision_unseen_comparison", f"{r['method']}_tx↓", float(100 * r["transmission_reduction_mean"]))


def fig_packet_loss(summary: pd.DataFrame, figdir: Path) -> None:
    apply_paper_style()
    sub = summary[summary["scenario"] == "packet_loss"].copy()
    keep = ["fixed_60", "fixed_75", "delta_plus_heartbeat", "semantic_expert", "cmappo_kl"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for method in keep:
        g = sub[sub["method"] == method].sort_values("packet_loss")
        if g.empty:
            continue
        yerr = g["mae_std"] if (g["n_runs"] > 1).any() else None
        ax.errorbar(
            100 * g["packet_loss"].astype(float),
            g["mae_mean"],
            yerr=yerr,
            marker="o",
            color=method_color(method),
            lw=1.5,
            capsize=3,
            label=DISPLAY_NAMES.get(method, method),
        )
        for _, r in g.iterrows():
            rec("fig_revision_packet_loss_comparison", f"{method}_mae_pl{int(100*r['packet_loss'])}", float(r["mae_mean"]))
    ax.set_xlabel("Packet loss (%)")
    ax.set_ylabel("CO$_2$ reconstruction MAE (ppm)")
    ax.axhline(EPS_MAE, color="#9CA3AF", ls="--", lw=1.1, label="MAE constraint (9 ppm)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.18)
    fig.tight_layout()
    fig.savefig(figdir / "fig_revision_packet_loss_comparison.png", dpi=300, facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for method in keep:
        g = sub[sub["method"] == method].sort_values("packet_loss")
        if g.empty:
            continue
        yerr = g["recall_std"] if (g["n_runs"] > 1).any() else None
        ax.errorbar(
            100 * g["packet_loss"].astype(float),
            g["recall_mean"],
            yerr=yerr,
            marker="o",
            color=method_color(method),
            lw=1.5,
            capsize=3,
            label=DISPLAY_NAMES.get(method, method),
        )
    ax.axhline(RECALL_MIN, color="#9CA3AF", ls="--", lw=1.1, label="Recall constraint (0.985)")
    ax.set_xlabel("Packet loss (%)")
    ax.set_ylabel("Event recall")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.18)
    fig.tight_layout()
    fig.savefig(figdir / "fig_revision_packet_loss_recall.png", dpi=300, facecolor="white")
    plt.close(fig)


def fig_contextual_relations(root: Path, figdir: Path) -> None:
    """Replacement for fig03 with communication-link disclaimer."""
    apply_paper_style()
    cfg = load_yaml(root / "configs" / "data.yaml")
    devices = load_devices(root / cfg["paths"]["raw_release"] / cfg["dataset"]["devices_file"]).copy()
    devices["deveui"] = devices["device_id"].astype(str).str.upper()
    ids = load_cohort("final")
    order = json.loads((root / "results" / "graphs" / "node_order.json").read_text(encoding="utf-8"))
    node_order = order["node_order"] if isinstance(order, dict) else order
    A = np.load(root / "results" / "graphs" / "adjacency_hybrid.npy")
    idx = [node_order.index(s) for s in ids]
    S = A[np.ix_(idx, idx)]
    np.fill_diagonal(S, 0)
    d = devices.set_index("deveui")
    lon = np.array([float(d.loc[s, "longitude"]) for s in ids])
    lat = np.array([float(d.loc[s, "latitude"]) for s in ids])
    fl = [str(d.loc[s, "floor"]) for s in ids]
    colors = {"-1": "#7C3AED", "1": "#2563EB", "2": "#0891B2", "3": "#059669", "4": "#D97706", "5": "#DC2626"}
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    others = devices[devices["device_type"].astype(str).str.contains("CO2", case=False, na=False) & ~devices["deveui"].isin(set(ids))]
    ax.scatter(others["longitude"], others["latitude"], s=10, c="#D9DCE1", alpha=0.55, zorder=1, label="Other campus CO$_2$ sensors")
    n_edges = 0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if S[i, j] > 0:
                n_edges += 1
                ax.plot([lon[i], lon[j]], [lat[i], lat[j]], color="#F87171", lw=1.0 + 2.2 * float(S[i, j]), alpha=0.75, zorder=2)
    deg = (S > 0).sum(1)
    for f in sorted(set(fl), key=lambda v: float(v)):
        m = np.array([x == f for x in fl])
        ax.scatter(lon[m], lat[m], s=60 + 55 * deg[m], c=colors.get(f, "#374151"), edgecolors="white", linewidths=0.6, zorder=3, label=f"Floor {f} ({int(m.sum())})")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.text(
        0.99,
        0.01,
        f"{len(ids)} agents · {n_edges} contextual relations\nSpatial/statistical contextual relations — not communication links.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#374151",
    )
    ax.grid(True, alpha=0.12)
    fig.tight_layout()
    fig.savefig(figdir / "fig_revision_contextual_relations.png", dpi=300, facecolor="white")
    plt.close(fig)
    rec("fig_revision_contextual_relations", "n_edges", n_edges)


def rel_change(new, old) -> float:
    old = float(old)
    new = float(new)
    if not np.isfinite(old) or abs(old) < 1e-12:
        return float("nan")
    return (new - old) / old


def effect_sizes(master: pd.DataFrame, summary: pd.DataFrame, out: Path) -> pd.DataFrame:
    rows = []

    sel_cfg = ""
    sel_path = ROOT / "results" / "revision" / "delta_heartbeat_selection.json"
    if sel_path.exists():
        sel_cfg = json.loads(sel_path.read_text(encoding="utf-8")).get("selected_configuration", "")

    def get_sum(scenario, split, method, packet_loss=0.0):
        hit = summary[
            (summary["scenario"] == scenario)
            & (summary["split"] == split)
            & (summary["method"] == method)
            & np.isclose(summary["packet_loss"].astype(float), packet_loss)
        ]
        if method == "delta_plus_heartbeat" and sel_cfg:
            hit = hit[hit["configuration"] == sel_cfg]
        return None if hit.empty else hit.iloc[0]

    def get_one(scenario, split, method):
        hit = master[
            (master["scenario"] == scenario)
            & (master["split"] == split)
            & (master["method"] == method)
            & np.isclose(master["packet_loss"].astype(float), 0.0)
            & (master["source"] == "revision_eval")
        ]
        if method == "delta_plus_heartbeat" and len(hit) > 1:
            sel = json.loads((ROOT / "results" / "revision" / "delta_heartbeat_selection.json").read_text(encoding="utf-8"))
            hit = hit[hit["configuration"] == sel["selected_configuration"]]
        return None if hit.empty else hit.iloc[0]

    for scenario, split, label in [
        ("validation_cohort", "val", "validation"),
        ("frozen_temporal_test", "test", "frozen_test"),
        ("heldout_sensors", "test", "heldout_test"),
    ]:
        kl = get_sum(scenario, split, "cmappo_kl")
        if kl is None:
            continue
        for other in ["fixed_60", "delta_plus_heartbeat", "semantic_expert", "campus_senserl_bc"]:
            ot = get_sum(scenario, split, other)
            if ot is None:
                ot_row = get_one(scenario, split, other)
                if ot_row is None:
                    continue
                ot = pd.Series(
                    {
                        "n_tx_attempts_mean": ot_row["n_tx_attempts"],
                        "mae_mean": ot_row["mae"],
                        "recall_mean": ot_row["recall"],
                        "event_fn_mean": ot_row["event_fn"],
                        "transmission_reduction_mean": ot_row["transmission_reduction"],
                        "n_runs": 1,
                    }
                )
            rows.append(
                {
                    "scenario": label,
                    "comparison": f"CAMPUS-SenseRL vs {DISPLAY_NAMES.get(other, other)}",
                    "kl_n_runs": int(kl["n_runs"]),
                    "other_n_runs": int(ot.get("n_runs", 1)),
                    "tx_attempts_kl": kl["n_tx_attempts_mean"],
                    "tx_attempts_other": ot["n_tx_attempts_mean"],
                    "relative_change_tx_attempts": rel_change(kl["n_tx_attempts_mean"], ot["n_tx_attempts_mean"]),
                    "absolute_tx_reduction_diff_pp": 100 * (kl["transmission_reduction_mean"] - ot["transmission_reduction_mean"]),
                    "mae_kl": kl["mae_mean"],
                    "mae_other": ot["mae_mean"],
                    "absolute_mae_diff_ppm": kl["mae_mean"] - ot["mae_mean"],
                    "relative_mae_improvement": -rel_change(kl["mae_mean"], ot["mae_mean"]),
                    "recall_kl": kl["recall_mean"],
                    "recall_other": ot["recall_mean"],
                    "absolute_recall_improvement": kl["recall_mean"] - ot["recall_mean"],
                    "fn_kl": kl["event_fn_mean"],
                    "fn_other": ot["event_fn_mean"],
                    "relative_missed_event_reduction": -rel_change(kl["event_fn_mean"], ot["event_fn_mean"]),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out / "effect_sizes.csv", index=False)
    return df


def write_summary_md(root: Path, master: pd.DataFrame, summary: pd.DataFrame, effects: pd.DataFrame, out: Path) -> None:
    sel_path = out / "delta_heartbeat_selection.json"
    sel = json.loads(sel_path.read_text(encoding="utf-8")) if sel_path.exists() else {}
    checks = json.loads((out / "checks.json").read_text(encoding="utf-8")) if (out / "checks.json").exists() else {}
    lines = [
        "# Revision evaluation summary",
        "",
        "No training was performed. Frozen KL-CMAPPO checkpoints were evaluated as-is.",
        "",
        "## Delta+heartbeat selection (validation only)",
        "",
        f"- Selected configuration: `{sel.get('selected_configuration')}`",
        f"- Reason: {sel.get('reason')}",
        f"- Criterion: {sel.get('criterion')}",
        "",
        "## Constraint thresholds",
        "",
        "- MAE ≤ 9 ppm",
        "- Event miss rate ≤ 0.015 (recall ≥ 0.985)",
        "- Raw AoI ≤ 3.5 intervals",
        "",
        "## Training",
        "",
        f"- Training performed: {checks.get('training_performed', False)}",
        f"- KL checkpoints: {checks.get('kl_checkpoints')}",
        f"- BC checkpoints: {checks.get('bc_checkpoints')}",
        "",
        "## Notes",
        "",
        "- Fig05 remains an illustrative window (`results/figures/fig05_window.json`), not primary evidence.",
        "- Replace fig03 with `fig_revision_contextual_relations.png` (relation graph, not a mesh).",
        "- Replace fig10/fig12/fig13 with the new `fig_revision_*` comparison figures.",
        "- BC checkpoints were missing in this clone; BC numbers may be imported from frozen CSVs and are labelled as such.",
        "",
    ]
    (out / "revision_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = repo_root()
    figdir = ensure_dir(root / "results" / "figures")
    out = ensure_dir(root / "results" / "revision")
    master, summary = load_revision(root)
    validate_master(master)
    fig_contextual_relations(root, figdir)
    fig_mae_vs_reduction(master, summary, figdir)
    fig_recall_vs_reduction(master, summary, figdir)
    fig_unseen(summary, figdir)
    fig_packet_loss(summary, figdir)
    effects = effect_sizes(master, summary, out)
    fv = pd.DataFrame(VALUES)
    fv.to_csv(figdir / "figure_values_revision.csv", index=False)
    fv.to_csv(out / "figure_values_revision.csv", index=False)
    validate_figure_values(master, fv)
    write_summary_md(root, master, summary, effects, out)
    print(f"[done] figures in {figdir} (fig_revision_*)")


if __name__ == "__main__":
    main()
