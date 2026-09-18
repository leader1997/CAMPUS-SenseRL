#!/usr/bin/env python
"""Redesigned one-question revision figures. No training."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from campus_senserl.data.audit import load_devices
from campus_senserl.data.cohort import load_cohort
from campus_senserl.evaluation.revision_checks import validate_figure_values, validate_master
from campus_senserl.evaluation.revision_export import (
    selected_delta_cfg,
    write_claim_audit,
    write_final_reproducibility_check,
    write_paper_ready_summary,
    write_protocol_json,
    write_reproducibility_manifest,
    write_tables,
)
from campus_senserl.evaluation.revision_metrics import (
    DISPLAY_NAMES,
    EPS_MAE,
    EPS_MISS,
    PAPER_METHOD_ORDER,
    backfill_master,
    pareto_mask,
)
from campus_senserl.utils import ensure_dir, load_yaml, repo_root
from campus_senserl.visualization.revision_style import (
    C_DELTA,
    C_EXPERT,
    C_GRID,
    C_KL,
    C_PARETO,
    C_PERIODIC,
    apply_revision_style,
    method_color,
    method_marker,
    save_revision_figure,
)

VALUES: list[dict] = []
FIG_TABLES: dict[str, pd.DataFrame] = {}


def rec(fig: str, key: str, value, note: str = "", source: str = "") -> None:
    VALUES.append({"figure": fig, "quantity": key, "value": value, "note": note, "source": source})


def dump_fig_csv(name: str, df: pd.DataFrame, valuedir: Path) -> None:
    FIG_TABLES[name] = df
    df.to_csv(valuedir / f"{name}.csv", index=False)


def load_revision(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rev = root / "results" / "revision"
    master = backfill_master(pd.read_csv(rev / "all_method_evaluations.csv"))
    master.to_csv(rev / "all_method_evaluations.csv", index=False)
    summary = pd.read_csv(rev / "all_method_summary.csv") if (rev / "all_method_summary.csv").exists() else pd.DataFrame()
    return master, summary


def val_zero_loss(master: pd.DataFrame) -> pd.DataFrame:
    return master[
        (master["scenario"] == "validation_cohort")
        & np.isclose(pd.to_numeric(master["packet_loss"], errors="coerce").fillna(0.0), 0.0)
        & (master["source"].astype(str) == "revision_eval")
    ].copy()


def val_summary(summary: pd.DataFrame) -> pd.DataFrame:
    return summary[
        (summary["scenario"] == "validation_cohort")
        & (summary["split"] == "val")
        & np.isclose(pd.to_numeric(summary["packet_loss"], errors="coerce").fillna(0.0), 0.0)
    ].copy()


def miss_pct(recall) -> float:
    r = float(recall)
    return float("nan") if not np.isfinite(r) else 100.0 * (1.0 - r)


def fig_tradeoff_mae(master: pd.DataFrame, summary: pd.DataFrame, figdir: Path, valuedir: Path, sel_cfg: str) -> None:
    """At high TX reduction, who keeps low reconstruction MAE?"""
    apply_revision_style()
    val = val_zero_loss(master)
    vs = val_summary(summary)
    fig, ax = plt.subplots(figsize=(6.8, 4.7))
    plotted = []

    per = val[val["method"].str.startswith("fixed_") & val["method"].ne("fixed_15")].sort_values("transmission_reduction")
    ax.plot(
        100 * per["transmission_reduction"],
        per["mae"],
        "s-",
        color=C_PERIODIC,
        lw=1.5,
        ms=7,
        label="Periodic (Fixed-30…90)",
        zorder=3,
    )
    for _, r in per.iterrows():
        rec("fig_revision_tradeoff_mae", r["method"], float(r["mae"]), f"tx={100*r['transmission_reduction']:.3f}%")
        plotted.append({"method": r["method"], "tx_reduction_pct": 100 * r["transmission_reduction"], "mae": r["mae"], "role": "periodic"})

    dlt = val[val["method"] == "delta_plus_heartbeat"].copy()
    dlt["pareto"] = pareto_mask(dlt)
    ax.scatter(
        100 * dlt["transmission_reduction"],
        dlt["mae"],
        s=26,
        c=C_DELTA,
        alpha=0.28,
        marker="P",
        label="Delta+heartbeat grid",
        zorder=2,
    )
    p = dlt[dlt["pareto"]]
    ax.scatter(
        100 * p["transmission_reduction"],
        p["mae"],
        s=64,
        facecolors="none",
        edgecolors=C_PARETO,
        linewidths=1.4,
        marker="P",
        label="Delta+heartbeat Pareto",
        zorder=4,
    )
    sel = dlt[dlt["configuration"] == sel_cfg]
    if len(sel):
        r = sel.iloc[0]
        ax.scatter(
            100 * r["transmission_reduction"],
            r["mae"],
            marker="P",
            s=110,
            c=C_DELTA,
            edgecolors="black",
            linewidths=0.6,
            zorder=6,
            label="Selected Delta+heartbeat",
        )
        rec("fig_revision_tradeoff_mae", "delta_selected", float(r["mae"]), sel_cfg)
        plotted.append({"method": "delta_plus_heartbeat_selected", "tx_reduction_pct": 100 * r["transmission_reduction"], "mae": r["mae"], "role": "selected"})

    def add_mean(method: str, label: str, ms: float):
        hit = vs[vs["method"] == method]
        if method == "delta_plus_heartbeat":
            return
        if hit.empty:
            raw = val[val["method"] == method]
            if raw.empty:
                return
            r = raw.iloc[0]
            ax.scatter(
                100 * r["transmission_reduction"],
                r["mae"],
                marker=method_marker(method),
                s=ms,
                c=method_color(method),
                zorder=6,
                label=label,
            )
            rec("fig_revision_tradeoff_mae", method, float(r["mae"]))
            plotted.append({"method": method, "tx_reduction_pct": 100 * r["transmission_reduction"], "mae": r["mae"], "role": "point"})
            return
        r = hit.iloc[0]
        ax.errorbar(
            100 * r["transmission_reduction_mean"],
            r["mae_mean"],
            xerr=100 * r["transmission_reduction_std"] if r["n_runs"] > 1 else None,
            yerr=r["mae_std"] if r["n_runs"] > 1 else None,
            fmt=method_marker(method),
            color=method_color(method),
            ms=11 if method == "cmappo_kl" else 9,
            capsize=3,
            zorder=7 if method == "cmappo_kl" else 5,
            label=label,
        )
        rec("fig_revision_tradeoff_mae", method, float(r["mae_mean"]), f"n={int(r['n_runs'])}")
        plotted.append(
            {
                "method": method,
                "tx_reduction_pct": 100 * r["transmission_reduction_mean"],
                "mae": r["mae_mean"],
                "mae_sd": r["mae_std"],
                "n_runs": int(r["n_runs"]),
                "role": "mean",
            }
        )

    add_mean("semantic_expert", "Semantic expert", 90)
    add_mean("campus_senserl_bc", "BC initialization (legacy)", 80)
    add_mean("cmappo_kl", "CAMPUS-SenseRL (KL-CMAPPO)", 120)
    ax.axhline(EPS_MAE, color=C_GRID, ls="--", lw=1.1, zorder=1)
    ax.text(
        0.98,
        EPS_MAE,
        f"MAE ≤ {EPS_MAE:.0f} ppm",
        transform=ax.get_yaxis_transform(),
        color="#4B5563",
        fontsize=8,
        ha="right",
        va="bottom",
    )
    ax.set_xlabel("Transmission reduction (%)")
    ax.set_ylabel("MAE (ppm)")
    ax.set_title("Who keeps low MAE at high communication savings?")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.grid(True, alpha=0.18)
    save_revision_figure(fig, figdir / "fig_revision_tradeoff_mae.png")
    dump_fig_csv("fig_revision_tradeoff_mae", pd.DataFrame(plotted), valuedir)


def fig_event_miss(master: pd.DataFrame, summary: pd.DataFrame, figdir: Path, valuedir: Path, sel_cfg: str) -> None:
    """Union event-miss rate as a horizontal bar chart (lower is better)."""
    apply_revision_style()
    val = val_zero_loss(master)
    vs = val_summary(summary)
    order = [
        ("fixed_60", "Fixed-60"),
        ("fixed_75", "Fixed-75"),
        ("delta_plus_heartbeat", "Selected Delta+heartbeat"),
        ("semantic_expert", "Semantic expert"),
        ("campus_senserl_bc", "BC initialization (legacy)"),
        ("cmappo_kl", "CAMPUS-SenseRL (KL-CMAPPO)"),
    ]
    plotted = []
    for method, label in order:
        hit = vs[vs["method"] == method]
        if method == "delta_plus_heartbeat":
            raw = val[(val["method"] == method) & (val["configuration"] == sel_cfg)]
            if raw.empty:
                continue
            r = raw.iloc[0]
            plotted.append(
                {
                    "method": method,
                    "label": label,
                    "event_miss_pct": miss_pct(r["recall"]),
                    "event_miss_sd": 0.0,
                    "n_runs": 1,
                    "result_source": r.get("result_source", "new_evaluation"),
                }
            )
            rec("fig_revision_event_miss", method, miss_pct(r["recall"]), sel_cfg)
            continue
        if hit.empty:
            raw = val[val["method"] == method]
            if raw.empty:
                continue
            r = raw.iloc[0]
            plotted.append(
                {
                    "method": method,
                    "label": label,
                    "event_miss_pct": miss_pct(r["recall"]),
                    "event_miss_sd": 0.0,
                    "n_runs": 1,
                    "result_source": r.get("result_source", ""),
                }
            )
            rec("fig_revision_event_miss", method, miss_pct(r["recall"]))
            continue
        r = hit.iloc[0]
        y = miss_pct(r["recall_mean"])
        ysd = 100.0 * float(r["recall_std"]) if int(r["n_runs"]) > 1 else 0.0
        plotted.append(
            {
                "method": method,
                "label": label,
                "event_miss_pct": y,
                "event_miss_sd": ysd,
                "n_runs": int(r["n_runs"]),
                "result_source": r.get("result_source", ""),
            }
        )
        rec("fig_revision_event_miss", method, y, f"n={int(r['n_runs'])}")

    df = pd.DataFrame(plotted)
    fig, ax = plt.subplots(figsize=(6.8, 4.4), layout="constrained")
    y = np.arange(len(df))
    colors = [method_color(m) for m in df["method"]]
    xerr = df["event_miss_sd"].to_numpy(dtype=float)
    xerr = np.where(np.isfinite(xerr) & (xerr > 1e-12), xerr, np.nan)
    ax.barh(y, df["event_miss_pct"], color=colors, edgecolor="white", height=0.62, xerr=xerr, capsize=3, zorder=3, error_kw={"ecolor": "#374151", "lw": 0.9})
    ax.axvline(100 * EPS_MISS, color=C_GRID, ls="--", lw=1.2, zorder=2, label="1.5% event-miss constraint")
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Union event miss rate (%)  =  100×(1 − recall)")
    ax.set_title("Who preserves high-CO₂ or rapid-rise events?  (lower is better)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(True, axis="x", alpha=0.18)
    save_revision_figure(fig, figdir / "fig_revision_event_miss.png")
    dump_fig_csv("fig_revision_event_miss", df, valuedir)


def fig_generalization_dumbbell(summary: pd.DataFrame, figdir: Path, valuedir: Path, sel_cfg: str) -> None:
    """MAE change: development TEST vs held-out TEST (same temporal window)."""
    apply_revision_style()
    methods = ["fixed_60", "fixed_75", "delta_plus_heartbeat", "semantic_expert", "campus_senserl_bc", "cmappo_kl"]
    rows = []
    for method in methods:
        dev = summary[
            (summary["scenario"] == "frozen_temporal_test")
            & (summary["split"] == "test")
            & (summary["method"] == method)
            & np.isclose(pd.to_numeric(summary["packet_loss"], errors="coerce").fillna(0.0), 0.0)
        ]
        ho = summary[
            (summary["scenario"] == "heldout_sensors")
            & (summary["split"] == "test")
            & (summary["method"] == method)
            & np.isclose(pd.to_numeric(summary["packet_loss"], errors="coerce").fillna(0.0), 0.0)
        ]
        if method == "delta_plus_heartbeat":
            if len(dev):
                dev = dev[dev["configuration"].astype(str) == sel_cfg]
            if len(ho):
                ho = ho[ho["configuration"].astype(str) == sel_cfg]
        if dev.empty or ho.empty:
            continue
        d, h = dev.iloc[0], ho.iloc[0]
        rows.append(
            {
                "method": method,
                "label": DISPLAY_NAMES.get(method, method),
                "mae_dev_test": float(d["mae_mean"]),
                "mae_heldout_test": float(h["mae_mean"]),
                "n_dev": int(d["n_runs"]),
                "n_heldout": int(h["n_runs"]),
            }
        )
        rec("fig_revision_generalization_dumbbell", f"{method}_dev_test", float(d["mae_mean"]))
        rec("fig_revision_generalization_dumbbell", f"{method}_heldout_test", float(h["mae_mean"]))
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    y = np.arange(len(df))
    for i, r in df.iterrows():
        c = method_color(r["method"])
        ax.plot([r["mae_dev_test"], r["mae_heldout_test"]], [i, i], color=c, lw=1.8, zorder=2)
        ax.scatter(r["mae_dev_test"], i, marker="o", s=42, c="white", edgecolors=c, linewidths=1.6, zorder=3, label="Development test" if i == 0 else None)
        ax.scatter(r["mae_heldout_test"], i, marker=method_marker(r["method"]), s=70, c=c, zorder=4, label="Held-out test" if i == 0 else None)
    ax.axvline(EPS_MAE, color=C_GRID, ls="--", lw=1.1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.set_xlabel("Reconstruction MAE (ppm)")
    ax.set_title("Does MAE change on unseen sensors? (same test window)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(True, axis="x", alpha=0.18)
    fig.tight_layout()
    save_revision_figure(fig, figdir / "fig_revision_generalization_dumbbell.png")
    dump_fig_csv("fig_revision_generalization_dumbbell", df, valuedir)


def _pl_keep(summary: pd.DataFrame) -> pd.DataFrame:
    sub = summary[summary["scenario"] == "packet_loss"].copy()
    keep = ["fixed_60", "fixed_75", "delta_plus_heartbeat", "semantic_expert", "cmappo_kl"]
    return sub[sub["method"].isin(keep)]


def fig_packet_loss_mae(summary: pd.DataFrame, figdir: Path, valuedir: Path) -> None:
    apply_revision_style()
    sub = _pl_keep(summary)
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    plotted = []
    for method in ["fixed_60", "fixed_75", "delta_plus_heartbeat", "semantic_expert", "cmappo_kl"]:
        g = sub[sub["method"] == method].sort_values("packet_loss")
        if g.empty:
            continue
        yerr = g["mae_std"] if (g["n_runs"] > 1).any() else None
        ax.errorbar(
            100 * g["packet_loss"].astype(float),
            g["mae_mean"],
            yerr=yerr,
            marker=method_marker(method),
            color=method_color(method),
            lw=1.6,
            ms=8 if method != "cmappo_kl" else 10,
            capsize=3,
            label=DISPLAY_NAMES.get(method, method),
            zorder=6 if method == "cmappo_kl" else 4,
        )
        for _, r in g.iterrows():
            rec("fig_revision_packet_loss_mae", f"{method}_pl{int(100*r['packet_loss'])}", float(r["mae_mean"]))
            plotted.append({"method": method, "packet_loss_pct": 100 * r["packet_loss"], "mae": r["mae_mean"], "mae_sd": r["mae_std"], "n_runs": int(r["n_runs"])})
    ax.axhline(EPS_MAE, color=C_GRID, ls="--", lw=1.1)
    ax.set_xlabel("Packet loss (%)")
    ax.set_ylabel("Reconstruction MAE (ppm)")
    ax.set_title("How does reconstruction MAE degrade under packet loss?")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.18)
    fig.tight_layout()
    save_revision_figure(fig, figdir / "fig_revision_packet_loss_mae.png")
    dump_fig_csv("fig_revision_packet_loss_mae", pd.DataFrame(plotted), valuedir)


def fig_packet_loss_event_miss(summary: pd.DataFrame, figdir: Path, valuedir: Path) -> None:
    apply_revision_style()
    sub = _pl_keep(summary)
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    plotted = []
    for method in ["fixed_60", "fixed_75", "delta_plus_heartbeat", "semantic_expert", "cmappo_kl"]:
        g = sub[sub["method"] == method].sort_values("packet_loss")
        if g.empty:
            continue
        y = 100.0 * (1.0 - g["recall_mean"])
        yerr = 100.0 * g["recall_std"] if (g["n_runs"] > 1).any() else None
        ax.errorbar(
            100 * g["packet_loss"].astype(float),
            y,
            yerr=yerr,
            marker=method_marker(method),
            color=method_color(method),
            lw=1.6,
            ms=8 if method != "cmappo_kl" else 10,
            capsize=3,
            label=DISPLAY_NAMES.get(method, method),
            zorder=6 if method == "cmappo_kl" else 4,
        )
        for _, r in g.iterrows():
            rec("fig_revision_packet_loss_event_miss", f"{method}_pl{int(100*r['packet_loss'])}", miss_pct(r["recall_mean"]))
            plotted.append({"method": method, "packet_loss_pct": 100 * r["packet_loss"], "event_miss_pct": miss_pct(r["recall_mean"]), "n_runs": int(r["n_runs"])})
    ax.axhline(100 * EPS_MISS, color=C_GRID, ls="--", lw=1.1)
    ax.set_xlabel("Packet loss (%)")
    ax.set_ylabel("Event miss rate (%)")
    ax.set_title("How does event miss rate change under packet loss?")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.18)
    fig.tight_layout()
    save_revision_figure(fig, figdir / "fig_revision_packet_loss_event_miss.png")
    dump_fig_csv("fig_revision_packet_loss_event_miss", pd.DataFrame(plotted), valuedir)


def fig_contextual_relations(root: Path, figdir: Path, valuedir: Path) -> None:
    apply_revision_style()
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
    colors = {"-1": "#7C3AED", "1": "#0072B2", "2": "#56B4E9", "3": "#009E73", "4": "#E69F00", "5": "#D55E00"}
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    others = devices[devices["device_type"].astype(str).str.contains("CO2", case=False, na=False) & ~devices["deveui"].isin(set(ids))]
    ax.scatter(others["longitude"], others["latitude"], s=10, c="#D9DCE1", alpha=0.55, zorder=1, label="Other campus CO$_2$ sensors")
    n_edges = 0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if S[i, j] > 0:
                n_edges += 1
                ax.plot(
                    [lon[i], lon[j]],
                    [lat[i], lat[j]],
                    color="#4B5563",
                    lw=0.4 + 1.6 * float(S[i, j]),
                    alpha=0.22,
                    zorder=2,
                )
    deg = (S > 0).sum(1)
    for f in sorted(set(fl), key=lambda v: float(v)):
        m = np.array([x == f for x in fl])
        ax.scatter(lon[m], lat[m], s=55 + 50 * deg[m], c=colors.get(f, "#374151"), edgecolors="white", linewidths=0.5, zorder=3, label=f"Floor {f} ({int(m.sum())})")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_aspect("equal", adjustable="box")
    ax.plot([], [], color="#4B5563", lw=1.4, alpha=0.45, label="contextual relation — not communication link")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.text(
        0.99,
        0.01,
        f"{len(ids)} logical per-sensor decision agents · {n_edges} contextual relations\n"
        "Faint edges: contextual relation — not communication link.\n"
        "Neighbour features use server-available monitoring states only.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#111827",
    )
    ax.set_title("Contextual relations among development-cohort sensors")
    ax.grid(True, alpha=0.12)
    fig.tight_layout()
    save_revision_figure(fig, figdir / "fig_revision_contextual_relations.png")
    rec("fig_revision_contextual_relations", "n_edges", n_edges)
    dump_fig_csv("fig_revision_contextual_relations", pd.DataFrame([{"n_agents": len(ids), "n_edges": n_edges}]), valuedir)


def fig_constraint_matrix(summary: pd.DataFrame, figdir: Path, valuedir: Path, sel_cfg: str) -> None:
    apply_revision_style()
    val = summary[
        (summary["scenario"] == "validation_cohort")
        & (summary["split"] == "val")
        & np.isclose(pd.to_numeric(summary["packet_loss"], errors="coerce").fillna(0.0), 0.0)
    ].copy()
    rows = []
    for method in PAPER_METHOD_ORDER:
        hit = val[val["method"] == method]
        if method == "delta_plus_heartbeat":
            hit = hit[hit["configuration"].astype(str) == sel_cfg]
        if hit.empty:
            continue
        r = hit.iloc[0]
        rows.append(
            {
                "method": method,
                "label": DISPLAY_NAMES.get(method, method),
                "MAE": "N/A" if method == "fixed_15" else str(r.get("mae_ok", "N/A")),
                "Event": str(r.get("event_ok", "N/A")),
                "AoI": str(r.get("aoi_ok", "N/A")),
                "All": "N/A" if method == "fixed_15" else str(r.get("all_constraints_satisfied", "N/A")),
                "k_n": "N/A (full-transmission reference)" if method == "fixed_15" else str(r.get("all_constraints_satisfied_kn", r.get("constraint_display", ""))),
            }
        )
    df = pd.DataFrame(rows)
    cols = ["MAE", "Event", "AoI", "All"]
    color_map = {"True": "#009E73", "False": "#D55E00", "N/A": "#C8C8C8"}
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_ylim(-0.5, len(df) - 0.5)
    for i, rec_row in df.iterrows():
        for j, col in enumerate(cols):
            val = rec_row[col]
            label = {"True": "Pass", "False": "Fail", "N/A": "N/A"}.get(val, val)
            ax.add_patch(
                plt.Rectangle((j - 0.45, i - 0.4), 0.9, 0.8, facecolor=color_map.get(val, "#EEE"), edgecolor="white", lw=1.5)
            )
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color="#111")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["label"])
    ax.invert_yaxis()
    ax.set_title("VAL constraint pass/fail (k/n in tables; Fixed-15 is N/A)")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(
        handles=[
            Patch(facecolor="#009E73", label="Pass"),
            Patch(facecolor="#D55E00", label="Fail"),
            Patch(facecolor="#C8C8C8", label="N/A"),
        ],
        frameon=False,
        loc="lower right",
        fontsize=8,
    )
    fig.tight_layout()
    save_revision_figure(fig, figdir / "fig_revision_constraint_matrix.png")
    dump_fig_csv("fig_revision_constraint_matrix", df, valuedir)


def rel_change(new, old) -> float:
    old = float(old)
    new = float(new)
    if not np.isfinite(old) or abs(old) < 1e-12:
        return float("nan")
    return (new - old) / old


def effect_sizes(master: pd.DataFrame, summary: pd.DataFrame, out: Path, sel_cfg: str) -> pd.DataFrame:
    rows = []

    def get_sum(scenario, split, method, packet_loss=0.0):
        hit = summary[
            (summary["scenario"] == scenario)
            & (summary["split"] == split)
            & (summary["method"] == method)
            & np.isclose(pd.to_numeric(summary["packet_loss"], errors="coerce").fillna(0.0), packet_loss)
        ]
        if method == "delta_plus_heartbeat" and sel_cfg:
            hit = hit[hit["configuration"] == sel_cfg]
        return None if hit.empty else hit.iloc[0]

    for scenario, split, label in [
        ("validation_cohort", "val", "validation"),
        ("frozen_temporal_test", "test", "frozen_test"),
        ("heldout_sensors", "test", "heldout_test"),
    ]:
        kl = get_sum(scenario, split, "cmappo_kl")
        if kl is None:
            continue
        for other in ["fixed_60", "fixed_75", "delta_plus_heartbeat", "semantic_expert", "campus_senserl_bc"]:
            ot = get_sum(scenario, split, other)
            if ot is None:
                continue
            rows.append(
                {
                    "scenario": label,
                    "comparison": f"CAMPUS-SenseRL vs {DISPLAY_NAMES.get(other, other)}",
                    "kl_n_runs": int(kl["n_runs"]),
                    "other_n_runs": int(ot.get("n_runs", 1)),
                    "tx_attempts_kl": kl.get("n_tx_attempts_mean"),
                    "tx_attempts_other": ot.get("n_tx_attempts_mean"),
                    "relative_change_tx_attempts": rel_change(kl.get("n_tx_attempts_mean"), ot.get("n_tx_attempts_mean")),
                    "absolute_tx_reduction_diff_pp": 100 * (kl["transmission_reduction_mean"] - ot["transmission_reduction_mean"]),
                    "mae_kl": kl["mae_mean"],
                    "mae_other": ot["mae_mean"],
                    "absolute_mae_diff_ppm": kl["mae_mean"] - ot["mae_mean"],
                    "relative_mae_improvement": -rel_change(kl["mae_mean"], ot["mae_mean"]),
                    "recall_kl": kl["recall_mean"],
                    "recall_other": ot["recall_mean"],
                    "absolute_recall_improvement": kl["recall_mean"] - ot["recall_mean"],
                    "fn_kl": kl.get("event_fn_mean"),
                    "fn_other": ot.get("event_fn_mean"),
                    "relative_missed_event_reduction": -rel_change(kl.get("event_fn_mean"), ot.get("event_fn_mean")),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out / "effect_sizes.csv", index=False)
    return df


def main() -> None:
    root = repo_root()
    figdir = ensure_dir(root / "results" / "figures")
    valuedir = ensure_dir(figdir / "revision_values")
    out = ensure_dir(root / "results" / "revision")
    master, _old_summary = load_revision(root)
    validate_master(master)
    write_protocol_json(out)
    summary = write_tables(master, out)
    sel_cfg = selected_delta_cfg(out, master)
    fig_contextual_relations(root, figdir, valuedir)
    fig_tradeoff_mae(master, summary, figdir, valuedir, sel_cfg)
    fig_event_miss(master, summary, figdir, valuedir, sel_cfg)
    fig_generalization_dumbbell(summary, figdir, valuedir, sel_cfg)
    fig_packet_loss_mae(summary, figdir, valuedir)
    fig_packet_loss_event_miss(summary, figdir, valuedir)
    fig_constraint_matrix(summary, figdir, valuedir, sel_cfg)
    effect_sizes(master, summary, out, sel_cfg)
    fv = pd.DataFrame(VALUES)
    fv.to_csv(figdir / "figure_values_revision.csv", index=False)
    fv.to_csv(out / "figure_values_revision.csv", index=False)
    fv.to_csv(valuedir / "figure_values_revision.csv", index=False)
    validate_figure_values(master, fv)
    checks = json.loads((out / "checks.json").read_text(encoding="utf-8")) if (out / "checks.json").exists() else {}
    write_claim_audit(summary, out)
    write_paper_ready_summary(summary, out, checks)
    write_final_reproducibility_check(out)
    write_reproducibility_manifest(out)
    print(f"[done] figures in {figdir} (fig_revision_*); values in {valuedir}")


if __name__ == "__main__":
    main()
