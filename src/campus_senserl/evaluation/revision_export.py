"""Paper tables, protocol JSON, and claim audit for the revision pass."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from campus_senserl.evaluation.revision_metrics import (
    DISPLAY_NAMES,
    EPS_AOI,
    EPS_MAE,
    EPS_MISS,
    PAPER_METHOD_ORDER,
    RECALL_MIN,
    aggregate_packet_loss,
    aggregate_runs,
    compact_method_rows,
    delta_grid_table,
    json_safe,
    parse_delta_config,
    select_delta_heartbeat,
)
from campus_senserl.utils import save_json

SEEDS = [42, 123, 2024, 3407, 9999]
PACKET_LOSS_MASK_SEED = 20260317
N_LOSS_MASKS = 5
TARGET_TX_REDUCTIONS = [0.75, 0.78, 0.80]
SELECTED_DELTA_DEFAULT = "delta_ppm=25,heartbeat=3"


def _is_num(x: Any) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def _fmt_mean_sd(mean, std, n, *, scale: float = 1.0, nd: int = 3) -> str:
    if not _is_num(mean):
        return "N/A"
    m = float(mean) * scale
    if int(n) > 1 and _is_num(std) and float(std) > 0:
        return f"{m:.{nd}f} ± {float(std) * scale:.{nd}f}"
    return f"{m:.{nd}f}"


def constraint_cell(rec: dict[str, Any]) -> str:
    method = str(rec.get("method", ""))
    if method == "fixed_15":
        return "N/A (full-transmission reference)"
    disp = rec.get("constraint_display")
    if disp and str(disp) not in {"", "nan", "None"}:
        return str(disp)
    kn = rec.get("all_constraints_satisfied_kn")
    if kn and str(kn) not in {"", "nan", "None"}:
        return str(kn)
    return str(rec.get("all_constraints_satisfied", "N/A"))


def paper_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rec in df.to_dict("records"):
        n = int(rec.get("n_runs", 1) or 1)
        if str(rec.get("runs", "")).find("mean_of_5") >= 0:
            n = max(n, 5)
        rows.append(
            {
                "Method": DISPLAY_NAMES.get(rec["method"], rec["method"]),
                "Configuration": rec.get("configuration", ""),
                "result_source": rec.get("result_source", rec.get("source", "")),
                "n_runs": n,
                "Transmission reduction (%)": _fmt_mean_sd(
                    rec.get("transmission_reduction_mean", rec.get("transmission_reduction")),
                    rec.get("transmission_reduction_std"),
                    n,
                    scale=100.0,
                    nd=2,
                ),
                "MAE (ppm)": _fmt_mean_sd(rec.get("mae_mean", rec.get("mae")), rec.get("mae_std"), n, nd=3),
                "Precision": _fmt_mean_sd(
                    rec.get("precision_mean", rec.get("precision")), rec.get("precision_std"), n, nd=4
                ),
                "Recall": _fmt_mean_sd(rec.get("recall_mean", rec.get("recall")), rec.get("recall_std"), n, nd=4),
                "F1": _fmt_mean_sd(rec.get("f1_mean", rec.get("f1")), rec.get("f1_std"), n, nd=4),
                "AoI": _fmt_mean_sd(rec.get("aoi_raw_mean", rec.get("aoi_raw")), rec.get("aoi_raw_std"), n, nd=3),
                "Constraints (k/n)": constraint_cell(rec),
                "MAE pass": rec.get("mae_ok_kn", rec.get("mae_ok", "N/A")),
                "Event pass": rec.get("event_ok_kn", rec.get("event_ok", "N/A")),
                "AoI pass": rec.get("aoi_ok_kn", rec.get("aoi_ok", "N/A")),
            }
        )
    return pd.DataFrame(rows)


def selected_delta_cfg(out_dir: Path, master: pd.DataFrame | None = None) -> str:
    for name in ["delta_selection.json", "delta_heartbeat_selection.json"]:
        path = out_dir / name
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            cfg = data.get("selected_configuration")
            if cfg:
                return str(cfg)
    if master is not None:
        val = master[
            (master["scenario"] == "validation_cohort")
            & (master["method"] == "delta_plus_heartbeat")
            & np.isclose(pd.to_numeric(master["packet_loss"], errors="coerce"), 0.0)
        ]
        if len(val):
            return str(select_delta_heartbeat(val)["selected_configuration"])
    return SELECTED_DELTA_DEFAULT


def evaluation_protocol_dict() -> dict[str, Any]:
    return {
        "training_performed": False,
        "frozen": [
            "methodology",
            "train/val/test splits",
            "development and held-out cohorts",
            "KL-CMAPPO checkpoints",
            "Delta+heartbeat VAL selection (delta_ppm=25, heartbeat=3)",
        ],
        "transmission": {
            "denominator": "N_locally_available",
            "numerator": "N_tx_attempts",
            "formula": "transmission_reduction = 1 - N_tx_attempts / N_locally_available",
            "natural_missingness": "excluded from the denominator; never locally available",
            "packet_loss_cost": "counted on TX attempts, not deliveries",
        },
        "periodic_fixed_k": {
            "clock": "env step / 15-min grid",
            "rule": "TRANSMIT iff t % k == 0, then SKIP if not local_available",
            "missingness": "does not shift later TX times; the counter still advances",
            "not": "every nth valid observation",
        },
        "primary_event": {
            "definition": "combined OR: y >= 1000 ppm OR rise >= 150 ppm vs previous finite GT",
            "server_state": "same rule on reconstructed monitoring value (TX ground truth or recon/LOCF)",
            "skip_semantics": "SKIP is not an automatic FN; skip + correct reconstruction can be TP",
            "high_co2_and_rapid_rise": "logged separately as secondary; revision tables report the combined event",
        },
        "packet_loss": {
            "mode": "independent_slots",
            "mask": "shared (t, sensor) Bernoulli mask, identical across methods for a given loss_mask_id",
            "n_replicates": N_LOSS_MASKS,
            "base_seed": PACKET_LOSS_MASK_SEED,
            "mask_seed": "PACKET_LOSS_MASK_SEED + loss_mask_id",
            "hierarchy": {
                "columns": ["policy_run", "loss_mask_id"],
                "deterministic": "mean and sample SD across shared loss-mask replicates",
                "kl_cmappo": "mean over masks within each frozen policy, then sample SD across the 5 policies",
                "not": "25 policy×mask rows are not 25 i.i.d. trained models",
            },
            "bc": "legacy per-attempt packet-loss rows are not used in the new shared-mask figure",
        },
        "delta_selection": {
            "split": "VAL only",
            "grid": "delta in {25,50,75,100} ppm × heartbeat in {3,4,5,6}",
            "feasible": "MAE<=9 and miss rate<=0.015 and AoI<=3.5",
            "if_none_feasible": "Pareto set → min violation_score → max TX reduction",
            "violation_score": "max(0, MAE/9-1) + max(0, miss/0.015-1) + max(0, AoI/3.5-1)",
            "frozen_selection": SELECTED_DELTA_DEFAULT,
            "reselection": False,
        },
        "constraints": {
            "mae_ppm": EPS_MAE,
            "event_miss_rate": EPS_MISS,
            "recall_min": RECALL_MIN,
            "aoi_raw": EPS_AOI,
            "aggregation": "k/n pass counts; False if any contributing replicate fails",
            "fixed_15": "N/A (full-transmission reference)",
            "do_not_judge_feasibility_from_mean_alone": True,
        },
        "matched_budget": {
            "targets": TARGET_TX_REDUCTIONS,
            "kl": "direct VAL tau sweep on frozen actors; report the measured tau nearest each target",
            "bc": "interpolated estimate from legacy frozen summary only; not primary",
        },
        "generalization_figure": {
            "comparison": "development cohort TEST vs held-out cohort TEST (same temporal test window)",
            "not": "VAL vs held-out TEST (would confound sensors with time)",
        },
    }


def statistical_conventions_dict() -> dict[str, Any]:
    return {
        "dispersion": "sample standard deviation",
        "ddof": 1,
        "n": "number of independent policy runs, except packet-loss deterministic methods where n is the number of shared loss-mask replicates",
        "kl_seeds": SEEDS,
        "frozen_test_seed_original_paper_report": 123,
        "packet_loss_kl": "SD is across 5 policies after averaging 5 shared masks per policy",
        "legacy_scientific_validation_csvs": {
            "path": "results/rl_final/scientific_validation/",
            "ddof": 0,
            "not_rewritten": True,
            "paper_facing_revision": "use results/revision/ with ddof=1",
        },
        "bc_val_test_rows": "legacy pre-averaged summaries may have n_runs=1 on a mean-of-5 row; labelled result_source=legacy_frozen",
    }


def matched_budget_protocol_dict() -> dict[str, Any]:
    return {
        "targets_tx_reduction": TARGET_TX_REDUCTIONS,
        "split": "val",
        "cohort": "final",
        "kl": {
            "type": "direct",
            "procedure": "Evaluate frozen KL-CMAPPO actors on a modest tau grid. For each target, select the measured tau whose mean TX reduction is closest to the target. Report that tau's measured MAE/recall/AoI (mean±sample SD over seeds). Do not interpolate MAE or recall.",
            "result_source": "direct_measured",
            "primary": True,
        },
        "bc": {
            "type": "interpolated_legacy",
            "procedure": "Linear interpolation of MAE/recall vs TX reduction from frozen scientific_validation CSVs (ddof=0 in the source file). Checkpoints were not present for a direct sweep.",
            "result_source": "interpolated_legacy",
            "primary": False,
        },
    }


def packet_loss_protocol_dict() -> dict[str, Any]:
    return {
        "n_loss_masks": N_LOSS_MASKS,
        "mask_ids": list(range(N_LOSS_MASKS)),
        "packet_loss_mask_seed": PACKET_LOSS_MASK_SEED,
        "seed_formula": "PACKET_LOSS_MASK_SEED + loss_mask_id",
        "mode": "independent_slots",
        "shared_across_methods": True,
        "rate_zero": "single evaluation; no mask",
        "aggregation": {
            "kl_cmappo": "mean over loss_mask_id within policy_run, then mean and sample SD (ddof=1) across policy_run",
            "deterministic": "mean and sample SD (ddof=1) across loss_mask_id",
        },
        "exclude_from_new_figure": ["campus_senserl_bc_legacy_per_attempt"],
    }


def write_protocol_json(out_dir: Path) -> None:
    save_json(evaluation_protocol_dict(), out_dir / "evaluation_protocol.json")
    save_json(statistical_conventions_dict(), out_dir / "statistical_conventions.json")
    save_json(matched_budget_protocol_dict(), out_dir / "matched_budget_protocol.json")
    save_json(packet_loss_protocol_dict(), out_dir / "packet_loss_protocol.json")


def write_delta_artifacts(master: pd.DataFrame, out_dir: Path, selection: dict[str, Any] | None = None) -> dict[str, Any]:
    val = master[
        (master["scenario"] == "validation_cohort")
        & (master["method"] == "delta_plus_heartbeat")
        & np.isclose(pd.to_numeric(master["packet_loss"], errors="coerce"), 0.0)
    ].copy()
    if val.empty:
        return selection or {}
    grid = delta_grid_table(val)
    grid.to_csv(out_dir / "delta_grid_full.csv", index=False)
    if selection is None:
        selection = select_delta_heartbeat(val)
    clean = {
        "criterion": selection.get("criterion"),
        "reason": selection.get("reason"),
        "selected_configuration": selection.get("selected_configuration"),
        "n_grid": selection.get("n_grid"),
        "n_feasible": selection.get("n_feasible"),
        "n_pareto": selection.get("n_pareto"),
        "pareto_configurations": selection.get("pareto_configurations"),
        "reselection": False,
        "selected_metrics": {},
    }
    row = selection.get("selected_row") or {}
    if row:
        d, h = parse_delta_config(str(row.get("configuration", selection.get("selected_configuration", "delta_ppm=0,heartbeat=0"))))
        clean["selected_metrics"] = {
            "delta_threshold": d,
            "heartbeat": h,
            "tx_reduction": row.get("transmission_reduction"),
            "mae": row.get("mae"),
            "recall": row.get("recall"),
            "aoi": row.get("aoi_raw"),
            "violation_score": row.get("viol"),
            "pareto": row.get("pareto"),
            "all_constraints_satisfied": row.get("all_constraints_satisfied"),
        }
    save_json(json_safe(clean), out_dir / "delta_selection.json")
    save_json(json_safe(selection), out_dir / "delta_heartbeat_selection.json")
    return selection


def write_tables(master: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    sel_cfg = selected_delta_cfg(out_dir, master)
    core = master[
        master["scenario"].isin(["validation_cohort", "frozen_temporal_test", "heldout_sensors"])
        & np.isclose(pd.to_numeric(master["packet_loss"], errors="coerce").fillna(0.0), 0.0)
    ].copy()
    # Native operating points only (exclude tau-sweep copies if any leaked)
    if "prob_threshold" in core.columns:
        pt = pd.to_numeric(core["prob_threshold"], errors="coerce")
        keep_tau = ~core["method"].eq("cmappo_kl") | ~np.isfinite(pt) | np.isclose(pt, 0.5) | core["configuration"].astype(str).str.contains(r"tau=0\.5", regex=True)
        core = core.loc[keep_tau]
    agg = aggregate_runs(core)
    pl_agg = aggregate_packet_loss(master)
    summary_parts = [agg]
    if len(pl_agg):
        summary_parts.append(pl_agg)
    mb = master[master["scenario"] == "matched_budget"].copy()
    if len(mb):
        summary_parts.append(aggregate_runs(mb))
    summary = pd.concat(summary_parts, ignore_index=True, sort=False)
    summary.to_csv(out_dir / "all_method_summary.csv", index=False)

    def scenario_compact(scenario: str, split: str, path: Path, packet_loss: float = 0.0) -> pd.DataFrame:
        sub = agg[(agg["scenario"] == scenario) & (agg["split"] == split)]
        sub = sub[np.isclose(pd.to_numeric(sub["packet_loss"], errors="coerce").fillna(0.0), packet_loss)]
        compact = compact_method_rows(sub, selected_delta_cfg=sel_cfg)
        table = paper_table(compact)
        table.to_csv(path, index=False)
        return compact

    scenario_compact("validation_cohort", "val", out_dir / "table_validation_all_methods.csv")
    scenario_compact("frozen_temporal_test", "test", out_dir / "table_test_all_methods.csv")
    ho = agg[(agg["scenario"] == "heldout_sensors") & (agg["split"] == "test")]
    ho = ho[np.isclose(pd.to_numeric(ho["packet_loss"], errors="coerce").fillna(0.0), 0.0)]
    paper_table(compact_method_rows(ho, selected_delta_cfg=sel_cfg)).to_csv(
        out_dir / "table_unseen_all_methods.csv", index=False
    )

    val_main = agg[(agg["scenario"] == "validation_cohort")]
    val_main = val_main[np.isclose(pd.to_numeric(val_main["packet_loss"], errors="coerce").fillna(0.0), 0.0)]
    paper_table(compact_method_rows(val_main, selected_delta_cfg=sel_cfg)).to_csv(
        out_dir / "table_validation_main_methods.csv", index=False
    )

    if len(pl_agg):
        pt = paper_table(pl_agg)
        pt.insert(1, "Packet loss (%)", (100.0 * pd.to_numeric(pl_agg["packet_loss"], errors="coerce")).round(0).astype(int))
        pt.to_csv(out_dir / "table_packet_loss_all_methods.csv", index=False)
        pl_agg.to_csv(out_dir / "table_packet_loss_all_methods_full.csv", index=False)

    if len(mb):
        mb_agg = aggregate_runs(mb)
        # Primary: KL direct + BC interpolated only
        keep = []
        for rec in mb_agg.to_dict("records"):
            method = str(rec.get("method", ""))
            cfg = str(rec.get("configuration", ""))
            rs = str(rec.get("result_source", ""))
            if method == "cmappo_kl" and "interpolated_target" in cfg:
                continue
            if method == "cmappo_kl" and "interpolat" in rs and "direct" not in rs:
                continue
            keep.append(rec)
        if keep:
            paper_table(pd.DataFrame(keep)).to_csv(out_dir / "table_matched_budget.csv", index=False)
        else:
            paper_table(mb_agg).to_csv(out_dir / "table_matched_budget.csv", index=False)

    paper_table(val_main).to_csv(out_dir / "table_validation_native_operating_points.csv", index=False)
    return summary


def _lookup(summary: pd.DataFrame, *, scenario: str, split: str, method: str, selected_cfg: str) -> pd.Series | None:
    hit = summary[
        (summary["scenario"] == scenario)
        & (summary["split"] == split)
        & (summary["method"] == method)
        & np.isclose(pd.to_numeric(summary["packet_loss"], errors="coerce").fillna(0.0), 0.0)
    ]
    if "target_reduction" in hit.columns:
        tr = pd.to_numeric(hit["target_reduction"], errors="coerce")
        native = hit.loc[~np.isfinite(tr)]
        if len(native):
            hit = native
    if method == "delta_plus_heartbeat" and selected_cfg:
        sel = hit[hit["configuration"].astype(str) == selected_cfg]
        if len(sel):
            hit = sel
    if hit.empty:
        return None
    return hit.iloc[0]


def _num(rec: pd.Series | None, key: str) -> float:
    if rec is None:
        return float("nan")
    return float(rec[key]) if _is_num(rec.get(key)) else float("nan")


def write_claim_audit(summary: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    sel_cfg = selected_delta_cfg(out_dir)
    rows: list[dict[str, Any]] = []

    def add(claim: str, supported: str, evidence: str, **nums: Any) -> None:
        rows.append({"claim": claim, "supported": supported, "evidence": evidence, **nums})

    def get(scenario, split, method):
        return _lookup(summary, scenario=scenario, split=split, method=method, selected_cfg=sel_cfg)

    kl_val = get("validation_cohort", "val", "cmappo_kl")
    kl_test = get("frozen_temporal_test", "test", "cmappo_kl")
    kl_ho = get("heldout_sensors", "test", "cmappo_kl")
    ex_val = get("validation_cohort", "val", "semantic_expert")
    f60_val = get("validation_cohort", "val", "fixed_60")
    f75_val = get("validation_cohort", "val", "fixed_75")
    dlt_val = get("validation_cohort", "val", "delta_plus_heartbeat")
    bc_val = get("validation_cohort", "val", "campus_senserl_bc")
    f60_test = get("frozen_temporal_test", "test", "fixed_60")
    f75_test = get("frozen_temporal_test", "test", "fixed_75")
    dlt_test = get("frozen_temporal_test", "test", "delta_plus_heartbeat")
    ex_test = get("frozen_temporal_test", "test", "semantic_expert")
    f60_ho = get("heldout_sensors", "test", "fixed_60")
    dlt_ho = get("heldout_sensors", "test", "delta_plus_heartbeat")
    ex_ho = get("heldout_sensors", "test", "semantic_expert")

    kl_val_ok = str(kl_val.get("all_constraints_satisfied")) == "True" if kl_val is not None else False
    add(
        "CAMPUS-SenseRL satisfies VAL constraints (MAE≤9, miss≤1.5%, AoI≤3.5)",
        "yes" if kl_val_ok else "no",
        f"VAL all_constraints={kl_val.get('all_constraints_satisfied') if kl_val is not None else 'missing'} "
        f"({kl_val.get('all_constraints_satisfied_kn') if kl_val is not None else ''})",
        mae=_num(kl_val, "mae_mean"),
        recall=_num(kl_val, "recall_mean"),
        tx_reduction=_num(kl_val, "transmission_reduction_mean"),
    )
    kl_test_ok = str(kl_test.get("all_constraints_satisfied")) == "True" if kl_test is not None else False
    add(
        "CAMPUS-SenseRL is feasible on the frozen temporal test split",
        "no" if not kl_test_ok else "yes",
        "Test event-miss constraint fails for KL and for every communicating baseline "
        f"(KL recall={_num(kl_test, 'recall_mean'):.4f}, required ≥ {RECALL_MIN:.3f}). "
        "Do not claim test feasibility.",
        mae=_num(kl_test, "mae_mean"),
        recall=_num(kl_test, "recall_mean"),
        constraints=str(kl_test.get("all_constraints_satisfied") if kl_test is not None else "missing"),
    )
    mae_kl = _num(kl_val, "mae_mean")
    mae_ex = _num(ex_val, "mae_mean")
    add(
        "CAMPUS-SenseRL beats the semantic expert on VAL reconstruction MAE",
        "no" if (np.isfinite(mae_kl) and np.isfinite(mae_ex) and mae_kl > mae_ex) else "yes" if mae_kl < mae_ex else "unknown",
        f"VAL MAE expert={mae_ex:.3f} vs KL={mae_kl:.3f} ppm. Expert has lower MAE; KL saves more TX than the expert.",
        mae_kl=mae_kl,
        mae_expert=mae_ex,
        tx_kl=_num(kl_val, "transmission_reduction_mean"),
        tx_expert=_num(ex_val, "transmission_reduction_mean"),
    )
    tx_kl = _num(kl_val, "transmission_reduction_mean")
    tx_bc = _num(bc_val, "transmission_reduction_mean")
    add(
        "CAMPUS-SenseRL always transmits less than BC initialization",
        "no" if (np.isfinite(tx_kl) and np.isfinite(tx_bc) and tx_kl < tx_bc) else "yes" if tx_kl > tx_bc else "unknown",
        f"VAL TX reduction KL={100*tx_kl:.2f}% vs BC={100*tx_bc:.2f}%. "
        "KL transmits more often than BC at tau=0.5; BC VAL row is legacy_frozen.",
        tx_kl=tx_kl,
        tx_bc=tx_bc,
        bc_result_source=str(bc_val.get("result_source") if bc_val is not None else ""),
    )
    add(
        "CAMPUS-SenseRL has lower VAL MAE than Fixed-60 at comparable TX reduction",
        "yes" if mae_kl < _num(f60_val, "mae_mean") else "no",
        f"VAL MAE KL={mae_kl:.3f} vs Fixed-60={_num(f60_val, 'mae_mean'):.3f}; "
        f"TX↓ KL={100*tx_kl:.2f}% vs Fixed-60={100*_num(f60_val, 'transmission_reduction_mean'):.2f}%",
        mae_kl=mae_kl,
        mae_fixed60=_num(f60_val, "mae_mean"),
    )
    add(
        "CAMPUS-SenseRL has lower VAL MAE than Fixed-75",
        "yes" if mae_kl < _num(f75_val, "mae_mean") else "no",
        f"VAL MAE KL={mae_kl:.3f} vs Fixed-75={_num(f75_val, 'mae_mean'):.3f}",
        mae_fixed75=_num(f75_val, "mae_mean"),
    )
    add(
        "CAMPUS-SenseRL has lower VAL MAE than VAL-selected Delta+heartbeat",
        "yes" if mae_kl < _num(dlt_val, "mae_mean") else "no",
        f"VAL MAE KL={mae_kl:.3f} vs Delta {sel_cfg}={_num(dlt_val, 'mae_mean'):.3f}; "
        f"Delta was not constraint-feasible on VAL (MAE>9)",
        mae_delta=_num(dlt_val, "mae_mean"),
        delta_cfg=sel_cfg,
    )
    add(
        "Held-out test MAE remains below 9 ppm for CAMPUS-SenseRL",
        "yes" if _num(kl_ho, "mae_mean") <= EPS_MAE else "no",
        f"Held-out TEST MAE={_num(kl_ho, 'mae_mean'):.3f} (n={int(kl_ho['n_runs']) if kl_ho is not None else 0})",
        mae_heldout=_num(kl_ho, "mae_mean"),
        mae_dev_test=_num(kl_test, "mae_mean"),
    )
    add(
        "Fig05 window plot is primary evidence",
        "no",
        "Fig05 remains illustrative only (results/figures/fig05_window.json).",
    )
    add(
        "Matched-budget 75/78/80% MAE/recall for KL are direct rollouts",
        "yes"
        if (
            "result_source" in summary.columns
            and (
                summary["scenario"].eq("matched_budget")
                & summary["method"].eq("cmappo_kl")
                & summary["result_source"].astype(str).str.contains("direct")
            ).any()
        )
        else "pending_or_no",
        "Primary KL matched-budget uses nearest measured tau on VAL. BC remains interpolated_legacy.",
    )
    add(
        "Periodic Fixed-k is time-grid aligned",
        "yes",
        "FixedIntervalPolicy increments the clock every env step; missing slots convert TRANSMIT→SKIP without shifting later TX times.",
    )

    # Carry numeric comparisons for paper text
    for label, a, b in [
        ("frozen_test_KL_vs_Fixed-60_MAE", kl_test, f60_test),
        ("frozen_test_KL_vs_Fixed-75_MAE", kl_test, f75_test),
        ("frozen_test_KL_vs_Delta_MAE", kl_test, dlt_test),
        ("frozen_test_KL_vs_expert_MAE", kl_test, ex_test),
        ("heldout_test_KL_vs_Fixed-60_MAE", kl_ho, f60_ho),
        ("heldout_test_KL_vs_Delta_MAE", kl_ho, dlt_ho),
        ("heldout_test_KL_vs_expert_MAE", kl_ho, ex_ho),
    ]:
        add(
            label,
            "comparison",
            f"KL MAE={_num(a, 'mae_mean'):.3f} vs other MAE={_num(b, 'mae_mean'):.3f}",
            mae_kl=_num(a, "mae_mean"),
            mae_other=_num(b, "mae_mean"),
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "claim_audit.csv", index=False)
    return df


def _md_row(rec: pd.Series | None) -> str:
    if rec is None:
        return "_missing_"
    n = int(rec.get("n_runs", 1) or 1)
    tx = _fmt_mean_sd(rec.get("transmission_reduction_mean"), rec.get("transmission_reduction_std"), n, scale=100.0, nd=2)
    mae = _fmt_mean_sd(rec.get("mae_mean"), rec.get("mae_std"), n, nd=3)
    rec_s = _fmt_mean_sd(rec.get("recall_mean"), rec.get("recall_std"), n, nd=4)
    return f"TX↓ {tx}%; MAE {mae} ppm; recall {rec_s}; constraints {constraint_cell(rec.to_dict())}; n={n}; source={rec.get('result_source', rec.get('source'))}"


def write_paper_ready_summary(summary: pd.DataFrame, out_dir: Path, checks: dict[str, Any] | None = None) -> None:
    sel_cfg = selected_delta_cfg(out_dir)
    sel = {}
    if (out_dir / "delta_selection.json").exists():
        sel = json.loads((out_dir / "delta_selection.json").read_text(encoding="utf-8"))
    checks = checks or {}
    get = lambda sc, sp, m: _lookup(summary, scenario=sc, split=sp, method=m, selected_cfg=sel_cfg)

    lines = [
        "# Paper-ready revision summary",
        "",
        "No training was performed. Splits, cohorts, and KL-CMAPPO checkpoints are frozen.",
        "Every numeric claim below is copied from `results/revision/` tables.",
        "",
        "## Protocol",
        "",
        "- Primary event: high-CO2 (**≥1000 ppm**) **OR** rapid rise (**≥150 ppm**) on ground truth vs the same rule on the server monitor (TX GT or reconstruction). SKIP is not an automatic FN.",
        "- TX reduction: `1 - N_tx_attempts / N_locally_available`. Packet-loss cost is counted on attempts.",
        "- Periodic Fixed-k is aligned to the 15-minute time grid; missingness does not shift later TX times.",
        "- Dispersion is **sample SD (ddof=1)**. Legacy `results/rl_final/` matched-budget files used ddof=0 and were not rewritten.",
        "- Fixed-15 MAE is **N/A (full-transmission reference)**.",
        "",
        "## Delta+heartbeat (VAL freeze, not re-selected)",
        "",
        f"- Selected: `{sel.get('selected_configuration', sel_cfg)}`",
        f"- Reason: {sel.get('reason', '')}",
        f"- Feasible VAL configs: {sel.get('n_feasible', 0)} / {sel.get('n_grid', 16)}",
        "",
        "## Validation (development cohort)",
        "",
        f"- CAMPUS-SenseRL: {_md_row(get('validation_cohort', 'val', 'cmappo_kl'))}",
        f"- Fixed-60: {_md_row(get('validation_cohort', 'val', 'fixed_60'))}",
        f"- Fixed-75: {_md_row(get('validation_cohort', 'val', 'fixed_75'))}",
        f"- Delta+heartbeat: {_md_row(get('validation_cohort', 'val', 'delta_plus_heartbeat'))}",
        f"- Semantic expert: {_md_row(get('validation_cohort', 'val', 'semantic_expert'))}",
        f"- BC initialization: {_md_row(get('validation_cohort', 'val', 'campus_senserl_bc'))}",
        "",
        "## Frozen temporal test (development cohort)",
        "",
        f"- CAMPUS-SenseRL: {_md_row(get('frozen_temporal_test', 'test', 'cmappo_kl'))}",
        f"- Fixed-60: {_md_row(get('frozen_temporal_test', 'test', 'fixed_60'))}",
        f"- Semantic expert: {_md_row(get('frozen_temporal_test', 'test', 'semantic_expert'))}",
        "",
        "**Unsupported on test:** constraint feasibility. Event recall on this slice is below 0.985 for KL and for the communicating baselines. Do not claim the test operating point is feasible.",
        "",
        "## Held-out sensors (same test window)",
        "",
        f"- CAMPUS-SenseRL: {_md_row(get('heldout_sensors', 'test', 'cmappo_kl'))}",
        f"- Fixed-60: {_md_row(get('heldout_sensors', 'test', 'fixed_60'))}",
        f"- Delta+heartbeat: {_md_row(get('heldout_sensors', 'test', 'delta_plus_heartbeat'))}",
        f"- Semantic expert: {_md_row(get('heldout_sensors', 'test', 'semantic_expert'))}",
        "",
        "The generalization figure compares **development TEST vs held-out TEST**, not VAL vs held-out.",
        "",
        "## Claims that are not supported",
        "",
        "- Beating the semantic expert on MAE (expert MAE is lower on VAL, test, and held-out).",
        "- Always transmitting less than BC (at tau=0.5, KL TX reduction is lower than BC; BC VAL/test rows may be legacy_frozen).",
        "- Test-set constraint feasibility.",
        "- Interpolated matched-budget numbers as if they were direct KL rollouts. Only the VAL tau-sweep rows labelled `direct_measured` are primary; BC remains `interpolated_legacy`.",
        "",
        "## Figures",
        "",
        "- `fig_revision_tradeoff_mae.png` — who keeps low MAE at high TX reduction?",
        "- `fig_revision_event_miss.png` — who preserves events (miss rate %, 1.5% line)?",
        "- `fig_revision_generalization_dumbbell.png` — MAE, development TEST vs unseen TEST.",
        "- `fig_revision_packet_loss_mae.png` / `fig_revision_packet_loss_event_miss.png` — shared-mask replicates; BC omitted.",
        "- `fig_revision_contextual_relations.png` — spatial/statistical relations, **not** communication links.",
        "- `fig_revision_constraint_matrix.png` — VAL Pass/Fail/N/A supplement.",
        "- Fig05 remains illustrative only.",
        "",
        f"- Training performed: {checks.get('training_performed', False)}",
        "",
    ]
    (out_dir / "paper_ready_summary.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "revision_summary.md").write_text("\n".join(lines), encoding="utf-8")
