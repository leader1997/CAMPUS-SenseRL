"""Paper tables, protocol JSON, and claim audit for the revision pass."""

from __future__ import annotations

import hashlib
import json
import re
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
from campus_senserl.utils import environment_fingerprint, git_commit_hash, load_yaml, repo_root, save_json

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


def _fmt_count(mean, std, n) -> str:
    if not _is_num(mean):
        return "N/A"
    m = float(mean)
    if int(n) > 1 and _is_num(std) and float(std) > 0:
        return f"{m:.1f} ± {float(std):.1f}"
    if abs(m - round(m)) < 1e-6:
        return str(int(round(m)))
    return f"{m:.1f}"


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
                "n_true_events": _fmt_count(
                    rec.get("n_true_events_mean", rec.get("n_true_events")), rec.get("n_true_events_std"), n
                ),
                "n_tp": _fmt_count(rec.get("event_tp_mean", rec.get("event_tp")), rec.get("event_tp_std"), n),
                "n_fn": _fmt_count(rec.get("event_fn_mean", rec.get("event_fn")), rec.get("event_fn_std"), n),
                "n_fp": _fmt_count(rec.get("event_fp_mean", rec.get("event_fp")), rec.get("event_fp_std"), n),
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
        "mixed_kl_direct_bc_interpolated": {
            "primary_evidence": False,
            "use_as": "supplementary / secondary evidence only",
            "reason": "KL matched-budget MAE/recall are directly measured; BC values are linearly interpolated from a legacy summary. Do not treat them as a matched-budget bake-off.",
        },
        "do_not_interpolate_mae_or_recall_for_kl": True,
        "threshold_selection": "validation only; freeze tau; evaluate that tau (no test-set tau search)",
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


def event_definition_dict() -> dict[str, Any]:
    return {
        "headline_event": "union",
        "headline_alias": "event_recall is event_recall_union",
        "training_and_evaluation_identical": True,
        "formulas": {
            "high_co2": "x_t >= 1000 ppm (finite x_t)",
            "rapid_rise": "(x_t - x_{t-1}) >= 150 ppm (both finite)",
            "union": "event_t = high_co2 OR rapid_rise",
        },
        "thresholds": {
            "primary_threshold_ppm": 1000.0,
            "rapid_increase_ppm": 150.0,
        },
        "server_state": (
            "The same rule is applied to the server monitoring value "
            "(delivered ground truth on successful TX, otherwise reconstruction/LOCF). "
            "SKIP is not an automatic FN: a skipped observation reconstructed as an event can be TP."
        ),
        "code": {
            "detector": "campus_senserl.environment.event_detector.EventDetector.detect_step / classify_detection",
            "training_constraint": "campus_senserl.rl.cmappo.CMAPPOTrainer._step_costs j_miss from info['missed_events'] (union FN)",
            "paper_eval": "campus_senserl.evaluation.rl_policy_eval.evaluate_policy event_recall == event_recall_union",
            "revision_tables": "results/revision/*.csv column recall = union event recall",
        },
        "secondary_metrics": [
            "event_recall_union",
            "event_precision_union",
            "event_f1_union",
            "high_co2_recall",
            "high_co2_precision",
            "high_co2_f1",
            "rapid_rise_recall",
            "rapid_rise_precision",
            "rapid_rise_f1",
        ],
        "do_not_change": "Original validated implementation uses the union; revision did not alter it.",
    }


def write_protocol_json(out_dir: Path) -> None:
    save_json(evaluation_protocol_dict(), out_dir / "evaluation_protocol.json")
    save_json(statistical_conventions_dict(), out_dir / "statistical_conventions.json")
    save_json(matched_budget_protocol_dict(), out_dir / "matched_budget_protocol.json")
    save_json(packet_loss_protocol_dict(), out_dir / "packet_loss_protocol.json")
    save_json(event_definition_dict(), out_dir / "event_definition.json")


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
    write_matched_budget_points(summary, out_dir)
    write_generalization_same_split(summary, out_dir, sel_cfg)
    write_packet_loss_raw(master, out_dir)
    return summary


def write_matched_budget_points(summary: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """One row per reported 75/78/80% matched-budget point with provenance."""
    mb = summary[summary["scenario"].astype(str) == "matched_budget"].copy()
    rows: list[dict[str, Any]] = []
    for rec in mb.to_dict("records"):
        method = str(rec.get("method", ""))
        cfg = str(rec.get("configuration", ""))
        rs = str(rec.get("result_source", rec.get("source", "")))
        if method == "cmappo_kl" and "interpolated_target" in cfg:
            continue
        eval_type = "direct" if "direct" in rs else "interpolated"
        tau = rec.get("prob_threshold", float("nan"))
        if not _is_num(tau):
            m = re.search(r"tau=([0-9.]+)", cfg)
            tau = float(m.group(1)) if m else float("nan")
        rows.append(
            {
                "method": method,
                "display_name": DISPLAY_NAMES.get(method, method),
                "target_reduction": rec.get("target_reduction"),
                "selected_probability_threshold": tau if _is_num(tau) else "N/A",
                "achieved_reduction": rec.get("transmission_reduction_mean", rec.get("transmission_reduction")),
                "mae": rec.get("mae_mean", rec.get("mae")),
                "recall": rec.get("recall_mean", rec.get("recall")),
                "precision": rec.get("precision_mean", rec.get("precision")),
                "f1": rec.get("f1_mean", rec.get("f1")),
                "aoi": rec.get("aoi_raw_mean", rec.get("aoi_raw")),
                "n_runs": rec.get("n_runs"),
                "source": rec.get("source"),
                "result_source": rs,
                "evaluation_type": eval_type,
                "primary_evidence": bool(method == "cmappo_kl" and eval_type == "direct"),
                "configuration": cfg,
            }
        )
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(out_dir / "matched_budget_points.csv", index=False)
    return df


def write_generalization_same_split(summary: pd.DataFrame, out_dir: Path, sel_cfg: str) -> pd.DataFrame:
    """Development TEST vs held-out TEST, same chronological test interval."""
    methods = ["fixed_60", "fixed_75", "delta_plus_heartbeat", "semantic_expert", "campus_senserl_bc", "cmappo_kl"]
    specs = [
        ("development", "frozen_temporal_test", "test"),
        ("heldout", "heldout_sensors", "test"),
    ]
    rows: list[dict[str, Any]] = []
    for method in methods:
        for cohort, scenario, split in specs:
            rec = _lookup(summary, scenario=scenario, split=split, method=method, selected_cfg=sel_cfg)
            if rec is None:
                continue
            n = int(rec.get("n_runs", 1) or 1)
            rows.append(
                {
                    "method": method,
                    "display_name": DISPLAY_NAMES.get(method, method),
                    "cohort": cohort,
                    "split": split,
                    "scenario": scenario,
                    "tx_reduction": rec.get("transmission_reduction_mean"),
                    "mae": rec.get("mae_mean"),
                    "recall": rec.get("recall_mean"),
                    "precision": rec.get("precision_mean"),
                    "f1": rec.get("f1_mean"),
                    "aoi": rec.get("aoi_raw_mean"),
                    "mean": rec.get("mae_mean"),
                    "sd": rec.get("mae_std"),
                    "tx_reduction_sd": rec.get("transmission_reduction_std"),
                    "mae_sd": rec.get("mae_std"),
                    "recall_sd": rec.get("recall_std"),
                    "precision_sd": rec.get("precision_std"),
                    "f1_sd": rec.get("f1_std"),
                    "aoi_sd": rec.get("aoi_raw_std"),
                    "n_runs": n,
                    "source": rec.get("source"),
                    "result_source": rec.get("result_source"),
                    "configuration": rec.get("configuration"),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "generalization_same_split.csv", index=False)
    return df


def write_packet_loss_raw(master: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    pl = master[master["scenario"].astype(str) == "packet_loss"].copy()
    if pl.empty:
        return pl
    # Shared-mask raw rows only. Legacy BC per-attempt summaries are incomplete and not primary.
    src = pl["source"].astype(str)
    rs = pl["result_source"].astype(str)
    keep_mask = src.eq("revision_eval") & ~rs.str.contains("legacy|interpolat", case=False, regex=True)
    pl = pl.loc[keep_mask].copy()
    pl = pl.copy()
    n_av = pd.to_numeric(pl.get("n_available"), errors="coerce")
    n_tx = pd.to_numeric(pl.get("n_tx_attempts"), errors="coerce")
    n_del = pd.to_numeric(pl.get("n_delivered"), errors="coerce")
    pl["attempt_rate"] = n_tx / n_av
    pl["delivery_rate"] = n_del / n_av
    pl["attempt_based_reduction"] = pd.to_numeric(pl.get("transmission_reduction"), errors="coerce")
    keep = [
        "method",
        "policy_run",
        "loss_mask_id",
        "packet_loss",
        "n_available",
        "n_tx_attempts",
        "n_delivered",
        "attempt_rate",
        "delivery_rate",
        "attempt_based_reduction",
        "transmission_reduction",
        "mae",
        "recall",
        "precision",
        "f1",
        "aoi_raw",
        "configuration",
        "result_source",
        "source",
        "packet_loss_mode",
        "notes",
    ]
    for c in keep:
        if c not in pl.columns:
            pl[c] = np.nan
    out = pl[keep].rename(columns={"recall": "event_recall", "aoi_raw": "aoi"})
    out.to_csv(out_dir / "packet_loss_raw.csv", index=False)
    return out


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

    def add(
        claim: str,
        supported: str,
        *,
        evidence_file: str,
        evidence_metric: str,
        notes: str,
    ) -> None:
        rows.append(
            {
                "claim": claim,
                "supported": supported,
                "evidence_file": evidence_file,
                "evidence_metric": evidence_metric,
                "notes": notes,
            }
        )

    def get(scenario, split, method):
        return _lookup(summary, scenario=scenario, split=split, method=method, selected_cfg=sel_cfg)

    kl_val = get("validation_cohort", "val", "cmappo_kl")
    kl_test = get("frozen_temporal_test", "test", "cmappo_kl")
    kl_ho = get("heldout_sensors", "test", "cmappo_kl")
    ex_val = get("validation_cohort", "val", "semantic_expert")
    ex_test = get("frozen_temporal_test", "test", "semantic_expert")
    ex_ho = get("heldout_sensors", "test", "semantic_expert")
    dlt_val = get("validation_cohort", "val", "delta_plus_heartbeat")

    kn_val = str(kl_val.get("all_constraints_satisfied_kn") if kl_val is not None else "")
    kn_test = str(kl_test.get("all_constraints_satisfied_kn") if kl_test is not None else "")
    kn_ho = str(kl_ho.get("all_constraints_satisfied_kn") if kl_ho is not None else "")
    kl_val_ok = str(kl_val.get("all_constraints_satisfied")) == "True" if kl_val is not None else False
    kl_test_ok = str(kl_test.get("all_constraints_satisfied")) == "True" if kl_test is not None else False
    kl_ho_ok = str(kl_ho.get("all_constraints_satisfied")) == "True" if kl_ho is not None else False

    add(
        "KL-CMAPPO satisfies all predefined constraints on validation.",
        "true" if kl_val_ok else "false",
        evidence_file="results/revision/table_validation_main_methods.csv",
        evidence_metric="all_constraints_satisfied_kn",
        notes=f"VAL k/n={kn_val}; MAE={_num(kl_val, 'mae_mean'):.3f}, recall={_num(kl_val, 'recall_mean'):.4f}, AoI={_num(kl_val, 'aoi_raw_mean'):.3f}; n=5.",
    )
    add(
        "KL-CMAPPO satisfies all predefined constraints on frozen temporal test.",
        "true" if kl_test_ok else "false",
        evidence_file="results/revision/table_test_all_methods.csv",
        evidence_metric="event_ok_kn",
        notes=(
            f"Test k/n={kn_test}. Event constraint fails (recall={_num(kl_test, 'recall_mean'):.4f} < {RECALL_MIN:.3f}); "
            "MAE and AoI pass 5/5. Do not claim frozen-test feasibility."
        ),
    )
    add(
        "KL-CMAPPO satisfies all predefined constraints on unseen sensors.",
        "true" if kl_ho_ok else "false",
        evidence_file="results/revision/table_unseen_all_methods.csv",
        evidence_metric="all_constraints_satisfied_kn",
        notes=f"Held-out TEST k/n={kn_ho}; MAE={_num(kl_ho, 'mae_mean'):.3f}, recall={_num(kl_ho, 'recall_mean'):.4f}; n=5. Same chronological test window as development TEST.",
    )

    mae_pairs = [
        ("val", _num(kl_val, "mae_mean"), _num(ex_val, "mae_mean")),
        ("test", _num(kl_test, "mae_mean"), _num(ex_test, "mae_mean")),
        ("heldout", _num(kl_ho, "mae_mean"), _num(ex_ho, "mae_mean")),
    ]
    kl_mae_lower = all(np.isfinite(a) and np.isfinite(b) and a < b for _, a, b in mae_pairs)
    add(
        "KL-CMAPPO has lower MAE than the semantic expert.",
        "true" if kl_mae_lower else "false",
        evidence_file="results/revision/table_validation_main_methods.csv",
        evidence_metric="mae_mean",
        notes=(
            "Expert MAE is lower on every comparable split: "
            + "; ".join(f"{name} KL={a:.3f} vs expert={b:.3f}" for name, a, b in mae_pairs)
            + "."
        ),
    )

    tx_pairs = [
        ("val", _num(kl_val, "transmission_reduction_mean"), _num(ex_val, "transmission_reduction_mean")),
        ("test", _num(kl_test, "transmission_reduction_mean"), _num(ex_test, "transmission_reduction_mean")),
        ("heldout", _num(kl_ho, "transmission_reduction_mean"), _num(ex_ho, "transmission_reduction_mean")),
    ]
    kl_tx_higher = all(np.isfinite(a) and np.isfinite(b) and a > b for _, a, b in tx_pairs)
    add(
        "KL-CMAPPO reduces communication more than the semantic expert.",
        "true" if kl_tx_higher else "false",
        evidence_file="results/revision/table_validation_main_methods.csv",
        evidence_metric="transmission_reduction_mean",
        notes=(
            "Attempt-based TX reduction: "
            + "; ".join(f"{name} KL={100*a:.2f}% vs expert={100*b:.2f}%" for name, a, b in tx_pairs)
            + "."
        ),
    )

    d_tx = _num(dlt_val, "transmission_reduction_mean")
    d_mae = _num(dlt_val, "mae_mean")
    d_rec = _num(dlt_val, "recall_mean")
    k_tx = _num(kl_val, "transmission_reduction_mean")
    k_mae = _num(kl_val, "mae_mean")
    k_rec = _num(kl_val, "recall_mean")
    beats_delta = (
        np.isfinite(k_tx) and k_tx > d_tx and np.isfinite(k_mae) and k_mae < d_mae and np.isfinite(k_rec) and k_rec > d_rec
    )
    add(
        "KL-CMAPPO outperforms selected Delta on validation reduction, MAE, and event recall.",
        "true" if beats_delta else "false",
        evidence_file="results/revision/table_validation_main_methods.csv",
        evidence_metric="transmission_reduction_mean,mae_mean,recall_mean",
        notes=(
            f"VAL selected Delta `{sel_cfg}`: TX↓ {100*d_tx:.2f}% MAE {d_mae:.3f} recall {d_rec:.4f} vs "
            f"KL TX↓ {100*k_tx:.2f}% MAE {k_mae:.3f} recall {k_rec:.4f}. Delta is not VAL-feasible (MAE>9)."
        ),
    )

    mae_dev = _num(kl_test, "mae_mean")
    mae_ho = _num(kl_ho, "mae_mean")
    rec_dev = _num(kl_test, "recall_mean")
    rec_ho = _num(kl_ho, "recall_mean")
    mae_delta = mae_ho - mae_dev
    add(
        "KL-CMAPPO generalizes without large degradation to unseen sensors on the same temporal split.",
        "qualified",
        evidence_file="results/revision/generalization_same_split.csv",
        evidence_metric="mae (development TEST vs heldout TEST)",
        notes=(
            f"Same chronological TEST window. KL MAE {mae_dev:.3f} → {mae_ho:.3f} (Δ={mae_delta:+.3f} ppm); "
            f"union recall {rec_dev:.4f} → {rec_ho:.4f}. MAE does not worsen. Qualification: development TEST "
            f"event recall is already below 0.985, so 'no degradation' is not the same as test feasibility."
        ),
    )
    add(
        "The final actor does not use any expert-derived logic during deployment.",
        "false",
        evidence_file="results/revision/deployment_actor_audit.md",
        evidence_metric="residual_scale * heuristic_logits_torch(obs)",
        notes=(
            "A: SemanticExpertPolicy is not called. B: a frozen expert-derived residual prior is still "
            "evaluated inside ResidualSharedActor.forward. Do not describe the deployed actor as a pure neural policy."
        ),
    )
    add(
        "The contextual relation graph is not a wireless mesh.",
        "true",
        evidence_file="results/revision/context_feature_audit.md",
        evidence_metric="adjacency used only for server-side neighbor_summary(recon)",
        notes="Edges are spatial/statistical contextual relations. No sensor-to-sensor radio exchange.",
    )
    add(
        "Packet-loss communication reduction is computed from attempted transmissions.",
        "true",
        evidence_file="results/revision/packet_loss_raw.csv",
        evidence_metric="attempt_based_reduction = 1 - n_tx_attempts / n_available",
        notes="evaluate_policy counts TRANSMIT decisions, not transmit_history deliveries.",
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
        "- Interpolated matched-budget numbers as if they were direct KL rollouts. Only the VAL tau-sweep rows labelled `direct_measured` are primary; BC remains `interpolated_legacy`. Do not present mixed KL-direct / BC-interpolated matched-budget as primary evidence.",
        "- That the deployed KL actor is a pure neural policy with no expert-derived logic. `SemanticExpertPolicy` is not called, but a frozen expert-informed residual component is still evaluated inside `ResidualSharedActor`.",
        "- That the relation graph is a wireless mesh or that sensors exchange packets with neighbours.",
        "",
        "## Figures",
        "",
        "- `fig_revision_tradeoff_mae.png` — communication reduction vs reconstruction MAE.",
        "- `fig_revision_event_miss.png` — horizontal bars of union event-miss rate (%); 1.5% line.",
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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tol_for(metric: str) -> float:
    if metric in {"all_constraints_satisfied_kn", "mae_ok_kn", "event_ok_kn", "aoi_ok_kn", "result_source", "evaluation_type"}:
        return 0.0
    if metric in {"mae_pass_count", "event_pass_count", "aoi_pass_count", "all_constraints_pass_count", "n_runs"}:
        return 0.0
    if metric in {"n_true_events", "n_true_events_mean", "event_tp_mean", "event_fn_mean", "event_fp_mean"}:
        return 1e-6
    if metric in {"mae", "mae_mean", "aoi", "aoi_raw_mean"}:
        return 1e-6
    return 1e-8


def _values_match(prev, new, tol: float) -> bool:
    if prev is None and new is None:
        return True
    ps, ns = str(prev), str(new)
    if ps in {"", "nan", "None", "N/A"} and ns in {"", "nan", "None", "N/A"}:
        return True
    try:
        a, b = float(prev), float(new)
        if not (np.isfinite(a) and np.isfinite(b)):
            return (not np.isfinite(a)) and (not np.isfinite(b))
        return abs(a - b) <= tol + 1e-15 * max(abs(a), abs(b), 1.0)
    except (TypeError, ValueError):
        return ps == ns


def write_final_reproducibility_check(out_dir: Path, prev_dir: Path | None = None) -> pd.DataFrame:
    """Compare paper-facing summaries to the pre_final_rerun snapshot."""
    prev_dir = prev_dir or (out_dir / "pre_final_rerun")
    rows: list[dict[str, Any]] = []
    if not prev_dir.exists() or not (prev_dir / "all_method_summary.csv").exists():
        pd.DataFrame(rows).to_csv(out_dir / "final_reproducibility_check.csv", index=False)
        return pd.DataFrame(rows)

    prev = pd.read_csv(prev_dir / "all_method_summary.csv")
    cur = pd.read_csv(out_dir / "all_method_summary.csv")
    sel_cfg = selected_delta_cfg(out_dir)

    specs = [
        ("validation", "validation_cohort", "val", 0.0),
        ("frozen_temporal_test", "frozen_temporal_test", "test", 0.0),
        ("heldout_test", "heldout_sensors", "test", 0.0),
    ]
    methods = ["fixed_15", "fixed_30", "fixed_45", "fixed_60", "fixed_75", "fixed_90", "delta_plus_heartbeat", "semantic_expert", "campus_senserl_bc", "cmappo_kl"]
    metrics = [
        "transmission_reduction_mean",
        "mae_mean",
        "recall_mean",
        "precision_mean",
        "f1_mean",
        "aoi_raw_mean",
        "n_true_events_mean",
        "event_tp_mean",
        "event_fn_mean",
        "event_fp_mean",
        "all_constraints_satisfied_kn",
        "mae_pass_count",
        "event_pass_count",
        "aoi_pass_count",
        "all_constraints_pass_count",
        "n_runs",
        "result_source",
    ]

    def pick(df: pd.DataFrame, scenario: str, split: str, method: str, packet_loss: float):
        return _lookup(df, scenario=scenario, split=split, method=method, selected_cfg=sel_cfg if method == "delta_plus_heartbeat" else "")

    for label, scenario, split, ploss in specs:
        for method in methods:
            a = pick(prev, scenario, split, method, ploss)
            b = pick(cur, scenario, split, method, ploss)
            if a is None and b is None:
                continue
            for metric in metrics:
                pv = None if a is None else a.get(metric)
                nv = None if b is None else b.get(metric)
                tol = _tol_for(metric)
                rows.append(
                    {
                        "metric": metric,
                        "scenario": label,
                        "method": method,
                        "previous_value": pv,
                        "rerun_value": nv,
                        "absolute_difference": (
                            abs(float(pv) - float(nv))
                            if _is_num(pv) and _is_num(nv)
                            else (0.0 if str(pv) == str(nv) else float("nan"))
                        ),
                        "tolerance": tol,
                        "match": _values_match(pv, nv, tol),
                    }
                )

    pl_prev = prev[prev["scenario"].astype(str) == "packet_loss"]
    pl_cur = cur[cur["scenario"].astype(str) == "packet_loss"]
    pl_methods = ["fixed_60", "fixed_75", "delta_plus_heartbeat", "semantic_expert", "cmappo_kl"]
    pl_metrics = ["transmission_reduction_mean", "mae_mean", "recall_mean", "n_tx_attempts_mean", "n_delivered_mean", "all_constraints_pass_count", "n_runs"]
    for method in pl_methods:
        for rate in [0.0, 0.1, 0.2, 0.4]:
            def _pl(df):
                hit = df[
                    (df["method"] == method)
                    & np.isclose(pd.to_numeric(df["packet_loss"], errors="coerce").fillna(-1), rate)
                ]
                if method == "delta_plus_heartbeat":
                    hit = hit[hit["configuration"].astype(str) == sel_cfg]
                return None if hit.empty else hit.iloc[0]

            a, b = _pl(pl_prev), _pl(pl_cur)
            if a is None and b is None:
                continue
            for metric in pl_metrics:
                pv = None if a is None else a.get(metric)
                nv = None if b is None else b.get(metric)
                tol = _tol_for(metric)
                rows.append(
                    {
                        "metric": metric,
                        "scenario": f"packet_loss_{int(100 * rate)}",
                        "method": method,
                        "previous_value": pv,
                        "rerun_value": nv,
                        "absolute_difference": (
                            abs(float(pv) - float(nv))
                            if _is_num(pv) and _is_num(nv)
                            else (0.0 if str(pv) == str(nv) else float("nan"))
                        ),
                        "tolerance": tol,
                        "match": _values_match(pv, nv, tol),
                    }
                )

    g_prev_p = prev_dir / "generalization_same_split.csv"
    g_cur_p = out_dir / "generalization_same_split.csv"
    if g_prev_p.exists() and g_cur_p.exists():
        gp, gc = pd.read_csv(g_prev_p), pd.read_csv(g_cur_p)
        for method in ["fixed_60", "fixed_75", "delta_plus_heartbeat", "semantic_expert", "cmappo_kl"]:
            for cohort in ["development", "heldout"]:
                def _g(df):
                    hit = df[(df["method"] == method) & (df["cohort"] == cohort) & (df["split"] == "test")]
                    return None if hit.empty else hit.iloc[0]

                a, b = _g(gp), _g(gc)
                if a is None and b is None:
                    continue
                for metric in ["tx_reduction", "mae", "recall", "precision", "f1", "aoi"]:
                    pv = None if a is None else a.get(metric)
                    nv = None if b is None else b.get(metric)
                    tol = _tol_for(metric)
                    rows.append(
                        {
                            "metric": metric,
                            "scenario": f"generalization_{cohort}_test",
                            "method": method,
                            "previous_value": pv,
                            "rerun_value": nv,
                            "absolute_difference": (
                                abs(float(pv) - float(nv))
                                if _is_num(pv) and _is_num(nv)
                                else (0.0 if str(pv) == str(nv) else float("nan"))
                            ),
                            "tolerance": tol,
                            "match": _values_match(pv, nv, tol),
                        }
                    )

    mb_prev_p = prev_dir / "matched_budget_points.csv"
    mb_cur_p = out_dir / "matched_budget_points.csv"
    if mb_prev_p.exists() and mb_cur_p.exists():
        mp, mc = pd.read_csv(mb_prev_p), pd.read_csv(mb_cur_p)
        for rec in mc.to_dict("records"):
            hit = mp[
                (mp["method"] == rec["method"])
                & np.isclose(pd.to_numeric(mp["target_reduction"], errors="coerce"), float(rec["target_reduction"]))
            ]
            a = None if hit.empty else hit.iloc[0]
            for metric in ["achieved_reduction", "mae", "recall", "evaluation_type", "result_source"]:
                pv = None if a is None else a.get(metric)
                nv = rec.get(metric)
                tol = _tol_for(metric)
                rows.append(
                    {
                        "metric": metric,
                        "scenario": f"matched_budget_{float(rec['target_reduction']):.2f}",
                        "method": rec["method"],
                        "previous_value": pv,
                        "rerun_value": nv,
                        "absolute_difference": (
                            abs(float(pv) - float(nv))
                            if _is_num(pv) and _is_num(nv)
                            else (0.0 if str(pv) == str(nv) else float("nan"))
                        ),
                        "tolerance": tol,
                        "match": _values_match(pv, nv, tol),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "final_reproducibility_check.csv", index=False)
    return df


def write_reproducibility_manifest(out_dir: Path) -> dict[str, Any]:
    root = repo_root()
    env = environment_fingerprint()
    data_cfg = load_yaml(root / "configs" / "data.yaml")
    rl_cfg = load_yaml(root / "configs" / "rl_cmappo.yaml")
    from campus_senserl.data.cohort import load_cohort
    from campus_senserl.rl.expert_policy import SemanticExpertPolicy

    final_ids = load_cohort("final")
    held_ids = load_cohort("heldout")
    expert = SemanticExpertPolicy()
    ckpts = {}
    for seed in SEEDS:
        p = root / "results" / "rl_final" / "cmappo_kl" / f"seed_{seed}" / "best_model.pt"
        ckpts[str(seed)] = {
            "path": str(p.relative_to(root)) if p.exists() else None,
            "sha256": _sha256_file(p) if p.exists() else None,
        }

    csv_names = [
        "all_method_evaluations.csv",
        "all_method_summary.csv",
        "table_validation_main_methods.csv",
        "table_test_all_methods.csv",
        "table_unseen_all_methods.csv",
        "table_matched_budget.csv",
        "table_packet_loss_all_methods.csv",
        "generalization_same_split.csv",
        "matched_budget_points.csv",
        "packet_loss_raw.csv",
        "claim_audit.csv",
        "final_reproducibility_check.csv",
    ]
    result_hashes = {}
    for name in csv_names:
        p = out_dir / name
        result_hashes[name] = _sha256_file(p) if p.exists() else None

    fig_dir = root / "results" / "figures" / "revision_values"
    figure_hashes = {}
    if fig_dir.exists():
        for p in sorted(fig_dir.glob("*.csv")):
            figure_hashes[p.name] = _sha256_file(p)

    delta = {}
    if (out_dir / "delta_selection.json").exists():
        delta = json.loads((out_dir / "delta_selection.json").read_text(encoding="utf-8"))

    manifest = {
        "git_commit": git_commit_hash() or env.get("git_commit"),
        "python_version": env.get("python"),
        "platform": env.get("platform"),
        "packages": env.get("packages"),
        "cuda_available": env.get("cuda_available"),
        "cuda_version": env.get("cuda_version"),
        "dataset": {
            "raw_release": data_cfg.get("paths", {}).get("raw_release"),
            "splits": data_cfg.get("splits"),
            "nominal_interval_sec": data_cfg.get("dataset", {}).get("nominal_interval_sec"),
        },
        "development_sensor_ids_sha256": _sha256_text("\n".join(final_ids)),
        "heldout_sensor_ids_sha256": _sha256_text("\n".join(held_ids)),
        "n_development_sensors": len(final_ids),
        "n_heldout_sensors": len(held_ids),
        "kl_checkpoints": ckpts,
        "semantic_expert_thresholds": {
            "delta_ppm": expert.delta_ppm,
            "aoi_threshold": expert.aoi_threshold,
            "co2_ppm": expert.co2_ppm,
            "disagreement_ppm": expert.disagreement_ppm,
        },
        "event_definition": {
            "headline": "union",
            "high_co2_ppm": float(rl_cfg.get("events", {}).get("primary_threshold_ppm", 1000)),
            "rapid_increase_ppm": float(rl_cfg.get("events", {}).get("rapid_increase_ppm", 150)),
        },
        "constraints": {
            "eps_mae": float(rl_cfg.get("constraints", {}).get("eps_mae", EPS_MAE)),
            "eps_miss": float(rl_cfg.get("constraints", {}).get("eps_miss", EPS_MISS)),
            "eps_aoi": float(rl_cfg.get("constraints", {}).get("eps_aoi", EPS_AOI)),
        },
        "delta_selected_configuration": delta.get("selected_configuration", SELECTED_DELTA_DEFAULT),
        "packet_loss": {
            "mask_seed": PACKET_LOSS_MASK_SEED,
            "n_masks": N_LOSS_MASKS,
            "seed_formula": "PACKET_LOSS_MASK_SEED + loss_mask_id",
            "mode": "independent_slots",
        },
        "sd_convention": {"dispersion": "sample standard deviation", "ddof": 1},
        "matched_budget": {
            "kl": "direct_measured",
            "bc": "interpolated_legacy",
        },
        "result_csv_sha256": result_hashes,
        "figure_source_csv_sha256": figure_hashes,
        "training_performed": False,
    }
    save_json(json_safe(manifest), out_dir / "reproducibility_manifest.json")
    return manifest
