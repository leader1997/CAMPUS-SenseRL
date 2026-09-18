"""Standardized revision-evaluation metrics (no training, no retuning).

Transmission reduction is always:
    1 - N_tx_attempts / N_locally_available
Natural missingness is excluded from the denominator because those slots
are never locally available.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

EPS_MAE = 9.0
EPS_MISS = 0.015  # event miss rate; recall >= 0.985
EPS_AOI = 3.5
RECALL_MIN = 1.0 - EPS_MISS

MASTER_COLUMNS = [
    "scenario",
    "split",
    "cohort",
    "method",
    "configuration",
    "run",
    "checkpoint",
    "packet_loss",
    "target_reduction",
    "n_available",
    "n_tx_attempts",
    "n_delivered",
    "transmission_rate",
    "transmission_reduction",
    "mae",
    "event_tp",
    "event_fp",
    "event_fn",
    "precision",
    "recall",
    "f1",
    "aoi_raw",
    "mae_ok",
    "event_ok",
    "aoi_ok",
    "all_constraints_satisfied",
    "n_true_events",
    "aggregation",
    "source",
    "packet_loss_mode",
    "notes",
    "policy_run",
    "loss_mask_id",
    "result_source",
    "prob_threshold",
]

DISPLAY_NAMES = {
    "fixed_15": "Fixed-15",
    "fixed_30": "Fixed-30",
    "fixed_45": "Fixed-45",
    "fixed_60": "Fixed-60",
    "fixed_75": "Fixed-75",
    "fixed_90": "Fixed-90",
    "delta_plus_heartbeat": "Delta + heartbeat",
    "semantic_expert": "Semantic expert",
    "campus_senserl_bc": "BC initialization",
    "cmappo_kl": "CAMPUS-SenseRL (KL-CMAPPO)",
}

PAPER_METHOD_ORDER = [
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

NA = "N/A"


def _as_number(val, kind: str = "float"):
    if val is None:
        return float("nan")
    try:
        x = float(val)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(x):
        return float("nan")
    return int(x) if kind == "int" else float(x)


def _finite(x: Any) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def _flag(ok: bool | None) -> str:
    if ok is None:
        return NA
    return "True" if ok else "False"


def constraint_flags(
    mae: float,
    recall: float,
    aoi: float,
) -> tuple[str, str, str, str]:
    mae_ok = None if not _finite(mae) else bool(float(mae) <= EPS_MAE + 1e-12)
    event_ok = None if not _finite(recall) else bool(float(recall) >= RECALL_MIN - 1e-12)
    aoi_ok = None if not _finite(aoi) else bool(float(aoi) <= EPS_AOI + 1e-12)
    if mae_ok is None or event_ok is None or aoi_ok is None:
        all_ok = None
    else:
        all_ok = bool(mae_ok and event_ok and aoi_ok)
    return _flag(mae_ok), _flag(event_ok), _flag(aoi_ok), _flag(all_ok)


def standardize_eval(
    raw: dict[str, Any],
    *,
    scenario: str,
    split: str,
    cohort: str,
    method: str,
    configuration: str,
    run: str,
    checkpoint: str = "",
    packet_loss: float = 0.0,
    target_reduction: float | None = None,
    aggregation: str = "single_run",
    source: str = "revision_eval",
    packet_loss_mode: str = "none",
    notes: str = "",
    policy_run: str = "",
    loss_mask_id: float | None = None,
    result_source: str = "",
    prob_threshold: float | None = None,
) -> dict[str, Any]:
    n_available = _as_number(raw.get("n_locally_available", raw.get("n_available")), "int")
    n_tx = _as_number(raw.get("n_tx", raw.get("n_tx_attempts", raw.get("tx_requested"))), "int")
    n_delivered = _as_number(raw.get("tx_delivered", raw.get("n_delivered")), "int")
    if np.isfinite(n_available) and n_available > 0 and np.isfinite(n_tx):
        tx_rate = float(n_tx) / float(n_available)
        reduction = 1.0 - tx_rate
    else:
        tx_rate = float("nan")
        reduction = float("nan")
    mae = float(raw.get("mae_skipped", raw.get("mae", float("nan"))))
    precision = float(raw.get("event_precision", raw.get("precision", float("nan"))))
    recall = float(raw.get("event_recall", raw.get("recall", float("nan"))))
    f1 = float(raw.get("event_f1", raw.get("f1", float("nan"))))
    aoi = float(raw.get("mean_aoi_raw", raw.get("aoi_raw", float("nan"))))
    mae_ok, event_ok, aoi_ok, all_ok = constraint_flags(mae, recall, aoi)
    return {
        "scenario": scenario,
        "split": split,
        "cohort": cohort,
        "method": method,
        "configuration": configuration,
        "run": str(run),
        "checkpoint": checkpoint,
        "packet_loss": float(packet_loss),
        "target_reduction": float("nan") if target_reduction is None else float(target_reduction),
        "n_available": n_available,
        "n_tx_attempts": n_tx,
        "n_delivered": n_delivered,
        "transmission_rate": float(tx_rate),
        "transmission_reduction": float(reduction),
        "mae": mae,
        "event_tp": _as_number(raw.get("tp", raw.get("event_tp")), "int"),
        "event_fp": _as_number(raw.get("fp", raw.get("event_fp")), "int"),
        "event_fn": _as_number(raw.get("fn", raw.get("event_fn")), "int"),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "aoi_raw": aoi,
        "mae_ok": mae_ok,
        "event_ok": event_ok,
        "aoi_ok": aoi_ok,
        "all_constraints_satisfied": all_ok,
        "n_true_events": _as_number(raw.get("n_true_events"), "int"),
        "aggregation": aggregation,
        "source": source,
        "packet_loss_mode": packet_loss_mode,
        "notes": notes,
        "policy_run": str(policy_run or run),
        "loss_mask_id": float("nan") if loss_mask_id is None else float(loss_mask_id),
        "result_source": result_source or (
            "legacy_frozen" if str(source).startswith("legacy") else "new_evaluation"
        ),
        "prob_threshold": float("nan") if prob_threshold is None else float(prob_threshold),
    }


def row_key(row: dict[str, Any]) -> tuple:
    def _num(x) -> str:
        try:
            v = float(x)
        except (TypeError, ValueError):
            return ""
        if not np.isfinite(v):
            return ""
        return f"{v:.6f}"

    return (
        str(row.get("scenario", "")),
        str(row.get("split", "")),
        str(row.get("cohort", "")),
        str(row.get("method", "")),
        str(row.get("configuration", "")),
        str(row.get("run", "")),
        _num(row.get("packet_loss", 0.0)),
        _num(row.get("target_reduction", "")),
        _num(row.get("loss_mask_id", "")),
    )


def pareto_mask(df: pd.DataFrame) -> np.ndarray:
    """Non-dominated points: max reduction, min MAE, max recall."""
    red = df["transmission_reduction"].to_numpy(dtype=float)
    mae = df["mae"].to_numpy(dtype=float)
    rec = df["recall"].to_numpy(dtype=float)
    n = len(df)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not (np.isfinite(red[i]) and np.isfinite(mae[i]) and np.isfinite(rec[i])):
            keep[i] = False
            continue
        for j in range(n):
            if i == j:
                continue
            if not (np.isfinite(red[j]) and np.isfinite(mae[j]) and np.isfinite(rec[j])):
                continue
            ge = (red[j] >= red[i] - 1e-12) and (mae[j] <= mae[i] + 1e-12) and (rec[j] >= rec[i] - 1e-12)
            gt = (red[j] > red[i] + 1e-12) or (mae[j] < mae[i] - 1e-12) or (rec[j] > rec[i] + 1e-12)
            if ge and gt:
                keep[i] = False
                break
    return keep


def violation_score(mae: float, recall: float, aoi: float) -> float:
    v = 0.0
    if np.isfinite(mae):
        v += max(0.0, (mae - EPS_MAE) / EPS_MAE)
    else:
        v += 10.0
    if np.isfinite(recall):
        v += max(0.0, (RECALL_MIN - recall) / EPS_MISS)
    else:
        v += 10.0
    if np.isfinite(aoi):
        v += max(0.0, (aoi - EPS_AOI) / EPS_AOI)
    else:
        v += 10.0
    return float(v)


def select_delta_heartbeat(val_df: pd.DataFrame) -> dict[str, Any]:
    """Validation-only selection. Never use test/held-out rows."""
    sub = val_df[val_df["method"] == "delta_plus_heartbeat"].copy()
    if sub.empty:
        raise ValueError("No Delta+heartbeat validation rows")
    sub["pareto"] = pareto_mask(sub)
    sub["viol"] = [
        violation_score(r.mae, r.recall, r.aoi_raw) for r in sub.itertuples(index=False)
    ]
    feasible = sub[sub["all_constraints_satisfied"] == "True"]
    criterion = (
        "Among VAL Delta+heartbeat grid points: (1) if any configuration satisfies "
        "MAE<=9, recall>=0.985, AoI<=3.5, choose the feasible point with highest "
        "transmission reduction (ties: lower MAE, then higher recall). "
        "(2) else choose the Pareto point with smallest constraint-violation score, "
        "then highest transmission reduction."
    )
    if len(feasible):
        feasible = feasible.sort_values(
            ["transmission_reduction", "mae", "recall"],
            ascending=[False, True, False],
        )
        chosen = feasible.iloc[0]
        reason = "highest TX reduction among constraint-feasible VAL configurations"
    else:
        cand = sub[sub["pareto"]].copy() if sub["pareto"].any() else sub
        cand = cand.sort_values(
            ["viol", "transmission_reduction", "mae"],
            ascending=[True, False, True],
        )
        chosen = cand.iloc[0]
        reason = "no VAL config met all constraints; selected min violation on Pareto set"
    return {
        "criterion": criterion,
        "reason": reason,
        "selected_configuration": str(chosen["configuration"]),
        "selected_row": chosen.to_dict(),
        "n_grid": int(len(sub)),
        "n_feasible": int(len(feasible)) if len(feasible) else 0,
        "n_pareto": int(sub["pareto"].sum()),
        "pareto_configurations": sub.loc[sub["pareto"], "configuration"].tolist(),
    }


def aggregate_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Mean±SD over runs; deterministic methods stay single-run."""
    metric_cols = [
        "n_available",
        "n_tx_attempts",
        "n_delivered",
        "transmission_rate",
        "transmission_reduction",
        "mae",
        "event_tp",
        "event_fp",
        "event_fn",
        "precision",
        "recall",
        "f1",
        "aoi_raw",
        "n_true_events",
    ]
    keys = ["scenario", "split", "cohort", "method", "configuration", "packet_loss", "target_reduction"]
    rows = []
    for key, g in df.groupby(keys, dropna=False):
        rec = {k: v for k, v in zip(keys, key)}
        rec["n_runs"] = int(len(g))
        rec["aggregation"] = "mean_of_n" if len(g) > 1 else "single_run"
        rec["runs"] = ",".join(sorted(g["run"].astype(str)))
        rec["source"] = ",".join(sorted(set(g["source"].astype(str))))
        if "result_source" in g.columns:
            rec["result_source"] = ",".join(sorted(set(g["result_source"].fillna("").astype(str))))
        else:
            rec["result_source"] = rec["source"]
        for col in metric_cols:
            vals = pd.to_numeric(g[col], errors="coerce")
            rec[f"{col}_mean"] = float(vals.mean()) if vals.notna().any() else float("nan")
            rec[f"{col}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 0.0
        # Constraint: True only if every run True; False if any False; else N/A
        # Also report k/n pass counts (do not judge feasibility from the mean alone).
        n = int(len(g))
        for ccol, pcol in [
            ("mae_ok", "mae_pass_count"),
            ("event_ok", "event_pass_count"),
            ("aoi_ok", "aoi_pass_count"),
            ("all_constraints_satisfied", "all_constraints_pass_count"),
        ]:
            flags = g[ccol].astype(str).tolist() if ccol in g.columns else ["N/A"] * n
            n_true = int(sum(x == "True" for x in flags))
            n_false = int(sum(x == "False" for x in flags))
            n_na = n - n_true - n_false
            rec[pcol] = n_true
            rec[f"{pcol}_n"] = n
            rec[f"{ccol}_kn"] = "N/A" if n_na == n else f"{n_true}/{n}"
            if n_false:
                rec[ccol] = "False"
            elif n_true == n:
                rec[ccol] = "True"
            else:
                rec[ccol] = NA
            if ccol == "all_constraints_satisfied" and n_na == n:
                rec["constraint_display"] = "N/A (full-transmission reference)"
            elif ccol == "all_constraints_satisfied":
                rec["constraint_display"] = rec[f"{ccol}_kn"]
        rows.append(rec)
    return pd.DataFrame(rows)


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if obj is None or (isinstance(obj, str) and obj.lower() == "nan"):
        return None
    return obj


def parse_delta_config(configuration: str) -> tuple[float, float]:
    cfg = str(configuration)
    d = float(cfg.split("delta_ppm=")[1].split(",")[0])
    h = float(cfg.split("heartbeat=")[1])
    return d, h


def delta_grid_table(val_df: pd.DataFrame) -> pd.DataFrame:
    sub = val_df[val_df["method"] == "delta_plus_heartbeat"].copy()
    if sub.empty:
        return sub
    sub["pareto_efficient"] = pareto_mask(sub)
    rows = []
    for r in sub.itertuples(index=False):
        d, h = parse_delta_config(r.configuration)
        mae = float(r.mae)
        rec = float(r.recall)
        aoi = float(r.aoi_raw)
        miss = 1.0 - rec if np.isfinite(rec) else float("nan")
        rows.append(
            {
                "delta_threshold": d,
                "heartbeat": h,
                "configuration": r.configuration,
                "tx_reduction": float(r.transmission_reduction),
                "mae": mae,
                "recall": rec,
                "precision": float(r.precision),
                "f1": float(r.f1),
                "aoi": aoi,
                "mae_violation": max(0.0, mae / EPS_MAE - 1.0) if np.isfinite(mae) else float("nan"),
                "event_violation": max(0.0, miss / EPS_MISS - 1.0) if np.isfinite(miss) else float("nan"),
                "aoi_violation": max(0.0, aoi / EPS_AOI - 1.0) if np.isfinite(aoi) else float("nan"),
                "total_violation": violation_score(mae, rec, aoi),
                "pareto_efficient": bool(r.pareto_efficient),
                "all_constraints_satisfied": r.all_constraints_satisfied,
            }
        )
    return pd.DataFrame(rows)


def sample_sd(values) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if len(arr) <= 1:
        return 0.0
    return float(arr.std(ddof=1))


def pad_master_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    str_cols = {"policy_run", "result_source", "notes", "checkpoint", "source", "packet_loss_mode", "configuration", "run"}
    for col in MASTER_COLUMNS:
        if col not in out.columns:
            out[col] = pd.Series([pd.NA] * len(out), dtype="object") if col in str_cols else np.nan
        elif col in str_cols:
            out[col] = out[col].astype("object")
    return out.reindex(columns=list(dict.fromkeys([*MASTER_COLUMNS, *out.columns])))


def _missing_str(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    return series.isna() | s.isin(["", "nan", "None", "<NA>"])


def backfill_master(df: pd.DataFrame) -> pd.DataFrame:
    """Fill provenance columns on rows written before the second revision pass."""
    out = pad_master_columns(df)
    src = out["source"].astype(str)

    miss_rs = _missing_str(out["result_source"])
    out.loc[miss_rs, "result_source"] = np.where(
        src.loc[miss_rs].str.startswith("legacy"),
        "legacy_frozen",
        "new_evaluation",
    )
    cfg = out["configuration"].astype(str)
    direct_mb = out["scenario"].astype(str).eq("matched_budget") & cfg.str.contains("nearest to", case=False, na=False)
    interp_mb = out["scenario"].astype(str).eq("matched_budget") & cfg.str.contains("interpolated_target", case=False, na=False)
    out.loc[direct_mb, "result_source"] = "direct_measured"
    out.loc[interp_mb, "result_source"] = "interpolated_legacy"
    sweep = out["scenario"].astype(str).eq("matched_budget_tau_sweep")
    out.loc[sweep, "result_source"] = "direct_measured"

    miss_pr = _missing_str(out["policy_run"])
    out.loc[miss_pr, "policy_run"] = out.loc[miss_pr, "run"].astype(str)

    pl = out["scenario"].astype(str).eq("packet_loss")
    mask_num = pd.to_numeric(out["loss_mask_id"], errors="coerce")
    miss_mask = pl & (~np.isfinite(mask_num) | _missing_str(out["loss_mask_id"]))
    out.loc[miss_mask, "loss_mask_id"] = 0.0
    return drop_key_duplicates(out)


def drop_key_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    seen: set[tuple] = set()
    keep: list[int] = []
    for i, rec in enumerate(df.to_dict("records")):
        k = row_key(rec)
        if k in seen:
            continue
        seen.add(k)
        keep.append(i)
    return df.iloc[keep].reset_index(drop=True)


def compact_method_rows(
    agg: pd.DataFrame,
    *,
    selected_delta_cfg: str | None = None,
    methods: list[str] | None = None,
) -> pd.DataFrame:
    """One row per method; VAL-selected Delta only."""
    methods = methods or PAPER_METHOD_ORDER
    rows = []
    for method in methods:
        hit = agg[agg["method"] == method]
        if hit.empty:
            continue
        if method == "delta_plus_heartbeat" and selected_delta_cfg:
            sel = hit[hit["configuration"].astype(str) == str(selected_delta_cfg)]
            if len(sel):
                hit = sel
        if method == "cmappo_kl" and len(hit) > 1:
            native = hit[hit["configuration"].astype(str).str.contains(r"tau=0\.5", regex=True)]
            if len(native):
                hit = native
        rows.append(hit.iloc[0].to_dict())
    return pd.DataFrame(rows)


def aggregate_packet_loss(df: pd.DataFrame) -> pd.DataFrame:
    """Shared-mask hierarchy: mean over masks, then sample SD across policies.

    Deterministic methods have one policy: SD is across the shared loss-mask
    replicates. KL-CMAPPO has five frozen policies: each policy is first
    averaged over masks, then mean±SD is across those five policy means.
    """
    pl = df[df["scenario"].astype(str) == "packet_loss"].copy()
    if pl.empty:
        return pl
    pl = backfill_master(pl)
    # Do not mix legacy per-attempt BC into the shared-mask hierarchy.
    legacy_bc = (
        pl["method"].astype(str).eq("campus_senserl_bc")
        & pl["result_source"].astype(str).str.contains("legacy|interpolat", case=False, regex=True)
    )
    pl = pl.loc[~legacy_bc].copy()
    if pl.empty:
        return pl

    pl["policy_run"] = pl["policy_run"].fillna(pl["run"]).astype(str)
    pl["loss_mask_id"] = pd.to_numeric(pl["loss_mask_id"], errors="coerce").fillna(0.0)
    metric_cols = [
        "n_available",
        "n_tx_attempts",
        "n_delivered",
        "transmission_rate",
        "transmission_reduction",
        "mae",
        "event_tp",
        "event_fp",
        "event_fn",
        "precision",
        "recall",
        "f1",
        "aoi_raw",
        "n_true_events",
    ]
    keys = ["scenario", "split", "cohort", "method", "configuration", "packet_loss"]
    rows = []
    for key, g in pl.groupby(keys, dropna=False):
        rec = {k: v for k, v in zip(keys, key)}
        rec["target_reduction"] = float("nan")
        n_pol = int(g["policy_run"].nunique())
        n_mask = int(g["loss_mask_id"].nunique())
        rec["n_policies"] = n_pol
        rec["n_loss_masks"] = n_mask
        rec["n_eval_rows"] = int(len(g))
        rec["source"] = ",".join(sorted(set(g["source"].astype(str))))
        rec["result_source"] = ",".join(sorted(set(g["result_source"].fillna("").astype(str))))
        rec["runs"] = ",".join(sorted(set(g["policy_run"].astype(str))))

        if n_pol > 1:
            per_pol = []
            for _, gp in g.groupby("policy_run"):
                item = {col: float(pd.to_numeric(gp[col], errors="coerce").mean()) for col in metric_cols}
                mae_ok, event_ok, aoi_ok, all_ok = constraint_flags(
                    item["mae"], item["recall"], item["aoi_raw"]
                )
                item["mae_ok"] = mae_ok
                item["event_ok"] = event_ok
                item["aoi_ok"] = aoi_ok
                item["all_constraints_satisfied"] = all_ok
                per_pol.append(item)
            work = pd.DataFrame(per_pol)
            rec["n_runs"] = n_pol
            rec["aggregation"] = "mean_over_masks_then_sd_across_policies"
        else:
            work = g.copy()
            rec["n_runs"] = max(n_mask, int(len(g)))
            rec["aggregation"] = "mean_sd_across_shared_loss_masks"

        n = int(len(work))
        rec["aggregation"] = rec["aggregation"] if n > 1 else "single_run"
        for col in metric_cols:
            vals = pd.to_numeric(work[col], errors="coerce")
            rec[f"{col}_mean"] = float(vals.mean()) if vals.notna().any() else float("nan")
            rec[f"{col}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 0.0
        for ccol, pcol in [
            ("mae_ok", "mae_pass_count"),
            ("event_ok", "event_pass_count"),
            ("aoi_ok", "aoi_pass_count"),
            ("all_constraints_satisfied", "all_constraints_pass_count"),
        ]:
            flags = work[ccol].astype(str).tolist() if ccol in work.columns else ["N/A"] * n
            n_true = int(sum(x == "True" for x in flags))
            n_false = int(sum(x == "False" for x in flags))
            n_na = n - n_true - n_false
            rec[pcol] = n_true
            rec[f"{pcol}_n"] = n
            rec[f"{ccol}_kn"] = "N/A" if n_na == n else f"{n_true}/{n}"
            if n_false:
                rec[ccol] = "False"
            elif n_true == n:
                rec[ccol] = "True"
            else:
                rec[ccol] = NA
            if ccol == "all_constraints_satisfied" and n_na == n:
                rec["constraint_display"] = "N/A (full-transmission reference)"
            elif ccol == "all_constraints_satisfied":
                rec["constraint_display"] = rec[f"{ccol}_kn"]
        rows.append(rec)
    return pd.DataFrame(rows)
