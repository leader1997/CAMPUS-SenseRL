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
        for col in metric_cols:
            vals = pd.to_numeric(g[col], errors="coerce")
            rec[f"{col}_mean"] = float(vals.mean()) if vals.notna().any() else float("nan")
            rec[f"{col}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 0.0
        # Constraint: True only if every run True; False if any False; else N/A
        for ccol in ["mae_ok", "event_ok", "aoi_ok", "all_constraints_satisfied"]:
            flags = g[ccol].astype(str).tolist()
            if any(x == "False" for x in flags):
                rec[ccol] = "False"
            elif all(x == "True" for x in flags):
                rec[ccol] = "True"
            else:
                rec[ccol] = NA
        rows.append(rec)
    return pd.DataFrame(rows)
