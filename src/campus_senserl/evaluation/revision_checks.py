"""Automated scientific checks for revision evaluation outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from campus_senserl.evaluation.revision_metrics import MASTER_COLUMNS


class RevisionCheckError(RuntimeError):
    pass


def _fail(msg: str) -> None:
    raise RevisionCheckError(msg)


def validate_master(df: pd.DataFrame) -> list[str]:
    """Return warnings; raise RevisionCheckError on hard violations."""
    warnings: list[str] = []
    missing = [c for c in MASTER_COLUMNS if c not in df.columns]
    if missing:
        _fail(f"master CSV missing columns: {missing}")

    red = pd.to_numeric(df["transmission_reduction"], errors="coerce")
    if ((red < -1e-9) | (red > 1 + 1e-9)).any():
        _fail("transmission_reduction outside [0, 1]")
    for col in ["precision", "recall", "f1"]:
        v = pd.to_numeric(df[col], errors="coerce")
        bad = v.notna() & ((v < -1e-9) | (v > 1 + 1e-9))
        if bad.any():
            _fail(f"{col} outside [0, 1]")

    n_tx = pd.to_numeric(df["n_tx_attempts"], errors="coerce")
    n_av = pd.to_numeric(df["n_available"], errors="coerce")
    if ((n_tx.notna() & n_av.notna()) & (n_tx > n_av + 1e-9)).any():
        _fail("n_tx_attempts > n_available")

    n_del = pd.to_numeric(df["n_delivered"], errors="coerce")
    if ((n_del.notna() & n_tx.notna()) & (n_del > n_tx + 1e-9)).any():
        _fail("n_delivered > n_tx_attempts")

    tp = pd.to_numeric(df["event_tp"], errors="coerce")
    fn = pd.to_numeric(df["event_fn"], errors="coerce")
    n_true = pd.to_numeric(df["n_true_events"], errors="coerce")
    mask = tp.notna() & fn.notna() & n_true.notna() & (n_true > 0)
    if mask.any():
        mismatch = mask & ((tp + fn - n_true).abs() > 1e-6)
        if mismatch.any():
            _fail("TP + FN does not equal n_true_events")

    # Natural missingness excluded: reduction uses n_available, not all slots.
    recon = np.abs(n_tx / n_av.replace(0, np.nan) - (1.0 - red))
    if (recon.notna() & (recon > 1e-6)).any():
        _fail("transmission_reduction inconsistent with 1 - n_tx/n_available")

    # Cohort separation
    held = df[df["cohort"].astype(str).str.contains("heldout", case=False, na=False)]
    if held["method"].eq("cmappo_kl").any() and (held["split"] == "train").any():
        _fail("held-out evaluation used train split")

    if df["source"].astype(str).str.contains("test_tuned", na=False).any():
        _fail("test-tuned hyperparameters detected")

    # Deterministic no-loss baselines should be unique
    det = df[
        df["method"].str.startswith("fixed_")
        & (pd.to_numeric(df["packet_loss"], errors="coerce") == 0)
        & (df["aggregation"] == "single_run")
        & (df["source"] == "revision_eval")
    ]
    dups = det.groupby(["scenario", "split", "cohort", "method", "configuration"]).size()
    if (dups > 1).any():
        warnings.append("duplicate deterministic baseline rows found")

    return warnings


def validate_figure_values(master: pd.DataFrame, fig_values: pd.DataFrame) -> None:
    if fig_values.empty:
        _fail("figure_values_revision.csv is empty")
    if "source_row_id" not in fig_values.columns and "value" not in fig_values.columns:
        _fail("figure values missing value column")
    # Every numeric value should be finite or explicitly NA.
    vals = pd.to_numeric(fig_values.get("value"), errors="coerce")
    if vals.notna().sum() == 0:
        _fail("no numeric figure values to trace")
