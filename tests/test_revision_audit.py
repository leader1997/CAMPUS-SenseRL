"""Regression checks on revision CSV provenance (no training)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from campus_senserl.evaluation.revision_export import (
    write_generalization_same_split,
    write_matched_budget_points,
)
from campus_senserl.evaluation.revision_metrics import sample_sd
from campus_senserl.utils import repo_root


REV = repo_root() / "results" / "revision"


@pytest.mark.skipif(not (REV / "all_method_summary.csv").exists(), reason="revision summary missing")
def test_matched_budget_provenance_kl_direct_bc_interpolated():
    summary = pd.read_csv(REV / "all_method_summary.csv")
    pts = write_matched_budget_points(summary, REV)
    kl = pts[pts["method"] == "cmappo_kl"]
    bc = pts[pts["method"] == "campus_senserl_bc"]
    assert len(kl) >= 3
    assert (kl["evaluation_type"] == "direct").all()
    assert (kl["primary_evidence"] == True).all()  # noqa: E712
    if len(bc):
        assert (bc["evaluation_type"] == "interpolated").all()
        assert (bc["primary_evidence"] == False).all()  # noqa: E712
        assert (bc["result_source"].astype(str).str.contains("interpolat|legacy")).all()


@pytest.mark.skipif(not (REV / "all_method_summary.csv").exists(), reason="revision summary missing")
def test_same_split_generalization_uses_test_not_val():
    summary = pd.read_csv(REV / "all_method_summary.csv")
    df = write_generalization_same_split(summary, REV, "delta_ppm=25,heartbeat=3")
    assert set(df["split"].unique()) == {"test"}
    assert "val" not in set(df["split"].unique())
    cohorts = set(df["cohort"].unique())
    assert "development" in cohorts and "heldout" in cohorts
    kl = df[df["method"] == "cmappo_kl"]
    assert set(kl["cohort"]) == {"development", "heldout"}


@pytest.mark.skipif(not (REV / "packet_loss_raw.csv").exists() and not (REV / "all_method_evaluations.csv").exists(), reason="no PL rows")
def test_packet_loss_policy_and_mask_are_separate():
    path = REV / "packet_loss_raw.csv"
    if not path.exists():
        master = pd.read_csv(REV / "all_method_evaluations.csv")
        pl = master[master["scenario"] == "packet_loss"]
    else:
        pl = pd.read_csv(path)
    kl = pl[(pl["method"] == "cmappo_kl") & np.isclose(pd.to_numeric(pl["packet_loss"], errors="coerce"), 0.1)]
    if kl.empty:
        pytest.skip("no KL packet-loss 10% rows")
    n_policy = kl["policy_run"].nunique()
    n_mask = pd.to_numeric(kl["loss_mask_id"], errors="coerce").nunique()
    assert n_policy == 5
    assert n_mask == 5
    assert len(kl) == 25  # 5 policies × 5 masks, not 25 trained policies


def test_sample_sd_ddof_one_is_not_population():
    vals = [10.0, 12.0, 8.0, 11.0, 9.0]
    assert sample_sd(vals) == pytest.approx(float(pd.Series(vals).std(ddof=1)))
    assert sample_sd(vals) != pytest.approx(float(np.std(vals, ddof=0)))
