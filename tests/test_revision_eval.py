"""Checks for revision evaluation: common packet-loss masks, TX accounting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from campus_senserl.environment.communication_model import TRANSMIT
from campus_senserl.environment.trace_environment import TraceDrivenCampusEnv, build_synthetic_trace
from campus_senserl.evaluation.revision_checks import RevisionCheckError, validate_master
from campus_senserl.evaluation.revision_metrics import constraint_flags, standardize_eval
from campus_senserl.evaluation.rl_policy_eval import evaluate_policy
from campus_senserl.rl.fixed_policies import FixedIntervalPolicy


def test_independent_slot_mask_is_shared(minimal_rl_cfg):
    trace = build_synthetic_trace(n_steps=16, n_sensors=3, seed=0)
    trace["natural_missing"][:] = False
    trace["local_available"] = np.isfinite(trace["ground_truth"])
    cfg = dict(minimal_rl_cfg)
    cfg.setdefault("environment", {})
    cfg["environment"]["graph"] = "identity"
    cfg["environment"]["packet_loss_rate"] = 0.5
    cfg["environment"]["packet_loss_mode"] = "independent_slots"
    cfg["environment"]["packet_loss_mask_seed"] = 99

    env1 = TraceDrivenCampusEnv(cfg=cfg, trace=trace, multi_agent=True)
    env2 = TraceDrivenCampusEnv(cfg=cfg, trace=trace, multi_agent=True)
    assert env1.packet_loss_mask is not None
    np.testing.assert_array_equal(env1.packet_loss_mask, env2.packet_loss_mask)

    obs, _ = env1.reset()
    actions = np.full(env1.n_sensors, TRANSMIT, dtype=int)
    _, _, _, _, info = env1.step(actions)
    expected_drop = int(env1.packet_loss_mask[0].sum())
    assert info["tx_requested"] == env1.n_sensors
    assert info["tx_dropped"] == expected_drop
    assert info["transmit_count"] == env1.n_sensors - expected_drop


def test_tx_reduction_excludes_natural_missing(minimal_rl_cfg):
    trace = build_synthetic_trace(n_steps=20, n_sensors=3, seed=1)
    cfg = dict(minimal_rl_cfg)
    cfg.setdefault("environment", {})["graph"] = "identity"
    env = TraceDrivenCampusEnv(cfg=cfg, trace=trace, multi_agent=True)
    pol = FixedIntervalPolicy(interval_steps=2)

    def act(obs, *, local_available=None):
        return pol.act(obs, local_available=local_available)

    m = evaluate_policy(env, act, max_steps=12, seed=0)
    assert 0.0 <= m["transmission_reduction_pct"] / 100.0 <= 1.0
    assert m["n_tx"] <= m["n_locally_available"]
    row = standardize_eval(
        m,
        scenario="validation_cohort",
        split="val",
        cohort="final",
        method="fixed_30",
        configuration="interval_steps=2",
        run="deterministic",
    )
    assert abs(row["transmission_reduction"] - (1.0 - row["n_tx_attempts"] / row["n_available"])) < 1e-9


def test_fixed15_mae_is_undefined_without_skips(minimal_rl_cfg):
    trace = build_synthetic_trace(n_steps=12, n_sensors=2, seed=2)
    trace["natural_missing"][:] = False
    trace["local_available"] = np.ones_like(trace["ground_truth"], dtype=bool)
    cfg = dict(minimal_rl_cfg)
    cfg.setdefault("environment", {})["graph"] = "identity"
    env = TraceDrivenCampusEnv(cfg=cfg, trace=trace, multi_agent=True)
    pol = FixedIntervalPolicy(interval_steps=1)

    def act(obs, *, local_available=None):
        return pol.act(obs, local_available=local_available)

    m = evaluate_policy(env, act, max_steps=8, seed=0)
    assert m["n_skipped_eval"] == 0
    assert not np.isfinite(m["mae_skipped"])
    flags = constraint_flags(m["mae_skipped"], m["event_recall"], m["mean_aoi_raw"])
    assert flags[0] == "N/A"
    assert flags[3] == "N/A"


def test_validate_master_rejects_bad_reduction():
    df = pd.DataFrame(
        [
            standardize_eval(
                {
                    "n_locally_available": 100,
                    "n_tx": 40,
                    "tx_delivered": 40,
                    "mae_skipped": 8.0,
                    "event_precision": 0.9,
                    "event_recall": 0.99,
                    "event_f1": 0.94,
                    "mean_aoi_raw": 2.0,
                    "tp": 99,
                    "fp": 1,
                    "fn": 1,
                    "n_true_events": 100,
                },
                scenario="validation_cohort",
                split="val",
                cohort="final",
                method="fixed_60",
                configuration="x",
                run="deterministic",
            )
        ]
    )
    df.loc[0, "transmission_reduction"] = 1.5
    with pytest.raises(RevisionCheckError):
        validate_master(df)
