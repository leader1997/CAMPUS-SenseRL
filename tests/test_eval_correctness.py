"""Regression tests for evaluation correctness (TX rate, graph fail-fast)."""

from __future__ import annotations

import numpy as np
import pytest

from campus_senserl.environment.communication_model import SKIP, TRANSMIT
from campus_senserl.environment.trace_environment import TraceDrivenCampusEnv, build_synthetic_trace
from campus_senserl.evaluation.rl_policy_eval import evaluate_policy
from campus_senserl.rl.fixed_policies import FixedIntervalPolicy


def test_fixed_interval_1_has_zero_reduction(minimal_rl_cfg):
    """Every available slot TX → reduction over local availability is ~0%."""
    trace = build_synthetic_trace(n_steps=20, n_sensors=3, seed=0)
    cfg = dict(minimal_rl_cfg)
    cfg.setdefault("environment", {})["graph"] = "identity"
    cfg.setdefault("environment", {})["graph_fail_fast"] = False
    cfg.setdefault("safety_shield", {})["enabled"] = False
    env = TraceDrivenCampusEnv(cfg=cfg, trace=trace, multi_agent=True)
    pol = FixedIntervalPolicy(interval_steps=1)

    def act(obs, *, local_available=None):
        return pol.act(obs, local_available=local_available)

    m = evaluate_policy(env, act, max_steps=15, seed=0)
    assert m["transmission_reduction_pct"] == pytest.approx(0.0, abs=1e-6)
    assert m["transmit_rate"] == pytest.approx(1.0, abs=1e-6)


def test_hybrid_graph_fail_fast_without_cohort(minimal_rl_cfg):
    """Unknown synthetic IDs + hybrid + fail_fast must raise (no silent identity)."""
    trace = build_synthetic_trace(n_steps=8, n_sensors=3, seed=1)
    cfg = dict(minimal_rl_cfg)
    cfg["environment"] = {
        **cfg.get("environment", {}),
        "graph": "hybrid",
        "graph_fail_fast": True,
        "reconstructor": "locf",
    }
    with pytest.raises(RuntimeError, match="hybrid"):
        TraceDrivenCampusEnv(cfg=cfg, trace=trace, multi_agent=True)


def test_packet_loss_keeps_local_available(minimal_rl_cfg):
    """Packet loss drops delivery but measurement still counted as locally available."""
    trace = build_synthetic_trace(n_steps=10, n_sensors=2, seed=2)
    # Force all slots available
    trace["natural_missing"][:] = False
    trace["local_available"] = np.isfinite(trace["ground_truth"])
    cfg = dict(minimal_rl_cfg)
    cfg.setdefault("environment", {})["packet_loss_rate"] = 1.0
    cfg.setdefault("environment", {})["graph"] = "identity"
    cfg.setdefault("safety_shield", {})["enabled"] = False
    env = TraceDrivenCampusEnv(cfg=cfg, trace=trace, multi_agent=True)
    obs, _ = env.reset()
    actions = np.full(env.n_sensors, TRANSMIT, dtype=int)
    obs, r, term, trunc, info = env.step(actions)
    assert info["tx_requested_after_shield"] == env.n_sensors
    assert info["transmit_count"] == 0
    assert info["tx_dropped"] == env.n_sensors
    # Local availability for that step remains True in info
    assert bool(np.all(info["local_available"]))
