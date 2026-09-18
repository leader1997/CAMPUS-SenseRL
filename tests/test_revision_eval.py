"""Checks for revision evaluation: common packet-loss masks, TX accounting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from campus_senserl.environment.communication_model import SKIP, TRANSMIT
from campus_senserl.environment.trace_environment import TraceDrivenCampusEnv, build_synthetic_trace
from campus_senserl.evaluation.revision_checks import RevisionCheckError, validate_master
from campus_senserl.evaluation.revision_metrics import constraint_flags, sample_sd, standardize_eval
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


def test_fixed_interval_clock_ignores_patterned_missingness(minimal_rl_cfg):
    """Periodic TX is time-grid aligned: missingness does not shift later slots."""
    n_steps, n_sensors, k = 16, 2, 3
    trace = build_synthetic_trace(n_steps=n_steps, n_sensors=n_sensors, seed=3)
    trace["natural_missing"][:] = False
    trace["local_available"] = np.ones((n_steps, n_sensors), dtype=bool)
    # Knock out a scheduled TX slot (t=3) and an off-grid slot (t=4) on sensor 0.
    trace["local_available"][3, 0] = False
    trace["local_available"][4, 0] = False
    trace["natural_missing"][3, 0] = True
    cfg = dict(minimal_rl_cfg)
    cfg.setdefault("environment", {})["graph"] = "identity"
    env = TraceDrivenCampusEnv(cfg=cfg, trace=trace, multi_agent=True)
    pol = FixedIntervalPolicy(interval_steps=k)
    obs, _ = env.reset(seed=0)
    pol.reset()
    scheduled = []
    attempted = []
    for t in range(n_steps - 1):
        local = env.local_available[t]
        actions = pol.act(obs, local_available=local)
        scheduled.append(t % k == 0)
        attempted.append(actions.copy())
        obs, _, terminated, truncated, _ = env.step(actions)
        if terminated or truncated:
            break
    attempted = np.asarray(attempted)
    for t, was_scheduled in enumerate(scheduled):
        if not was_scheduled:
            assert np.all(attempted[t] == SKIP)
        else:
            for i in range(n_sensors):
                if trace["local_available"][t, i]:
                    assert attempted[t, i] == TRANSMIT
                else:
                    assert attempted[t, i] == SKIP
    # t=3 was a scheduled slot but missing on sensor 0; t=6 is still a TX slot.
    assert 3 % k == 0 and 6 % k == 0
    assert attempted[3, 0] == SKIP
    assert attempted[6, 0] == TRANSMIT
    assert attempted[4, 0] == SKIP  # off-grid missingness does not create a catch-up TX


def test_tx_denominator_is_locally_available_slots_only(minimal_rl_cfg):
    n_steps, n_sensors = 12, 3
    trace = build_synthetic_trace(n_steps=n_steps, n_sensors=n_sensors, seed=4)
    trace["natural_missing"][:] = False
    trace["local_available"] = np.ones((n_steps, n_sensors), dtype=bool)
    trace["local_available"][1, :] = False
    trace["local_available"][5, 0] = False
    cfg = dict(minimal_rl_cfg)
    cfg.setdefault("environment", {})["graph"] = "identity"
    env = TraceDrivenCampusEnv(cfg=cfg, trace=trace, multi_agent=True)
    pol = FixedIntervalPolicy(interval_steps=2)

    def act(obs, *, local_available=None):
        return pol.act(obs, local_available=local_available)

    max_steps = 10
    m = evaluate_policy(env, act, max_steps=max_steps, seed=0)
    expected_available = int(trace["local_available"][:max_steps].sum())
    assert m["n_locally_available"] == expected_available
    assert m["n_tx"] <= expected_available
    assert m["transmission_reduction_pct"] == pytest.approx(
        (1.0 - m["n_tx"] / m["n_locally_available"]) * 100.0, abs=1e-9
    )


def test_sample_sd_uses_ddof_one():
    assert sample_sd([1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert sample_sd([1.0, 2.0, 3.0]) != float(np.std([1.0, 2.0, 3.0], ddof=0))


def test_fixed60_packet_loss_does_not_change_tx_attempts(minimal_rl_cfg):
    """Deterministic Fixed-60: loss changes deliveries/MAE, not N_tx_attempts."""
    n_steps, n_sensors = 20, 3
    trace = build_synthetic_trace(n_steps=n_steps, n_sensors=n_sensors, seed=7)
    trace["natural_missing"][:] = False
    trace["local_available"] = np.ones((n_steps, n_sensors), dtype=bool)
    cfg0 = dict(minimal_rl_cfg)
    cfg0.setdefault("environment", {})
    cfg0["environment"] = dict(cfg0["environment"])
    cfg0["environment"]["graph"] = "identity"
    cfg0["environment"]["packet_loss_rate"] = 0.0
    cfg_l = dict(cfg0)
    cfg_l["environment"] = dict(cfg0["environment"])
    cfg_l["environment"]["packet_loss_rate"] = 0.4
    cfg_l["environment"]["packet_loss_mode"] = "independent_slots"
    cfg_l["environment"]["packet_loss_mask_seed"] = 11

    def run(cfg):
        env = TraceDrivenCampusEnv(cfg=cfg, trace=trace, multi_agent=True)
        pol = FixedIntervalPolicy(interval_steps=4)

        def act(obs, *, local_available=None):
            return pol.act(obs, local_available=local_available)

        return evaluate_policy(env, act, max_steps=16, seed=0)

    m0 = run(cfg0)
    m1 = run(cfg_l)
    assert m0["n_tx_attempts"] == m1["n_tx_attempts"] == m0["n_tx"]
    assert m1["n_delivered"] < m1["n_tx_attempts"]
    assert m1["attempt_based_reduction"] == pytest.approx(m0["attempt_based_reduction"])
    assert m1["event_recall"] == m1["event_recall_union"]


def test_evaluate_policy_aliases_union_event(minimal_rl_cfg):
    trace = build_synthetic_trace(n_steps=12, n_sensors=2, seed=8)
    cfg = dict(minimal_rl_cfg)
    cfg.setdefault("environment", {})["graph"] = "identity"
    env = TraceDrivenCampusEnv(cfg=cfg, trace=trace, multi_agent=True)
    pol = FixedIntervalPolicy(interval_steps=2)

    def act(obs, *, local_available=None):
        return pol.act(obs, local_available=local_available)

    m = evaluate_policy(env, act, max_steps=8, seed=0)
    assert m["event_recall"] == m["event_recall_union"]
    assert m["n_tx"] == m["n_tx_attempts"]
    row = standardize_eval(
        m,
        scenario="validation_cohort",
        split="val",
        cohort="final",
        method="fixed_30",
        configuration="interval_steps=2",
        run="deterministic",
    )
    assert row["event_recall_union"] == row["recall"]
    assert abs(row["attempt_based_reduction"] - row["transmission_reduction"]) < 1e-12


def test_shared_loss_mask_identical_across_two_envs(minimal_rl_cfg):
    trace = build_synthetic_trace(n_steps=10, n_sensors=3, seed=9)
    trace["natural_missing"][:] = False
    cfg = dict(minimal_rl_cfg)
    cfg.setdefault("environment", {})
    cfg["environment"] = dict(cfg["environment"])
    cfg["environment"]["graph"] = "identity"
    cfg["environment"]["packet_loss_rate"] = 0.3
    cfg["environment"]["packet_loss_mode"] = "independent_slots"
    cfg["environment"]["packet_loss_mask_seed"] = 20260317
    env_a = TraceDrivenCampusEnv(cfg=cfg, trace=trace, multi_agent=True)
    env_b = TraceDrivenCampusEnv(cfg=cfg, trace=trace, multi_agent=True)
    np.testing.assert_array_equal(env_a.packet_loss_mask, env_b.packet_loss_mask)


def test_neighbor_summary_does_not_use_hidden_true_after_skip(minimal_rl_cfg):
    n_steps, n_sensors = 8, 2
    trace = build_synthetic_trace(n_steps=n_steps, n_sensors=n_sensors, seed=10)
    trace["ground_truth"][:, :] = 400.0
    trace["ground_truth"][1, 1] = 1800.0  # current true of neighbor at t=1
    trace["natural_missing"][:] = False
    trace["local_available"][:] = True
    cfg = dict(minimal_rl_cfg)
    cfg.setdefault("environment", {})["graph"] = "identity"
    env = TraceDrivenCampusEnv(cfg=cfg, trace=trace, multi_agent=True)
    env.adjacency = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    env.reset(seed=0)
    # t=0: both transmit 400 so server recon is 400
    obs, _, _, _, _ = env.step(np.full(n_sensors, TRANSMIT, dtype=int))
    # obs is now t=1: neighbor 1 has hidden GT=1800 but server recon=400
    neigh0 = float(obs[0, 13])
    hidden_norm = 1800.0 / 1500.0
    last_tx_norm = 400.0 / 1500.0
    assert abs(neigh0 - hidden_norm) > 0.2
    assert neigh0 == pytest.approx(last_tx_norm, abs=0.05)
    # skipping the neighbor must not inject its current GT into server recon
    env.step(np.array([TRANSMIT, SKIP], dtype=int))
    assert env.server_state.last_transmitted[1] == pytest.approx(400.0)


def test_production_obs_path_does_not_include_neighbor_disagreement():
    import inspect

    src = inspect.getsource(TraceDrivenCampusEnv)
    assert "_neighbor_summary" in src
    assert "_neighbor_disagreement" not in src
    assert not hasattr(TraceDrivenCampusEnv, "_neighbor_disagreement")
    obs_src = inspect.getsource(TraceDrivenCampusEnv._agent_observation)
    assert "_neighbor_summary" in obs_src
    assert "disagreement" not in obs_src


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
