"""CRITICAL leakage tests: skipped GT never in server state; masking; past-only features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from campus_senserl.data.features import add_rolling_features
from campus_senserl.environment.communication_model import SKIP, TRANSMIT
from campus_senserl.environment.trace_environment import TraceDrivenCampusEnv, build_synthetic_trace
from campus_senserl.models.graph_reconstruction import apply_mask_scheme


def test_skipped_ground_truth_not_in_server_values(minimal_rl_cfg, synthetic_trace):
    trace = build_synthetic_trace(n_steps=12, n_sensors=3, seed=7)
    cfg = dict(minimal_rl_cfg)
    cfg["safety_shield"] = {"enabled": False}
    env = TraceDrivenCampusEnv(cfg=cfg, trace=trace, multi_agent=True)
    env.reset(seed=0)

    skip_actions = np.full(env.n_sensors, SKIP, dtype=int)
    for step in range(env.n_steps - 1):
        t = env._t
        gt = env.ground_truth[t].copy()
        local = env.local_available[t]
        _, _, _, _, info = env.step(skip_actions)

        assert env.server_state is not None
        for i in range(env.n_sensors):
            if not local[i] or not np.isfinite(gt[i]):
                continue
            sv = env.server_state.server_values[i]
            if env.server_state.input_mask[i]:
                pytest.fail("input_mask should be False after SKIP")
            if np.isfinite(sv) and np.isclose(sv, gt[i], rtol=0, atol=1e-3):
                if not env.transmit_history[: t + 1, i].any():
                    pytest.fail(
                        f"Skipped GT leaked into server_values at t={t}, sensor={i}: {sv}"
                    )


def test_transmit_updates_server_with_current_gt(minimal_rl_cfg):
    trace = build_synthetic_trace(n_steps=8, n_sensors=2, seed=1)
    cfg = dict(minimal_rl_cfg)
    cfg["safety_shield"] = {"enabled": False}
    env = TraceDrivenCampusEnv(cfg=cfg, trace=trace, multi_agent=True)
    env.reset(seed=0)

    tx = np.full(env.n_sensors, TRANSMIT, dtype=int)
    t = env._t
    gt = env.ground_truth[t].copy()
    env.step(tx)
    for i in range(env.n_sensors):
        if env.local_available[t, i] and np.isfinite(gt[i]):
            assert env.server_state.input_mask[i]
            assert np.isclose(env.server_state.server_values[i], gt[i])


def test_apply_mask_scheme_hides_values_from_input():
    rng = np.random.default_rng(0)
    observed = rng.random((20, 5)) < 0.9
    mask = apply_mask_scheme(observed, "random", rate=0.3, seed=123)
    assert mask.dtype == bool
    assert not np.any(mask & ~observed)
    visible = observed & ~mask
    hidden = observed & mask
    assert hidden.any()
    assert not np.any(visible & hidden)


def test_masking_only_on_observed_positions():
    observed = np.zeros((10, 4), dtype=bool)
    observed[2:8, 1] = True
    mask = apply_mask_scheme(observed, "random", rate=0.5, seed=99)
    assert not np.any(mask & ~observed)


def test_rolling_features_exclude_current_timestep():
    slots = pd.date_range("2024-01-01", periods=10, freq="15min", tz="UTC")
    rows = []
    for s in ["A", "B"]:
        for i, slot in enumerate(slots):
            rows.append({"slot": slot, "deveui": s, "co2": 400.0 + i * 10 + (100 if s == "B" else 0)})
    df = pd.DataFrame(rows)
    out = add_rolling_features(df, "co2", "deveui", "slot", windows=[4])
    for deveui in ["A", "B"]:
        sub = out[out["deveui"] == deveui].reset_index(drop=True)
        for i in range(1, len(sub)):
            roll = sub.loc[i, "co2_rollmean_4"]
            if i == 1:
                assert np.isnan(sub.loc[0, "co2_rollmean_4"]) or sub.loc[0, "co2_rollmean_4"] == sub.loc[0, "co2_lag1"]
            if i >= 2 and np.isfinite(roll):
                past_vals = sub.loc[: i - 1, "co2"].to_numpy()
                expected = np.mean(past_vals[-4:])
                assert np.isclose(roll, expected, rtol=1e-5), f"roll at i={i} used future data"


def test_rolling_delta_uses_lag_only():
    """delta must be lag1 - lag2 (past only), never current - lag1."""
    slots = pd.date_range("2024-01-01", periods=5, freq="15min", tz="UTC")
    df = pd.DataFrame({"slot": slots, "deveui": "X", "co2": [500, 520, 540, 560, 580]})
    out = add_rolling_features(df, "co2", "deveui", "slot", windows=[2])
    assert np.isnan(out.loc[0, "co2_delta1"])
    assert np.isnan(out.loc[1, "co2_delta1"])
    for i in range(2, len(out)):
        expected = out.loc[i, "co2_lag1"] - out.loc[i, "co2_lag2"]
        assert np.isclose(out.loc[i, "co2_delta1"], expected)
        # Must NOT equal current - lag1 (that would leak current co2)
        leaked = out.loc[i, "co2"] - out.loc[i, "co2_lag1"]
        # For this arithmetic series both equal 20, so use a non-linear series below


def test_rolling_delta_no_current_leakage():
    slots = pd.date_range("2024-01-01", periods=5, freq="15min", tz="UTC")
    df = pd.DataFrame({"slot": slots, "deveui": "X", "co2": [500, 510, 530, 600, 610]})
    out = add_rolling_features(df, "co2", "deveui", "slot", windows=[2])
    # at i=3: lag1=530, lag2=510 => delta=20; current-lag1=70
    assert np.isclose(out.loc[3, "co2_delta1"], 20.0)
    assert not np.isclose(out.loc[3, "co2_delta1"], 600 - 530)
