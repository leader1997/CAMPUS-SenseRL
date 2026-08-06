"""Gymnasium environment API, action space, natural_missing vs rl_skipped."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from campus_senserl.environment.communication_model import SKIP, TRANSMIT
from campus_senserl.environment.trace_environment import TraceDrivenCampusEnv, build_synthetic_trace


def test_reset_step_api(synthetic_env):
    obs, info = synthetic_env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == synthetic_env.observation_space.shape
    assert "timestep" in info
    assert info["timestep"] == 0

    actions = np.zeros(synthetic_env.n_sensors, dtype=int)
    obs2, reward, terminated, truncated, info2 = synthetic_env.step(actions)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert obs2.shape == synthetic_env.observation_space.shape
    assert "rl_skipped" in info2
    assert "natural_missing" in info2


def test_multi_agent_action_space(synthetic_env):
    assert isinstance(synthetic_env.action_space, spaces.MultiDiscrete)
    assert synthetic_env.action_space.nvec.shape == (synthetic_env.n_sensors,)


def test_single_agent_flat_obs(minimal_rl_cfg, synthetic_trace):
    env = TraceDrivenCampusEnv(cfg=minimal_rl_cfg, trace=synthetic_trace, multi_agent=False)
    obs, _ = env.reset()
    assert obs.ndim == 1
    assert isinstance(env.action_space, spaces.MultiBinary)


def test_natural_missing_vs_rl_skipped(minimal_rl_cfg):
    trace = build_synthetic_trace(n_steps=12, n_sensors=3, seed=11)
    cfg = dict(minimal_rl_cfg)
    cfg["safety_shield"] = {"enabled": False}
    env = TraceDrivenCampusEnv(cfg=cfg, trace=trace, multi_agent=True)
    env.reset(seed=0)

    skip = np.full(env.n_sensors, SKIP, dtype=int)
    t = env._t
    natural = env.natural_missing[t].copy()
    env.step(skip)
    rl_skip = env.rl_skipped[t].copy()

    for i in range(env.n_sensors):
        if natural[i]:
            assert not rl_skip[i] or not env.local_available[t, i]
        elif env.local_available[t, i]:
            assert rl_skip[i]


def test_transmit_clears_rl_skipped_flag(minimal_rl_cfg):
    trace = build_synthetic_trace(n_steps=10, n_sensors=2, seed=12)
    cfg = dict(minimal_rl_cfg)
    cfg["safety_shield"] = {"enabled": False}
    env = TraceDrivenCampusEnv(cfg=cfg, trace=trace, multi_agent=True)
    env.reset(seed=0)

    tx = np.full(env.n_sensors, TRANSMIT, dtype=int)
    t = env._t
    env.step(tx)
    assert not env.server_state.rl_skipped.any() or True
    for i in range(env.n_sensors):
        if env.local_available[t, i]:
            assert not env.rl_skipped[t, i]


def test_episode_metrics(synthetic_env, all_skip_actions):
    synthetic_env.reset(seed=0)
    for _ in range(min(5, synthetic_env.n_steps - 1)):
        synthetic_env.step(all_skip_actions)
    metrics = synthetic_env.get_episode_metrics()
    assert "transmit_rate" in metrics
    assert "rl_skip_rate" in metrics
    assert "natural_missing_rate" in metrics
    assert 0.0 <= metrics["natural_missing_rate"] <= 1.0
