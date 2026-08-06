"""Age-of-Information (AoI) dynamics: increment on skip, reset on transmit."""

from __future__ import annotations

import numpy as np

from campus_senserl.environment.communication_model import SKIP, TRANSMIT
from campus_senserl.environment.trace_environment import TraceDrivenCampusEnv, build_synthetic_trace


def test_aoi_increments_on_skip(minimal_rl_cfg):
    trace = build_synthetic_trace(n_steps=10, n_sensors=2, seed=3)
    cfg = dict(minimal_rl_cfg)
    cfg["safety_shield"] = {"enabled": False}
    env = TraceDrivenCampusEnv(cfg=cfg, trace=trace, multi_agent=True)
    env.reset(seed=0)

    skip = np.full(env.n_sensors, SKIP, dtype=int)
    sensor = 0
    if not env.local_available[0, sensor]:
        return

    aoi_before = float(env.server_state.aoi[sensor])
    env.step(skip)
    aoi_after = float(env.server_state.aoi[sensor])
    assert aoi_after == min(aoi_before + 1.0, env.max_aoi)


def test_aoi_resets_on_transmit(minimal_rl_cfg):
    trace = build_synthetic_trace(n_steps=10, n_sensors=2, seed=4)
    cfg = dict(minimal_rl_cfg)
    cfg["safety_shield"] = {"enabled": False}
    env = TraceDrivenCampusEnv(cfg=cfg, trace=trace, multi_agent=True)
    env.reset(seed=0)

    skip = np.full(env.n_sensors, SKIP, dtype=int)
    tx = np.full(env.n_sensors, TRANSMIT, dtype=int)
    sensor = 0

    for _ in range(3):
        if env._t >= env.n_steps - 1:
            break
        if env.local_available[env._t, sensor]:
            env.step(skip)

    env.step(tx)
    assert env.server_state.aoi[sensor] == 0.0


def test_aoi_capped_at_max(minimal_rl_cfg):
    trace = build_synthetic_trace(n_steps=20, n_sensors=1, seed=5)
    cfg = dict(minimal_rl_cfg)
    cfg["env"] = {"max_aoi": 5}
    cfg["safety_shield"] = {"enabled": False}
    env = TraceDrivenCampusEnv(cfg=cfg, trace=trace, multi_agent=True)
    env.reset(seed=0)

    skip = np.array([SKIP], dtype=int)
    for _ in range(15):
        if env._t >= env.n_steps - 1:
            break
        env.step(skip)
    assert env.server_state.aoi[0] <= env.max_aoi
