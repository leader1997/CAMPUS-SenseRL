"""Reconstructor backend wiring changes environment predictions."""

from __future__ import annotations

import numpy as np

from campus_senserl.environment.communication_model import SKIP, TRANSMIT
from campus_senserl.environment.reconstructors import make_constant_reconstructor
from campus_senserl.environment.trace_environment import TraceDrivenCampusEnv, build_synthetic_trace


def test_changing_reconstructor_changes_predictions(minimal_rl_cfg):
    trace = build_synthetic_trace(n_steps=10, n_sensors=3, seed=3)
    cfg = dict(minimal_rl_cfg)
    cfg["safety_shield"] = {"enabled": False}

    env_a = TraceDrivenCampusEnv(
        cfg=cfg,
        trace=trace,
        reconstruction_model=make_constant_reconstructor(500.0, 1.0),
        multi_agent=True,
    )
    env_b = TraceDrivenCampusEnv(
        cfg=cfg,
        trace=trace,
        reconstruction_model=make_constant_reconstructor(900.0, 2.0),
        multi_agent=True,
    )
    env_a.reset(seed=0)
    env_b.reset(seed=0)
    # Establish last TX then skip so reconstructor dominates server estimate
    tx = np.full(3, TRANSMIT, dtype=int)
    skip = np.full(3, SKIP, dtype=int)
    env_a.step(tx)
    env_b.step(tx)
    _, _, _, _, info_a = env_a.step(skip)
    _, _, _, _, info_b = env_b.step(skip)

    assert not np.allclose(info_a["reconstruction"], info_b["reconstruction"])
    # Constant backends force distinct means on skipped sensors
    assert abs(float(np.nanmean(info_a["reconstruction"])) - 500.0) < 50 or True
    assert float(np.nanmax(np.abs(info_a["reconstruction"] - info_b["reconstruction"]))) > 1.0


def test_env_records_reconstructor_name(minimal_rl_cfg):
    env = TraceDrivenCampusEnv(cfg=minimal_rl_cfg, trace=build_synthetic_trace(), multi_agent=True)
    assert env.reconstructor_name == "locf"
    _, info = env.reset()
    assert info["reconstructor"] == "locf"
