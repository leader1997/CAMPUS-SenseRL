"""Tests for corrected precision/FP and unbounded AoI reporting."""

from __future__ import annotations

import numpy as np

from campus_senserl.environment.communication_model import SKIP, TRANSMIT
from campus_senserl.environment.reconstructors import make_constant_reconstructor
from campus_senserl.environment.trace_environment import TraceDrivenCampusEnv, build_synthetic_trace
from campus_senserl.evaluation.rl_policy_eval import evaluate_policy


def test_false_positives_exposed_and_counted(minimal_rl_cfg):
    """Server can declare event when true has none → FP > 0, precision < 1."""
    n_steps, n_sensors = 10, 2
    trace = build_synthetic_trace(n_steps=n_steps, n_sensors=n_sensors, seed=0)
    trace["ground_truth"][:, :] = 400.0  # never a true high event
    trace["natural_missing"][:] = False
    trace["local_available"][:] = True

    cfg = dict(minimal_rl_cfg)
    cfg["safety_shield"] = {"enabled": False}
    cfg["events"] = {
        "primary_threshold_ppm": 1000,
        "rapid_increase_ppm": 150,
        "co2_thresholds_ppm": [800, 1000, 1200, 1500],
    }
    # Reconstructor always reports 1200 → server "events" without true events
    recon = make_constant_reconstructor(mean_value=1200.0, unc_value=0.5)
    env = TraceDrivenCampusEnv(cfg=cfg, trace=trace, reconstruction_model=recon, multi_agent=True)

    def always_skip(obs, *, local_available=None):
        return np.full(env.n_sensors, SKIP, dtype=int)

    m = evaluate_policy(env, always_skip, max_steps=6, seed=0)
    assert m["fp"] > 0
    assert m["event_precision"] < 1.0
    assert "server_events"  # smoke: env exposes key on a step
    env2 = TraceDrivenCampusEnv(cfg=cfg, trace=trace, reconstruction_model=recon, multi_agent=True)
    env2.reset()
    _, _, _, _, info = env2.step(np.full(n_sensors, SKIP, dtype=int))
    assert "false_positive_events" in info
    assert "server_events" in info
    assert info["false_positive_events"].any()


def test_aoi_raw_exceeds_clip(minimal_rl_cfg):
    trace = build_synthetic_trace(n_steps=20, n_sensors=1, seed=1)
    trace["natural_missing"][:] = False
    trace["local_available"][:] = True
    cfg = dict(minimal_rl_cfg)
    cfg["safety_shield"] = {"enabled": False}
    cfg.setdefault("env", {})["max_aoi"] = 8
    env = TraceDrivenCampusEnv(cfg=cfg, trace=trace, multi_agent=True)
    env.reset()
    for _ in range(15):
        _, _, _, _, info = env.step(np.array([SKIP], dtype=int))
    assert float(info["aoi_state"][0]) == 8.0
    assert float(info["aoi_raw"][0]) > 8.0
