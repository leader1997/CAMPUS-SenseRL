"""Shared fixtures for CAMPUS-SenseRL tests."""

from __future__ import annotations

import numpy as np
import pytest

from campus_senserl.environment.communication_model import SKIP, TRANSMIT
from campus_senserl.environment.trace_environment import (
    TraceDrivenCampusEnv,
    build_synthetic_trace,
)


@pytest.fixture
def minimal_rl_cfg() -> dict:
    return {
        "seed": 42,
        "env": {"max_aoi": 8},
        "communication": {"transmit_cost": 1.0, "skip_cost": 0.0},
        "reward": {
            "w_tx": 1.0,
            "w_err": 1.0,
            "w_aoi": 0.5,
            "w_unc": 0.5,
            "w_miss": 5.0,
            "w_event": 1.0,
            "clip": 10.0,
            "normalize_components": True,
        },
        "safety_shield": {
            "enabled": True,
            "force_transmit_if": {
                "aoi_exceeds": 8,
                "uncertainty_exceeds": 2.0,
                "co2_exceeds": 1200,
                "co2_rate_exceeds": 150,
                "neighbor_disagreement_exceeds": 200,
            },
        },
        "events": {
            "co2_thresholds_ppm": [800, 1000, 1200, 1500],
            "primary_threshold_ppm": 1000,
            "rapid_increase_ppm": 150,
            "persistent_high_intervals": 3,
        },
    }


@pytest.fixture
def synthetic_trace() -> dict:
    return build_synthetic_trace(n_steps=24, n_sensors=4, seed=42)


@pytest.fixture
def synthetic_env(minimal_rl_cfg, synthetic_trace) -> TraceDrivenCampusEnv:
    return TraceDrivenCampusEnv(
        cfg=minimal_rl_cfg,
        trace=synthetic_trace,
        multi_agent=True,
    )


@pytest.fixture
def all_skip_actions(synthetic_env) -> np.ndarray:
    return np.full(synthetic_env.n_sensors, SKIP, dtype=int)


@pytest.fixture
def all_transmit_actions(synthetic_env) -> np.ndarray:
    return np.full(synthetic_env.n_sensors, TRANSMIT, dtype=int)
