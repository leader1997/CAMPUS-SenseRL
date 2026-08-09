"""Graph-enabled environments must not silently use identity adjacency."""

from __future__ import annotations

import numpy as np
import pytest

from campus_senserl.environment.graph_utils import assert_not_identity, load_adjacency_for_sensors
from campus_senserl.environment.trace_environment import TraceDrivenCampusEnv, build_synthetic_trace


def test_assert_not_identity_raises_on_eye():
    with pytest.raises(AssertionError):
        assert_not_identity(np.eye(4, dtype=np.float32))


def test_assert_not_identity_passes_on_connected():
    a = np.eye(3, dtype=np.float32)
    a[0, 1] = a[1, 0] = 1.0
    assert_not_identity(a)


def test_env_with_explicit_graph_is_not_identity(minimal_rl_cfg):
    trace = build_synthetic_trace(n_steps=8, n_sensors=4, seed=5)
    adj = np.eye(4, dtype=np.float32)
    adj[0, 1] = adj[1, 0] = 0.8
    adj[2, 3] = adj[3, 2] = 0.5
    cfg = dict(minimal_rl_cfg)
    cfg["environment"] = {"reconstructor": "locf", "graph": "hybrid"}  # overridden by explicit adj
    env = TraceDrivenCampusEnv(cfg=cfg, trace=trace, adjacency=adj, multi_agent=True)
    assert_not_identity(env.adjacency)
    assert env.adjacency.shape == (4, 4)


def test_load_adjacency_identity_shortcut():
    a = load_adjacency_for_sensors("identity", ["a", "b", "c"])
    assert np.allclose(a, np.eye(3))
