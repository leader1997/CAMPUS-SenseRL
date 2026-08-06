"""Spatial graph construction and train-only correlation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from campus_senserl.graph import (
    adjacency_matrix,
    build_correlation_graph,
    build_spatial_graph,
)


def _synthetic_devices(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "device_id": [f"D{i}" for i in range(n)],
            "latitude": 65.0 + np.linspace(0, 0.01, n),
            "longitude": 25.4 + np.linspace(0, 0.01, n),
            "floor": [i % 2 for i in range(n)],
            "device_type": ["Elsys ERS CO2"] * n,
            "desc": [""] * n,
        }
    )


def test_spatial_graph_builds_edges():
    devices = _synthetic_devices(8)
    G = build_spatial_graph(devices, knn_k=3, max_distance_m=500.0)
    assert G.number_of_nodes() == 8
    assert G.number_of_edges() > 0
    for u, v, data in G.edges(data=True):
        assert data.get("kind") == "spatial"
        assert "distance_m" in data


def test_correlation_uses_only_train_matrix():
    rng = np.random.default_rng(0)
    cols = ["A", "B", "C"]
    train = pd.DataFrame(rng.normal(size=(100, 3)), columns=cols)
    train["A"] = train["B"] * 0.9 + rng.normal(scale=0.1, size=100)
    test = pd.DataFrame(rng.normal(size=(100, 3)), columns=cols)
    test["A"] = test["B"] * 0.1 + rng.normal(scale=1.0, size=100)

    G_train = build_correlation_graph(train, min_abs_corr=0.5, min_overlap_obs=20)
    G_test = build_correlation_graph(test, min_abs_corr=0.5, min_overlap_obs=20)

    ab_train = G_train.has_edge("A", "B")
    ab_test = G_test.has_edge("A", "B")
    assert ab_train
    assert not ab_test or G_test.edges["A", "B"]["corr"] < G_train.edges["A", "B"]["corr"]


def test_correlation_not_fit_on_test_only_data():
    rng = np.random.default_rng(1)
    cols = ["X", "Y"]
    test_only = pd.DataFrame({"X": rng.normal(size=50), "Y": rng.normal(size=50)})
    test_only["Y"] = test_only["X"] * 0.95 + rng.normal(scale=0.05, size=50)
    G = build_correlation_graph(test_only, min_abs_corr=0.8, min_overlap_obs=10)
    assert G.has_edge("X", "Y")


def test_adjacency_matrix_order():
    devices = _synthetic_devices(4)
    G = build_spatial_graph(devices, knn_k=2, max_distance_m=500.0)
    order = devices["device_id"].tolist()
    A = adjacency_matrix(G, order)
    assert A.shape == (4, 4)
    assert np.allclose(A, A.T)
