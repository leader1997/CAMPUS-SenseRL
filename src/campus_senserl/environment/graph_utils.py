"""Graph loading utilities for RL / reconstruction wiring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from campus_senserl.utils import repo_root


def load_adjacency_for_sensors(
    graph_type: str = "hybrid",
    sensor_ids: list[str] | None = None,
    *,
    graphs_dir: str | Path | None = None,
) -> np.ndarray:
    """Load a saved adjacency and reorder/subset to ``sensor_ids``.

    Parameters
    ----------
    graph_type:
        ``spatial``, ``correlation``, ``hybrid``, or ``identity``.
    sensor_ids:
        Environment sensor order (deveui strings). If None, return full matrix
        in artifact node order.
    """
    gtype = str(graph_type or "identity").lower().strip()
    if gtype in {"identity", "none", "eye", ""}:
        n = len(sensor_ids) if sensor_ids is not None else 1
        return np.eye(n, dtype=np.float32)

    root = repo_root()
    gdir = Path(graphs_dir) if graphs_dir is not None else root / "results" / "graphs"
    if not gdir.is_absolute():
        gdir = root / gdir

    adj_path = gdir / f"adjacency_{gtype}.npy"
    order_path = gdir / "node_order.json"
    if not adj_path.exists():
        raise FileNotFoundError(
            f"Graph artifact missing: {adj_path}. Run scripts/03_build_graph.py first."
        )

    adj = np.load(adj_path).astype(np.float32)
    if sensor_ids is None:
        return adj

    import json

    if order_path.exists():
        with open(order_path, encoding="utf-8") as f:
            node_order = json.load(f)
        if isinstance(node_order, dict):
            node_order = node_order.get("node_order") or node_order.get("nodes") or []
        node_order = [str(x) for x in node_order]
    else:
        # Fall back to assuming artifact order matches sorted sensor ids of full graph.
        node_order = [str(x) for x in sensor_ids]

    index = {d: i for i, d in enumerate(node_order)}
    n = len(sensor_ids)
    out = np.eye(n, dtype=np.float32)
    missing = 0
    for i, sid in enumerate(sensor_ids):
        si = index.get(str(sid))
        if si is None or si >= adj.shape[0]:
            missing += 1
            continue
        for j, sjd in enumerate(sensor_ids):
            sj = index.get(str(sjd))
            if sj is None or sj >= adj.shape[1]:
                continue
            out[i, j] = adj[si, sj]
    if missing == n:
        raise RuntimeError(
            f"None of the env sensors matched graph node_order ({order_path})."
        )
    # Ensure self-loops for neighbour iteration conventions.
    np.fill_diagonal(out, 1.0)
    return out


def assert_not_identity(adjacency: np.ndarray, *, atol: float = 1e-8) -> None:
    """Raise if adjacency is (numerically) identity — used in tests."""
    n = adjacency.shape[0]
    eye = np.eye(n, dtype=adjacency.dtype)
    if adjacency.shape != (n, n):
        raise AssertionError(f"adjacency shape {adjacency.shape} is not square")
    if np.allclose(adjacency, eye, atol=atol):
        raise AssertionError("adjacency is identity; graph-enabled run misconfigured")
