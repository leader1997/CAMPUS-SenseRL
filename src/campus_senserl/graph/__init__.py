"""Campus sensor graph construction.

Edges are a server-derived spatial/statistical relation graph: contextual
sensor relations for features/reconstruction. They are NOT LoRaWAN links,
sensor-to-sensor packet exchange, ventilation/airflow topology, or a
physical mesh network. Correlation edges are fit on TRAINING DATA ONLY.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from campus_senserl.utils import (
    ensure_dir,
    haversine_m,
    load_yaml,
    repo_root,
    save_json,
    set_seed,
)


def load_device_meta(path: Path | None = None) -> pd.DataFrame:
    root = repo_root()
    path = path or (root / "data" / "interim" / "devices.parquet")
    return pd.read_parquet(path)


def build_spatial_graph(
    devices: pd.DataFrame,
    knn_k: int = 8,
    max_distance_m: float = 80.0,
    same_floor_boost: bool = True,
    device_ids: list[str] | None = None,
) -> nx.Graph:
    """Graph A: geographic kNN with optional same-floor preference."""
    df = devices.copy()
    if device_ids is not None:
        df = df[df["device_id"].isin(device_ids)]
    df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

    G = nx.Graph()
    for _, r in df.iterrows():
        G.add_node(
            r["device_id"],
            floor=r.get("floor"),
            lat=float(r["latitude"]),
            lon=float(r["longitude"]),
            device_type=r.get("device_type"),
            desc=r.get("desc"),
        )

    ids = df["device_id"].tolist()
    coords = df[["latitude", "longitude"]].to_numpy(dtype=float)
    floors = df["floor"].astype(str).tolist()
    n = len(ids)
    for i in range(n):
        dists = []
        for j in range(n):
            if i == j:
                continue
            d = haversine_m(coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1])
            if same_floor_boost and floors[i] == floors[j]:
                score = d * 0.7  # prefer same floor
            else:
                score = d
            if d <= max_distance_m:
                dists.append((score, d, j))
        dists.sort()
        for _, d, j in dists[:knn_k]:
            G.add_edge(ids[i], ids[j], weight=1.0 / (1.0 + d), distance_m=d, kind="spatial")
    return G


def build_correlation_graph(
    wide_train: pd.DataFrame,
    min_abs_corr: float = 0.55,
    max_neighbors: int = 10,
    min_overlap_obs: int = 200,
) -> nx.Graph:
    """Graph B: pairwise Pearson correlation on TRAINING wide matrix only."""
    cols = list(wide_train.columns)
    G = nx.Graph()
    for c in cols:
        G.add_node(c)

    # Pairwise correlation with pairwise deletion
    corr = wide_train.corr(min_periods=min_overlap_obs)
    for i, a in enumerate(cols):
        scores = []
        for j, b in enumerate(cols):
            if j <= i:
                continue
            r = corr.loc[a, b]
            if pd.isna(r):
                continue
            if abs(r) >= min_abs_corr:
                scores.append((abs(r), float(r), b))
        scores.sort(reverse=True)
        for ar, r, b in scores[:max_neighbors]:
            G.add_edge(a, b, weight=ar, corr=r, kind="correlation")
    return G


def build_hybrid_graph(
    spatial: nx.Graph,
    correlation: nx.Graph,
    spatial_weight: float = 0.4,
    correlation_weight: float = 0.6,
    knn_k: int = 8,
    min_edge_score: float = 0.25,
) -> nx.Graph:
    """Graph C: combine spatial proximity and historical correlation."""
    nodes = set(spatial.nodes()) | set(correlation.nodes())
    G = nx.Graph()
    for n in nodes:
        attrs = {}
        if n in spatial.nodes:
            attrs.update(spatial.nodes[n])
        G.add_node(n, **attrs)

    scores: dict[tuple[str, str], dict[str, float]] = {}

    def _key(u, v):
        return (u, v) if u < v else (v, u)

    for u, v, data in spatial.edges(data=True):
        k = _key(u, v)
        scores.setdefault(k, {"spatial": 0.0, "corr": 0.0})
        scores[k]["spatial"] = float(data.get("weight", 0.0))
    for u, v, data in correlation.edges(data=True):
        k = _key(u, v)
        scores.setdefault(k, {"spatial": 0.0, "corr": 0.0})
        scores[k]["corr"] = float(data.get("weight", 0.0))

    # Per-node top-k by hybrid score
    per_node: dict[str, list[tuple[float, str, str]]] = {n: [] for n in nodes}
    for (u, v), sc in scores.items():
        hybrid = spatial_weight * sc["spatial"] + correlation_weight * sc["corr"]
        if hybrid < min_edge_score:
            continue
        per_node[u].append((hybrid, u, v))
        per_node[v].append((hybrid, u, v))

    added = set()
    for n, lst in per_node.items():
        lst.sort(reverse=True)
        for hybrid, u, v in lst[:knn_k]:
            k = _key(u, v)
            if k in added:
                continue
            added.add(k)
            G.add_edge(
                u,
                v,
                weight=hybrid,
                spatial=scores[k]["spatial"],
                corr=scores[k]["corr"],
                kind="hybrid",
            )
    return G


def adjacency_matrix(G: nx.Graph, node_order: list[str]) -> np.ndarray:
    idx = {n: i for i, n in enumerate(node_order)}
    A = np.zeros((len(node_order), len(node_order)), dtype=np.float32)
    for u, v, data in G.edges(data=True):
        if u in idx and v in idx:
            w = float(data.get("weight", 1.0))
            A[idx[u], idx[v]] = w
            A[idx[v], idx[u]] = w
    return A


def visualize_graph(
    G: nx.Graph,
    devices: pd.DataFrame,
    out_path: Path,
    title: str,
    max_nodes: int = 120,
) -> None:
    ensure_dir(out_path.parent)
    H = G
    if G.number_of_nodes() > max_nodes:
        # Subsample for readability: take highest-degree nodes
        deg = sorted(G.degree(), key=lambda x: -x[1])[:max_nodes]
        H = G.subgraph([n for n, _ in deg]).copy()

    pos = {}
    meta = devices.set_index("device_id")
    for n in H.nodes():
        if n in meta.index and pd.notna(meta.loc[n, "longitude"]):
            pos[n] = (float(meta.loc[n, "longitude"]), float(meta.loc[n, "latitude"]))
    if not pos:
        pos = nx.spring_layout(H, seed=42)

    fig, ax = plt.subplots(figsize=(8, 7))
    floors = []
    for n in H.nodes():
        fl = meta.loc[n, "floor"] if n in meta.index else "NA"
        floors.append(str(fl))
    uniq = sorted(set(floors))
    cmap = plt.cm.tab10
    color_map = {f: cmap(i % 10) for i, f in enumerate(uniq)}
    node_colors = [color_map[f] for f in floors]
    nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.25, width=0.6)
    nx.draw_networkx_nodes(H, pos, ax=ax, node_size=28, node_color=node_colors, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[f], markersize=8, label=f"Floor {f}")
        for f in uniq
    ]
    ax.legend(handles=handles, fontsize=8, title="Floor")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def save_graph(G: nx.Graph, path: Path) -> None:
    ensure_dir(path.parent)
    nx.write_gml(G, path)


def run_build_graphs(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    root = repo_root()
    cfg = cfg or load_yaml(root / "configs" / "model.yaml")
    set_seed(int(cfg.get("seed", 42)))
    out_dir = ensure_dir(root / cfg.get("outputs", {}).get("dir", "results/graphs"))

    devices = load_device_meta()
    panel_path = root / "data" / "processed" / "co2_panel_15min.parquet"
    wide_path = root / "data" / "processed" / "co2_wide_observed.parquet"
    if not panel_path.exists():
        raise FileNotFoundError("Run scripts/02_preprocess.py first.")

    panel = pd.read_parquet(panel_path, columns=["deveui", "split"])
    co2_ids = sorted(panel["deveui"].unique().tolist())
    devices_co2 = devices[devices["device_id"].isin(co2_ids)].copy()

    print("[graph] Building spatial graph…")
    G_spatial = build_spatial_graph(
        devices_co2,
        knn_k=int(cfg["spatial"]["knn_k"]),
        max_distance_m=float(cfg["spatial"]["max_distance_m"]),
        same_floor_boost=bool(cfg["spatial"]["same_floor_boost"]),
        device_ids=co2_ids,
    )

    print("[graph] Building correlation graph (TRAIN only)…")
    wide = pd.read_parquet(wide_path)
    # Restrict columns to train-overlapping sensors
    train_slots = panel[panel["split"] == "train"]
    # Use wide index split via panel unique slots
    panel_slots = pd.read_parquet(panel_path, columns=["slot", "split"]).drop_duplicates()
    train_slot_set = set(panel_slots.loc[panel_slots["split"] == "train", "slot"])
    wide_train = wide.loc[wide.index.isin(train_slot_set)]
    G_corr = build_correlation_graph(
        wide_train,
        min_abs_corr=float(cfg["correlation"]["min_abs_corr"]),
        max_neighbors=int(cfg["correlation"]["max_neighbors"]),
        min_overlap_obs=int(cfg["correlation"]["min_overlap_obs"]),
    )

    print("[graph] Building hybrid graph…")
    G_hybrid = build_hybrid_graph(
        G_spatial,
        G_corr,
        spatial_weight=float(cfg["hybrid"]["spatial_weight"]),
        correlation_weight=float(cfg["hybrid"]["correlation_weight"]),
        knn_k=int(cfg["hybrid"]["knn_k"]),
        min_edge_score=float(cfg["hybrid"]["min_edge_score"]),
    )

    node_order = co2_ids
    for name, G in [("spatial", G_spatial), ("correlation", G_corr), ("hybrid", G_hybrid)]:
        save_graph(G, out_dir / f"graph_{name}.graphml")
        A = adjacency_matrix(G, node_order)
        np.save(out_dir / f"adjacency_{name}.npy", A)
        visualize_graph(
            G,
            devices_co2,
            out_dir / f"graph_{name}.png",
            title=f"Campus sensor graph ({name})",
        )
        print(f"[graph] {name}: nodes={G.number_of_nodes()} edges={G.number_of_edges()}")

    save_json({"node_order": node_order}, out_dir / "node_order.json")
    summary = {
        "n_nodes": len(node_order),
        "spatial_edges": G_spatial.number_of_edges(),
        "correlation_edges": G_corr.number_of_edges(),
        "hybrid_edges": G_hybrid.number_of_edges(),
        "note": "Edges are statistical/spatial relationships, not ventilation topology.",
        "correlation_fit_split": "train",
    }
    save_json(summary, out_dir / "graph_summary.json")
    print("[graph] Done.")
    return summary


if __name__ == "__main__":
    run_build_graphs()
