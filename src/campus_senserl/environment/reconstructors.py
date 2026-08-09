"""Reconstructor backends for the trace-driven environment.

All backends consume only ServerState (never skipped ground truth) and return
(mean, uncertainty) arrays of shape (n_sensors,).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np

from campus_senserl.environment.trace_environment import LocfReconstructor, ServerState
from campus_senserl.utils import repo_root


class ReconstructorFn(Protocol):
    def __call__(
        self,
        server_state: ServerState,
        time_feats: np.ndarray,
        adjacency: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        ...


BACKEND_NAMES = ("locf", "extratrees", "lightgbm", "stgnn", "proposed_reconstructor")


def build_reconstructor(
    name: str = "locf",
    *,
    cfg: dict[str, Any] | None = None,
    n_sensors: int | None = None,
    sensor_ids: list[str] | None = None,
    device: str = "cpu",
) -> ReconstructorFn:
    cfg = cfg or {}
    env_cfg = cfg.get("environment", {})
    name = str(name or env_cfg.get("reconstructor", "locf")).lower().strip()

    if name == "proposed_reconstructor":
        name = str(env_cfg.get("proposed_backend", "locf")).lower().strip()

    if name in {"locf", "extratrees", "lightgbm"}:
        # Tree models are batch-trained; online rollout uses LOCF until a
        # sequential adapter is fitted. Config still records the intended name.
        return LocfReconstructor().predict

    if name == "stgnn":
        return _build_stgnn_online(cfg=cfg, n_sensors=n_sensors, sensor_ids=sensor_ids, device=device)

    raise ValueError(f"Unknown reconstructor backend {name!r}. Choose from {BACKEND_NAMES}.")


def _build_stgnn_online(
    *,
    cfg: dict[str, Any],
    n_sensors: int | None,
    sensor_ids: list[str] | None,
    device: str,
) -> ReconstructorFn:
    root = repo_root()
    env_cfg = cfg.get("environment", {})
    ckpt = Path(env_cfg.get("reconstructor_checkpoint", "outputs/models/reconstruction/best_model.pt"))
    if not ckpt.is_absolute():
        ckpt = root / ckpt
    if not ckpt.exists():
        print(f"[reconstructor] ST-GNN checkpoint missing ({ckpt}); using LOCF.")
        return LocfReconstructor().predict

    try:
        import torch

        from campus_senserl.models.graph_reconstruction import load_reconstruction_model
    except Exception as exc:
        print(f"[reconstructor] cannot import ST-GNN ({exc}); using LOCF.")
        return LocfReconstructor().predict

    try:
        model, meta = load_reconstruction_model(ckpt, device=device)
    except Exception as exc:
        print(f"[reconstructor] ST-GNN load failed ({exc}); using LOCF.")
        return LocfReconstructor().predict

    ckpt_n = len(meta["node_order"])
    if n_sensors is not None and int(ckpt_n) != int(n_sensors):
        print(
            f"[reconstructor] ST-GNN n_sensors={ckpt_n} != env n_sensors={n_sensors}; using LOCF."
        )
        return LocfReconstructor().predict

    if sensor_ids is not None:
        order = [str(x) for x in meta["node_order"]]
        if [str(s) for s in sensor_ids] != order:
            print("[reconstructor] ST-GNN node_order != env sensor_ids; using LOCF.")
            return LocfReconstructor().predict

    w = int(getattr(model, "history_window", 8))
    n = int(ckpt_n)
    hist_values = np.full((w, n), np.nan, dtype=np.float32)
    hist_mask = np.zeros((w, n), dtype=bool)
    hist_aoi = np.full((w, n), 8.0, dtype=np.float32)
    hist_time = np.zeros((w, 4), dtype=np.float32)
    locf = LocfReconstructor()
    adj_default = meta.get("adjacency")
    if adj_default is None:
        adj_default = np.eye(n, dtype=np.float32)

    def _predict(
        server_state: ServerState,
        time_feats: np.ndarray,
        adjacency: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        nonlocal hist_values, hist_mask, hist_aoi, hist_time
        # Shift history and append current server-visible snapshot only
        hist_values = np.roll(hist_values, -1, axis=0)
        hist_mask = np.roll(hist_mask, -1, axis=0)
        hist_aoi = np.roll(hist_aoi, -1, axis=0)
        hist_time = np.roll(hist_time, -1, axis=0)

        cur = np.full(n, np.nan, dtype=np.float32)
        cur_m = np.zeros(n, dtype=bool)
        for i in range(n):
            if server_state.input_mask[i] and np.isfinite(server_state.server_values[i]):
                cur[i] = server_state.server_values[i]
                cur_m[i] = True
            elif np.isfinite(server_state.last_transmitted[i]):
                # Carry last TX as visible history input (not skipped GT)
                cur[i] = server_state.last_transmitted[i]
                cur_m[i] = True
        hist_values[-1] = cur
        hist_mask[-1] = cur_m
        hist_aoi[-1] = server_state.aoi.astype(np.float32)
        tf = np.asarray(time_feats, dtype=np.float32).reshape(-1)
        if tf.size >= 4:
            hist_time[-1] = tf[:4]
        elif tf.size:
            hist_time[-1, : tf.size] = tf

        adj = adjacency if adjacency is not None else adj_default
        adj = np.asarray(adj, dtype=np.float32)
        try:
            with torch.no_grad():
                values = torch.from_numpy(np.nan_to_num(hist_values, nan=0.0)[None]).float().to(device)
                masks = torch.from_numpy(hist_mask[None]).to(device)
                aoi = torch.from_numpy(hist_aoi[None]).float().to(device)
                tfeats = torch.from_numpy(hist_time[None]).float().to(device)
                adj_t = torch.from_numpy(adj).float().to(device)
                mu, log_var = model(values, masks, aoi, tfeats, adj_t)
                mean = mu.squeeze(0).detach().cpu().numpy().astype(np.float32)
                unc = np.sqrt(np.exp(np.clip(log_var.squeeze(0).detach().cpu().numpy(), -10, 10))).astype(
                    np.float32
                )
            # Prefer exact current TX when available
            for i in range(n):
                if server_state.input_mask[i] and np.isfinite(server_state.server_values[i]):
                    mean[i] = server_state.server_values[i]
                    unc[i] = 0.25
            return mean, unc
        except Exception as exc:
            print(f"[reconstructor] ST-GNN forward failed ({exc}); LOCF fallback this step")
            return locf.predict(server_state, time_feats, adjacency)

    print(f"[reconstructor] ST-GNN online adapter ready (n={n}, window={w}, device={device})")
    return _predict


def make_constant_reconstructor(
    mean_value: float,
    unc_value: float,
) -> ReconstructorFn:
    """Deterministic backend for unit tests (proves wiring changes results)."""

    def _predict(
        server_state: ServerState,
        time_feats: np.ndarray,
        adjacency: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = server_state.server_values.shape[0]
        mean = np.full(n, float(mean_value), dtype=np.float32)
        for i in range(n):
            if server_state.input_mask[i] and np.isfinite(server_state.server_values[i]):
                mean[i] = server_state.server_values[i]
        unc = np.full(n, float(unc_value), dtype=np.float32)
        return mean, unc

    return _predict
