"""Masked spatio-temporal graph reconstruction with heteroscedastic outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from campus_senserl.models.uncertainty import gaussian_nll_masked, summarize_uncertainty_metrics
from campus_senserl.utils import ensure_dir, load_yaml, repo_root, save_json, set_seed


class MaskingScheme(str, Enum):
    RANDOM = "random"
    TEMPORAL_BLOCK = "temporal_block"
    SENSOR_OUTAGE = "sensor_outage"
    FLOOR_OUTAGE = "floor_outage"


def apply_mask_scheme(
    observed: np.ndarray,
    scheme: str | MaskingScheme,
    *,
    rate: float = 0.2,
    seed: int = 42,
    sensor_floors: np.ndarray | None = None,
    block_len: int = 4,
) -> np.ndarray:
    """Return boolean mask where True => hidden from model input (held out).

    Only positions with observed ground truth may be masked.
    """
    rng = np.random.default_rng(seed)
    obs = observed.astype(bool)
    mask = np.zeros_like(obs, dtype=bool)
    scheme = MaskingScheme(scheme)
    valid = obs.copy()

    if scheme == MaskingScheme.RANDOM:
        rand = rng.random(obs.shape)
        mask = valid & (rand < rate)
        return mask

    t_steps, n_sensors = obs.shape
    if scheme == MaskingScheme.TEMPORAL_BLOCK:
        n_blocks = max(1, int(t_steps * rate / max(block_len, 1)))
        for _ in range(n_blocks):
            s = int(rng.integers(0, max(1, t_steps - block_len + 1)))
            e = min(t_steps, s + block_len)
            cols = rng.choice(n_sensors, size=max(1, int(n_sensors * 0.3)), replace=False)
            for c in cols:
                mask[s:e, c] = valid[s:e, c]
        return mask

    if scheme == MaskingScheme.SENSOR_OUTAGE:
        n_out = max(1, int(n_sensors * rate))
        sensors = rng.choice(n_sensors, size=n_out, replace=False)
        for c in sensors:
            start = int(rng.integers(0, t_steps // 2))
            end = min(t_steps, start + int(rng.integers(block_len, t_steps // 2 + block_len)))
            mask[start:end, c] = valid[start:end, c]
        return mask

    if scheme == MaskingScheme.FLOOR_OUTAGE:
        if sensor_floors is None:
            # Fallback to sensor outage
            return apply_mask_scheme(
                observed, MaskingScheme.SENSOR_OUTAGE, rate=rate, seed=seed, block_len=block_len
            )
        floors = np.unique(sensor_floors)
        n_floor = max(1, int(len(floors) * rate))
        chosen = rng.choice(floors, size=min(n_floor, len(floors)), replace=False)
        for fl in chosen:
            cols = np.where(sensor_floors == fl)[0]
            if len(cols) == 0:
                continue
            start = int(rng.integers(0, max(1, t_steps - block_len)))
            end = min(t_steps, start + block_len * 3)
            mask[start:end, cols] = valid[start:end, cols]
        return mask

    raise ValueError(f"Unknown masking scheme: {scheme}")


class GraphMessageLayer(nn.Module):
    """GraphSAGE-style mean aggregation with optional GAT attention."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        n_heads: int = 4,
        dropout: float = 0.1,
        use_gat: bool = True,
    ) -> None:
        super().__init__()
        self.use_gat = use_gat
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        if use_gat:
            self.lin = nn.Linear(in_dim, out_dim * n_heads, bias=False)
            self.att_src = nn.Parameter(torch.empty(n_heads, out_dim))
            self.att_dst = nn.Parameter(torch.empty(n_heads, out_dim))
            nn.init.xavier_uniform_(self.att_src)
            nn.init.xavier_uniform_(self.att_dst)
            self.out_proj = nn.Linear(out_dim * n_heads, out_dim)
        else:
            self.lin_self = nn.Linear(in_dim, out_dim)
            self.lin_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, F) or (B, N, F); adj: (N, N) non-negative weights with self-loops
        if x.dim() == 3:
            # (B, N, F)
            x_in = x.unsqueeze(1)
            squeeze = True
        else:
            x_in = x
            squeeze = False
        b, t, n, _ = x_in.shape
        # Ensure self-loops
        adj = adj.clone()
        eye = torch.eye(n, device=adj.device, dtype=adj.dtype)
        adj = adj + eye
        deg = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        adj_norm = adj / deg

        if self.use_gat:
            h = self.lin(x_in).view(b, t, n, self.n_heads, -1)  # (B,T,N,H,D)
            # Attention scores e_ij
            e_i = (h * self.att_src.view(1, 1, 1, self.n_heads, -1)).sum(-1)  # (B,T,N,H)
            e_j = (h * self.att_dst.view(1, 1, 1, self.n_heads, -1)).sum(-1)
            logits = F.leaky_relu(e_i.unsqueeze(3) + e_j.unsqueeze(2), 0.2)  # (B,T,N,N,H)
            mask = (adj > 0).view(1, 1, n, n, 1)
            logits = logits.masked_fill(~mask, -1e9)
            alpha = torch.softmax(logits, dim=3)
            alpha = torch.nan_to_num(alpha, nan=0.0)
            alpha = self.dropout(alpha)
            # Aggregate over neighbours j: sum_j alpha_ij * h_j
            agg = torch.einsum("btijh,btjhd->btihd", alpha, h)
            out = agg.reshape(b, t, n, -1)
            out = F.relu(self.out_proj(out))
        else:
            # GraphSAGE mean
            flat = x_in.reshape(b * t, n, -1)
            neigh = torch.bmm(adj_norm.unsqueeze(0).expand(b * t, -1, -1), flat)
            neigh = neigh.view(b, t, n, -1)
            out = F.relu(self.lin_self(x_in) + self.lin_neigh(neigh))
        return out.squeeze(1) if squeeze else out


class MaskedSpatioTemporalGraphNetwork(nn.Module):
    """Masked ST-GNN with GRU temporal encoder and heteroscedastic head."""

    def __init__(
        self,
        n_sensors: int,
        *,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        history_window: int = 8,
        use_gat: bool = False,
        n_aux_features: int = 4,
    ) -> None:
        super().__init__()
        self.n_sensors = n_sensors
        self.history_window = history_window
        self.sensor_emb = nn.Embedding(n_sensors, hidden_dim)
        self.time_emb = nn.Linear(4, hidden_dim)
        self.value_proj = nn.Linear(2 + n_aux_features, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.graph_layers = nn.ModuleList(
            [
                GraphMessageLayer(
                    hidden_dim,
                    hidden_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                    use_gat=use_gat,
                )
                for _ in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.logvar_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        values: torch.Tensor,
        input_mask: torch.Tensor,
        aoi: torch.Tensor,
        time_feats: torch.Tensor,
        adj: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        values: (B, T, N) server-visible values with NaN/0 for hidden positions
        input_mask: (B, T, N) True where value is available to the model
        aoi: (B, T, N) age of information
        time_feats: (B, T, 4) sin/cos hour/dow
        adj: (N, N)
        """
        b, t, n = values.shape
        device = values.device
        sensor_ids = torch.arange(n, device=device).unsqueeze(0).expand(b * t, n)

        val = values.clone()
        # Normalize CO2-scale inputs for stability (approx ppm / 1000)
        val = torch.where(input_mask, val / 1000.0, torch.zeros_like(val))
        aoi_f = aoi / aoi.new_tensor(float(max(self.history_window, 1)))
        feat = torch.stack([val, input_mask.float(), aoi_f.clamp(0, 5), torch.log1p(aoi)], dim=-1)
        if feat.shape[-1] < self.value_proj.in_features:
            pad = self.value_proj.in_features - feat.shape[-1]
            feat = F.pad(feat, (0, pad))
        elif feat.shape[-1] > self.value_proj.in_features:
            feat = feat[..., : self.value_proj.in_features]

        x = self.value_proj(feat)
        x = x + self.sensor_emb(sensor_ids).view(b, t, n, -1)
        x = x + self.time_emb(time_feats).unsqueeze(2)
        x = self.dropout(F.relu(x))

        seq = x.reshape(b * n, t, -1)
        seq, _ = self.gru(seq)
        x = seq.view(b, n, t, -1).transpose(1, 2)

        for layer in self.graph_layers:
            x = layer(x, adj) + x

        h = x[:, -1]
        residual = self.mean_head(h).squeeze(-1) * 100.0
        # Causal LOCF from last visible server value in the window
        locf = torch.zeros(b, n, device=device)
        for ti in range(t):
            locf = torch.where(input_mask[:, ti], values[:, ti], locf)
        mean = locf + residual
        log_var = self.logvar_head(h).squeeze(-1).clamp(-4.0, 6.0)
        return mean, log_var

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        *,
        adj: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        mean, log_var = self.forward(
            batch["values"],
            batch["input_mask"],
            batch["aoi"],
            batch["time_feats"],
            adj,
        )
        target = batch["target"][:, -1]
        eval_mask = batch["eval_mask"][:, -1]
        nll = gaussian_nll_masked(mean, log_var, target, eval_mask)
        if eval_mask.any():
            mae_loss = (mean[eval_mask] - target[eval_mask]).abs().mean()
        else:
            mae_loss = mean.new_tensor(0.0)
        loss = mae_loss + 0.1 * nll
        with torch.no_grad():
            mae = float(mae_loss.detach().cpu()) if eval_mask.any() else 0.0
        return loss, {"loss": float(loss.detach().cpu()), "mae": mae}


class ReconstructionDataset(Dataset):
    """Sliding-window dataset over wide CO2 matrix with causal server-visible inputs."""

    def __init__(
        self,
        wide: np.ndarray,
        observed: np.ndarray,
        time_feats: np.ndarray,
        *,
        history_window: int = 8,
        mask_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        seed: int = 42,
    ) -> None:
        self.wide = wide.astype(np.float32)
        self.observed = observed.astype(bool)
        self.time_feats = time_feats.astype(np.float32)
        self.history_window = history_window
        self.mask_fn = mask_fn
        self.seed = seed
        self.indices: list[int] = []
        t = wide.shape[0]
        for i in range(history_window - 1, t):
            if self.observed[i].any():
                self.indices.append(i)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        t = self.indices[idx]
        w = self.history_window
        sl = slice(t - w + 1, t + 1)
        gt = self.wide[sl].copy()
        obs = self.observed[sl].copy()
        holdout = self.mask_fn(obs) if self.mask_fn else np.zeros_like(obs, dtype=bool)
        holdout = holdout & obs

        # Server-visible: hide held-out AND naturally missing
        input_mask = obs & ~holdout
        server_values = np.where(input_mask, gt, np.nan)
        server_values = np.nan_to_num(server_values, nan=0.0)

        aoi = np.zeros_like(gt, dtype=np.float32)
        last_seen = np.full(gt.shape[1], -1, dtype=int)
        for ti in range(w):
            for si in range(gt.shape[1]):
                if input_mask[ti, si]:
                    last_seen[si] = ti
                    aoi[ti, si] = 0.0
                elif last_seen[si] >= 0:
                    aoi[ti, si] = float(ti - last_seen[si])
                else:
                    aoi[ti, si] = float(ti + 1)

        return {
            "values": torch.from_numpy(server_values),
            "input_mask": torch.from_numpy(input_mask),
            "aoi": torch.from_numpy(aoi),
            "time_feats": torch.from_numpy(self.time_feats[sl]),
            "target": torch.from_numpy(np.nan_to_num(gt, nan=0.0)),
            "eval_mask": torch.from_numpy(holdout),
            "observed": torch.from_numpy(obs),
        }


def _load_graph_data(
    split: str = "train",
    max_sensors: int | None = None,
) -> dict[str, Any]:
    root = repo_root()
    wide_path = root / "data" / "processed" / "co2_wide_observed.parquet"
    panel_path = root / "data" / "processed" / "co2_panel_15min.parquet"
    graph_dir = root / "results" / "graphs"
    if not wide_path.exists():
        raise FileNotFoundError("Run scripts/02_preprocess.py first.")
    wide = pd.read_parquet(wide_path)
    panel = pd.read_parquet(panel_path, columns=["slot", "deveui", "split", "observed", "floor", "hour_sin", "hour_cos", "dow_sin", "dow_cos"])
    slots = wide.index
    slot_split = panel.drop_duplicates("slot").set_index("slot")["split"]
    split_mask = slot_split.reindex(slots) == split
    node_order_path = graph_dir / "node_order.json"
    if node_order_path.exists():
        node_order = json.loads(node_order_path.read_text(encoding="utf-8"))["node_order"]
    else:
        node_order = list(wide.columns.astype(str))
    cols = [c for c in node_order if c in wide.columns]
    if max_sensors is not None and len(cols) > max_sensors:
        cols = cols[:max_sensors]
    wide = wide[cols].reindex(slots)
    obs_mask = (
        panel.pivot_table(index="slot", columns="deveui", values="observed", aggfunc="max")
        .reindex(index=slots, columns=cols)
        .fillna(0)
        .astype(bool)
        .to_numpy()
    )
    cal = panel.drop_duplicates("slot").set_index("slot").reindex(slots)
    time_feats = np.stack(
        [
            cal["hour_sin"].fillna(0).to_numpy(dtype=np.float32),
            cal["hour_cos"].fillna(0).to_numpy(dtype=np.float32),
            cal["dow_sin"].fillna(0).to_numpy(dtype=np.float32),
            cal["dow_cos"].fillna(0).to_numpy(dtype=np.float32),
        ],
        axis=-1,
    )
    floors = (
        panel.drop_duplicates("deveui")
        .set_index("deveui")
        .reindex(cols)["floor"]
        .fillna(-1)
        .to_numpy()
    )
    adj_path = graph_dir / "adjacency_hybrid.npy"
    if adj_path.exists():
        A = np.load(adj_path)
        idx_map = {n: i for i, n in enumerate(json.loads(node_order_path.read_text())["node_order"])}
        sel = [idx_map[c] for c in cols]
        A = A[np.ix_(sel, sel)].astype(np.float32)
    else:
        A = np.eye(len(cols), dtype=np.float32)
    return {
        "wide": wide.to_numpy(dtype=np.float32),
        "observed": obs_mask,
        "time_feats": time_feats,
        "split_mask": split_mask.fillna(False).to_numpy(),
        "adjacency": A,
        "node_order": cols,
        "floors": floors,
    }


def train_reconstruction_model(
    cfg: dict[str, Any] | None = None,
    *,
    max_sensors: int | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    root = repo_root()
    cfg = cfg or load_yaml(root / "configs" / "reconstruction.yaml")
    set_seed(int(cfg.get("seed", 42)))
    neural = cfg.get("neural", {})
    max_sensors = max_sensors if max_sensors is not None else cfg.get("max_sensors")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("[recon] CUDA unavailable; falling back to CPU")
        device = "cpu"
    print(f"[recon] device={device} max_sensors={max_sensors}")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    train_data = _load_graph_data("train", max_sensors=max_sensors)
    val_data = _load_graph_data("val", max_sensors=max_sensors)
    adj = torch.from_numpy(train_data["adjacency"]).to(device)
    n_sensors = len(train_data["node_order"])
    model = MaskedSpatioTemporalGraphNetwork(
        n_sensors,
        hidden_dim=int(neural.get("hidden_dim", 64)),
        n_layers=int(neural.get("n_layers", 2)),
        n_heads=int(neural.get("n_heads", 4)),
        dropout=float(neural.get("dropout", 0.1)),
        history_window=int(cfg.get("history_window", 8)),
    ).to(device)

    schemes = cfg.get("mask_schemes", ["random"])
    rates = cfg.get("mask_rates", [0.2])
    history = int(cfg.get("history_window", 8))

    def make_loader(data: dict[str, Any], split_mask: np.ndarray, train: bool) -> DataLoader:
        wide = data["wide"][split_mask]
        obs = data["observed"][split_mask]
        tf = data["time_feats"][split_mask]
        scheme = schemes[0]
        rate = rates[0]
        seed = int(cfg.get("seed", 42))

        def mask_fn(obs_arr: np.ndarray) -> np.ndarray:
            return apply_mask_scheme(
                obs_arr,
                scheme,
                rate=rate,
                seed=seed,
                sensor_floors=data["floors"],
            )

        ds = ReconstructionDataset(
            wide,
            obs,
            tf,
            history_window=history,
            mask_fn=mask_fn if train else lambda o: apply_mask_scheme(o, scheme, rate=rate, seed=seed + 1),
        )
        return DataLoader(
            ds,
            batch_size=int(neural.get("batch_size", 64)),
            shuffle=train,
            drop_last=False,
            pin_memory=device.startswith("cuda"),
            num_workers=0,
        )

    train_loader = make_loader(train_data, train_data["split_mask"], True)
    val_loader = make_loader(val_data, val_data["split_mask"], False)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(neural.get("lr", 1e-3)),
        weight_decay=float(neural.get("weight_decay", 1e-5)),
    )
    best_val = float("inf")
    patience = int(neural.get("early_stopping_patience", 8))
    bad = 0
    out_dir = ensure_dir(root / "results" / "models" / "reconstruction")
    metrics_log: list[dict[str, Any]] = []

    for epoch in range(int(neural.get("epochs", 40))):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            loss, stats = model.training_step(batch, adj=adj)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(neural.get("grad_clip", 1.0)))
            opt.step()
            train_loss += stats["loss"]
            n_batches += 1
        train_loss /= max(n_batches, 1)

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                loss, stats = model.training_step(batch, adj=adj)
                val_loss += stats["loss"]
                val_batches += 1
        val_loss /= max(val_batches, 1)
        metrics_log.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[recon] epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            bad = 0
            ckpt = {
                "model_state": model.state_dict(),
                "node_order": train_data["node_order"],
                "cfg": cfg,
                "adjacency": train_data["adjacency"],
            }
            torch.save(ckpt, out_dir / "best_model.pt")
        else:
            bad += 1
            if bad >= patience:
                print("[recon] Early stopping.")
                break

    save_json({"metrics": metrics_log, "best_val_loss": best_val, "device": device}, out_dir / "train_metrics.json")
    return {
        "best_val_loss": best_val,
        "checkpoint": str(out_dir / "best_model.pt"),
        "device": device,
    }


def load_reconstruction_model(
    checkpoint: str | Path | None = None,
    device: str | None = None,
) -> tuple[MaskedSpatioTemporalGraphNetwork, dict[str, Any]]:
    root = repo_root()
    path = Path(checkpoint) if checkpoint else root / "results" / "models" / "reconstruction" / "best_model.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", {})
    neural = cfg.get("neural", {})
    n_sensors = len(ckpt["node_order"])
    model = MaskedSpatioTemporalGraphNetwork(
        n_sensors,
        hidden_dim=int(neural.get("hidden_dim", 64)),
        n_layers=int(neural.get("n_layers", 2)),
        n_heads=int(neural.get("n_heads", 4)),
        dropout=float(neural.get("dropout", 0.1)),
        history_window=int(cfg.get("history_window", 8)),
    )
    model.load_state_dict(ckpt["model_state"])
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    meta = {
        "node_order": ckpt["node_order"],
        "adjacency": ckpt.get("adjacency"),
        "cfg": cfg,
        "device": device,
    }
    return model, meta


@torch.no_grad()
def predict_reconstruction(
    model: MaskedSpatioTemporalGraphNetwork,
    server_values: np.ndarray,
    input_mask: np.ndarray,
    aoi: np.ndarray,
    time_feats: np.ndarray,
    adjacency: np.ndarray,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Run model on server-visible inputs only."""
    model.eval()
    t, n = server_values.shape[-2], server_values.shape[-1]
    w = model.history_window
    if server_values.ndim == 2:
        server_values = server_values[None]
        input_mask = input_mask[None]
        aoi = aoi[None]
        time_feats = time_feats[None]
    b = server_values.shape[0]
    means = []
    logvars = []
    adj = torch.from_numpy(adjacency).float().to(device)
    for start in range(0, t - w + 1):
        sl = slice(start, start + w)
        batch = {
            "values": torch.from_numpy(server_values[:, sl]).float().to(device),
            "input_mask": torch.from_numpy(input_mask[:, sl]).to(device),
            "aoi": torch.from_numpy(aoi[:, sl]).float().to(device),
            "time_feats": torch.from_numpy(time_feats[:, sl]).float().to(device),
        }
        mu, lv = model(
            batch["values"],
            batch["input_mask"],
            batch["aoi"],
            batch["time_feats"],
            adj,
        )
        means.append(mu.cpu().numpy())
        logvars.append(lv.cpu().numpy())
    if not means:
        return np.zeros((b, n)), np.zeros((b, n))
    return np.stack(means, axis=1), np.stack(logvars, axis=1)
