"""Multi-objective reward for communication scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from campus_senserl.environment.communication_model import TRANSMIT, CommunicationCostModel


@dataclass
class RewardComputer:
    w_tx: float = 1.0
    w_err: float = 1.0
    w_aoi: float = 0.5
    w_unc: float = 0.5
    w_miss: float = 5.0
    w_event: float = 1.0
    clip: float = 10.0
    normalize_components: bool = True

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "RewardComputer":
        r = cfg.get("reward", cfg)
        return cls(
            w_tx=float(r.get("w_tx", 1.0)),
            w_err=float(r.get("w_err", 1.0)),
            w_aoi=float(r.get("w_aoi", 0.5)),
            w_unc=float(r.get("w_unc", 0.5)),
            w_miss=float(r.get("w_miss", 5.0)),
            w_event=float(r.get("w_event", 1.0)),
            clip=float(r.get("clip", 10.0)),
            normalize_components=bool(r.get("normalize_components", True)),
        )

    def compute_step(
        self,
        *,
        actions: np.ndarray,
        ground_truth: np.ndarray,
        reconstruction: np.ndarray,
        uncertainty: np.ndarray,
        aoi: np.ndarray,
        local_available: np.ndarray,
        events: np.ndarray,
        missed_event_mask: np.ndarray,
        comm_model: CommunicationCostModel | None = None,
    ) -> tuple[float, dict[str, float]]:
        comm_model = comm_model or CommunicationCostModel()
        actions = np.asarray(actions, dtype=int)
        gt = np.asarray(ground_truth, dtype=float)
        recon = np.asarray(reconstruction, dtype=float)
        unc = np.asarray(uncertainty, dtype=float)
        aoi = np.asarray(aoi, dtype=float)
        local_available = np.asarray(local_available, dtype=bool)
        events = np.asarray(events, dtype=bool)
        missed = np.asarray(missed_event_mask, dtype=bool)

        m = local_available & np.isfinite(gt) & np.isfinite(recon)
        if m.any():
            err = np.abs(gt[m] - recon[m]) / 400.0
            err_term = float(err.mean())
        else:
            err_term = 0.0

        tx_cost = float(np.mean([comm_model.cost(a) for a in actions[local_available]])) if local_available.any() else 0.0
        aoi_term = float(np.mean(aoi[local_available] / max(aoi.max(), 1.0))) if local_available.any() else 0.0
        unc_term = float(np.mean(unc[local_available] / max(unc.max(), 1.0))) if local_available.any() else 0.0
        miss_term = float(missed.sum()) / max(local_available.sum(), 1)
        event_term = float(events[local_available].sum()) / max(local_available.sum(), 1)

        parts = {
            "tx": -self.w_tx * tx_cost,
            "err": -self.w_err * err_term,
            "aoi": -self.w_aoi * aoi_term,
            "unc": -self.w_unc * unc_term,
            "miss": -self.w_miss * miss_term,
            "event": -self.w_event * event_term,
        }
        total = sum(parts.values())
        if self.normalize_components:
            total = total / max(
                self.w_tx + self.w_err + self.w_aoi + self.w_unc + self.w_miss + self.w_event,
                1e-6,
            )
        total = float(np.clip(total, -self.clip, self.clip))
        return total, parts


def compute_reward(
    *,
    actions: np.ndarray,
    ground_truth: np.ndarray,
    reconstruction: np.ndarray,
    uncertainty: np.ndarray,
    aoi: np.ndarray,
    local_available: np.ndarray,
    events: np.ndarray,
    missed_event_mask: np.ndarray,
    cfg: dict[str, Any] | None = None,
    comm_model: CommunicationCostModel | None = None,
) -> tuple[float, dict[str, float]]:
    computer = RewardComputer.from_config(cfg or {})
    return computer.compute_step(
        actions=actions,
        ground_truth=ground_truth,
        reconstruction=reconstruction,
        uncertainty=uncertainty,
        aoi=aoi,
        local_available=local_available,
        events=events,
        missed_event_mask=missed_event_mask,
        comm_model=comm_model,
    )
