"""Communication cost model for SKIP vs TRANSMIT actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Action constants
SKIP = 0
TRANSMIT = 1


@dataclass
class CommunicationCostModel:
    """Proxy communication cost model.

    Default costs are dimensionless proxies (transmit=1, skip=0).
    Optional estimated energy uses LoRa parameters when available and is
    clearly labeled as an estimate, not measured ground truth.
    """

    transmit_cost: float = 1.0
    skip_cost: float = 0.0
    use_estimated_energy: bool = False
    # Estimated LoRa airtime/energy parameters (placeholders; tune from datasheet)
    estimated_tx_energy_mj: float = 50.0
    estimated_rx_energy_mj: float = 10.0
    payload_bytes: int = 12

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "CommunicationCostModel":
        comm = cfg.get("communication", cfg)
        return cls(
            transmit_cost=float(comm.get("transmit_cost", 1.0)),
            skip_cost=float(comm.get("skip_cost", 0.0)),
            use_estimated_energy=bool(comm.get("use_estimated_energy", False)),
        )

    def cost(self, action: int) -> float:
        if int(action) == TRANSMIT:
            return float(self.transmit_cost)
        return float(self.skip_cost)

    def estimated_energy_mj(self, action: int) -> float | None:
        """Return estimated energy in millijoules if enabled, else None."""
        if not self.use_estimated_energy:
            return None
        if int(action) == TRANSMIT:
            # Simple linear model: base TX + per-byte overhead (estimate only)
            return float(self.estimated_tx_energy_mj + 0.5 * self.payload_bytes)
        return float(self.estimated_rx_energy_mj * 0.1)


def action_cost(
    actions: np.ndarray,
    model: CommunicationCostModel | None = None,
) -> np.ndarray:
    """Vectorized action costs."""
    model = model or CommunicationCostModel()
    actions = np.asarray(actions, dtype=int)
    costs = np.where(actions == TRANSMIT, model.transmit_cost, model.skip_cost)
    return costs.astype(float)
