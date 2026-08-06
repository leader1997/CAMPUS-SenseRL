"""Safety shield: override SKIP with TRANSMIT under reliability constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from campus_senserl.environment.communication_model import SKIP, TRANSMIT


@dataclass
class ShieldDecision:
    original_action: int
    final_action: int
    overridden: bool
    reasons: list[str]


@dataclass
class SafetyShield:
    """Force TRANSMIT when AoI, uncertainty, CO2, rate, or neighbor disagreement exceed limits."""

    enabled: bool = True
    aoi_exceeds: float = 8.0
    uncertainty_exceeds: float = 2.0
    co2_exceeds: float = 1200.0
    co2_rate_exceeds: float = 150.0
    neighbor_disagreement_exceeds: float = 200.0

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "SafetyShield":
        shield = cfg.get("safety_shield", cfg)
        force = shield.get("force_transmit_if", {})
        return cls(
            enabled=bool(shield.get("enabled", True)),
            aoi_exceeds=float(force.get("aoi_exceeds", 8)),
            uncertainty_exceeds=float(force.get("uncertainty_exceeds", 2.0)),
            co2_exceeds=float(force.get("co2_exceeds", 1200)),
            co2_rate_exceeds=float(force.get("co2_rate_exceeds", 150)),
            neighbor_disagreement_exceeds=float(
                force.get("neighbor_disagreement_exceeds", 200)
            ),
        )

    def apply(
        self,
        action: int,
        *,
        aoi: float,
        uncertainty: float = 0.0,
        local_co2: float | None = None,
        co2_rate: float = 0.0,
        neighbor_disagreement: float = 0.0,
        local_available: bool = True,
    ) -> ShieldDecision:
        """Return possibly overridden action and audit trail."""
        action = int(action)
        reasons: list[str] = []
        if not self.enabled or action == TRANSMIT:
            return ShieldDecision(action, action, False, reasons)

        if aoi >= self.aoi_exceeds:
            reasons.append("aoi")
        if uncertainty >= self.uncertainty_exceeds:
            reasons.append("uncertainty")
        if local_available and local_co2 is not None and np.isfinite(local_co2):
            if local_co2 >= self.co2_exceeds:
                reasons.append("co2")
        if abs(co2_rate) >= self.co2_rate_exceeds:
            reasons.append("co2_rate")
        if neighbor_disagreement >= self.neighbor_disagreement_exceeds:
            reasons.append("neighbor_disagreement")

        if reasons:
            return ShieldDecision(action, TRANSMIT, True, reasons)
        return ShieldDecision(action, action, False, reasons)

    def apply_vector(
        self,
        actions: np.ndarray,
        *,
        aoi: np.ndarray,
        uncertainty: np.ndarray | None = None,
        local_co2: np.ndarray | None = None,
        co2_rate: np.ndarray | None = None,
        neighbor_disagreement: np.ndarray | None = None,
        local_available: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[ShieldDecision]]:
        n = len(actions)
        uncertainty = np.zeros(n) if uncertainty is None else uncertainty
        co2_rate = np.zeros(n) if co2_rate is None else co2_rate
        neighbor_disagreement = np.zeros(n) if neighbor_disagreement is None else neighbor_disagreement
        local_available = np.ones(n, dtype=bool) if local_available is None else local_available.astype(bool)
        final = np.asarray(actions, dtype=int).copy()
        decisions: list[ShieldDecision] = []
        for i in range(n):
            d = self.apply(
                int(actions[i]),
                aoi=float(aoi[i]),
                uncertainty=float(uncertainty[i]),
                local_co2=None if local_co2 is None else float(local_co2[i]),
                co2_rate=float(co2_rate[i]),
                neighbor_disagreement=float(neighbor_disagreement[i]),
                local_available=bool(local_available[i]),
            )
            final[i] = d.final_action
            decisions.append(d)
        return final, decisions
