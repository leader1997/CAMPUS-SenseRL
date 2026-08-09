"""Packet delivery ratio helpers.

Requested transmissions are policy TRANSMIT decisions (attempted uplinks).
Delivered transmissions are those that succeed after simulated packet loss.
"""

from __future__ import annotations

import numpy as np


def packet_delivery_ratio(
    n_requested: int | float,
    n_delivered: int | float,
) -> float:
    """Return PDR in [0, 1].

    Parameters
    ----------
    n_requested:
        Policy TRANSMIT decisions (attempted uplinks).
    n_delivered:
        Successfully delivered TRANSMIT outcomes after packet-loss simulation.
    """
    req = float(n_requested)
    deliv = float(n_delivered)
    if req < 0 or deliv < 0:
        raise ValueError("n_requested and n_delivered must be non-negative")
    if deliv > req + 1e-9:
        raise ValueError(
            f"n_delivered ({deliv}) cannot exceed n_requested ({req}); "
            "count requested TX from the policy, delivered after loss."
        )
    if req <= 0:
        return 0.0
    pdr = deliv / req
    return float(np.clip(pdr, 0.0, 1.0))


def count_requested_transmits(final_actions: np.ndarray, transmit_code: int = 1) -> int:
    """Count policy TRANSMIT decisions."""
    a = np.asarray(final_actions)
    return int(np.sum(a == transmit_code))
