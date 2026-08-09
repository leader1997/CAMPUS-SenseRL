"""PDR must always lie in [0, 1] and use post-shield requests."""

from __future__ import annotations

import pytest

from campus_senserl.evaluation.pdr import count_requested_transmits, packet_delivery_ratio


def test_pdr_in_unit_interval():
    assert packet_delivery_ratio(10, 10) == 1.0
    assert packet_delivery_ratio(10, 0) == 0.0
    assert packet_delivery_ratio(10, 5) == 0.5
    assert packet_delivery_ratio(0, 0) == 0.0


def test_pdr_rejects_delivered_gt_requested():
    with pytest.raises(ValueError):
        packet_delivery_ratio(5, 6)


def test_pdr_rejects_negative():
    with pytest.raises(ValueError):
        packet_delivery_ratio(-1, 0)


def test_count_requested_transmits():
    import numpy as np

    assert count_requested_transmits(np.array([1, 0, 1, 1]), transmit_code=1) == 3
