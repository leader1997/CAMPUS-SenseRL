"""Safety shield force-transmit conditions."""

from __future__ import annotations

import numpy as np

from campus_senserl.environment.communication_model import SKIP, TRANSMIT
from campus_senserl.environment.safety_shield import SafetyShield


def test_shield_forces_transmit_on_high_aoi():
    shield = SafetyShield(enabled=True, aoi_exceeds=8.0)
    d = shield.apply(SKIP, aoi=8.0, local_available=True, local_co2=600.0)
    assert d.final_action == TRANSMIT
    assert d.overridden
    assert "aoi" in d.reasons


def test_shield_forces_transmit_on_high_uncertainty():
    shield = SafetyShield(enabled=True, uncertainty_exceeds=2.0)
    d = shield.apply(SKIP, aoi=0.0, uncertainty=2.5, local_co2=600.0, local_available=True)
    assert d.final_action == TRANSMIT
    assert "uncertainty" in d.reasons


def test_shield_forces_transmit_on_high_co2():
    shield = SafetyShield(enabled=True, co2_exceeds=1200.0)
    d = shield.apply(SKIP, aoi=0.0, local_co2=1300.0, local_available=True)
    assert d.final_action == TRANSMIT
    assert "co2" in d.reasons


def test_shield_forces_transmit_on_co2_rate():
    shield = SafetyShield(enabled=True, co2_rate_exceeds=150.0)
    d = shield.apply(SKIP, aoi=0.0, co2_rate=200.0, local_co2=800.0, local_available=True)
    assert d.final_action == TRANSMIT
    assert "co2_rate" in d.reasons


def test_shield_forces_transmit_on_neighbor_disagreement():
    shield = SafetyShield(enabled=True, neighbor_disagreement_exceeds=200.0)
    d = shield.apply(
        SKIP,
        aoi=0.0,
        neighbor_disagreement=250.0,
        local_co2=800.0,
        local_available=True,
    )
    assert d.final_action == TRANSMIT
    assert "neighbor_disagreement" in d.reasons


def test_shield_respects_skip_when_safe():
    shield = SafetyShield(enabled=True)
    d = shield.apply(SKIP, aoi=1.0, uncertainty=0.5, local_co2=600.0, local_available=True)
    assert d.final_action == SKIP
    assert not d.overridden


def test_shield_disabled_passes_through():
    shield = SafetyShield(enabled=False)
    d = shield.apply(SKIP, aoi=100.0, local_co2=2000.0, local_available=True)
    assert d.final_action == SKIP
    assert not d.overridden


def test_apply_vector_overrides_per_sensor():
    shield = SafetyShield(enabled=True, aoi_exceeds=5.0)
    actions = np.array([SKIP, SKIP, TRANSMIT])
    aoi = np.array([6.0, 1.0, 0.0])
    final, decisions = shield.apply_vector(actions, aoi=aoi, local_available=np.ones(3, dtype=bool))
    assert final[0] == TRANSMIT
    assert final[1] == SKIP
    assert final[2] == TRANSMIT
    assert decisions[0].overridden
    assert not decisions[1].overridden
