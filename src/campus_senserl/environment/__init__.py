"""Trace-driven RL environment and communication primitives."""

from campus_senserl.environment.communication_model import (
    SKIP,
    TRANSMIT,
    CommunicationCostModel,
    action_cost,
)
from campus_senserl.environment.event_detector import EventDetector

__all__ = [
    "SKIP",
    "TRANSMIT",
    "CommunicationCostModel",
    "action_cost",
    "EventDetector",
    "TraceDrivenCampusEnv",
    "load_trace_tensors",
]


def __getattr__(name: str):
    if name in {"TraceDrivenCampusEnv", "load_trace_tensors"}:
        from campus_senserl.environment.trace_environment import (
            TraceDrivenCampusEnv,
            load_trace_tensors,
        )

        return TraceDrivenCampusEnv if name == "TraceDrivenCampusEnv" else load_trace_tensors
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
