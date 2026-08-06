"""Reinforcement learning components for CAMPUS-SenseRL."""

from campus_senserl.rl.fixed_policies import FixedIntervalPolicy, make_fixed_policies
from campus_senserl.rl.heuristic_policies import (
    AoIThresholdPolicy,
    CO2ThresholdPolicy,
    ChangeThresholdPolicy,
    UncertaintyThresholdPolicy,
    make_heuristic_policies,
)
from campus_senserl.rl.info_value import InfoValueWeights, information_value
from campus_senserl.rl.reward import RewardComputer, compute_reward

__all__ = [
    "FixedIntervalPolicy",
    "make_fixed_policies",
    "AoIThresholdPolicy",
    "CO2ThresholdPolicy",
    "ChangeThresholdPolicy",
    "UncertaintyThresholdPolicy",
    "make_heuristic_policies",
    "InfoValueWeights",
    "information_value",
    "RewardComputer",
    "compute_reward",
]


def __getattr__(name: str):
    if name == "PPOTrainer":
        from campus_senserl.rl.ppo import PPOTrainer

        return PPOTrainer
    if name == "MAPPOTrainer":
        from campus_senserl.rl.mappo import MAPPOTrainer

        return MAPPOTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
