"""Strategy interface. Every strategy maps the current per-tick
observation to a single `HighLevelAction` for the main-lane decision
vehicle — never raw throttle/steer (see controller.py for why).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from merge_sim.controller import HighLevelAction


class Strategy(ABC):
    name: str = "base"

    @abstractmethod
    def decide(self, main_lane_state, ramp_state, t: float) -> HighLevelAction:
        """`main_lane_state`: MainLaneObservableState (scenario.py).
        `ramp_state`: RampObservableState (scenario.py). `t`: episode time (s).
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Called once per episode before the first `decide()` call.
        Override for strategies that carry state across ticks."""
        return None
