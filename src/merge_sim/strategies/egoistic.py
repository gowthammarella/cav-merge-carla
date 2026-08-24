"""Baseline strategy: pure IDM car-following in its own lane. The
main-lane vehicle never reacts to the ramp vehicle's request — the ramp
vehicle must find (or force) its own gap. This is the "no cooperation"
reference point every other strategy is measured against.
"""
from __future__ import annotations

from merge_sim.controller import HighLevelAction
from merge_sim.strategies.base import Strategy


class EgoisticStrategy(Strategy):
    name = "egoistic"

    def decide(self, main_lane_state, ramp_state, t: float) -> HighLevelAction:
        return HighLevelAction(target_speed_mps=main_lane_state.desired_speed_mps)
