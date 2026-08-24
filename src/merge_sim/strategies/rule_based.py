"""Reactive rule-based cooperation: the main-lane vehicle yields once the
ramp vehicle is within a fixed trigger distance AND the projected
time-to-collision drops below a fixed safety threshold. No negotiation, no
utility function, no message exchange — a hand-tuned if/else, matching
what most prior rule-based merging papers implement. Exists in this study
purely as the middle rung between egoistic and negotiation.
"""
from __future__ import annotations

from merge_sim.controller import HighLevelAction
from merge_sim.strategies.base import Strategy

TTC_YIELD_THRESHOLD_S = 3.0
TRIGGER_DISTANCE_M = 40.0
YIELD_SPEED_MARGIN_MPS = 3.0


class RuleBasedStrategy(Strategy):
    name = "rule_based"

    def decide(self, main_lane_state, ramp_state, t: float) -> HighLevelAction:
        if ramp_state.distance_to_merge_point_m > TRIGGER_DISTANCE_M:
            return HighLevelAction(target_speed_mps=main_lane_state.desired_speed_mps)

        closing_speed = main_lane_state.own_speed_mps - ramp_state.ramp_speed_mps
        gap_m = max(ramp_state.gap_to_ramp_m, 0.1)
        ttc = gap_m / closing_speed if closing_speed > 0 else float("inf")

        if ttc < TTC_YIELD_THRESHOLD_S:
            yield_speed = max(0.0, ramp_state.ramp_speed_mps - YIELD_SPEED_MARGIN_MPS)
            return HighLevelAction(target_speed_mps=yield_speed)

        return HighLevelAction(target_speed_mps=main_lane_state.desired_speed_mps)
