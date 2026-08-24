"""Wraps `negotiation_protocol.select_response` as a Strategy: builds a
MergeRequest from the current ramp observation each tick, gets back a
MergeResponse (with a logged utility breakdown), and converts it into a
HighLevelAction for the shared controller. This is the study's central
contribution — see negotiation_protocol.py for the utility model itself.
"""
from __future__ import annotations

from typing import Optional

from merge_sim.controller import HighLevelAction
from merge_sim.negotiation_protocol import (
    CooperationWeights,
    GapOption,
    HDVUncertainty,
    MainLaneState,
    MergeRequest,
    MergeResponse,
    ResponseType,
    select_response,
)
from merge_sim.strategies.base import Strategy

REQUEST_ZONE_M = 60.0
YIELD_SPEED_MARGIN_MPS = 3.0
ACCELERATE_SPEED_FACTOR = 1.1

# Default belief about HDV follower behavior: the follower's speed is known
# only to within a few m/s, and there's a real (if modest) chance it acts
# non-cooperatively rather than holding a predictable gap. These are the
# strategy's own *beliefs*, independent of whatever noise scenario.py's
# background traffic actually injects into the simulation — deliberately
# so, since a real deployed vehicle would also be reasoning under a belief
# model rather than ground truth. Pass an explicit `uncertainty=` (e.g.
# `HDVUncertainty()` for the fully deterministic degenerate case) to run
# the ablation against the non-uncertainty-aware version of this strategy.
DEFAULT_HDV_UNCERTAINTY = HDVUncertainty(
    follower_speed_std_mps=1.5,
    noncompliance_prob=0.1,
    noncompliance_speed_delta_mps=4.0,
)


class NegotiationStrategy(Strategy):
    name = "negotiation"

    def __init__(
        self,
        weights: Optional[CooperationWeights] = None,
        uncertainty: Optional[HDVUncertainty] = DEFAULT_HDV_UNCERTAINTY,
        name: Optional[str] = None,
    ):
        self.weights = weights or CooperationWeights()
        self.uncertainty = uncertainty
        self.last_response: Optional[MergeResponse] = None
        if name is not None:
            self.name = name

    def reset(self) -> None:
        self.last_response = None

    def decide(self, main_lane_state, ramp_state, t: float) -> HighLevelAction:
        if ramp_state.distance_to_merge_point_m > REQUEST_ZONE_M:
            self.last_response = None
            return HighLevelAction(target_speed_mps=main_lane_state.desired_speed_mps)

        urgency = 1.0 - min(1.0, ramp_state.distance_to_merge_point_m / REQUEST_ZONE_M)
        gap = GapOption(
            gap_id="primary",
            distance_m=ramp_state.gap_to_ramp_m,
            size_m=ramp_state.gap_to_ramp_m,
            follower_speed_mps=main_lane_state.own_speed_mps,
        )
        request = MergeRequest(
            requester_id="ramp_0",
            urgency=urgency,
            target_gap=gap,
            ramp_speed_mps=ramp_state.ramp_speed_mps,
            deadline_s=ramp_state.distance_to_merge_point_m / max(ramp_state.ramp_speed_mps, 0.1),
            requested_at_s=t,
        )
        state = MainLaneState(
            own_speed_mps=main_lane_state.own_speed_mps,
            distance_to_merge_point_m=ramp_state.distance_to_merge_point_m,
            gap_ahead_m=main_lane_state.gap_ahead_m,
        )
        response = select_response(request, state, self.weights, uncertainty=self.uncertainty)
        self.last_response = response

        if response.type is ResponseType.YIELD:
            target = max(0.0, ramp_state.ramp_speed_mps - YIELD_SPEED_MARGIN_MPS)
        elif response.type is ResponseType.ACCELERATE:
            target = main_lane_state.desired_speed_mps * ACCELERATE_SPEED_FACTOR
        else:  # HOLD or COUNTER_OFFER
            target = main_lane_state.desired_speed_mps

        return HighLevelAction(target_speed_mps=target)
