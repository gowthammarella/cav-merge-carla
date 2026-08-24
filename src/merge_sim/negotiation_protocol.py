"""Formal negotiation protocol between a ramp (merging) vehicle and the
main-lane vehicle it is requesting a gap from.

This module has no CARLA dependency and is fully unit-testable. It is the
paper's core contribution: instead of an ad-hoc "vehicle yields if X", the
main-lane vehicle evaluates a small, explicit utility function over a fixed
response vocabulary and the weights are exposed so their sensitivity can be
swept as a standalone experiment.

Collision-risk estimation additionally supports reasoning under uncertain
human-driven-vehicle (HDV) intent (`HDVUncertainty` /
`estimate_collision_risk_under_uncertainty`): rather than assuming the
follower vehicle's speed is known exactly, risk is computed as an
expectation over a distribution of plausible follower behavior, including
an explicit probability that the follower behaves non-cooperatively. This
targets the same "CAVs interacting with HDVs of uncertain intentions"
framing used in recent cooperative-merging literature, without requiring a
full joint/optimization-based multi-vehicle controller.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class ResponseType(Enum):
    YIELD = "yield"
    HOLD = "hold"
    ACCELERATE = "accelerate"
    COUNTER_OFFER = "counter_offer"


@dataclass(frozen=True)
class GapOption:
    """One candidate gap in the main lane the ramp vehicle could merge into."""

    gap_id: str
    # signed distance (m) from the ramp vehicle's current position to the
    # gap's midpoint at the moment of the request
    distance_m: float
    # size of the gap (m) between the lead and follow vehicle bounding it
    size_m: float
    # speed (m/s) of the vehicle immediately behind this gap in the main lane
    follower_speed_mps: float


@dataclass(frozen=True)
class MergeRequest:
    """Sent by the ramp vehicle once it enters the negotiation zone."""

    requester_id: str
    urgency: float  # 0..1, rises as the ramp vehicle nears the end of the ramp
    target_gap: GapOption
    ramp_speed_mps: float
    deadline_s: float  # seconds until the ramp vehicle runs out of ramp
    requested_at_s: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.urgency <= 1.0:
            raise ValueError(f"urgency must be in [0, 1], got {self.urgency}")
        if self.deadline_s < 0:
            raise ValueError(f"deadline_s must be >= 0, got {self.deadline_s}")


@dataclass(frozen=True)
class MainLaneState:
    """Kinematic situation of the responding main-lane vehicle."""

    own_speed_mps: float
    distance_to_merge_point_m: float
    gap_ahead_m: float  # distance to the vehicle ahead in the main lane


@dataclass(frozen=True)
class CooperationWeights:
    """Sweepable utility weights. Defaults chosen to be mildly cooperative
    (safety weighted highest) but every field is meant to be varied in the
    weight-sensitivity experiment.
    """

    w_own_delay: float = 1.0
    w_collision_risk: float = 4.0
    w_courtesy: float = 0.5
    # tie-break margin: if the best two responses' utilities are within this
    # margin, prefer the safer one (YIELD) rather than pure argmax
    safety_tie_margin: float = 0.05


@dataclass(frozen=True)
class HDVUncertainty:
    """Models what the main-lane decision-maker does NOT know for certain
    about the human-driven vehicle bounding its target gap: its true
    instantaneous speed (measurement/prediction noise) and the chance it
    behaves non-cooperatively (e.g. an impatient HDV closes the gap instead
    of holding it). Both default to zero, in which case risk estimation is
    bit-for-bit identical to the fully-deterministic point estimate.
    """

    follower_speed_std_mps: float = 0.0
    noncompliance_prob: float = 0.0
    # signed adjustment applied to the sampled follower speed when the
    # non-compliance branch is drawn (positive = HDV unexpectedly speeds up
    # and closes the gap faster than a cooperative driver would)
    noncompliance_speed_delta_mps: float = 4.0
    n_samples: int = 300
    seed: int = 0

    def __post_init__(self) -> None:
        if self.follower_speed_std_mps < 0:
            raise ValueError("follower_speed_std_mps must be >= 0")
        if not 0.0 <= self.noncompliance_prob <= 1.0:
            raise ValueError("noncompliance_prob must be in [0, 1]")
        if self.n_samples < 1:
            raise ValueError("n_samples must be >= 1")

    @property
    def is_trivial(self) -> bool:
        """True when this reduces to the deterministic point estimate."""
        return self.follower_speed_std_mps == 0.0 and self.noncompliance_prob == 0.0


@dataclass(frozen=True)
class MergeResponse:
    """The main-lane vehicle's decision, with the full utility breakdown
    retained for failure-mode analysis and reproducibility.
    """

    type: ResponseType
    counter_gap: Optional[GapOption]
    utilities: dict = field(default_factory=dict)  # ResponseType.value -> utility
    chosen_by_tie_break: bool = False


# ---------------------------------------------------------------------------
# Cost / benefit terms
# ---------------------------------------------------------------------------

def estimate_own_delay_s(response: ResponseType, state: MainLaneState) -> float:
    """Rough time cost (s) the main-lane vehicle incurs by choosing this
    response, relative to doing nothing. YIELD costs the most (it slows to
    open a gap), ACCELERATE costs ~0 (it speeds up to close the gap and
    proceeds), HOLD costs a small amount (it must monitor/maintain), and
    COUNTER_OFFER costs a small negotiation overhead.
    """
    if response is ResponseType.YIELD:
        # decelerating and re-accelerating costs roughly speed / comfortable_decel
        comfortable_decel = 2.0  # m/s^2
        return state.own_speed_mps / comfortable_decel * 0.5
    if response is ResponseType.HOLD:
        return 0.5
    if response is ResponseType.ACCELERATE:
        return 0.0
    if response is ResponseType.COUNTER_OFFER:
        return 1.0
    raise ValueError(f"unknown response type {response}")


def _collision_risk_point(
    response: ResponseType,
    request: MergeRequest,
    state: MainLaneState,
    follower_speed_mps: float,
) -> float:
    """Heuristic collision-risk proxy in [0, 1] for one point estimate of
    the follower's speed. Higher means riskier. Modeled as inversely
    related to the resulting gap size relative to a speed-dependent
    safe-following distance (a simplified time-headway rule).
    """
    safe_headway_s = 1.5
    safe_gap_m = safe_headway_s * max(state.own_speed_mps, request.ramp_speed_mps, follower_speed_mps)
    if safe_gap_m <= 0:
        return 0.0

    if response is ResponseType.YIELD:
        resulting_gap = request.target_gap.size_m * 1.6  # yielding enlarges the gap
    elif response is ResponseType.HOLD:
        resulting_gap = request.target_gap.size_m
    elif response is ResponseType.ACCELERATE:
        resulting_gap = request.target_gap.size_m * 0.5  # closing the gap shrinks it
    elif response is ResponseType.COUNTER_OFFER:
        resulting_gap = request.target_gap.size_m  # neutral, risk deferred to new gap
    else:
        raise ValueError(f"unknown response type {response}")

    deficit = max(0.0, safe_gap_m - resulting_gap)
    risk = 1.0 - math.exp(-deficit / safe_gap_m) if safe_gap_m > 0 else 0.0
    return min(1.0, max(0.0, risk))


def estimate_collision_risk(
    response: ResponseType, request: MergeRequest, state: MainLaneState
) -> float:
    """Deterministic point-estimate risk, using the follower's advertised
    speed exactly as given. This is what `compute_utility` uses when no
    `HDVUncertainty` is supplied — kept as a separate entry point so
    existing callers/tests are unaffected by the uncertainty-aware path.
    """
    return _collision_risk_point(response, request, state, request.target_gap.follower_speed_mps)


def estimate_collision_risk_under_uncertainty(
    response: ResponseType,
    request: MergeRequest,
    state: MainLaneState,
    uncertainty: Optional[HDVUncertainty],
) -> float:
    """Expected collision risk under uncertainty about the HDV follower's
    true speed. With probability `uncertainty.noncompliance_prob`, the
    follower is drawn as behaving non-cooperatively (closing the gap
    faster than expected) rather than merely noisy — this is what lets the
    model distinguish "the sensor is a bit noisy" from "human drivers
    sometimes just don't cooperate", which plain Gaussian noise alone does
    not capture.

    Reduces exactly to `estimate_collision_risk` (bit-for-bit) when
    `uncertainty` is None or trivial, so this is a strict superset of the
    deterministic model, not a replacement.
    """
    if uncertainty is None or uncertainty.is_trivial:
        return estimate_collision_risk(response, request, state)

    base_speed = request.target_gap.follower_speed_mps
    rng = np.random.default_rng(uncertainty.seed)
    noise = rng.normal(0.0, uncertainty.follower_speed_std_mps, size=uncertainty.n_samples)
    noncompliant = rng.random(uncertainty.n_samples) < uncertainty.noncompliance_prob

    sampled_speeds = base_speed + noise
    sampled_speeds = sampled_speeds + noncompliant * uncertainty.noncompliance_speed_delta_mps
    sampled_speeds = np.clip(sampled_speeds, 0.0, None)

    risks = np.array(
        [_collision_risk_point(response, request, state, float(v)) for v in sampled_speeds]
    )
    return float(risks.mean())


def estimate_courtesy_bonus(response: ResponseType, request: MergeRequest) -> float:
    """Reward term scaling with the requester's urgency, so cooperative
    responses matter more when the ramp vehicle is close to running out of
    road. This is what lets courtesy weight trade off against delay/risk in
    the sweep experiment.
    """
    if response is ResponseType.YIELD:
        return request.urgency
    if response is ResponseType.COUNTER_OFFER:
        return 0.5 * request.urgency
    if response is ResponseType.HOLD:
        return 0.0
    if response is ResponseType.ACCELERATE:
        return -request.urgency  # actively unhelpful when urgency is high
    raise ValueError(f"unknown response type {response}")


def compute_utility(
    response: ResponseType,
    request: MergeRequest,
    state: MainLaneState,
    weights: CooperationWeights,
    uncertainty: Optional[HDVUncertainty] = None,
) -> float:
    delay = estimate_own_delay_s(response, state)
    risk = estimate_collision_risk_under_uncertainty(response, request, state, uncertainty)
    courtesy = estimate_courtesy_bonus(response, request)
    return (
        -weights.w_own_delay * delay
        - weights.w_collision_risk * risk
        + weights.w_courtesy * courtesy
    )


# ---------------------------------------------------------------------------
# Response selection
# ---------------------------------------------------------------------------

_CANDIDATE_RESPONSES = (
    ResponseType.YIELD,
    ResponseType.HOLD,
    ResponseType.ACCELERATE,
)


def select_response(
    request: MergeRequest,
    state: MainLaneState,
    weights: CooperationWeights = CooperationWeights(),
    alternative_gaps: Optional[list[GapOption]] = None,
    uncertainty: Optional[HDVUncertainty] = None,
) -> MergeResponse:
    """Selects the main-lane vehicle's response by maximizing utility over
    the fixed response vocabulary, with a safety-biased tie-break: if the
    top response is within ``safety_tie_margin`` of YIELD's utility, YIELD
    wins even if it is not the strict argmax.

    If ``alternative_gaps`` is supplied and non-empty, COUNTER_OFFER also
    competes, using the best-utility alternative gap.

    If ``uncertainty`` is supplied, collision risk is computed as an
    expectation over plausible HDV follower behavior (see
    `HDVUncertainty`) instead of a single point estimate — this is what
    lets the negotiation strategy account for not knowing a human driver's
    true intent with certainty.
    """
    utilities: dict[str, float] = {
        r.value: compute_utility(r, request, state, weights, uncertainty) for r in _CANDIDATE_RESPONSES
    }

    best_counter_gap: Optional[GapOption] = None
    if alternative_gaps:
        utilities[ResponseType.COUNTER_OFFER.value] = compute_utility(
            ResponseType.COUNTER_OFFER, request, state, weights, uncertainty
        )
        # prefer the alternative gap closest to the requester (cheapest to reach)
        best_counter_gap = min(alternative_gaps, key=lambda g: abs(g.distance_m))

    best_type = ResponseType(max(utilities, key=utilities.get))
    chosen_by_tie_break = False

    yield_utility = utilities.get(ResponseType.YIELD.value)
    if (
        best_type is not ResponseType.YIELD
        and yield_utility is not None
        and utilities[best_type.value] - yield_utility <= weights.safety_tie_margin
    ):
        best_type = ResponseType.YIELD
        chosen_by_tie_break = True

    return MergeResponse(
        type=best_type,
        counter_gap=best_counter_gap if best_type is ResponseType.COUNTER_OFFER else None,
        utilities=utilities,
        chosen_by_tie_break=chosen_by_tie_break,
    )
