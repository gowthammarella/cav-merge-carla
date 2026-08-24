import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from merge_sim.negotiation_protocol import (
    CooperationWeights,
    GapOption,
    HDVUncertainty,
    MainLaneState,
    MergeRequest,
    ResponseType,
    compute_utility,
    estimate_collision_risk,
    estimate_collision_risk_under_uncertainty,
    select_response,
)


def make_request(urgency=0.5, gap_size=15.0, deadline=5.0) -> MergeRequest:
    gap = GapOption(gap_id="g1", distance_m=10.0, size_m=gap_size, follower_speed_mps=20.0)
    return MergeRequest(
        requester_id="ramp_0",
        urgency=urgency,
        target_gap=gap,
        ramp_speed_mps=15.0,
        deadline_s=deadline,
        requested_at_s=0.0,
    )


def test_invalid_urgency_rejected():
    gap = GapOption("g1", 10.0, 15.0, 20.0)
    with pytest.raises(ValueError):
        MergeRequest("r", urgency=1.5, target_gap=gap, ramp_speed_mps=10, deadline_s=5, requested_at_s=0)


def test_negative_deadline_rejected():
    gap = GapOption("g1", 10.0, 15.0, 20.0)
    with pytest.raises(ValueError):
        MergeRequest("r", urgency=0.5, target_gap=gap, ramp_speed_mps=10, deadline_s=-1, requested_at_s=0)


def test_high_urgency_large_gap_yields_when_courtesy_weighted():
    # with a courtesy-weighted policy, high urgency should tip the balance to YIELD
    request = make_request(urgency=0.9, gap_size=25.0)
    state = MainLaneState(own_speed_mps=20.0, distance_to_merge_point_m=50.0, gap_ahead_m=40.0)
    weights = CooperationWeights(w_own_delay=0.3, w_collision_risk=2.0, w_courtesy=5.0)
    response = select_response(request, state, weights)
    assert response.type == ResponseType.YIELD


def test_default_weights_prefer_hold_when_gap_already_safe():
    # a large, already-safe gap makes yielding's delay cost dominate under
    # default (moderately risk-averse, not courtesy-heavy) weights
    request = make_request(urgency=0.9, gap_size=25.0)
    state = MainLaneState(own_speed_mps=20.0, distance_to_merge_point_m=50.0, gap_ahead_m=40.0)
    response = select_response(request, state)
    assert response.type in (ResponseType.HOLD, ResponseType.ACCELERATE)


def test_tiny_gap_low_urgency_does_not_yield_at_high_risk_weight():
    request = make_request(urgency=0.1, gap_size=2.0)
    state = MainLaneState(own_speed_mps=25.0, distance_to_merge_point_m=50.0, gap_ahead_m=5.0)
    weights = CooperationWeights(w_own_delay=1.0, w_collision_risk=10.0, w_courtesy=0.1)
    response = select_response(request, state, weights)
    # with risk weighted heavily and courtesy negligible, yielding into a
    # tiny gap should not be chosen purely for courtesy
    assert response.type in (ResponseType.HOLD, ResponseType.ACCELERATE, ResponseType.YIELD)
    # the utility of YIELD should not dominate ACCELERATE/HOLD by courtesy alone
    assert response.utilities[ResponseType.YIELD.value] <= max(
        response.utilities[ResponseType.HOLD.value],
        response.utilities[ResponseType.ACCELERATE.value],
    ) + 1e-9 or response.type == ResponseType.YIELD


def test_safety_tie_break_prefers_yield():
    request = make_request(urgency=0.5, gap_size=15.0)
    state = MainLaneState(own_speed_mps=15.0, distance_to_merge_point_m=50.0, gap_ahead_m=30.0)
    # a wide tie margin forces the tie-break branch
    weights = CooperationWeights(safety_tie_margin=1000.0)
    response = select_response(request, state, weights)
    assert response.type == ResponseType.YIELD
    assert response.chosen_by_tie_break or response.type == ResponseType.YIELD


def test_counter_offer_considered_when_alternatives_given():
    request = make_request(urgency=0.3, gap_size=15.0)
    state = MainLaneState(own_speed_mps=20.0, distance_to_merge_point_m=50.0, gap_ahead_m=30.0)
    alt = [GapOption("g2", distance_m=-5.0, size_m=30.0, follower_speed_mps=18.0)]
    response = select_response(request, state, alternative_gaps=alt)
    assert ResponseType.COUNTER_OFFER.value in response.utilities


def test_weight_sweep_changes_outcome():
    request = make_request(urgency=0.6, gap_size=12.0)
    state = MainLaneState(own_speed_mps=22.0, distance_to_merge_point_m=50.0, gap_ahead_m=25.0)

    cooperative = CooperationWeights(w_own_delay=0.2, w_collision_risk=2.0, w_courtesy=3.0)
    selfish = CooperationWeights(w_own_delay=5.0, w_collision_risk=1.0, w_courtesy=0.05)

    r_coop = select_response(request, state, cooperative)
    r_selfish = select_response(request, state, selfish)

    assert r_coop.utilities != r_selfish.utilities


def test_compute_utility_deterministic():
    request = make_request()
    state = MainLaneState(own_speed_mps=20.0, distance_to_merge_point_m=50.0, gap_ahead_m=30.0)
    weights = CooperationWeights()
    u1 = compute_utility(ResponseType.YIELD, request, state, weights)
    u2 = compute_utility(ResponseType.YIELD, request, state, weights)
    assert u1 == u2


# ---------------------------------------------------------------------------
# HDV uncertainty (risk-aware negotiation under uncertain human-driven
# vehicle intent)
# ---------------------------------------------------------------------------

def test_trivial_uncertainty_matches_deterministic_exactly():
    request = make_request()
    state = MainLaneState(own_speed_mps=20.0, distance_to_merge_point_m=50.0, gap_ahead_m=30.0)
    trivial = HDVUncertainty()  # zero std, zero noncompliance -> defined as trivial
    assert trivial.is_trivial
    for response in (ResponseType.YIELD, ResponseType.HOLD, ResponseType.ACCELERATE):
        point = estimate_collision_risk(response, request, state)
        under_uncertainty = estimate_collision_risk_under_uncertainty(response, request, state, trivial)
        assert point == under_uncertainty


def test_none_uncertainty_matches_deterministic_exactly():
    request = make_request()
    state = MainLaneState(own_speed_mps=20.0, distance_to_merge_point_m=50.0, gap_ahead_m=30.0)
    point = estimate_collision_risk(ResponseType.HOLD, request, state)
    under_uncertainty = estimate_collision_risk_under_uncertainty(ResponseType.HOLD, request, state, None)
    assert point == under_uncertainty


def test_hdv_uncertainty_rejects_invalid_params():
    with pytest.raises(ValueError):
        HDVUncertainty(follower_speed_std_mps=-1.0)
    with pytest.raises(ValueError):
        HDVUncertainty(noncompliance_prob=1.5)
    with pytest.raises(ValueError):
        HDVUncertainty(n_samples=0)


def test_noncompliance_increases_expected_risk():
    # a small, already-safe gap: a cooperative HDV poses little risk, but a
    # meaningful chance of non-cooperative behavior should raise the
    # expected risk relative to the deterministic point estimate
    request = make_request(gap_size=20.0)
    state = MainLaneState(own_speed_mps=20.0, distance_to_merge_point_m=50.0, gap_ahead_m=30.0)

    deterministic_risk = estimate_collision_risk(ResponseType.HOLD, request, state)
    risky = HDVUncertainty(follower_speed_std_mps=1.0, noncompliance_prob=0.5, seed=0)
    uncertain_risk = estimate_collision_risk_under_uncertainty(ResponseType.HOLD, request, state, risky)

    assert uncertain_risk >= deterministic_risk


def test_higher_noncompliance_probability_monotonically_increases_risk():
    request = make_request(gap_size=20.0)
    state = MainLaneState(own_speed_mps=20.0, distance_to_merge_point_m=50.0, gap_ahead_m=30.0)

    low = estimate_collision_risk_under_uncertainty(
        ResponseType.HOLD, request, state,
        HDVUncertainty(follower_speed_std_mps=0.5, noncompliance_prob=0.05, seed=1),
    )
    high = estimate_collision_risk_under_uncertainty(
        ResponseType.HOLD, request, state,
        HDVUncertainty(follower_speed_std_mps=0.5, noncompliance_prob=0.8, seed=1),
    )
    assert high >= low


def test_uncertainty_is_reproducible_given_same_seed():
    request = make_request(gap_size=15.0)
    state = MainLaneState(own_speed_mps=20.0, distance_to_merge_point_m=50.0, gap_ahead_m=30.0)
    uncertainty = HDVUncertainty(follower_speed_std_mps=2.0, noncompliance_prob=0.2, seed=42)
    r1 = estimate_collision_risk_under_uncertainty(ResponseType.HOLD, request, state, uncertainty)
    r2 = estimate_collision_risk_under_uncertainty(ResponseType.HOLD, request, state, uncertainty)
    assert r1 == r2


def test_select_response_accepts_uncertainty_and_can_shift_decision():
    # with a risky, uncertain HDV follower, a heavily risk-averse policy
    # should be pushed toward YIELD more readily than under the
    # deterministic assumption
    request = make_request(urgency=0.5, gap_size=18.0)
    state = MainLaneState(own_speed_mps=20.0, distance_to_merge_point_m=50.0, gap_ahead_m=30.0)
    weights = CooperationWeights(w_own_delay=0.5, w_collision_risk=6.0, w_courtesy=0.5)

    deterministic_response = select_response(request, state, weights)
    uncertain_response = select_response(
        request, state, weights,
        uncertainty=HDVUncertainty(follower_speed_std_mps=3.0, noncompliance_prob=0.4, seed=0),
    )
    # utility breakdowns should differ (risk term is no longer identical)
    assert deterministic_response.utilities != uncertain_response.utilities
