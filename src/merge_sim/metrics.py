"""Per-episode metric computation from a recorded trajectory log.

No CARLA dependency: `EpisodeLog` is a plain container of numpy arrays so
this module (and its tests) run anywhere, including a machine with no GPU.
`episode_runner.py` (CARLA-side) is responsible for populating an
`EpisodeLog` from real simulation ticks and handing it to `compute_metrics`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

HARD_BRAKE_THRESHOLD_MPS2 = -3.0  # deceleration below this counts as "hard braking"
DANGEROUS_TTC_S = 1.5  # min TTC below this is a near-miss even without a collision


@dataclass
class EpisodeLog:
    """Recorded trajectory for one (strategy, density, seed) episode.

    All arrays share the same time base `t` (seconds, uniform dt).
    Positions are along-road longitudinal coordinates (m); same axis for
    ramp and main-lane vehicles once the ramp vehicle has entered the lane.
    """

    t: np.ndarray
    ramp_x: np.ndarray
    ramp_v: np.ndarray
    main_x: np.ndarray  # the main-lane "decision" vehicle governed by the strategy
    main_v: np.ndarray
    main_a: np.ndarray
    follower_v: np.ndarray  # vehicle immediately behind the main-lane decision vehicle
    follower_a: np.ndarray
    follower2_v: np.ndarray  # second vehicle back — used for the string-effect metric
    baseline_follower2_speed_mps: float  # free-flow speed of follower2 before interaction
    merge_success: bool
    collision: bool
    merge_time_s: Optional[float]  # time index at which merge completed, if it did
    request_time_s: float  # time the negotiation/decision window opened

    def __post_init__(self) -> None:
        n = len(self.t)
        for name in (
            "ramp_x", "ramp_v", "main_x", "main_v", "main_a",
            "follower_v", "follower_a", "follower2_v",
        ):
            arr = getattr(self, name)
            if len(arr) != n:
                raise ValueError(f"{name} has length {len(arr)}, expected {n}")


def _dt(t: np.ndarray) -> float:
    if len(t) < 2:
        return 0.0
    return float(np.median(np.diff(t)))


def time_to_collision(
    pos_rear: np.ndarray, vel_rear: np.ndarray, pos_front: np.ndarray, vel_front: np.ndarray
) -> np.ndarray:
    """Per-tick TTC (s) between a rear vehicle and the vehicle ahead of it.

    Returns +inf where the rear vehicle is not closing (safe) or already
    past the front vehicle's position (undefined / already merged).
    """
    gap = pos_front - pos_rear
    closing_speed = vel_rear - vel_front
    ttc = np.full_like(gap, np.inf, dtype=float)
    closing_mask = (closing_speed > 1e-6) & (gap > 0)
    ttc[closing_mask] = gap[closing_mask] / closing_speed[closing_mask]
    return ttc


def min_ttc(log: EpisodeLog) -> float:
    """Minimum TTC between the ramp vehicle and the main-lane decision
    vehicle over the episode. Returns +inf if never closing (fully safe).
    """
    ttc = time_to_collision(log.ramp_x, log.ramp_v, log.main_x, log.main_v)
    ttc_reverse = time_to_collision(log.main_x, log.main_v, log.ramp_x, log.ramp_v)
    combined = np.minimum(ttc, ttc_reverse)
    finite = combined[np.isfinite(combined)]
    return float(np.min(finite)) if finite.size else float("inf")


def rms_jerk(accel: np.ndarray, dt: float) -> float:
    if dt <= 0 or len(accel) < 2:
        return 0.0
    jerk = np.diff(accel) / dt
    return float(np.sqrt(np.mean(jerk**2)))


def hard_braking_count(accel: np.ndarray, threshold: float = HARD_BRAKE_THRESHOLD_MPS2) -> int:
    below = accel < threshold
    # count distinct events (contiguous below-threshold runs), not ticks
    if not below.any():
        return 0
    edges = np.diff(below.astype(int))
    starts = int(below[0]) + int(np.sum(edges == 1))
    return starts


def maneuver_completion_time_s(log: EpisodeLog) -> Optional[float]:
    if not log.merge_success or log.merge_time_s is None:
        return None
    return float(log.merge_time_s - log.request_time_s)


def string_effect_speed_drop(log: EpisodeLog) -> float:
    """Max speed drop (m/s) of the SECOND vehicle behind the yielding
    main-lane vehicle, relative to its pre-interaction free-flow speed.
    Positive values mean the disturbance propagated two cars back.
    """
    if log.follower2_v.size == 0:
        return 0.0
    drop = log.baseline_follower2_speed_mps - np.min(log.follower2_v)
    return float(max(0.0, drop))


def compute_metrics(log: EpisodeLog) -> dict:
    dt = _dt(log.t)
    return {
        "merge_success": bool(log.merge_success),
        "collision": bool(log.collision),
        "min_ttc_s": min_ttc(log),
        "completion_time_s": maneuver_completion_time_s(log),
        "hard_braking_main": hard_braking_count(log.main_a),
        "hard_braking_follower": hard_braking_count(log.follower_a),
        "jerk_rms_main_mps3": rms_jerk(log.main_a, dt),
        "jerk_rms_follower_mps3": rms_jerk(log.follower_a, dt),
        "string_effect_speed_drop_mps": string_effect_speed_drop(log),
        "dangerous_near_miss": bool(
            (not log.collision) and min_ttc(log) < DANGEROUS_TTC_S
        ),
    }
