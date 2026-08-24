"""Classifies each episode into a failure-mode taxonomy so results report
*why* a strategy failed, not just aggregate success rate.

Priority order (first match wins): a collision is always the worst outcome
regardless of anything else; a completed-but-dangerous merge is worse than
a clean failure to find a gap; a successful merge that still disturbed
downstream traffic is flagged separately from a fully clean success.
"""
from __future__ import annotations

from enum import Enum

from merge_sim.metrics import EpisodeLog, compute_metrics, DANGEROUS_TTC_S

STRING_EFFECT_THRESHOLD_MPS = 2.0  # follower-2 speed drop above this = cascade


class FailureMode(Enum):
    SUCCESS_CLEAN = "success_clean"
    SUCCESS_WITH_DISTURBANCE = "success_with_disturbance"  # merged, but caused a brake cascade
    LATE_FORCED_MERGE = "late_forced_merge"  # merged, but with a dangerous near-miss
    COLLISION = "collision"
    NO_GAP_FOUND = "no_gap_found"  # episode ended, ramp vehicle never merged, no collision
    TIMEOUT = "timeout"  # episode hit the max duration without resolving either way


def classify(log: EpisodeLog, max_episode_duration_s: float) -> FailureMode:
    metrics = compute_metrics(log)

    if metrics["collision"]:
        return FailureMode.COLLISION

    if metrics["merge_success"]:
        if metrics["min_ttc_s"] < DANGEROUS_TTC_S:
            return FailureMode.LATE_FORCED_MERGE
        if metrics["string_effect_speed_drop_mps"] > STRING_EFFECT_THRESHOLD_MPS:
            return FailureMode.SUCCESS_WITH_DISTURBANCE
        return FailureMode.SUCCESS_CLEAN

    episode_span = float(log.t[-1] - log.t[0]) if len(log.t) else 0.0
    if episode_span >= max_episode_duration_s:
        return FailureMode.TIMEOUT
    return FailureMode.NO_GAP_FOUND


def classify_batch(
    logs: list[EpisodeLog], max_episode_duration_s: float
) -> dict[FailureMode, int]:
    counts: dict[FailureMode, int] = {mode: 0 for mode in FailureMode}
    for log in logs:
        counts[classify(log, max_episode_duration_s)] += 1
    return counts
