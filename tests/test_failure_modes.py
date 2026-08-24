import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from merge_sim.failure_modes import FailureMode, classify
from merge_sim.metrics import EpisodeLog

MAX_DURATION = 15.0


def _log(n=50, dt=0.1, **overrides) -> EpisodeLog:
    t = np.arange(n) * dt
    defaults = dict(
        t=t,
        ramp_x=np.linspace(0, 20, n),
        ramp_v=np.full(n, 20.0),
        main_x=np.linspace(30, 50, n),
        main_v=np.full(n, 20.0),
        main_a=np.zeros(n),
        follower_v=np.full(n, 20.0),
        follower_a=np.zeros(n),
        follower2_v=np.full(n, 20.0),
        baseline_follower2_speed_mps=20.0,
        merge_success=True,
        collision=False,
        merge_time_s=float(t[-1]),
        request_time_s=0.0,
    )
    defaults.update(overrides)
    return EpisodeLog(**defaults)


def test_collision_takes_priority():
    log = _log(collision=True, merge_success=False, merge_time_s=None)
    assert classify(log, MAX_DURATION) == FailureMode.COLLISION


def test_clean_success():
    log = _log()
    assert classify(log, MAX_DURATION) == FailureMode.SUCCESS_CLEAN


def test_success_with_disturbance():
    n = 30
    follower2_v = np.array([20.0] * 10 + [10.0] * 10 + [20.0] * 10)
    log = _log(
        n=n, t=np.arange(n) * 0.1,
        ramp_x=np.linspace(0, 30, n), ramp_v=np.full(n, 20.0),
        main_x=np.linspace(30, 60, n), main_v=np.full(n, 20.0),
        follower2_v=follower2_v, merge_time_s=float(np.arange(n)[-1] * 0.1),
    )
    assert classify(log, MAX_DURATION) == FailureMode.SUCCESS_WITH_DISTURBANCE


def test_late_forced_merge_dangerous_ttc():
    n = 30
    t = np.arange(n) * 0.1
    log = _log(
        n=n, t=t,
        ramp_x=np.linspace(0, 40, n), ramp_v=np.full(n, 25.0),
        main_x=np.full(n, 15.0) + np.linspace(0, 5, n), main_v=np.full(n, 5.0),
        merge_time_s=float(t[-1]),
    )
    assert classify(log, MAX_DURATION) == FailureMode.LATE_FORCED_MERGE


def test_no_gap_found_when_never_merged_and_episode_short():
    log = _log(n=20, t=np.arange(20) * 0.1, merge_success=False, merge_time_s=None)
    assert classify(log, MAX_DURATION) == FailureMode.NO_GAP_FOUND


def test_timeout_when_episode_hits_max_duration():
    n = 200
    t = np.arange(n) * 0.1  # spans 19.9s > MAX_DURATION
    log = _log(
        n=n, t=t,
        ramp_x=np.linspace(0, 20, n), ramp_v=np.full(n, 20.0),
        main_x=np.linspace(30, 50, n), main_v=np.full(n, 20.0),
        merge_success=False, merge_time_s=None,
    )
    assert classify(log, MAX_DURATION) == FailureMode.TIMEOUT
