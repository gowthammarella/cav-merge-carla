import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from merge_sim.metrics import (
    EpisodeLog,
    compute_metrics,
    hard_braking_count,
    min_ttc,
    rms_jerk,
    string_effect_speed_drop,
    time_to_collision,
)


def _flat_log(n=50, dt=0.1, **overrides) -> EpisodeLog:
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


def test_ttc_infinite_when_not_closing():
    pos_rear = np.array([0.0, 5.0, 10.0])
    vel_rear = np.array([10.0, 10.0, 10.0])
    pos_front = np.array([20.0, 25.0, 30.0])
    vel_front = np.array([15.0, 15.0, 15.0])
    ttc = time_to_collision(pos_rear, vel_rear, pos_front, vel_front)
    assert np.all(np.isinf(ttc))


def test_ttc_finite_when_closing():
    pos_rear = np.array([0.0])
    vel_rear = np.array([20.0])
    pos_front = np.array([20.0])
    vel_front = np.array([10.0])
    ttc = time_to_collision(pos_rear, vel_rear, pos_front, vel_front)
    assert np.isclose(ttc[0], 2.0)


def test_min_ttc_on_safe_episode_is_large():
    log = _flat_log()
    assert min_ttc(log) == float("inf")


def test_min_ttc_detects_closing_scenario():
    n = 30
    t = np.arange(n) * 0.1
    log = _flat_log(
        n=n,
        t=t,
        ramp_x=np.linspace(0, 40, n),
        ramp_v=np.full(n, 25.0),
        main_x=np.full(n, 15.0) + np.linspace(0, 5, n),
        main_v=np.full(n, 5.0),
    )
    result = min_ttc(log)
    assert result < float("inf")
    assert result > 0


def test_rms_jerk_zero_for_constant_accel():
    accel = np.full(20, 1.5)
    assert rms_jerk(accel, dt=0.1) == 0.0


def test_rms_jerk_positive_for_varying_accel():
    accel = np.array([0.0, 3.0, -3.0, 3.0, -3.0])
    assert rms_jerk(accel, dt=0.1) > 0


def test_hard_braking_count_counts_events_not_ticks():
    # one 3-tick braking event followed by one 2-tick braking event
    accel = np.array([0.0, -4.0, -4.0, -4.0, 0.0, -5.0, -5.0, 0.0])
    assert hard_braking_count(accel) == 2


def test_hard_braking_count_zero_when_no_braking():
    accel = np.zeros(10)
    assert hard_braking_count(accel) == 0


def test_string_effect_zero_when_no_slowdown():
    log = _flat_log()
    assert string_effect_speed_drop(log) == 0.0


def test_string_effect_positive_when_follower2_slows():
    log = _flat_log(follower2_v=np.array([20.0] * 10 + [15.0] * 10 + [20.0] * 10),
                     n=30, t=np.arange(30) * 0.1,
                     ramp_x=np.linspace(0, 30, 30), ramp_v=np.full(30, 20.0),
                     main_x=np.linspace(30, 60, 30), main_v=np.full(30, 20.0),
                     main_a=np.zeros(30), follower_v=np.full(30, 20.0),
                     follower_a=np.zeros(30))
    assert string_effect_speed_drop(log) == 5.0


def test_compute_metrics_smoke():
    log = _flat_log()
    metrics = compute_metrics(log)
    assert metrics["merge_success"] is True
    assert metrics["collision"] is False
    assert metrics["completion_time_s"] is not None
    assert "min_ttc_s" in metrics
    assert "jerk_rms_main_mps3" in metrics


def test_completion_time_none_when_not_merged():
    log = _flat_log(merge_success=False, merge_time_s=None)
    metrics = compute_metrics(log)
    assert metrics["merge_success"] is False
    assert metrics["completion_time_s"] is None
