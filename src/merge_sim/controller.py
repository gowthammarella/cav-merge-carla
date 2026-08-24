"""Single shared low-level controller used by EVERY strategy, so that
strategy comparisons measure decision quality (when to yield/hold/
accelerate) and never driving skill (how well it tracks a speed/waypoint).

Longitudinal target speed comes from the strategy's HighLevelAction; actual
throttle/brake/steer are always produced by the same IDM + PID combination
here. No strategy is allowed to touch `carla.VehicleControl` directly.

Requires `carla`; first executable on Colab.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import carla

IDM_PARAMS = dict(
    a_max=1.5,      # max acceleration, m/s^2
    b_comfort=2.0,  # comfortable braking, m/s^2
    delta=4,        # acceleration exponent
    s0=2.0,          # minimum bumper-to-bumper gap, m
    T=1.5,           # desired time headway, s
)


def idm_desired_gap(v_mps: float, delta_v_mps: float, params: dict = IDM_PARAMS) -> float:
    a_max, b, s0, T = params["a_max"], params["b_comfort"], params["s0"], params["T"]
    return s0 + max(0.0, v_mps * T + (v_mps * delta_v_mps) / (2 * (a_max * b) ** 0.5))


def idm_acceleration(
    v_mps: float, v0_mps: float, gap_m: float, delta_v_mps: float, params: dict = IDM_PARAMS
) -> float:
    """v_mps: current speed. v0_mps: desired/target speed (the strategy's
    high-level output). gap_m: net distance to the lead vehicle.
    delta_v_mps: v_mps - v_lead_mps (positive = closing on the lead).
    """
    a_max, delta = params["a_max"], params["delta"]
    gap_m = max(gap_m, 0.1)
    s_star = idm_desired_gap(v_mps, delta_v_mps, params)
    return a_max * (1 - (v_mps / max(v0_mps, 0.1)) ** delta - (s_star / gap_m) ** 2)


@dataclass(frozen=True)
class HighLevelAction:
    """The ONLY thing a Strategy is allowed to output. `target_speed_mps`
    encodes the decision uniformly: YIELD -> low target speed, HOLD ->
    match current desired speed, ACCELERATE -> raised target speed.
    """

    target_speed_mps: float


class SharedController:
    """Wraps `carla.VehiclePIDController` (lateral: waypoint tracking;
    longitudinal: speed tracking toward `HighLevelAction.target_speed_mps`).
    Every strategy's action passes through this same object.
    """

    def __init__(
        self,
        vehicle: "carla.Vehicle",
        args_lateral: Optional[dict] = None,
        args_longitudinal: Optional[dict] = None,
    ):
        self._vehicle = vehicle
        self._pid = carla.VehiclePIDController(
            vehicle,
            args_lateral=args_lateral or {"K_P": 1.5, "K_I": 0.05, "K_D": 0.1, "dt": 0.05},
            args_longitudinal=args_longitudinal or {"K_P": 1.0, "K_I": 0.05, "K_D": 0.0, "dt": 0.05},
        )

    def step(self, action: HighLevelAction, waypoint: "carla.Waypoint") -> "carla.VehicleControl":
        target_kmh = max(0.0, action.target_speed_mps) * 3.6
        return self._pid.run_step(target_kmh, waypoint)
