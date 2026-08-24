"""Defines the merge scenario: a stream of main-lane background traffic
plus one designated main-lane "decision" vehicle and one ramp vehicle
attempting to merge in front of it, at a fixed on-ramp point.

Requires a live CARLA connection — first executable on Colab. The one
piece that cannot be fully generic is the exact on-ramp waypoint for a
given town/map version; `find_merge_point_candidates()` below auto-detects
junction candidates from the map topology, but you should visually confirm
the chosen one in Colab (see notebooks/00_colab_setup.ipynb, "calibrate
merge point" cell) before running the full experiment grid, and hardcode
the confirmed transform into `MANUAL_MERGE_POINT` if the auto-pick is
wrong for your CARLA version's Town04/Town06.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import carla
import numpy as np

from merge_sim import carla_utils
from merge_sim.controller import HighLevelAction, SharedController, idm_acceleration
from merge_sim.metrics import EpisodeLog

TOWN = "Town04"
FIXED_DELTA_SECONDS = 0.05
MAX_EPISODE_DURATION_S = 25.0

DENSITY_INTER_ARRIVAL_S = {"light": 6.0, "medium": 3.5, "heavy": 1.8}
DESIRED_SPEED_MPS = {"light": 25.0, "medium": 22.0, "heavy": 18.0}

# Manual override: set to a carla.Transform once you've confirmed the
# correct on-ramp point in Colab. Leave None to use auto-detection.
MANUAL_MERGE_POINT: Optional["carla.Transform"] = None

MERGE_COMPLETE_LATERAL_TOLERANCE_M = 1.0
FOLLOWER_SEARCH_RADIUS_M = 100.0


@dataclass(frozen=True)
class MainLaneObservableState:
    own_speed_mps: float
    desired_speed_mps: float
    gap_ahead_m: float


@dataclass(frozen=True)
class RampObservableState:
    ramp_speed_mps: float
    distance_to_merge_point_m: float
    gap_to_ramp_m: float  # size of the gap the ramp vehicle is targeting


@dataclass
class _FrameRecord:
    t: float
    ramp_x: float
    ramp_v: float
    main_x: float
    main_v: float
    main_a: float
    follower_v: float
    follower_a: float
    follower2_v: float


def find_merge_point_candidates(world: "carla.World", max_candidates: int = 5) -> list:
    """Returns up to `max_candidates` junction waypoints where multiple
    incoming lanes converge, as candidates for the on-ramp merge point.
    Print these in Colab and visually confirm (spectator camera) which one
    is the intended highway on-ramp before relying on auto-detection.
    """
    carla_map = world.get_map()
    topology = carla_map.get_topology()
    junction_ids = {}
    for wp_start, wp_end in topology:
        junction = wp_start.get_junction() or wp_end.get_junction()
        if junction is not None:
            junction_ids.setdefault(junction.id, junction)

    candidates = []
    for junction in junction_ids.values():
        lanes = junction.get_waypoints(carla.LaneType.Driving)
        if len(lanes) >= 2:
            entry_wp = lanes[0][0]
            candidates.append(entry_wp)
        if len(candidates) >= max_candidates:
            break
    return candidates


def _speed_mps(actor) -> float:
    v = actor.get_velocity()
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)


def _forward_distance(from_actor, to_actor) -> float:
    ego_loc = from_actor.get_transform().location
    other_loc = to_actor.get_transform().location
    forward = from_actor.get_transform().get_forward_vector()
    delta = other_loc - ego_loc
    return delta.x * forward.x + delta.y * forward.y + delta.z * forward.z


class MergeScenario:
    """Owns the CARLA world/actors for one episode.

    Usage:
        scenario = MergeScenario(client)
        scenario.reset(seed=0, density="medium")
        strategy.reset()
        while not scenario.is_done():
            scenario.tick(strategy)
        log = scenario.get_episode_log()
    """

    def __init__(self, client: "carla.Client", town: str = TOWN):
        self.client = client
        self.world = carla_utils.load_world(client, town)
        self.traffic_manager = client.get_trafficmanager()
        self.traffic_manager.set_synchronous_mode(True)
        self._actors: list = []
        self._background_vehicles: list = []
        self._merge_point: Optional["carla.Transform"] = None
        self._t = 0.0
        self._merged = False
        self._collided = False
        self._merge_time_s: Optional[float] = None
        self._request_time_s = 0.0
        self._frames: list[_FrameRecord] = []
        self._baseline_follower2_speed_mps = 0.0
        self._desired_speed = DESIRED_SPEED_MPS["medium"]

        self._main_vehicle = None
        self._ramp_vehicle = None
        self._follower = None
        self._follower2 = None
        self._controller: Optional[SharedController] = None
        self._collision_sensor = None
        self._prev_main_accel = 0.0

    # -- lifecycle -----------------------------------------------------
    def reset(self, seed: int, density: str) -> None:
        if density not in DENSITY_INTER_ARRIVAL_S:
            raise ValueError(f"unknown density {density!r}, expected one of {list(DENSITY_INTER_ARRIVAL_S)}")

        carla_utils.destroy_actors(self._actors)
        self._actors = []
        self._background_vehicles = []
        carla_utils.set_seed(seed)
        self.traffic_manager.set_random_device_seed(seed)

        self._t = 0.0
        self._merged = False
        self._collided = False
        self._merge_time_s = None
        self._frames = []
        self._prev_main_accel = 0.0
        self._desired_speed = DESIRED_SPEED_MPS[density]

        if self._merge_point is None:
            self._merge_point = self._resolve_merge_point()

        self._spawn_background_traffic(density, seed)
        self._main_vehicle = self._spawn_main_lane_vehicle()
        self._ramp_vehicle = self._spawn_ramp_vehicle()
        self._actors.extend([self._main_vehicle, self._ramp_vehicle])

        self._controller = SharedController(self._main_vehicle)
        self._collision_sensor = self._attach_collision_sensor(self._main_vehicle)
        self._actors.append(self._collision_sensor)

        self.world.tick()  # let physics settle one tick before recording
        self._follower, _ = carla_utils.nearest_vehicle_behind(
            self.world, self._main_vehicle, FOLLOWER_SEARCH_RADIUS_M
        )
        self._follower2, _ = (
            carla_utils.nearest_vehicle_behind(self.world, self._follower, FOLLOWER_SEARCH_RADIUS_M)
            if self._follower is not None
            else (None, float("inf"))
        )
        self._baseline_follower2_speed_mps = (
            _speed_mps(self._follower2) if self._follower2 is not None else self._desired_speed
        )
        self._request_time_s = self._t

    def close(self) -> None:
        carla_utils.destroy_actors(self._actors)
        carla_utils.destroy_actors(self._background_vehicles)
        self._actors = []
        self._background_vehicles = []

    def is_done(self) -> bool:
        return self._merged or self._collided or self._t >= MAX_EPISODE_DURATION_S

    def tick(self, strategy) -> None:
        main_state = self._observe_main_lane_state()
        ramp_state = self._observe_ramp_state()
        action: HighLevelAction = strategy.decide(main_state, ramp_state, self._t)

        waypoint = self.world.get_map().get_waypoint(self._main_vehicle.get_location())
        control = self._controller.step(action, waypoint)
        self._main_vehicle.apply_control(control)

        # ramp vehicle: simple IDM toward the gap it's negotiating for
        self._step_ramp_vehicle()

        self.world.tick()
        self._t += FIXED_DELTA_SECONDS

        self._check_merge_complete()
        self._record_frame()

    def get_episode_log(self) -> EpisodeLog:
        if not self._frames:
            raise RuntimeError("no frames recorded — call tick() at least once before get_episode_log()")

        t = np.array([f.t for f in self._frames])
        return EpisodeLog(
            t=t,
            ramp_x=np.array([f.ramp_x for f in self._frames]),
            ramp_v=np.array([f.ramp_v for f in self._frames]),
            main_x=np.array([f.main_x for f in self._frames]),
            main_v=np.array([f.main_v for f in self._frames]),
            main_a=np.array([f.main_a for f in self._frames]),
            follower_v=np.array([f.follower_v for f in self._frames]),
            follower_a=np.array([f.follower_a for f in self._frames]),
            follower2_v=np.array([f.follower2_v for f in self._frames]),
            baseline_follower2_speed_mps=self._baseline_follower2_speed_mps,
            merge_success=self._merged,
            collision=self._collided,
            merge_time_s=self._merge_time_s,
            request_time_s=self._request_time_s,
        )

    # -- setup helpers ---------------------------------------------------
    def _resolve_merge_point(self) -> "carla.Transform":
        if MANUAL_MERGE_POINT is not None:
            return MANUAL_MERGE_POINT
        candidates = find_merge_point_candidates(self.world)
        if not candidates:
            raise RuntimeError(
                "no junction candidates found for the merge point; set "
                "MANUAL_MERGE_POINT in scenario.py after inspecting the map in Colab"
            )
        return candidates[0].transform

    def _spawn_background_traffic(self, density: str, seed: int, n_vehicles: int = 15) -> None:
        """Spawns background traffic with per-vehicle randomized behavior
        so the simulated HDVs actually exhibit the kind of uncertain
        intent the negotiation strategy's `HDVUncertainty` belief model
        (strategies/negotiation.py) is reasoning about — rather than every
        background vehicle following an identical, perfectly predictable
        policy. A fixed fraction are marked "impatient" (faster, shorter
        following distance, more prone to ignore right-of-way), mirroring
        the `noncompliance_prob` used on the belief side; the two are
        deliberately independent constants (not the same variable) since a
        real deployed policy's belief about the world is never guaranteed
        to be perfectly calibrated to it — that miscalibration gap is
        itself worth reporting, not hidden by tying them together.
        """
        IMPATIENT_FRACTION = 0.15
        spawn_points = self.world.get_map().get_spawn_points()
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(spawn_points), size=min(n_vehicles, len(spawn_points)), replace=False)
        for idx in chosen:
            vehicle = carla_utils.spawn_vehicle(self.world, spawn_points[int(idx)])
            if vehicle is None:
                continue
            vehicle.set_autopilot(True, self.traffic_manager.get_port())

            if rng.random() < IMPATIENT_FRACTION:
                # impatient HDV: faster than the flow, tailgates, and
                # occasionally ignores another vehicle's right-of-way
                self.traffic_manager.vehicle_percentage_speed_difference(
                    vehicle, float(rng.uniform(-25.0, -10.0))
                )
                self.traffic_manager.distance_to_leading_vehicle(vehicle, float(rng.uniform(0.5, 1.5)))
                self.traffic_manager.ignore_vehicles_percentage(vehicle, float(rng.uniform(0.0, 15.0)))
            else:
                # typical HDV: mild, zero-mean-ish variability around the
                # flow speed and a normal following distance
                self.traffic_manager.vehicle_percentage_speed_difference(
                    vehicle, float(rng.uniform(-5.0, 10.0))
                )
                self.traffic_manager.distance_to_leading_vehicle(vehicle, float(rng.uniform(1.5, 3.0)))

            self._background_vehicles.append(vehicle)
        # inter-arrival calibration is approximated via TM's global distance
        # setting; finer control requires per-vehicle spawn scheduling,
        # left as a documented extension point for the Colab run.
        self.traffic_manager.set_global_distance_to_leading_vehicle(
            max(2.0, DENSITY_INTER_ARRIVAL_S[density])
        )

    def _spawn_main_lane_vehicle(self):
        transform = carla.Transform(
            self._merge_point.location - self._merge_point.get_forward_vector() * 30.0,
            self._merge_point.rotation,
        )
        vehicle = carla_utils.spawn_vehicle(self.world, transform, role_name="main_decision")
        if vehicle is None:
            raise RuntimeError("failed to spawn main-lane decision vehicle at the merge point")
        return vehicle

    def _spawn_ramp_vehicle(self):
        # offset laterally onto the on-ramp; exact lane offset is
        # town-geometry-dependent — verify visually in Colab and adjust
        # RAMP_LATERAL_OFFSET_M if the ramp vehicle doesn't land on-ramp.
        RAMP_LATERAL_OFFSET_M = 3.5
        right = self._merge_point.get_right_vector()
        location = (
            self._merge_point.location
            - self._merge_point.get_forward_vector() * 45.0
            + right * RAMP_LATERAL_OFFSET_M
        )
        transform = carla.Transform(location, self._merge_point.rotation)
        vehicle = carla_utils.spawn_vehicle(self.world, transform, role_name="ramp_vehicle")
        if vehicle is None:
            raise RuntimeError("failed to spawn ramp vehicle — check RAMP_LATERAL_OFFSET_M for this town")
        return vehicle

    def _attach_collision_sensor(self, vehicle):
        bp = self.world.get_blueprint_library().find("sensor.other.collision")
        sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=vehicle)
        sensor.listen(lambda event: self._on_collision(event))
        return sensor

    def _on_collision(self, event) -> None:
        self._collided = True

    # -- per-tick helpers -------------------------------------------------
    def _observe_main_lane_state(self) -> MainLaneObservableState:
        lead, gap_ahead = carla_utils.nearest_vehicle_ahead(self.world, self._main_vehicle)
        return MainLaneObservableState(
            own_speed_mps=_speed_mps(self._main_vehicle),
            desired_speed_mps=self._desired_speed,
            gap_ahead_m=gap_ahead,
        )

    def _observe_ramp_state(self) -> RampObservableState:
        distance_to_merge = _forward_distance(self._ramp_vehicle, self._main_vehicle) * -1.0
        distance_to_merge = max(0.0, distance_to_merge)
        gap_to_ramp = abs(_forward_distance(self._ramp_vehicle, self._main_vehicle))
        return RampObservableState(
            ramp_speed_mps=_speed_mps(self._ramp_vehicle),
            distance_to_merge_point_m=distance_to_merge,
            gap_to_ramp_m=gap_to_ramp,
        )

    def _step_ramp_vehicle(self) -> None:
        v = _speed_mps(self._ramp_vehicle)
        lead, gap = carla_utils.nearest_vehicle_ahead(self.world, self._ramp_vehicle)
        lead_v = _speed_mps(lead) if lead is not None else v
        target_v = self._desired_speed
        accel = idm_acceleration(v, target_v, gap, v - lead_v)
        throttle = max(0.0, min(1.0, accel / 1.5))
        brake = max(0.0, min(1.0, -accel / 3.0))
        waypoint = self.world.get_map().get_waypoint(self._ramp_vehicle.get_location())
        next_wps = waypoint.next(2.0)
        steer = 0.0
        if next_wps:
            target_loc = next_wps[0].transform.location
            ego_tf = self._ramp_vehicle.get_transform()
            forward = ego_tf.get_forward_vector()
            to_target = target_loc - ego_tf.location
            steer = math.atan2(
                forward.x * to_target.y - forward.y * to_target.x,
                forward.x * to_target.x + forward.y * to_target.y,
            )
            steer = max(-1.0, min(1.0, steer))
        self._ramp_vehicle.apply_control(
            carla.VehicleControl(throttle=throttle, brake=brake, steer=steer)
        )

    def _check_merge_complete(self) -> None:
        if self._merged or self._collided:
            return
        ramp_wp = self.world.get_map().get_waypoint(self._ramp_vehicle.get_location())
        main_wp = self.world.get_map().get_waypoint(self._main_vehicle.get_location())
        if ramp_wp.lane_id == main_wp.lane_id and ramp_wp.road_id == main_wp.road_id:
            self._merged = True
            self._merge_time_s = self._t

    def _record_frame(self) -> None:
        main_v = _speed_mps(self._main_vehicle)
        main_a = (main_v - (self._frames[-1].main_v if self._frames else main_v)) / FIXED_DELTA_SECONDS
        follower_v = _speed_mps(self._follower) if self._follower is not None else main_v
        follower_a = (
            (follower_v - self._frames[-1].follower_v) / FIXED_DELTA_SECONDS if self._frames else 0.0
        )
        follower2_v = _speed_mps(self._follower2) if self._follower2 is not None else follower_v

        self._frames.append(
            _FrameRecord(
                t=self._t,
                ramp_x=self._ramp_vehicle.get_location().x,
                ramp_v=_speed_mps(self._ramp_vehicle),
                main_x=self._main_vehicle.get_location().x,
                main_v=main_v,
                main_a=main_a,
                follower_v=follower_v,
                follower_a=follower_a,
                follower2_v=follower2_v,
            )
        )
        self._prev_main_accel = main_a
