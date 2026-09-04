"""Two-agent (main-lane CAV + ramp CAV) merge scenario with per-agent RGB
camera perception and configurable V2V communication richness, used by the
vision + communication-cost experiment track. Reuses merge geometry helpers
from `merge_sim.scenario` so both tracks share one notion of "where the
on-ramp is" and one set of density/speed presets. Requires `carla`,
`ultralytics`, and `torch`; first executable on Colab.

Design note on what "communication" means here: every vehicle always knows
its OWN true kinematic state (that's not perception). What varies by
condition is what a vehicle learns about the OTHER agent:
  A. local-only  — nothing; only what its own camera detects.
  B. state       — the other agent's own broadcast state (position, speed,
                   accel, lane, intent) — cheap, but only covers the sender.
  C. object      — the other agent's full locally-detected-object list —
                   larger, but can reveal a third vehicle the receiver's own
                   camera cannot see (occlusion), matching the mentor
                   brief's stated benefit of semantic/object sharing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import carla
import numpy as np

from merge_sim import carla_utils
from merge_sim.controller import HighLevelAction, SharedController
from merge_sim.scenario import (
    DENSITY_INTER_ARRIVAL_S,
    DESIRED_SPEED_MPS,
    MANUAL_MERGE_POINT,
    MAX_EPISODE_DURATION_S,
    TOWN,
    find_merge_point_candidates,
)
from vision_comm.comm import (
    CommCondition,
    DetectedObject,
    ObjectMessage,
    StateMessage,
    message_size_for_condition,
)
from vision_comm.comm_metrics import communication_cost, count_id_switches, match_detections
from vision_comm.perception import CameraPerception

FIXED_DELTA_SECONDS = 0.05
AGENT_IDS = ["main", "ramp"]
OTHER_AGENT = {"main": "ramp", "ramp": "main"}

ACTION_HOLD, ACTION_YIELD, ACTION_ACCELERATE = 0, 1, 2
YIELD_SPEED_MARGIN_MPS = 3.0
ACCELERATE_SPEED_FACTOR = 1.1

HARD_BRAKE_PENALTY = -1.0
COLLISION_PENALTY = -50.0
SUCCESS_REWARD = 20.0
STEP_PENALTY = -0.01

# not-observed sentinel for the "other agent" slot of the observation
# vector, when nothing was perceived or communicated about them this tick
NOT_OBSERVED_DISTANCE_M = 200.0

OBS_DIM = 6  # [own_speed, distance_to_merge, other_rel_x, other_rel_y, other_rel_vx, observed_flag]

PSEUDO_BOX_HALF_SIZE_M = 1.0  # for reusing comm_metrics.match_detections on world positions


def _speed_mps(actor) -> float:
    v = actor.get_velocity()
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)


def _position_to_pseudo_box(position_m: tuple[float, float]) -> tuple[float, float, float, float]:
    """Turns a world (x, y) point into a small fixed-size box so we can
    reuse `comm_metrics.match_detections`'s tested IoU-matching logic for
    position-based (rather than pixel-IoU-based) detection accuracy — a
    simplification documented in the project plan; full 3D-bbox-to-image
    ground-truth projection is out of scope for this phase.
    """
    x, y = position_m
    return (x - PSEUDO_BOX_HALF_SIZE_M, y - PSEUDO_BOX_HALF_SIZE_M,
            x + PSEUDO_BOX_HALF_SIZE_M, y + PSEUDO_BOX_HALF_SIZE_M)


@dataclass
class _EpisodeRecord:
    message_sizes_bytes: list = field(default_factory=list)
    detection_precisions: list = field(default_factory=list)
    detection_recalls: list = field(default_factory=list)
    id_switch_sequence: dict = field(default_factory=dict)  # agent_id -> [track_id,...] for the other agent


class MultiAgentMergeScenario:
    """Satisfies `vision_comm.ippo.MultiAgentEnv`: fixed `agent_ids`,
    `obs_dim`, `n_actions`, and `reset()` / `step(actions)`.
    """

    agent_ids = AGENT_IDS
    n_actions = 3  # HOLD, YIELD, ACCELERATE — same vocabulary as merge_sim's strategies
    obs_dim = OBS_DIM

    def __init__(self, client: "carla.Client", condition: CommCondition, town: str = TOWN):
        self.client = client
        self.world = carla_utils.load_world(client, town)
        self.traffic_manager = client.get_trafficmanager()
        self.traffic_manager.set_synchronous_mode(True)
        self.condition = condition

        self._actors: list = []
        self._background_vehicles: list = []
        self._merge_point: Optional["carla.Transform"] = None
        self._t = 0.0
        self._collided = {aid: False for aid in AGENT_IDS}
        self._merged = False
        self._merge_time_s: Optional[float] = None
        self._desired_speed = DESIRED_SPEED_MPS["medium"]

        self._vehicles: dict = {}
        self._controllers: dict[str, SharedController] = {}
        self._perception: dict[str, CameraPerception] = {}
        self._record = _EpisodeRecord()

    # -- lifecycle -----------------------------------------------------
    def reset(self, seed: int = 0, density: str = "medium") -> dict[str, np.ndarray]:
        carla_utils.destroy_actors(self._actors)
        self._actors, self._background_vehicles = [], []
        carla_utils.set_seed(seed)
        self.traffic_manager.set_random_device_seed(seed)

        self._t = 0.0
        self._collided = {aid: False for aid in AGENT_IDS}
        self._merged = False
        self._merge_time_s = None
        self._desired_speed = DESIRED_SPEED_MPS[density]
        self._record = _EpisodeRecord(id_switch_sequence={aid: [] for aid in AGENT_IDS})

        if self._merge_point is None:
            self._merge_point = self._resolve_merge_point()

        self._spawn_background_traffic(density, seed)
        self._vehicles["main"] = self._spawn_agent_vehicle(behind_m=30.0, right_m=0.0, role="main_cav")
        self._vehicles["ramp"] = self._spawn_agent_vehicle(behind_m=45.0, right_m=3.5, role="ramp_cav")

        for aid, vehicle in self._vehicles.items():
            self._actors.append(vehicle)
            self._controllers[aid] = SharedController(vehicle)
            self._perception[aid] = CameraPerception(self.world, vehicle)
            self._actors.append(self._perception[aid].camera)
            sensor = self._attach_collision_sensor(aid, vehicle)
            self._actors.append(sensor)

        self.world.tick()
        return self._build_observations()

    def close(self) -> None:
        for perception in self._perception.values():
            perception.close()
        carla_utils.destroy_actors(self._actors)
        carla_utils.destroy_actors(self._background_vehicles)
        self._actors, self._background_vehicles = [], []

    def is_done(self) -> bool:
        return self._merged or any(self._collided.values()) or self._t >= MAX_EPISODE_DURATION_S

    # -- MultiAgentEnv protocol -----------------------------------------
    def step(
        self, actions: dict[str, int]
    ) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, bool], dict]:
        for aid, vehicle in self._vehicles.items():
            action = HighLevelAction(target_speed_mps=self._action_to_target_speed(actions[aid]))
            waypoint = self.world.get_map().get_waypoint(vehicle.get_location())
            control = self._controllers[aid].step(action, waypoint)
            vehicle.apply_control(control)

        self.world.tick()
        self._t += FIXED_DELTA_SECONDS
        self._check_merge_complete()

        obs = self._build_observations()
        done = self.is_done()
        dones = {aid: done for aid in AGENT_IDS}
        rewards = {aid: self._reward_for_agent(aid) for aid in AGENT_IDS}
        return obs, rewards, dones, {}

    # -- observation / communication ------------------------------------
    def _build_observations(self) -> dict[str, np.ndarray]:
        local_tracks = {
            aid: self._perception[aid].perceive(FIXED_DELTA_SECONDS) for aid in AGENT_IDS
        }
        self._record_perception_accuracy(local_tracks)

        obs = {}
        for aid in AGENT_IDS:
            other_aid = OTHER_AGENT[aid]
            other_vehicle = self._vehicles[other_aid]
            own_vehicle = self._vehicles[aid]

            other_relative = self._communicated_other_position(aid, other_aid, local_tracks[aid])
            observed_flag = 1.0 if other_relative is not None else 0.0
            rel_x, rel_y, rel_vx = other_relative if other_relative is not None else (
                NOT_OBSERVED_DISTANCE_M, 0.0, 0.0,
            )

            own_speed = _speed_mps(own_vehicle)
            distance_to_merge = self._distance_to_merge_point(own_vehicle)
            obs[aid] = np.array(
                [own_speed, distance_to_merge, rel_x, rel_y, rel_vx, observed_flag], dtype=np.float32
            )
        return obs

    def _communicated_other_position(
        self, receiver_id: str, sender_id: str, receiver_local_tracks: list
    ) -> Optional[tuple[float, float, float]]:
        """Returns (relative_x, relative_y, relative_vx) for what `receiver_id`
        knows about `sender_id`'s position this tick, under the active
        condition — from its own camera (always tried first), else from a
        communicated message if the condition allows it. Tracks the
        message bytes sent for the communication-cost metric.
        """
        receiver_vehicle = self._vehicles[receiver_id]
        sender_vehicle = self._vehicles[sender_id]
        receiver_loc = receiver_vehicle.get_location()

        own_perceived = self._match_track_to_actor(receiver_local_tracks, sender_vehicle)
        if own_perceived is not None:
            rel_x = own_perceived.position_m[0] - receiver_loc.x
            rel_y = own_perceived.position_m[1] - receiver_loc.y
            return rel_x, rel_y, own_perceived.velocity_mps[0]

        if self.condition is CommCondition.LOCAL_ONLY:
            self._record.message_sizes_bytes.append(0)
            return None

        if self.condition is CommCondition.STATE:
            sender_loc = sender_vehicle.get_location()
            sender_speed = _speed_mps(sender_vehicle)
            waypoint = self.world.get_map().get_waypoint(sender_loc)
            message = StateMessage(
                vehicle_id=sender_id,
                position_m=(sender_loc.x, sender_loc.y),
                speed_mps=sender_speed,
                accel_mps2=0.0,  # not tracked per-tick at this layer; acceptable for message-size accounting
                lane_id=waypoint.lane_id,
                intent="unknown",
            )
            self._record.message_sizes_bytes.append(
                message_size_for_condition(self.condition, state=message)
            )
            return sender_loc.x - receiver_loc.x, sender_loc.y - receiver_loc.y, sender_speed

        if self.condition is CommCondition.OBJECT:
            sender_tracks = self._perception[sender_id].perceive(FIXED_DELTA_SECONDS)
            detected_objects = tuple(
                DetectedObject(
                    object_type=t.object_type,
                    position_m=t.position_m,
                    velocity_mps=t.velocity_mps,
                    confidence=t.confidence,
                )
                for t in sender_tracks
            )
            message = ObjectMessage(sender_id=sender_id, objects=detected_objects)
            self._record.message_sizes_bytes.append(
                message_size_for_condition(self.condition, objects=message)
            )
            # does the sender's object list happen to include the receiver
            # itself (mutual visibility), which the receiver can use as a
            # position fix on the sender via the sender's own broadcast loc
            sender_loc = sender_vehicle.get_location()
            sender_speed = _speed_mps(sender_vehicle)
            return sender_loc.x - receiver_loc.x, sender_loc.y - receiver_loc.y, sender_speed

        raise NotImplementedError(f"condition {self.condition} not supported by this scenario")

    def _match_track_to_actor(self, tracks: list, actor) -> Optional[object]:
        actor_loc = actor.get_location()
        best, best_dist = None, 5.0  # meters — association gate
        for track in tracks:
            dist = math.hypot(track.position_m[0] - actor_loc.x, track.position_m[1] - actor_loc.y)
            if dist < best_dist:
                best, best_dist = track, dist
        return best

    def _record_perception_accuracy(self, local_tracks: dict[str, list]) -> None:
        for aid in AGENT_IDS:
            other_aid = OTHER_AGENT[aid]
            other_actor_loc = self._vehicles[other_aid].get_location()
            gt_box = _position_to_pseudo_box((other_actor_loc.x, other_actor_loc.y))
            pred_boxes = [_position_to_pseudo_box(t.position_m) for t in local_tracks[aid]]
            precision, recall = match_detections(pred_boxes, [gt_box])
            self._record.detection_precisions.append(precision)
            self._record.detection_recalls.append(recall)

            matched = self._match_track_to_actor(local_tracks[aid], self._vehicles[other_aid])
            self._record.id_switch_sequence[aid].append(matched.track_id if matched else None)

    # -- reward / termination --------------------------------------------
    def _reward_for_agent(self, agent_id: str) -> float:
        reward = STEP_PENALTY
        if self._collided[agent_id]:
            reward += COLLISION_PENALTY
        if self._merged and self.is_done():
            reward += SUCCESS_REWARD
        return reward

    def _action_to_target_speed(self, action: int) -> float:
        if action == ACTION_YIELD:
            return max(0.0, self._desired_speed - YIELD_SPEED_MARGIN_MPS)
        if action == ACTION_ACCELERATE:
            return self._desired_speed * ACCELERATE_SPEED_FACTOR
        return self._desired_speed

    def _distance_to_merge_point(self, vehicle) -> float:
        loc = vehicle.get_location()
        mp = self._merge_point.location
        return math.hypot(loc.x - mp.x, loc.y - mp.y)

    def _check_merge_complete(self) -> None:
        if self._merged:
            return
        main_wp = self.world.get_map().get_waypoint(self._vehicles["main"].get_location())
        ramp_wp = self.world.get_map().get_waypoint(self._vehicles["ramp"].get_location())
        if main_wp.lane_id == ramp_wp.lane_id and main_wp.road_id == ramp_wp.road_id:
            self._merged = True
            self._merge_time_s = self._t

    # -- setup helpers ----------------------------------------------------
    def _resolve_merge_point(self) -> "carla.Transform":
        if MANUAL_MERGE_POINT is not None:
            return MANUAL_MERGE_POINT
        candidates = find_merge_point_candidates(self.world)
        if not candidates:
            raise RuntimeError(
                "no junction candidates found; set MANUAL_MERGE_POINT in merge_sim/scenario.py"
            )
        return candidates[0].transform

    def _spawn_background_traffic(self, density: str, seed: int, n_vehicles: int = 12) -> None:
        spawn_points = self.world.get_map().get_spawn_points()
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(spawn_points), size=min(n_vehicles, len(spawn_points)), replace=False)
        for idx in chosen:
            vehicle = carla_utils.spawn_vehicle(self.world, spawn_points[int(idx)])
            if vehicle is None:
                continue
            vehicle.set_autopilot(True, self.traffic_manager.get_port())
            self._background_vehicles.append(vehicle)
        self.traffic_manager.set_global_distance_to_leading_vehicle(
            max(2.0, DENSITY_INTER_ARRIVAL_S[density])
        )

    def _spawn_agent_vehicle(self, behind_m: float, right_m: float, role: str):
        right = self._merge_point.get_right_vector()
        location = (
            self._merge_point.location
            - self._merge_point.get_forward_vector() * behind_m
            + right * right_m
        )
        transform = carla.Transform(location, self._merge_point.rotation)
        vehicle = carla_utils.spawn_vehicle(self.world, transform, role_name=role)
        if vehicle is None:
            raise RuntimeError(f"failed to spawn {role} — check merge point/offset geometry for this town")
        return vehicle

    def _attach_collision_sensor(self, agent_id: str, vehicle):
        bp = self.world.get_blueprint_library().find("sensor.other.collision")
        sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=vehicle)
        sensor.listen(lambda event, aid=agent_id: self._on_collision(aid, event))
        return sensor

    def _on_collision(self, agent_id: str, event) -> None:
        self._collided[agent_id] = True

    # -- episode-end metrics ----------------------------------------------
    def get_episode_metrics(self) -> dict:
        comm_cost = communication_cost(self._record.message_sizes_bytes, FIXED_DELTA_SECONDS)
        id_switches = {
            aid: count_id_switches(self._record.id_switch_sequence[aid]) for aid in AGENT_IDS
        }
        return {
            "condition": self.condition.value,
            "merge_success": self._merged,
            "collision": any(self._collided.values()),
            "merge_time_s": self._merge_time_s,
            "mean_detection_precision": float(np.mean(self._record.detection_precisions))
            if self._record.detection_precisions else float("nan"),
            "mean_detection_recall": float(np.mean(self._record.detection_recalls))
            if self._record.detection_recalls else float("nan"),
            "id_switches_main": id_switches["main"],
            "id_switches_ramp": id_switches["ramp"],
            **{f"comm_{k}": v for k, v in comm_cost.items()},
        }
