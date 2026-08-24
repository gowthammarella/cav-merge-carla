"""Thin CARLA connection/world helpers shared by scenario.py and
episode_runner.py.

Requires a running CARLA server and the `carla` Python package — this
module cannot be imported or exercised on a machine without CARLA. First
real execution happens on Colab; see notebooks/00_colab_setup.ipynb.
"""
from __future__ import annotations

import random
from contextlib import contextmanager

import carla


def connect(host: str = "localhost", port: int = 2000, timeout: float = 20.0) -> "carla.Client":
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    return client


@contextmanager
def synchronous_mode(world: "carla.World", fixed_delta_seconds: float = 0.05):
    """All episodes run in CARLA's synchronous mode so that `world.tick()`
    advances the simulation by exactly `fixed_delta_seconds`, keeping
    metrics (dt-dependent: jerk, TTC) reproducible across runs and not
    subject to wall-clock/render-time jitter.
    """
    settings = world.get_settings()
    original = carla.WorldSettings(
        synchronous_mode=settings.synchronous_mode,
        fixed_delta_seconds=settings.fixed_delta_seconds,
    )
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = fixed_delta_seconds
    world.apply_settings(settings)
    try:
        yield
    finally:
        world.apply_settings(original)


def load_world(client: "carla.Client", town: str) -> "carla.World":
    world = client.get_world()
    current = world.get_map().name.split("/")[-1]
    if current != town:
        world = client.load_world(town)
    return world


def spawn_vehicle(
    world: "carla.World",
    transform: "carla.Transform",
    model: str = "vehicle.tesla.model3",
    role_name: str | None = None,
):
    bp_library = world.get_blueprint_library()
    candidates = bp_library.filter(model)
    if not candidates:
        candidates = bp_library.filter("vehicle.*")
    bp = candidates[0]
    if role_name and bp.has_attribute("role_name"):
        bp.set_attribute("role_name", role_name)
    return world.try_spawn_actor(bp, transform)


def destroy_actors(actors: list) -> None:
    for actor in actors:
        if actor is not None and actor.is_alive:
            actor.destroy()


def set_seed(seed: int) -> None:
    random.seed(seed)


def nearest_vehicle_ahead(world: "carla.World", vehicle, max_distance_m: float = 100.0):
    """Finds the nearest other vehicle ahead of `vehicle` in roughly the
    same lane, by projecting onto the vehicle's forward vector. Used to
    locate the "follower" vehicles for the string-effect metric and to
    build IDM's gap-ahead observation.
    """
    ego_tf = vehicle.get_transform()
    ego_loc = ego_tf.location
    forward = ego_tf.get_forward_vector()

    best = None
    best_dist = max_distance_m
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.id == vehicle.id:
            continue
        other_loc = actor.get_transform().location
        delta = other_loc - ego_loc
        along = delta.x * forward.x + delta.y * forward.y + delta.z * forward.z
        if along <= 0:
            continue  # behind us
        lateral_sq = (delta.x - along * forward.x) ** 2 + (delta.y - along * forward.y) ** 2
        if lateral_sq > 6.25:  # > ~2.5 m off the driving axis -> different lane
            continue
        if along < best_dist:
            best_dist = along
            best = actor
    return best, best_dist if best is not None else float("inf")


def nearest_vehicle_behind(world: "carla.World", vehicle, max_distance_m: float = 100.0):
    """Mirror of `nearest_vehicle_ahead`, used to find the immediate and
    second follower for the string-effect metric.
    """
    ego_tf = vehicle.get_transform()
    ego_loc = ego_tf.location
    forward = ego_tf.get_forward_vector()

    best = None
    best_dist = max_distance_m
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.id == vehicle.id:
            continue
        other_loc = actor.get_transform().location
        delta = ego_loc - other_loc
        along = delta.x * forward.x + delta.y * forward.y + delta.z * forward.z
        if along <= 0:
            continue  # ahead of us, not behind
        lateral_sq = (delta.x - along * forward.x) ** 2 + (delta.y - along * forward.y) ** 2
        if lateral_sq > 6.25:
            continue
        if along < best_dist:
            best_dist = along
            best = actor
    return best, best_dist if best is not None else float("inf")
