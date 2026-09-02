"""Standalone CARLA connectivity smoke test.

Must be run with the Python 3.10 venv's interpreter (CARLA 0.9.15's client
wheel only supports Python 3.7-3.10; Colab's default Python is newer), e.g.:

    /content/carla_venv/bin/python /content/fag-project/scripts/smoke_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import carla  # noqa: E402
from merge_sim import carla_utils  # noqa: E402


def main() -> None:
    client = carla_utils.connect(timeout=30.0)
    print("Server version:", client.get_server_version())

    world = client.get_world()
    with carla_utils.synchronous_mode(world, fixed_delta_seconds=0.05):
        spawn_points = world.get_map().get_spawn_points()
        vehicle = carla_utils.spawn_vehicle(world, spawn_points[0])
        assert vehicle is not None, "spawn failed — server may not be fully ready yet, wait longer and retry"
        world.tick()
        print("SMOKE TEST OK — spawned", vehicle.type_id, "at", vehicle.get_location())
        vehicle.destroy()


if __name__ == "__main__":
    main()
