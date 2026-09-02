"""Prints junction candidates for the on-ramp merge point so you can
visually confirm the auto-detected one is correct (or hardcode the right
`carla.Transform` into `MANUAL_MERGE_POINT` in `src/merge_sim/scenario.py`
if not). Run with the Python 3.10 venv's interpreter:

    /content/carla_venv/bin/python /content/fag-project/scripts/calibrate_merge_point.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from merge_sim import carla_utils  # noqa: E402
from merge_sim.scenario import find_merge_point_candidates  # noqa: E402


def main() -> None:
    client = carla_utils.connect(timeout=30.0)
    world = carla_utils.load_world(client, "Town04")
    candidates = find_merge_point_candidates(world)
    if not candidates:
        print("No junction candidates found.")
        return
    for i, wp in enumerate(candidates):
        loc = wp.transform.location
        print(f"[{i}] road_id={wp.road_id} lane_id={wp.lane_id} location=({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})")


if __name__ == "__main__":
    main()
