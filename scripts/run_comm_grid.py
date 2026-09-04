"""Evaluates trained IPPO policy pairs across the communication-condition x
density x seed grid. Must be run with the Python 3.10 venv's interpreter,
AFTER training each condition with scripts/train_ippo.py:

    /content/carla_venv/bin/python /content/fag-project/scripts/run_comm_grid.py --seeds 1
    /content/carla_venv/bin/python /content/fag-project/scripts/run_comm_grid.py --seeds 20

Resumable: safe to re-run after a Colab disconnect — already-completed
(condition, density, seed) rows in results/comm_episodes.csv are skipped.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from merge_sim import carla_utils  # noqa: E402
from vision_comm.comm import CommCondition  # noqa: E402
from vision_comm.comm_experiment_grid import RESULTS_PATH, run_grid  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20, help="seeds per (condition, density) cell")
    parser.add_argument(
        "--conditions", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"]
    )
    args = parser.parse_args()

    client = carla_utils.connect(timeout=30.0)
    print("Connected:", client.get_server_version())
    print("Results ->", RESULTS_PATH)

    conditions = tuple(CommCondition(c) for c in args.conditions)
    run_grid(client, conditions=conditions, seeds=range(args.seeds))


if __name__ == "__main__":
    main()
