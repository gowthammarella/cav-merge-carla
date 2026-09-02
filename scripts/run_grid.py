"""Runs the strategy x density x seed experiment grid. Must be run with
the Python 3.10 venv's interpreter:

    /content/carla_venv/bin/python /content/fag-project/scripts/run_grid.py --seeds 1
    /content/carla_venv/bin/python /content/fag-project/scripts/run_grid.py --seeds 20

Resumable: safe to re-run after a Colab disconnect — already-completed
(strategy, density, seed) rows in results/raw_episodes.csv are skipped.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from merge_sim import carla_utils  # noqa: E402
from merge_sim.experiment_grid import RESULTS_PATH, default_strategy_factories, run_grid  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20, help="seeds per (strategy, density) cell")
    parser.add_argument("--learned-model", type=str, default=None, help="path to a trained PPO .zip, optional")
    args = parser.parse_args()

    client = carla_utils.connect(timeout=30.0)
    print("Connected:", client.get_server_version())
    print("Results ->", RESULTS_PATH)

    factories = default_strategy_factories(learned_model_path=args.learned_model)
    run_grid(client, strategy_factories=factories, seeds=range(args.seeds))


if __name__ == "__main__":
    main()
