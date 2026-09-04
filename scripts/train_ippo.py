"""Trains one IPPO policy pair for one communication condition. Must be
run with the Python 3.10 venv's interpreter:

    /content/carla_venv/bin/python /content/fag-project/scripts/train_ippo.py --condition B --iterations 50

Run once per condition you want in the comparison (A/B/C for the MVP; D is
not implemented — see src/vision_comm/comm.py).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from merge_sim import carla_utils  # noqa: E402
from vision_comm.comm import CommCondition  # noqa: E402
from vision_comm.train import train_condition  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["A", "B", "C"], required=True)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--steps-per-rollout", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    client = carla_utils.connect(timeout=30.0)
    print("Connected:", client.get_server_version())

    condition = CommCondition(args.condition)
    out_path = train_condition(
        client, condition,
        n_iterations=args.iterations,
        steps_per_rollout=args.steps_per_rollout,
        seed=args.seed,
    )
    print("Saved trained policy pair to", out_path)


if __name__ == "__main__":
    main()
