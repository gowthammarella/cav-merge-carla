"""Enumerates the full strategy x density x seed experiment grid, skips
combinations already present in results/raw_episodes.csv (so a Colab
disconnect only costs the one in-flight episode), and writes each new
result row to disk immediately (flushed + fsynced).

Requires `carla`; first executable on Colab.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Optional

from merge_sim.episode_runner import run_and_record
from merge_sim.negotiation_protocol import HDVUncertainty
from merge_sim.strategies.egoistic import EgoisticStrategy
from merge_sim.strategies.negotiation import NegotiationStrategy
from merge_sim.strategies.rule_based import RuleBasedStrategy

DENSITIES = ("light", "medium", "heavy")
DEFAULT_SEEDS = range(20)

RESULTS_PATH = Path(__file__).resolve().parents[2] / "results" / "raw_episodes.csv"

RESULT_FIELDS = [
    "strategy", "density", "seed", "ticks", "wall_clock_s", "failure_mode",
    "merge_success", "collision", "min_ttc_s", "completion_time_s",
    "hard_braking_main", "hard_braking_follower", "jerk_rms_main_mps3",
    "jerk_rms_follower_mps3", "string_effect_speed_drop_mps",
    "dangerous_near_miss", "last_response_type", "chosen_by_tie_break",
]


def default_strategy_factories(learned_model_path: Optional[str] = None) -> dict:
    """The core grid. Includes an uncertainty-ablation pair
    (`negotiation` vs `negotiation_no_uncertainty`) so the paper can
    directly quantify what risk-aware reasoning about HDV intent buys you,
    holding the utility weights and everything else fixed.
    """
    factories = {
        "egoistic": EgoisticStrategy,
        "rule_based": RuleBasedStrategy,
        "negotiation": NegotiationStrategy,
        "negotiation_no_uncertainty": lambda: NegotiationStrategy(
            uncertainty=HDVUncertainty(), name="negotiation_no_uncertainty"
        ),
    }
    if learned_model_path:
        from merge_sim.strategies.learned import LearnedStrategy

        factories["learned"] = lambda: LearnedStrategy(learned_model_path)
    return factories


def _load_completed_keys(path: Path) -> set:
    if not path.exists():
        return set()
    completed = set()
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add((row["strategy"], row["density"], int(row["seed"])))
    return completed


def _make_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    f = path.open("a", newline="")
    writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
    if write_header:
        writer.writeheader()
        f.flush()

    def write(row: dict) -> None:
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())

    return write, f


def run_grid(
    client,
    strategy_factories: Optional[dict] = None,
    densities=DENSITIES,
    seeds=DEFAULT_SEEDS,
    results_path: Path = RESULTS_PATH,
) -> None:
    """Resumable: safe to re-run after a Colab disconnect — already
    completed (strategy, density, seed) rows in `results_path` are skipped.
    """
    factories = strategy_factories or default_strategy_factories()
    completed = _load_completed_keys(results_path)
    writer, fh = _make_writer(results_path)
    try:
        for strategy_name, factory in factories.items():
            for density in densities:
                for seed in seeds:
                    key = (strategy_name, density, seed)
                    if key in completed:
                        continue
                    strategy = factory()
                    row = run_and_record(client, strategy, density, seed, writer)
                    print(
                        f"done: {key} -> success={row['merge_success']} "
                        f"failure_mode={row['failure_mode']}"
                    )
    finally:
        fh.close()
