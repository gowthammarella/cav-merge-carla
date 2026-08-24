"""Runs one (strategy, density, seed) episode end-to-end in CARLA and
returns a flat metrics dict, tagged with the episode's identifying keys and
the negotiation strategy's response-type trace (when applicable) for
failure-mode / utility-breakdown analysis later.

Requires `carla`; first executable on Colab.
"""
from __future__ import annotations

import time
from typing import Optional

from merge_sim import carla_utils
from merge_sim.failure_modes import classify
from merge_sim.metrics import compute_metrics
from merge_sim.scenario import MAX_EPISODE_DURATION_S, MergeScenario
from merge_sim.strategies.negotiation import NegotiationStrategy


def run_episode(
    scenario: MergeScenario,
    strategy,
    density: str,
    seed: int,
    max_ticks: int = 2000,
) -> dict:
    scenario.reset(seed=seed, density=density)
    strategy.reset()

    ticks = 0
    wall_start = time.time()
    while not scenario.is_done() and ticks < max_ticks:
        scenario.tick(strategy)
        ticks += 1

    log = scenario.get_episode_log()
    metrics = compute_metrics(log)
    failure_mode = classify(log, MAX_EPISODE_DURATION_S)

    row = {
        "strategy": strategy.name,
        "density": density,
        "seed": seed,
        "ticks": ticks,
        "wall_clock_s": time.time() - wall_start,
        "failure_mode": failure_mode.value,
        **metrics,
    }
    if isinstance(strategy, NegotiationStrategy) and strategy.last_response is not None:
        row["last_response_type"] = strategy.last_response.type.value
        row["chosen_by_tie_break"] = strategy.last_response.chosen_by_tie_break

    return row


def run_and_record(
    client,
    strategy,
    density: str,
    seed: int,
    results_writer,
) -> dict:
    """Convenience wrapper: builds a fresh MergeScenario, runs one episode,
    writes the result row immediately via `results_writer(row)`, and
    cleans up actors. Use this from experiment_grid.py so a Colab
    disconnect mid-grid never loses more than the one in-flight episode.
    """
    scenario = MergeScenario(client)
    try:
        with carla_utils.synchronous_mode(scenario.world):
            row = run_episode(scenario, strategy, density, seed)
    finally:
        scenario.close()
    results_writer(row)
    return row
