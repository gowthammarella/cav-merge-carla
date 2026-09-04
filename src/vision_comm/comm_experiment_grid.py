"""Resumable strategy-grid runner for the vision + communication-richness
track: evaluates each trained IPPO policy pair (one per CommCondition)
across densities x seeds, mirroring `merge_sim/experiment_grid.py`'s
append-only-CSV resumability so a Colab disconnect only costs the one
in-flight episode. Requires `carla`, `ultralytics`, `torch`; first
executable on Colab.
"""
from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import torch

from vision_comm.comm import CommCondition
from vision_comm.scenario import MultiAgentMergeScenario
from vision_comm.train import DEFAULT_MODELS_DIR, load_trained_agents

DENSITIES = ("light", "medium", "heavy")
DEFAULT_SEEDS = range(20)
DEFAULT_CONDITIONS = (CommCondition.LOCAL_ONLY, CommCondition.STATE, CommCondition.OBJECT)

RESULTS_PATH = Path(__file__).resolve().parents[2] / "results" / "comm_episodes.csv"

RESULT_FIELDS = [
    "condition", "density", "seed", "ticks", "wall_clock_s",
    "merge_success", "collision", "merge_time_s",
    "mean_detection_precision", "mean_detection_recall",
    "id_switches_main", "id_switches_ramp",
    "comm_total_bytes", "comm_mean_bytes_per_message",
    "comm_messages_per_second", "comm_bytes_per_second",
]


def _greedy_action(net, obs) -> int:
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        logits, _ = net.forward(obs_t)
        return int(torch.argmax(logits).item())


def run_episode(
    scenario: MultiAgentMergeScenario, nets: dict, density: str, seed: int, max_ticks: int = 2000
) -> dict:
    obs = scenario.reset(seed=seed, density=density)
    ticks = 0
    wall_start = time.time()
    while not scenario.is_done() and ticks < max_ticks:
        actions = {aid: _greedy_action(nets[aid], obs[aid]) for aid in scenario.agent_ids}
        obs, _rewards, _dones, _info = scenario.step(actions)
        ticks += 1

    metrics = scenario.get_episode_metrics()
    return {
        "density": density,
        "seed": seed,
        "ticks": ticks,
        "wall_clock_s": time.time() - wall_start,
        **metrics,
    }


def _load_completed_keys(path: Path) -> set:
    if not path.exists():
        return set()
    completed = set()
    with path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            completed.add((row["condition"], row["density"], int(row["seed"])))
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
    conditions=DEFAULT_CONDITIONS,
    densities=DENSITIES,
    seeds=DEFAULT_SEEDS,
    models_dir: Path = DEFAULT_MODELS_DIR,
    results_path: Path = RESULTS_PATH,
) -> None:
    completed = _load_completed_keys(results_path)
    writer, fh = _make_writer(results_path)
    try:
        for condition in conditions:
            model_path = models_dir / f"ippo_condition_{condition.value}.pkl"
            if condition is not CommCondition.LOCAL_ONLY and not model_path.exists():
                print(f"skipping {condition.value}: no trained model at {model_path} — run train.py first")
                continue

            scenario = MultiAgentMergeScenario(client, condition)
            try:
                if model_path.exists():
                    nets, _ = load_trained_agents(model_path)
                else:
                    # LOCAL_ONLY can be evaluated with an untrained (random-init)
                    # policy pair too, but for a fair comparison you should still
                    # train it — this branch only fires if you skip that step.
                    from vision_comm.ippo import ActorCritic, IPPOConfig

                    config = IPPOConfig(obs_dim=scenario.obs_dim, n_actions=scenario.n_actions)
                    nets = {
                        aid: ActorCritic(config.obs_dim, config.n_actions, config.hidden_dim)
                        for aid in scenario.agent_ids
                    }

                for density in densities:
                    for seed in seeds:
                        key = (condition.value, density, seed)
                        if key in completed:
                            continue
                        row = run_episode(scenario, nets, density, seed)
                        row["condition"] = condition.value
                        writer(row)
                        print(f"done: {key} -> success={row['merge_success']}")
            finally:
                scenario.close()
    finally:
        fh.close()
