"""Trains one IPPO policy pair (main-lane CAV + ramp CAV) for a single
communication condition. Each condition gets its own trained policy pair,
since the observation each agent sees differs by condition (A/B/C give
different information about the other agent) — the trained policies are
not interchangeable across conditions. Requires `carla`, `ultralytics`,
`torch`; first executable on Colab.
"""
from __future__ import annotations

import copy
import pickle
from pathlib import Path

from vision_comm.comm import CommCondition
from vision_comm.ippo import IPPOConfig, IPPOTrainer
from vision_comm.scenario import MultiAgentMergeScenario

DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def train_condition(
    client,
    condition: CommCondition,
    n_iterations: int = 50,
    steps_per_rollout: int = 256,
    seed: int = 0,
    models_dir: Path = DEFAULT_MODELS_DIR,
) -> Path:
    scenario = MultiAgentMergeScenario(client, condition)
    try:
        config = IPPOConfig(obs_dim=scenario.obs_dim, n_actions=scenario.n_actions)
        trainer = IPPOTrainer(scenario, config, seed=seed)

        # Checkpoint the BEST iteration, not the last one. PPO does not
        # improve monotonically, so saving whatever the final iteration
        # happened to produce can ship a policy that was worse than one
        # reached earlier in the same run.
        history = []
        best_score = float("-inf")
        best_state_dicts = None
        best_iteration = -1

        for iteration in range(n_iterations):
            rollout_stats = trainer.collect_rollout(steps_per_rollout)
            update_stats = trainer.update()
            history.append({**rollout_stats, "update": update_stats})

            returns = [
                rollout_stats[f"{aid}_mean_episode_return"] for aid in scenario.agent_ids
            ]
            # NaN means no episode finished inside this rollout — not a score
            if any(r != r for r in returns):
                continue
            score = sum(returns) / len(returns)
            if score > best_score:
                best_score = score
                best_iteration = iteration
                best_state_dicts = {
                    aid: copy.deepcopy(trainer.agents[aid].net.state_dict())
                    for aid in scenario.agent_ids
                }

        if best_state_dicts is None:  # no rollout ever completed an episode
            best_state_dicts = {
                aid: trainer.agents[aid].net.state_dict() for aid in scenario.agent_ids
            }

        models_dir.mkdir(parents=True, exist_ok=True)
        out_path = models_dir / f"ippo_condition_{condition.value}.pkl"
        with out_path.open("wb") as f:
            pickle.dump(
                {
                    "condition": condition.value,
                    "agent_state_dicts": best_state_dicts,
                    "config": config,
                    "history": history,
                    "best_iteration": best_iteration,
                    "best_mean_return": best_score,
                },
                f,
            )
        return out_path
    finally:
        scenario.close()


def load_trained_agents(path: Path):
    """Rebuilds `{agent_id: ActorCritic}` from a saved checkpoint, for use
    by `comm_experiment_grid.py`'s evaluation runs.
    """
    import torch

    from vision_comm.ippo import ActorCritic

    with path.open("rb") as f:
        checkpoint = pickle.load(f)

    config = checkpoint["config"]
    nets = {}
    for agent_id, state_dict in checkpoint["agent_state_dicts"].items():
        net = ActorCritic(config.obs_dim, config.n_actions, config.hidden_dim)
        net.load_state_dict(state_dict)
        net.eval()
        nets[agent_id] = net
    return nets, checkpoint["condition"]
