"""Trains one IPPO policy pair (main-lane CAV + ramp CAV) for a single
communication condition. Each condition gets its own trained policy pair,
since the observation each agent sees differs by condition (A/B/C give
different information about the other agent) — the trained policies are
not interchangeable across conditions. Requires `carla`, `ultralytics`,
`torch`; first executable on Colab.
"""
from __future__ import annotations

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
        history = trainer.train(n_iterations=n_iterations, steps_per_rollout=steps_per_rollout)

        models_dir.mkdir(parents=True, exist_ok=True)
        out_path = models_dir / f"ippo_condition_{condition.value}.pkl"
        with out_path.open("wb") as f:
            pickle.dump(
                {
                    "condition": condition.value,
                    "agent_state_dicts": {
                        aid: trainer.agents[aid].net.state_dict() for aid in scenario.agent_ids
                    },
                    "config": config,
                    "history": history,
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
