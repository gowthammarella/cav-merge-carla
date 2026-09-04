import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from vision_comm.ippo import IPPOConfig, IPPOTrainer, RolloutBuffer, compute_gae


class TinyMatchEnv:
    """A trivial 2-agent contextual-bandit env: each agent observes a
    random binary target and must output the matching action to get
    reward +1 (else -1). Fully solvable by a linear policy, so IPPO should
    converge to near-perfect accuracy in a handful of iterations — this is
    a correctness check on the PPO math, not a CARLA stand-in.
    """

    agent_ids = ["a0", "a1"]
    obs_dim = 1
    n_actions = 2

    def __init__(self, episode_len: int = 10, seed: int = 0):
        self.episode_len = episode_len
        self.rng = np.random.default_rng(seed)
        self._t = 0
        self._targets = {aid: int(self.rng.integers(0, 2)) for aid in self.agent_ids}

    def reset(self):
        self._t = 0
        self._targets = {aid: int(self.rng.integers(0, 2)) for aid in self.agent_ids}
        return {aid: np.array([float(self._targets[aid])], dtype=np.float32) for aid in self.agent_ids}

    def step(self, actions):
        rewards = {aid: (1.0 if actions[aid] == self._targets[aid] else -1.0) for aid in self.agent_ids}
        self._t += 1
        done = self._t >= self.episode_len
        dones = {aid: done for aid in self.agent_ids}
        self._targets = {aid: int(self.rng.integers(0, 2)) for aid in self.agent_ids}
        obs = {aid: np.array([float(self._targets[aid])], dtype=np.float32) for aid in self.agent_ids}
        return obs, rewards, dones, {}


def test_compute_gae_zero_reward_zero_advantage():
    advantages, returns = compute_gae(
        rewards=[0.0, 0.0, 0.0], values=[0.0, 0.0, 0.0], dones=[False, False, True],
        last_value=0.0, gamma=0.99, lam=0.95,
    )
    assert np.allclose(advantages, 0.0)
    assert np.allclose(returns, 0.0)


def test_compute_gae_single_step_matches_td_error():
    # single-step GAE reduces to the plain TD residual
    advantages, returns = compute_gae(
        rewards=[1.0], values=[0.0], dones=[False], last_value=0.0, gamma=0.9, lam=0.9,
    )
    assert np.isclose(advantages[0], 1.0)


def test_rollout_buffer_add_and_clear():
    buf = RolloutBuffer()
    buf.add(obs=np.array([0.0]), action=0, log_prob=-0.5, reward=1.0, value=0.1, done=False)
    assert len(buf) == 1
    buf.clear()
    assert len(buf) == 0


def test_ippo_trainer_runs_without_error():
    env = TinyMatchEnv(episode_len=5, seed=1)
    config = IPPOConfig(obs_dim=1, n_actions=2, minibatch_size=16)
    trainer = IPPOTrainer(env, config, seed=1)
    history = trainer.train(n_iterations=2, steps_per_rollout=32)
    assert len(history) == 2
    for entry in history:
        assert "update" in entry
        for agent_id in env.agent_ids:
            assert f"{agent_id}_mean_episode_return" in entry


def test_ippo_learns_the_trivial_matching_task():
    torch.manual_seed(0)
    env = TinyMatchEnv(episode_len=10, seed=0)
    config = IPPOConfig(obs_dim=1, n_actions=2, lr=1e-2, minibatch_size=32, n_epochs=4)
    trainer = IPPOTrainer(env, config, seed=0)
    trainer.train(n_iterations=25, steps_per_rollout=64)

    # after training, the greedy (argmax) policy should reliably match the
    # observed target for both agents
    correct = 0
    total = 0
    for agent_id in env.agent_ids:
        net = trainer.agents[agent_id].net
        for target in (0, 1):
            obs = torch.as_tensor([float(target)], dtype=torch.float32)
            logits, _ = net.forward(obs)
            predicted = int(torch.argmax(logits).item())
            correct += int(predicted == target)
            total += 1
    accuracy = correct / total
    assert accuracy >= 0.75, f"expected IPPO to mostly solve the trivial task, got accuracy={accuracy}"
