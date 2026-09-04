"""A minimal, from-scratch Independent PPO (IPPO) trainer in PyTorch.

IPPO is the standard MARL baseline the mentor's brief names: each agent
runs its own PPO learner (own actor-critic network, own rollout buffer, own
clipped-surrogate update) while all agents step a SHARED, synchronized
environment together. This is deliberately not stable-baselines3 — SB3
assumes it alone drives its env's stepping, which doesn't fit two agents
that must each see the same CARLA tick's consequences of the OTHER agent's
action. This module has no CARLA dependency: it operates on any env that
exposes the small `MultiAgentEnv` protocol below, so it's tested here
against a tiny synthetic 2-agent env, not CARLA.

Scope note: this is intentionally the plain/vanilla version of PPO (no
GAE-lambda tuning knobs beyond the standard ones, no LSTM/recurrent
policies, no parameter sharing across agents) — matching the mentor's
"use a standard MARL baseline" instruction rather than a research-grade
PPO implementation with all the extensions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiAgentEnv(Protocol):
    """Minimal environment protocol IPPO needs. `agent_ids` is fixed for
    the env's lifetime. `reset()` returns {agent_id: obs}. `step(actions)`
    takes {agent_id: action} and returns
    ({agent_id: obs}, {agent_id: reward}, {agent_id: done}, info).
    """

    agent_ids: list[str]
    obs_dim: int
    n_actions: int

    def reset(self) -> dict[str, np.ndarray]: ...

    def step(
        self, actions: dict[str, int]
    ) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, bool], dict]: ...


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(obs)
        return self.policy_head(features), self.value_head(features).squeeze(-1)

    def act(self, obs: torch.Tensor) -> tuple[int, float, float]:
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return int(action.item()), float(dist.log_prob(action).item()), float(value.item())

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        logits, values = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values


@dataclass
class RolloutBuffer:
    obs: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    values: list = field(default_factory=list)
    dones: list = field(default_factory=list)

    def add(self, obs, action, log_prob, reward, value, done) -> None:
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self) -> None:
        self.__init__()

    def __len__(self) -> int:
        return len(self.obs)


def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    last_value: float,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation. Returns (advantages, returns)."""
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    gae = 0.0
    next_value = last_value
    for t in reversed(range(n)):
        mask = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
        next_value = values[t]
    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


@dataclass
class IPPOConfig:
    obs_dim: int
    n_actions: int
    hidden_dim: int = 64
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    n_epochs: int = 4
    minibatch_size: int = 32
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5


class IPPOAgent:
    """One agent's independent PPO learner: its own network, optimizer,
    and rollout buffer.
    """

    def __init__(self, config: IPPOConfig, seed: int = 0):
        torch.manual_seed(seed)
        self.config = config
        self.net = ActorCritic(config.obs_dim, config.n_actions, config.hidden_dim)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config.lr)
        self.buffer = RolloutBuffer()

    def act(self, obs: np.ndarray) -> tuple[int, float, float]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        return self.net.act(obs_t.squeeze(0)) if obs_t.dim() == 1 else self.net.act(obs_t)

    def update(self, last_value: float) -> dict:
        buf = self.buffer
        advantages, returns = compute_gae(
            buf.rewards, buf.values, buf.dones, last_value,
            gamma=self.config.gamma, lam=self.config.gae_lambda,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs = torch.as_tensor(np.array(buf.obs), dtype=torch.float32)
        actions = torch.as_tensor(np.array(buf.actions), dtype=torch.long)
        old_log_probs = torch.as_tensor(np.array(buf.log_probs), dtype=torch.float32)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32)
        returns_t = torch.as_tensor(returns, dtype=torch.float32)

        n = len(buf)
        losses = []
        for _ in range(self.config.n_epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.config.minibatch_size):
                batch_idx = indices[start : start + self.config.minibatch_size]
                batch_idx_t = torch.as_tensor(batch_idx, dtype=torch.long)

                new_log_probs, entropy, values = self.net.evaluate(
                    obs[batch_idx_t], actions[batch_idx_t]
                )
                ratio = torch.exp(new_log_probs - old_log_probs[batch_idx_t])
                batch_adv = advantages_t[batch_idx_t]

                surrogate_1 = ratio * batch_adv
                surrogate_2 = torch.clamp(
                    ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio
                ) * batch_adv
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
                value_loss = F.mse_loss(values, returns_t[batch_idx_t])
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    + self.config.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                losses.append(float(loss.item()))

        self.buffer.clear()
        return {"mean_loss": float(np.mean(losses)) if losses else 0.0}


class IPPOTrainer:
    """Drives N `IPPOAgent`s over a shared `MultiAgentEnv` for a fixed
    number of environment steps per rollout, then updates every agent.
    """

    def __init__(self, env: MultiAgentEnv, config: IPPOConfig, seed: int = 0):
        self.env = env
        self.agents = {
            agent_id: IPPOAgent(config, seed=seed + i)
            for i, agent_id in enumerate(env.agent_ids)
        }
        self._obs = env.reset()

    def collect_rollout(self, n_steps: int) -> dict[str, float]:
        episode_returns = {agent_id: 0.0 for agent_id in self.env.agent_ids}
        completed_returns: dict[str, list[float]] = {aid: [] for aid in self.env.agent_ids}

        for _ in range(n_steps):
            actions, log_probs, values = {}, {}, {}
            for agent_id in self.env.agent_ids:
                action, log_prob, value = self.agents[agent_id].act(self._obs[agent_id])
                actions[agent_id] = action
                log_probs[agent_id] = log_prob
                values[agent_id] = value

            next_obs, rewards, dones, _info = self.env.step(actions)

            for agent_id in self.env.agent_ids:
                self.agents[agent_id].buffer.add(
                    self._obs[agent_id],
                    actions[agent_id],
                    log_probs[agent_id],
                    rewards[agent_id],
                    values[agent_id],
                    dones[agent_id],
                )
                episode_returns[agent_id] += rewards[agent_id]
                if dones[agent_id]:
                    completed_returns[agent_id].append(episode_returns[agent_id])
                    episode_returns[agent_id] = 0.0

            self._obs = next_obs

        return {
            f"{agent_id}_mean_episode_return": (
                float(np.mean(completed_returns[agent_id])) if completed_returns[agent_id] else float("nan")
            )
            for agent_id in self.env.agent_ids
        }

    def update(self) -> dict[str, dict]:
        results = {}
        for agent_id in self.env.agent_ids:
            last_value = self.agents[agent_id].net.act(
                torch.as_tensor(self._obs[agent_id], dtype=torch.float32)
            )[2]
            results[agent_id] = self.agents[agent_id].update(last_value)
        return results

    def train(self, n_iterations: int, steps_per_rollout: int) -> list[dict]:
        history = []
        for _ in range(n_iterations):
            rollout_stats = self.collect_rollout(steps_per_rollout)
            update_stats = self.update()
            history.append({**rollout_stats, "update": update_stats})
        return history
