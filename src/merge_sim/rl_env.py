"""Gymnasium wrapper around MergeScenario for training the stretch
"learned" high-level decision policy with PPO (stable-baselines3).

Deliberately scoped to a 3-action discrete space (HOLD/YIELD/ACCELERATE) —
identical vocabulary to the negotiation strategy and to
strategies/learned.py's action mapping — so the comparison between learned
and negotiation strategies is head-to-head, not confounded by a richer
action space. End-to-end control (steering/throttle) is explicitly out of
scope: the shared low-level controller (controller.py) still handles that,
exactly as it does for every other strategy.

Requires `carla` and `gymnasium`; first executable on Colab.
"""
from __future__ import annotations

import random
from typing import Optional

import gymnasium as gym
import numpy as np

from merge_sim.controller import HighLevelAction
from merge_sim.failure_modes import classify
from merge_sim.metrics import compute_metrics
from merge_sim.scenario import (
    DENSITY_INTER_ARRIVAL_S,
    MAX_EPISODE_DURATION_S,
    MergeScenario,
)

ACTION_HOLD = 0
ACTION_YIELD = 1
ACTION_ACCELERATE = 2

YIELD_SPEED_MARGIN_MPS = 3.0
ACCELERATE_SPEED_FACTOR = 1.1

HARD_BRAKE_PENALTY = -1.0
NEAR_MISS_PENALTY = -5.0
COLLISION_PENALTY = -50.0
SUCCESS_REWARD = 20.0
STEP_PENALTY = -0.01  # small per-tick cost so the policy doesn't stall indefinitely


class _PassthroughStrategy:
    """Adapter so MergeScenario.tick(strategy) can be driven one action at
    a time from the Gym env's step(), instead of a Strategy deciding
    autonomously every tick.
    """

    name = "learned_training"

    def __init__(self):
        self.pending_action: Optional[HighLevelAction] = None

    def reset(self):
        self.pending_action = None

    def decide(self, main_lane_state, ramp_state, t):
        return self.pending_action


class MergeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, client, densities=tuple(DENSITY_INTER_ARRIVAL_S), seed: Optional[int] = None):
        super().__init__()
        self.client = client
        self.densities = densities
        self._rng = random.Random(seed)
        self.scenario = MergeScenario(client)
        self._strategy = _PassthroughStrategy()

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([50.0, 200.0, 50.0, 200.0, 200.0], dtype=np.float32),
        )

    def _obs(self):
        main_state = self.scenario._observe_main_lane_state()
        ramp_state = self.scenario._observe_ramp_state()
        return np.array(
            [
                main_state.own_speed_mps,
                main_state.gap_ahead_m,
                ramp_state.ramp_speed_mps,
                ramp_state.gap_to_ramp_m,
                ramp_state.distance_to_merge_point_m,
            ],
            dtype=np.float32,
        ), main_state, ramp_state

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        episode_seed = seed if seed is not None else self._rng.randint(0, 2**31 - 1)
        density = self._rng.choice(self.densities)
        self.scenario.reset(seed=episode_seed, density=density)
        self._strategy.reset()
        obs, _, _ = self._obs()
        return obs, {"density": density, "seed": episode_seed}

    def step(self, action: int):
        obs, main_state, ramp_state = self._obs()

        if action == ACTION_YIELD:
            target = max(0.0, ramp_state.ramp_speed_mps - YIELD_SPEED_MARGIN_MPS)
        elif action == ACTION_ACCELERATE:
            target = main_state.desired_speed_mps * ACCELERATE_SPEED_FACTOR
        else:
            target = main_state.desired_speed_mps

        self._strategy.pending_action = HighLevelAction(target_speed_mps=target)
        self.scenario.tick(self._strategy)

        terminated = self.scenario.is_done()
        truncated = False
        reward = STEP_PENALTY

        if terminated:
            log = self.scenario.get_episode_log()
            metrics = compute_metrics(log)
            failure_mode = classify(log, MAX_EPISODE_DURATION_S)

            if metrics["collision"]:
                reward += COLLISION_PENALTY
            elif metrics["merge_success"]:
                reward += SUCCESS_REWARD
                if metrics["dangerous_near_miss"]:
                    reward += NEAR_MISS_PENALTY
            reward += HARD_BRAKE_PENALTY * metrics["hard_braking_main"]

            next_obs, _, _ = self._obs()
            info = {"metrics": metrics, "failure_mode": failure_mode.value}
            return next_obs, reward, terminated, truncated, info

        next_obs, _, _ = self._obs()
        return next_obs, reward, terminated, truncated, {}

    def close(self):
        self.scenario.close()
