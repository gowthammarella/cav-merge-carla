"""Stretch strategy: wraps a trained stable-baselines3 policy behind the
Strategy interface, using the SAME discrete action vocabulary
(HOLD/YIELD/ACCELERATE) as `negotiation.py` so the comparison is
head-to-head rather than apples-to-oranges. The action-index mapping here
MUST match `rl_env.py`'s action space exactly.

`stable_baselines3` is imported lazily so this module (and the rest of the
package) stays importable without it installed until the learned strategy
is actually used.
"""
from __future__ import annotations

import numpy as np

from merge_sim.controller import HighLevelAction
from merge_sim.strategies.base import Strategy

YIELD_SPEED_MARGIN_MPS = 3.0
ACCELERATE_SPEED_FACTOR = 1.1

ACTION_HOLD = 0
ACTION_YIELD = 1
ACTION_ACCELERATE = 2


class LearnedStrategy(Strategy):
    name = "learned"

    def __init__(self, model_path: str):
        from stable_baselines3 import PPO

        self._model = PPO.load(model_path)

    def decide(self, main_lane_state, ramp_state, t: float) -> HighLevelAction:
        obs = np.array(
            [
                main_lane_state.own_speed_mps,
                main_lane_state.gap_ahead_m,
                ramp_state.ramp_speed_mps,
                ramp_state.gap_to_ramp_m,
                ramp_state.distance_to_merge_point_m,
            ],
            dtype=np.float32,
        )
        action, _ = self._model.predict(obs, deterministic=True)
        action = int(action)

        if action == ACTION_YIELD:
            target = max(0.0, ramp_state.ramp_speed_mps - YIELD_SPEED_MARGIN_MPS)
        elif action == ACTION_ACCELERATE:
            target = main_lane_state.desired_speed_mps * ACCELERATE_SPEED_FACTOR
        else:
            target = main_lane_state.desired_speed_mps

        return HighLevelAction(target_speed_mps=target)
