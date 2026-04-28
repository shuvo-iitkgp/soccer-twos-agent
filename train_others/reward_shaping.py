from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import gym
import numpy as np


@dataclass(frozen=True)
class RewardShapingConfig:
    """Configuration for dense reward shaping terms."""

    alpha: float = 0.2
    beta: float = 0.1
    direction_coeff: float = 0.01
    ball_touch_bonus: float = 0.2

    max_dist: float = 30.0
    touch_threshold: float = 1.5
    touch_cooldown_steps: int = 15

    proximity_clip: Tuple[float, float] = (-0.1, 0.1)
    goal_progress_clip: Tuple[float, float] = (-0.1, 0.1)
    direction_clip: Tuple[float, float] = (-0.01, 0.01)
    shaping_clip: Tuple[float, float] = (-0.3, 0.3)

    opponent_goal_x: float = 16.0
    opponent_goal_y: float = 0.0

    debug_print_every: int = 0


def opponent_goal_position(config: RewardShapingConfig) -> np.ndarray:
    return np.array([config.opponent_goal_x, config.opponent_goal_y], dtype=np.float32)


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


class RewardShapingWrapper(gym.Wrapper):
    """Gym wrapper that adds clipped reward-shaping terms to the env reward."""

    def __init__(self, env: gym.Env, config: Optional[RewardShapingConfig] = None):
        super().__init__(env)
        self.config = config or RewardShapingConfig()

        self._prev_player_pos: Optional[np.ndarray] = None
        self._prev_ball_pos: Optional[np.ndarray] = None
        self._prev_ball_vel: Optional[np.ndarray] = None

        self._debug_step_count = 0
        self._last_touch_step = -10**9

    def reset(self, **kwargs):
        self._prev_player_pos = None
        self._prev_ball_pos = None
        self._prev_ball_vel = None

        reset_out = self.env.reset(**kwargs)

        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            observation, info = reset_out
            return observation, info

        return reset_out

    def step(self, action):
        observation, env_reward, done, info = self.env.step(action)
        info_dict: Dict = dict(info) if isinstance(info, dict) else {}

        proximity_reward = 0.0
        goal_progress_reward = 0.0
        direction_reward = 0.0
        ball_touch_reward = 0.0

        proximity_delta = None
        goal_delta = None
        proximity_raw = None
        goal_progress_raw = None
        direction_raw = None

        player_info = info_dict.get("player_info", {})
        ball_info = info_dict.get("ball_info", {})

        if "position" in player_info and "position" in ball_info:
            curr_player_pos = np.asarray(player_info["position"], dtype=np.float32)
            curr_ball_pos = np.asarray(ball_info["position"], dtype=np.float32)
            curr_ball_vel = np.asarray(ball_info.get("velocity", [0.0, 0.0]), dtype=np.float32)

            goal = opponent_goal_position(self.config)
            max_dist = max(float(self.config.max_dist), 1e-6)

            if self._prev_player_pos is not None and self._prev_ball_pos is not None:
                proximity_delta = (
                    _distance(self._prev_player_pos, self._prev_ball_pos)
                    - _distance(curr_player_pos, curr_ball_pos)
                ) 

                goal_delta = (
                    _distance(self._prev_ball_pos, goal)
                    - _distance(curr_ball_pos, goal)
                ) 
                

                proximity_raw = self.config.alpha * proximity_delta
                goal_progress_raw = self.config.beta * goal_delta

                proximity_reward = float(
                    np.clip(proximity_raw, self.config.proximity_clip[0], self.config.proximity_clip[1])
                )
                goal_progress_reward = float(
                    np.clip(goal_progress_raw, self.config.goal_progress_clip[0], self.config.goal_progress_clip[1])
                )

                goal_direction = goal - curr_ball_pos
                goal_direction = goal_direction / (np.linalg.norm(goal_direction) + 1e-6)

                direction_raw = self.config.direction_coeff * float(np.dot(curr_ball_vel, goal_direction))
                direction_reward = float(
                    np.clip(direction_raw, self.config.direction_clip[0], self.config.direction_clip[1])
                )

            dist_agent_ball = np.linalg.norm(curr_player_pos - curr_ball_pos)
            if (
                dist_agent_ball < self.config.touch_threshold
                and (self._debug_step_count - self._last_touch_step) > self.config.touch_cooldown_steps
            ):
                ball_touch_reward = float(self.config.ball_touch_bonus)
                self._last_touch_step = self._debug_step_count

            self._prev_player_pos = curr_player_pos
            self._prev_ball_pos = curr_ball_pos
            self._prev_ball_vel = curr_ball_vel

        shaping_unclipped = (
            proximity_reward
            + goal_progress_reward
            + direction_reward
            + ball_touch_reward
        )
        shaping_reward = float(
            np.clip(shaping_unclipped, self.config.shaping_clip[0], self.config.shaping_clip[1])
        )
        shaped_reward = float(env_reward) + shaping_reward

        info_dict["env_reward"] = float(env_reward)
        info_dict["proximity_reward"] = float(proximity_reward)
        info_dict["goal_progress_reward"] = float(goal_progress_reward)
        info_dict["direction_reward"] = float(direction_reward)
        info_dict["ball_touch_reward"] = float(ball_touch_reward)
        info_dict["shaping_reward"] = float(shaping_reward)
        info_dict["shaped_reward"] = float(shaped_reward)
        info_dict["opponent_goal_x"] = float(self.config.opponent_goal_x)
        info_dict["opponent_goal_y"] = float(self.config.opponent_goal_y)

        self._debug_step_count += 1

        if self.config.debug_print_every > 0 and self._debug_step_count % self.config.debug_print_every == 0:
            print(
                f"[shape-debug] step={self._debug_step_count} "
                f"has_player_pos={'position' in player_info} "
                f"has_ball_pos={'position' in ball_info} "
                f"env={float(env_reward):+.6f} "
                f"prox_delta={proximity_delta} "
                f"goal_delta={goal_delta} "
                f"prox_raw={proximity_raw} "
                f"goal_raw={goal_progress_raw} "
                f"dir_raw={direction_raw} "
                f"prox_clip={proximity_reward:+.6f} "
                f"goal_clip={goal_progress_reward:+.6f} "
                f"dir={direction_reward:+.6f} "
                f"touch={ball_touch_reward:+.6f} "
                f"shape={shaping_reward:+.6f} "
                f"total={shaped_reward:+.6f}"
            )

        return observation, shaped_reward, done, info_dict
