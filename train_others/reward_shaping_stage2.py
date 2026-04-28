from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import gym
import numpy as np


@dataclass(frozen=True)
class RewardShapingConfig:
    """
    Configuration for dense reward shaping terms.

    Stage 1 (vs random, offensive focus):
        alpha=0.2, beta=0.1, direction_coeff=0.01, ball_touch_bonus=0.2

    Stage 2 (curriculum, defensive added):
        Offensive weights reduced ~40-50%, three defensive terms added.
    """

    # --- Offensive terms (reduced ~40% from Stage 1) ---
    alpha: float = 0.12          # proximity: was 0.2
    beta: float = 0.06           # goal progress: was 0.1
    direction_coeff: float = 0.006  # direction: was 0.01
    ball_touch_bonus: float = 0.12  # touch: was 0.2

    max_dist: float = 30.0
    touch_threshold: float = 1.5
    touch_cooldown_steps: int = 15

    # --- Offensive clips (tightened to make room for defensive signal) ---
    proximity_clip: Tuple[float, float] = (-0.06, 0.06)
    goal_progress_clip: Tuple[float, float] = (-0.06, 0.06)
    direction_clip: Tuple[float, float] = (-0.006, 0.006)
    shaping_clip: Tuple[float, float] = (-0.3, 0.3)

    # --- Goal coordinates ---
    opponent_goal_x: float = 16.0
    opponent_goal_y: float = 0.0

    # ------------------------------------------------------------------ #
    # Defensive terms (NEW in Stage 2)
    # ------------------------------------------------------------------ #

    # A. Goal-side reward
    # Small positive reward when agent is goal-side of the ball
    # (between ball and own goal) while ball is in defensive half.
    goal_side_coeff: float = 0.02
    goal_side_clip: Tuple[float, float] = (-0.02, 0.02)
    # Ball must be in defensive half (signed x < defensive_zone_x) to activate
    defensive_zone_x: float = 0.0   # ball x must be < 0 (own half)

    # B. Clearance reward
    # Reward ball velocity away from own goal in defensive danger zone.
    clearance_coeff: float = 0.03
    clearance_clip: Tuple[float, float] = (-0.03, 0.03)
    # Only activates when ball is within danger_radius of own goal
    danger_radius: float = 8.0

    # C. Danger touch penalty
    # Negative reward if agent touches ball near own goal AND ball moves toward own goal.
    danger_touch_penalty: float = -0.03
    danger_touch_clip: Tuple[float, float] = (-0.03, 0.0)

    debug_print_every: int = 0


def opponent_goal_position(config: RewardShapingConfig) -> np.ndarray:
    return np.array([config.opponent_goal_x, config.opponent_goal_y], dtype=np.float32)


def own_goal_position(config: RewardShapingConfig) -> np.ndarray:
    """Own goal is the opposite end from the opponent goal."""
    return np.array([-config.opponent_goal_x, config.opponent_goal_y], dtype=np.float32)


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


class RewardShapingWrapper(gym.Wrapper):
    """
    Stage 2 curriculum reward shaping wrapper.

    Offensive terms are reduced ~40% from Stage 1.
    Three new defensive terms are added:
      - goal_side_reward: be goal-side of ball in defensive half
      - clearance_reward: kick ball away from own goal in danger zone
      - danger_touch_penalty: penalise bad touches near own goal
    """

    def __init__(self, env: gym.Env, config: Optional[RewardShapingConfig] = None):
        super().__init__(env)
        self.config = config or RewardShapingConfig()

        self._prev_player_pos: Optional[np.ndarray] = None
        self._prev_ball_pos: Optional[np.ndarray] = None
        self._prev_ball_vel: Optional[np.ndarray] = None

        self._debug_step_count = 0
        self._last_touch_step = -10 ** 9

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

        # ------------------------------------------------------------------ #
        # Initialise all shaping terms to 0
        # ------------------------------------------------------------------ #
        proximity_reward = 0.0
        goal_progress_reward = 0.0
        direction_reward = 0.0
        ball_touch_reward = 0.0
        goal_side_reward = 0.0
        clearance_reward = 0.0
        danger_touch_penalty = 0.0

        player_info = info_dict.get("player_info", {})
        ball_info = info_dict.get("ball_info", {})

        if "position" in player_info and "position" in ball_info:
            curr_player_pos = np.asarray(player_info["position"], dtype=np.float32)
            curr_ball_pos = np.asarray(ball_info["position"], dtype=np.float32)
            curr_ball_vel = np.asarray(
                ball_info.get("velocity", [0.0, 0.0]), dtype=np.float32
            )

            opp_goal = opponent_goal_position(self.config)
            own_goal = own_goal_position(self.config)
            max_dist = max(float(self.config.max_dist), 1e-6)

            # -------------------------------------------------------------- #
            # Offensive terms (same logic, reduced coefficients)
            # -------------------------------------------------------------- #
            if self._prev_player_pos is not None and self._prev_ball_pos is not None:
                proximity_delta = (
                    _distance(self._prev_player_pos, self._prev_ball_pos)
                    - _distance(curr_player_pos, curr_ball_pos)
                )
                goal_delta = (
                    _distance(self._prev_ball_pos, opp_goal)
                    - _distance(curr_ball_pos, opp_goal)
                )

                proximity_raw = self.config.alpha * proximity_delta
                goal_progress_raw = self.config.beta * goal_delta

                proximity_reward = float(
                    np.clip(
                        proximity_raw,
                        self.config.proximity_clip[0],
                        self.config.proximity_clip[1],
                    )
                )
                goal_progress_reward = float(
                    np.clip(
                        goal_progress_raw,
                        self.config.goal_progress_clip[0],
                        self.config.goal_progress_clip[1],
                    )
                )

                goal_dir = opp_goal - curr_ball_pos
                goal_dir = goal_dir / (np.linalg.norm(goal_dir) + 1e-6)
                direction_raw = self.config.direction_coeff * float(
                    np.dot(curr_ball_vel, goal_dir)
                )
                direction_reward = float(
                    np.clip(
                        direction_raw,
                        self.config.direction_clip[0],
                        self.config.direction_clip[1],
                    )
                )

            # Touch bonus (offensive)
            dist_agent_ball = float(np.linalg.norm(curr_player_pos - curr_ball_pos))
            touch_this_step = (
                dist_agent_ball < self.config.touch_threshold
                and (self._debug_step_count - self._last_touch_step)
                > self.config.touch_cooldown_steps
            )
            if touch_this_step:
                ball_touch_reward = float(self.config.ball_touch_bonus)
                self._last_touch_step = self._debug_step_count

            # -------------------------------------------------------------- #
            # A. Goal-side reward (NEW)
            # Reward agent for being between ball and own goal
            # when ball is in defensive half.
            # -------------------------------------------------------------- #
            attack_sign = 1.0 if self.config.opponent_goal_x >= 0.0 else -1.0
            ball_signed_x = attack_sign * float(curr_ball_pos[0])
            ball_in_defensive_half = ball_signed_x < (
                attack_sign * self.config.defensive_zone_x
            )

            if ball_in_defensive_half:
                # Agent is "goal-side" if it is closer to own goal than ball is
                agent_dist_own = _distance(curr_player_pos, own_goal)
                ball_dist_own = _distance(curr_ball_pos, own_goal)
                if agent_dist_own < ball_dist_own:
                    # Scale reward by how far ball is into defensive half
                    depth = max(0.0, -ball_signed_x) / max(abs(self.config.opponent_goal_x), 1e-6)
                    goal_side_raw = self.config.goal_side_coeff * depth
                    goal_side_reward = float(
                        np.clip(
                            goal_side_raw,
                            self.config.goal_side_clip[0],
                            self.config.goal_side_clip[1],
                        )
                    )

            # -------------------------------------------------------------- #
            # B. Clearance reward (NEW)
            # Reward ball velocity away from own goal in danger zone.
            # -------------------------------------------------------------- #
            dist_ball_own = _distance(curr_ball_pos, own_goal)
            ball_in_danger = dist_ball_own < self.config.danger_radius

            if ball_in_danger:
                # Direction from own goal toward opponent goal (clearance direction)
                clearance_dir = opp_goal - own_goal
                clearance_dir = clearance_dir / (np.linalg.norm(clearance_dir) + 1e-6)
                ball_vel_toward_clear = float(np.dot(curr_ball_vel, clearance_dir))
                clearance_raw = self.config.clearance_coeff * ball_vel_toward_clear
                clearance_reward = float(
                    np.clip(
                        clearance_raw,
                        self.config.clearance_clip[0],
                        self.config.clearance_clip[1],
                    )
                )

            # -------------------------------------------------------------- #
            # C. Danger touch penalty (NEW)
            # Penalise touching ball near own goal if ball moves toward own goal.
            # -------------------------------------------------------------- #
            if touch_this_step and ball_in_danger:
                # Check ball velocity direction relative to own goal
                toward_own_goal_dir = own_goal - curr_ball_pos
                toward_own_goal_dir = toward_own_goal_dir / (
                    np.linalg.norm(toward_own_goal_dir) + 1e-6
                )
                ball_vel_toward_own = float(np.dot(curr_ball_vel, toward_own_goal_dir))
                if ball_vel_toward_own > 0:
                    # Ball is moving toward own goal after touch — penalise
                    danger_touch_raw = self.config.danger_touch_penalty * (
                        ball_vel_toward_own / (np.linalg.norm(curr_ball_vel) + 1e-6)
                    )
                    danger_touch_penalty = float(
                        np.clip(
                            danger_touch_raw,
                            self.config.danger_touch_clip[0],
                            self.config.danger_touch_clip[1],
                        )
                    )

            # Update previous state
            self._prev_player_pos = curr_player_pos
            self._prev_ball_pos = curr_ball_pos
            self._prev_ball_vel = curr_ball_vel

        # ------------------------------------------------------------------ #
        # Combine all shaping terms
        # ------------------------------------------------------------------ #
        shaping_unclipped = (
            proximity_reward
            + goal_progress_reward
            + direction_reward
            + ball_touch_reward
            + goal_side_reward
            + clearance_reward
            + danger_touch_penalty
        )
        shaping_reward = float(
            np.clip(
                shaping_unclipped,
                self.config.shaping_clip[0],
                self.config.shaping_clip[1],
            )
        )
        shaped_reward = float(env_reward) + shaping_reward

        # ------------------------------------------------------------------ #
        # Log everything to info_dict
        # ------------------------------------------------------------------ #
        info_dict["env_reward"] = float(env_reward)
        info_dict["proximity_reward"] = float(proximity_reward)
        info_dict["goal_progress_reward"] = float(goal_progress_reward)
        info_dict["direction_reward"] = float(direction_reward)
        info_dict["ball_touch_reward"] = float(ball_touch_reward)
        info_dict["goal_side_reward"] = float(goal_side_reward)
        info_dict["clearance_reward"] = float(clearance_reward)
        info_dict["danger_touch_penalty"] = float(danger_touch_penalty)
        info_dict["shaping_reward"] = float(shaping_reward)
        info_dict["shaped_reward"] = float(shaped_reward)
        info_dict["opponent_goal_x"] = float(self.config.opponent_goal_x)
        info_dict["opponent_goal_y"] = float(self.config.opponent_goal_y)

        self._debug_step_count += 1

        if (
            self.config.debug_print_every > 0
            and self._debug_step_count % self.config.debug_print_every == 0
        ):
            print(
                f"[shape-debug] step={self._debug_step_count} "
                f"env={float(env_reward):+.4f} "
                f"prox={proximity_reward:+.4f} "
                f"goal={goal_progress_reward:+.4f} "
                f"dir={direction_reward:+.4f} "
                f"touch={ball_touch_reward:+.4f} "
                f"goal_side={goal_side_reward:+.4f} "
                f"clear={clearance_reward:+.4f} "
                f"danger={danger_touch_penalty:+.4f} "
                f"shape={shaping_reward:+.4f} "
                f"total={shaped_reward:+.4f}"
            )

        return observation, shaped_reward, done, info_dict
