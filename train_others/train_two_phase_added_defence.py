import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Tuple
from my_agent_2 import RayAgent
import numpy as np

import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.ppo import PPOTrainer
from soccer_twos import EnvType

from utils import create_rllib_env

try:
    import gym
except Exception:  # pragma: no cover
    import gymnasium as gym


def _flatten_player_action(action: np.ndarray) -> int:
    """
    Convert a 3-branch per-player action array with values in {0,1,2}
    into a single flattened discrete action in [0, 26].

    Example:
        [a, b, c] -> a * 9 + b * 3 + c
    """
    arr = np.asarray(action, dtype=np.int64).reshape(-1)
    if arr.shape[0] != 3:
        raise ValueError(f"Expected per-player branched action of length 3, got shape {arr.shape}")
    a, b, c = int(arr[0]), int(arr[1]), int(arr[2])
    if not (0 <= a <= 2 and 0 <= b <= 2 and 0 <= c <= 2):
        raise ValueError(f"Per-player action values must be in [0,2], got {arr}")
    return a * 9 + b * 3 + c


class TrainingModerateOpponent:
    def __init__(self, env):
        self.env = env
        self.agent = RayAgent(env)
        self._pending_second_action = None

    def __call__(self, _single_obs):
        # Return cached action for player 3 on the second call of the pair.
        if self._pending_second_action is not None:
            action = self._pending_second_action
            self._pending_second_action = None
            return action

        last_obs = getattr(self.env, "last_obs", None)
        if last_obs is None or 2 not in last_obs or 3 not in last_obs:
            zero = 0
            self._pending_second_action = zero
            return zero

        try:
            actions = self.agent.act(last_obs)
        except Exception as exc:
            print(f"[curriculum] moderate opponent act() failed, returning zero action: {exc}")
            zero = 0
            self._pending_second_action = zero
            return zero

        action_p2 = actions.get(2, np.zeros(3, dtype=np.int64))
        action_p3 = actions.get(3, np.zeros(3, dtype=np.int64))

        flat_p2 = _flatten_player_action(action_p2)
        flat_p3 = _flatten_player_action(action_p3)

        self._pending_second_action = flat_p3
        return flat_p2


PITCH_X_HALF = 16.5
PITCH_Y_HALF = 10.5


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _vec2(value: Any) -> np.ndarray:
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 2:
        return np.array([_safe_float(value[0]), _safe_float(value[1])], dtype=np.float32)
    return np.zeros(2, dtype=np.float32)


def _unwrap_info(info: Any) -> Dict[str, Any]:
    if isinstance(info, dict):
        if "player_info" in info or "ball_info" in info:
            return info
        for v in info.values():
            if isinstance(v, dict) and ("player_info" in v or "ball_info" in v):
                return v
    return {}


class CurriculumRewardWrapper(gym.Wrapper):
    """
    Single wrapper with stage-switch support and original-schema shaping metrics.

    Stage 1  — vs random opponent:  full attacking shaping, ball_touch=0.2, goal_side active.
    Stage 2  — vs agent_2 opponent: lighter attacking backbone + active defense terms.
                                     goal_side removed, ball_touch lowered to 0.05,
                                     pressure_reward and idle_defense_penalty added,
                                     clearance_reward tightened to defensive-third only.
    """

    TOUCH_COOLDOWN_STEPS = 15

    def __init__(self, env, stage: int = 1):
        super().__init__(env)
        self.stage = 1

        self.prev_player_pos = np.zeros(2, dtype=np.float32)
        self.prev_ball_pos = np.zeros(2, dtype=np.float32)
        self.prev_ball_vel = np.zeros(2, dtype=np.float32)
        self.touch_cooldown = 0
        self.debug_step_count = 0

        self._moderate_opponent = None
        self._moderate_agent = None
        self.set_stage(stage)

    def _get_moderate_opponent(self):
        if self._moderate_opponent is None:
            print("[curriculum] loading stage 2 moderate opponent from RayAgent checkpoint")
            self._moderate_agent = TrainingModerateOpponent(self.env)
            self._moderate_opponent = self._moderate_agent
        return self._moderate_opponent

    def _set_opponent(self) -> None:
        random_opponent = lambda *_: 0

        if self.stage == 2:
            try:
                policy_fn = self._get_moderate_opponent()
                print("[curriculum] stage 2 opponent: RayAgent moderate checkpoint")
            except Exception as exc:
                print(f"[curriculum] failed to load moderate opponent, fallback-to-random: {exc}")
                policy_fn = random_opponent
        else:
            policy_fn = random_opponent

        if hasattr(self.env, "set_policies") and callable(getattr(self.env, "set_policies")):
            try:
                self.env.set_policies(policy_fn)
                return
            except Exception:
                pass

        if hasattr(self.env, "set_opponent") and callable(getattr(self.env, "set_opponent")):
            try:
                self.env.set_opponent(policy_fn)
                return
            except Exception:
                pass

    def set_stage(self, stage: int) -> None:
        if stage not in (1, 2):
            raise ValueError("stage must be 1 or 2")
        self.stage = stage
        self._set_opponent()

    def reset(self, **kwargs):
        self.touch_cooldown = 0
        self.debug_step_count = 0

        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}

        p, b, v = self._extract_state(_unwrap_info(info))
        self.prev_player_pos, self.prev_ball_pos, self.prev_ball_vel = p, b, v

        if isinstance(out, tuple) and len(out) == 2:
            return obs, info
        return obs

    def step(self, action):
        out = self.env.step(action)
        gymnasium_api = len(out) == 5
        if gymnasium_api:
            obs, env_reward, terminated, truncated, info = out
            done = bool(terminated or truncated)
        else:
            obs, env_reward, done, info = out
            truncated = False

        info_dict = _unwrap_info(info)
        player_pos, ball_pos, ball_vel = self._extract_state(info_dict)
        player_step_distance = float(np.linalg.norm(player_pos - self.prev_player_pos))
        if self.touch_cooldown > 0:
            self.touch_cooldown -= 1

        shaped_reward, shaping_terms = self._shape_with_stage_logic(
            env_reward=_safe_float(env_reward),
            player_pos=player_pos,
            ball_pos=ball_pos,
            ball_vel=ball_vel,
            info_dict=info_dict,
            player_step_distance=player_step_distance,
        )

        base_payload = {
            "player_info": info_dict.get("player_info", {}),
            "ball_info": info_dict.get("ball_info", {}),
            **shaping_terms,
            "stage": self.stage,
        }

        if isinstance(info, dict) and "agent0" in info:
            info_out = {"agent0": base_payload}
        else:
            info_out = base_payload

        self.prev_player_pos = player_pos
        self.prev_ball_pos = ball_pos
        self.prev_ball_vel = ball_vel
        self.debug_step_count += 1

        if gymnasium_api:
            return obs, shaped_reward, terminated, truncated, info_out
        return obs, shaped_reward, done, info_out

    def _extract_state(self, info_dict: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        player_info = info_dict.get("player_info", {})
        ball_info = info_dict.get("ball_info", {})

        player_pos = _vec2(player_info.get("position"))
        ball_pos = _vec2(ball_info.get("position"))
        ball_vel = _vec2(ball_info.get("velocity"))
        return player_pos, ball_pos, ball_vel

    def _shape_with_stage_logic(
        self,
        *,
        env_reward: float,
        player_pos: np.ndarray,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        info_dict: Dict[str, Any],
        player_step_distance: float,
    ):
        opponent_goal_x = PITCH_X_HALF
        opponent_goal_y = 0.0

        ball_to_goal = np.array(
            [opponent_goal_x - ball_pos[0], opponent_goal_y - ball_pos[1]], dtype=np.float32
        )

        touched_ball = bool(info_dict.get("player_info", {}).get("touched_ball", False)) or bool(
            info_dict.get("touched_ball", False)
        )

        # ------------------------------------------------------------------ #
        # Stage 1 — full attacking shaping vs random opponent                 #
        # ------------------------------------------------------------------ #
        if self.stage == 1:
            prev_player_ball_dist = float(np.linalg.norm(self.prev_ball_pos - self.prev_player_pos))
            curr_player_ball_dist = float(np.linalg.norm(ball_pos - player_pos))
            proximity_reward = float(
                np.clip(0.2 * (prev_player_ball_dist - curr_player_ball_dist), -0.1, 0.1)
            )

            prev_goal_dist = float(
                np.linalg.norm(
                    np.array(
                        [opponent_goal_x - self.prev_ball_pos[0], opponent_goal_y - self.prev_ball_pos[1]]
                    )
                )
            )
            curr_goal_dist = float(np.linalg.norm(ball_to_goal))
            goal_progress_reward = float(np.clip(0.1 * (prev_goal_dist - curr_goal_dist), -0.12, 0.12))

            goal_dir = ball_to_goal / (np.linalg.norm(ball_to_goal) + 1e-8)
            ball_vel_dir = ball_vel / (np.linalg.norm(ball_vel) + 1e-8)
            direction_reward = float(np.clip(0.01 * float(np.dot(goal_dir, ball_vel_dir)), -0.01, 0.01))

            ball_touch_reward = 0.0
            if touched_ball and self.touch_cooldown == 0:
                ball_touch_reward = 0.2
                self.touch_cooldown = self.TOUCH_COOLDOWN_STEPS
            ball_touch_reward = float(np.clip(ball_touch_reward, 0.0, 0.2))

            # goal_side_reward = float(np.clip(0.08 * (float(ball_pos[0]) / PITCH_X_HALF), -0.08, 0.08))

            shaping_reward = (
                proximity_reward
                + goal_progress_reward
                + direction_reward
                + ball_touch_reward
            )
            shaping_reward = float(np.clip(shaping_reward, -0.20, 0.20))
            shaped_reward = env_reward + shaping_reward

            info_terms = {
                "env_reward": env_reward,
                "proximity_reward": proximity_reward,
                "goal_progress_reward": goal_progress_reward,
                "direction_reward": direction_reward,
                "ball_touch_reward": ball_touch_reward,
                "support_reward": 0.0,
                "off_ball_penalty": 0.0,
                "clearance_reward": 0.0,
                "danger_touch_penalty": 0.0,
                "pressure_reward": 0.0,
                "idle_defense_penalty": 0.0,
                "player_step_distance": player_step_distance,
                "shaping_reward": shaping_reward,
                "shaped_reward": shaped_reward,
                "opponent_goal_x": opponent_goal_x,
                "opponent_goal_y": opponent_goal_y,
            }
            return shaped_reward, info_terms

                # ------------------------------------------------------------------ #
        # Stage 2 — lighter attacking backbone + active defense                #
        #                                                                      #
        # Changes vs stage 1:                                                  #
        #   - proximity coeff 0.2 -> 0.15, tighter clip                       #
        #   - goal_progress coeff 0.1 -> 0.08                                  #
        #   - direction coeff 0.01 -> 0.008                                   #
        #   - ball_touch_reward 0.2 -> 0.05 (stop rewarding random contact)   #
        #   - goal_side_reward REMOVED (passive, not behavior-linked)          #
        #   - clearance_reward: tightened to defensive third only              #
        #   - pressure_reward: reward moving toward ball when in own half      #
        #   - idle_defense_penalty: penalise standing far away in danger zone  #
        #   - danger_touch_penalty: kept but reduced to -0.03                  #
        #   - NEW: support_reward + off_ball_penalty to reduce idle teammate   #
        #   - total shaping clip tightened from ±0.20 -> ±0.15                #
        # ------------------------------------------------------------------ #

        # --- Attacking backbone (weakened) --------------------------------- #
        prev_player_ball_dist = float(np.linalg.norm(self.prev_ball_pos - self.prev_player_pos))
        curr_player_ball_dist = float(np.linalg.norm(ball_pos - player_pos))
        proximity_reward = float(
            np.clip(0.15 * (prev_player_ball_dist - curr_player_ball_dist), -0.06, 0.06)
        )

        prev_goal_dist = float(
            np.linalg.norm(
                np.array(
                    [opponent_goal_x - self.prev_ball_pos[0], opponent_goal_y - self.prev_ball_pos[1]]
                )
            )
        )
        curr_goal_dist = float(np.linalg.norm(ball_to_goal))
        goal_progress_reward = float(np.clip(0.08 * (prev_goal_dist - curr_goal_dist), -0.08, 0.08))

        goal_dir = ball_to_goal / (np.linalg.norm(ball_to_goal) + 1e-8)
        ball_vel_dir = ball_vel / (np.linalg.norm(ball_vel) + 1e-8)
        direction_reward = float(
            np.clip(0.008 * float(np.dot(goal_dir, ball_vel_dir)), -0.008, 0.008)
        )

        # Lowered touch reward — stop over-rewarding random contact
        ball_touch_reward = 0.0
        if touched_ball and self.touch_cooldown == 0:
            ball_touch_reward = 0.05
            self.touch_cooldown = self.TOUCH_COOLDOWN_STEPS
        ball_touch_reward = float(np.clip(ball_touch_reward, 0.0, 0.05))

        # --- Active defense terms ------------------------------------------ #

        # pressure_reward: reward closing in on the ball when it is in own half
        pressure_reward = 0.0
        if ball_pos[0] < 0.0:
            prev_dist = float(np.linalg.norm(self.prev_ball_pos - self.prev_player_pos))
            curr_dist = float(np.linalg.norm(ball_pos - player_pos))
            pressure_reward = float(np.clip(0.10 * (prev_dist - curr_dist), -0.04, 0.04))

        # clearance_reward: reward moving the ball toward midfield, but only
        # when it started in the defensive third (not just defensive half)
        clearance_reward = 0.0
        if self.prev_ball_pos[0] < -0.25 * PITCH_X_HALF:
            moved_toward_mid = float(ball_pos[0] - self.prev_ball_pos[0])
            clearance_reward = float(np.clip(0.35 * max(0.0, moved_toward_mid), 0.0, 0.10))

        # idle_defense_penalty: penalise standing far from the ball in danger zone
        idle_defense_penalty = 0.0
        dist_to_ball = float(np.linalg.norm(ball_pos - player_pos))
        if ball_pos[0] < -0.35 * PITCH_X_HALF:
            if dist_to_ball > 4.0:
                idle_defense_penalty = -0.02

        # NEW: reward useful support distance, penalise being completely uninvolved
        # This is meant to wake up the second agent without causing full swarming.
        support_reward = 0.0
        off_ball_penalty = 0.0

        if 3.0 <= dist_to_ball <= 7.0:
            support_reward = 0.005
        elif dist_to_ball > 8.0 and ball_pos[0] < 0.8 * PITCH_X_HALF:
            off_ball_penalty = -0.01

        # danger_touch_penalty: small penalty for bad touches in own box
        danger_touch_penalty = 0.0
        if touched_ball and ball_pos[0] < -0.35 * PITCH_X_HALF and _safe_float(ball_vel[0]) < 0.0:
            danger_touch_penalty = -0.03
        danger_touch_penalty = float(np.clip(danger_touch_penalty, -0.03, 0.0))

        shaping_reward = (
            proximity_reward
            + goal_progress_reward
            + direction_reward
            + ball_touch_reward
            + pressure_reward
            + clearance_reward
            + idle_defense_penalty
            + support_reward
            + off_ball_penalty
            + danger_touch_penalty
        )
        shaping_reward = float(np.clip(shaping_reward, -0.15, 0.15))
        shaped_reward = env_reward + shaping_reward

        info_terms = {
            "env_reward": env_reward,
            "proximity_reward": proximity_reward,
            "goal_progress_reward": goal_progress_reward,
            "direction_reward": direction_reward,
            "ball_touch_reward": ball_touch_reward,
            "goal_side_reward": 0.0,          # removed in stage 2
            "support_reward": support_reward,
            "off_ball_penalty": off_ball_penalty,
            "clearance_reward": clearance_reward,
            "danger_touch_penalty": danger_touch_penalty,
            "pressure_reward": pressure_reward,
            "idle_defense_penalty": idle_defense_penalty,
            "player_step_distance": player_step_distance,
            "shaping_reward": shaping_reward,
            "shaped_reward": shaped_reward,
            "opponent_goal_x": opponent_goal_x,
            "opponent_goal_y": opponent_goal_y,
        }
        return shaped_reward, info_terms


# --------------------------------------------------------------------------- #
# Callbacks                                                                     #
# --------------------------------------------------------------------------- #

class CurriculumMetricsCallback(DefaultCallbacks):
    """Compute per-episode metrics from info schema. Tracks all reward components."""

    def on_episode_start(self, *, episode, **kwargs):
        episode.user_data["env_reward_sum"] = 0.0
        episode.user_data["step_count"] = 0
        episode.user_data["field_tilt_vals"] = []
        episode.user_data["defensive_zone_vals"] = []
        episode.user_data["goal_side_vals"] = []
        episode.user_data["goal_side_reward_vals"] = []
        episode.user_data["clearance_reward_vals"] = []
        episode.user_data["danger_touch_penalty_vals"] = []
        episode.user_data["env_reward_vals"] = []
        episode.user_data["shaping_reward_vals"] = []
        episode.user_data["proximity_reward_vals"] = []
        episode.user_data["goal_progress_reward_vals"] = []
        episode.user_data["direction_reward_vals"] = []
        episode.user_data["ball_touch_reward_vals"] = []
        # New stage-2 defense metrics
        episode.user_data["pressure_reward_vals"] = []
        episode.user_data["idle_defense_penalty_vals"] = []
        episode.user_data["support_reward_vals"] = []
        episode.user_data["off_ball_penalty_vals"] = []
        episode.user_data["player_step_distance_vals"] = []

    def on_episode_step(self, *, episode, **kwargs):
        info = _unwrap_info(episode.last_info_for())
        if not info:
            return
        

        player_info = info.get("player_info", {})
        ball_info = info.get("ball_info", {})
        player_pos = _vec2(player_info.get("position"))
        ball_pos = _vec2(ball_info.get("position"))

        episode.user_data["env_reward_sum"] += _safe_float(info.get("env_reward", 0.0))
        episode.user_data["step_count"] += 1

        player_ball_dist = float(np.linalg.norm(player_pos - ball_pos))
        possession_step = 1.0 if player_ball_dist <= 1.5 else 0.0

        own_goal = np.array([-PITCH_X_HALF, 0.0], dtype=np.float32)
        player_to_own_goal = float(np.linalg.norm(player_pos - own_goal))
        ball_to_own_goal = float(np.linalg.norm(ball_pos - own_goal))
        goal_side_step = 1.0 if player_to_own_goal < ball_to_own_goal else 0.0

        episode.user_data["field_tilt_vals"].append(float(ball_pos[0]) / PITCH_X_HALF)
        episode.user_data["defensive_zone_vals"].append(1.0 if ball_pos[0] < 0.0 else 0.0)
        episode.user_data["goal_side_vals"].append(goal_side_step)
        episode.user_data.setdefault("possession_vals", []).append(possession_step)

        episode.user_data["goal_side_reward_vals"].append(_safe_float(info.get("goal_side_reward", 0.0)))
        episode.user_data["clearance_reward_vals"].append(_safe_float(info.get("clearance_reward", 0.0)))
        episode.user_data["danger_touch_penalty_vals"].append(
            _safe_float(info.get("danger_touch_penalty", 0.0))
        )
        episode.user_data["env_reward_vals"].append(_safe_float(info.get("env_reward", 0.0)))
        episode.user_data["shaping_reward_vals"].append(_safe_float(info.get("shaping_reward", 0.0)))
        episode.user_data["proximity_reward_vals"].append(_safe_float(info.get("proximity_reward", 0.0)))
        episode.user_data["goal_progress_reward_vals"].append(
            _safe_float(info.get("goal_progress_reward", 0.0))
        )
        episode.user_data["direction_reward_vals"].append(_safe_float(info.get("direction_reward", 0.0)))
        episode.user_data["ball_touch_reward_vals"].append(_safe_float(info.get("ball_touch_reward", 0.0)))
        # New
        episode.user_data["pressure_reward_vals"].append(_safe_float(info.get("pressure_reward", 0.0)))
        episode.user_data["idle_defense_penalty_vals"].append(
            _safe_float(info.get("idle_defense_penalty", 0.0))
        )
        episode.user_data["player_step_distance_vals"].append(
            _safe_float(info.get("player_step_distance", 0.0))
        )
        episode.user_data["support_reward_vals"].append(_safe_float(info.get("support_reward", 0.0)))
        episode.user_data["off_ball_penalty_vals"].append(_safe_float(info.get("off_ball_penalty", 0.0)))

    def on_episode_end(self, *, episode, **kwargs):
        steps = max(1, int(episode.user_data["step_count"]))
        possession_pct = float(np.mean(episode.user_data.get("possession_vals", [0.0])))

        episode.custom_metrics["possession_pct"] = possession_pct
        episode.custom_metrics["field_tilt"] = float(
            np.mean(episode.user_data["field_tilt_vals"] or [0.0])
        )
        episode.custom_metrics["defensive_zone_pct"] = float(
            np.mean(episode.user_data["defensive_zone_vals"] or [0.0])
        )
        episode.custom_metrics["goal_side_pct"] = float(
            np.mean(episode.user_data["goal_side_vals"] or [0.0])
        )
        episode.custom_metrics["goal_side_reward"] = float(
            np.mean(episode.user_data["goal_side_reward_vals"] or [0.0])
        )
        episode.custom_metrics["clearance_reward"] = float(
            np.mean(episode.user_data["clearance_reward_vals"] or [0.0])
        )
        episode.custom_metrics["danger_touch_penalty"] = float(
            np.mean(episode.user_data["danger_touch_penalty_vals"] or [0.0])
        )
        episode.custom_metrics["player_distance_travelled"] = float(
            np.sum(episode.user_data["player_step_distance_vals"] or [0.0])
        )
        episode.custom_metrics["player_step_distance_mean"] = float(
            np.mean(episode.user_data["player_step_distance_vals"] or [0.0])
        )

        env_sum = float(episode.user_data["env_reward_sum"])
        episode.custom_metrics["win"] = 1.0 if env_sum > 0 else 0.0
        episode.custom_metrics["loss"] = 1.0 if env_sum < 0 else 0.0
        episode.custom_metrics["draw"] = 1.0 if env_sum == 0 else 0.0
        episode.custom_metrics["episode_length"] = float(steps)

        episode.custom_metrics["env_reward_mean"] = float(
            np.mean(episode.user_data["env_reward_vals"] or [0.0])
        )
        episode.custom_metrics["shaping_reward_mean"] = float(
            np.mean(episode.user_data["shaping_reward_vals"] or [0.0])
        )
        episode.custom_metrics["proximity_reward_mean"] = float(
            np.mean(episode.user_data["proximity_reward_vals"] or [0.0])
        )
        episode.custom_metrics["goal_progress_reward_mean"] = float(
            np.mean(episode.user_data["goal_progress_reward_vals"] or [0.0])
        )
        episode.custom_metrics["direction_reward_mean"] = float(
            np.mean(episode.user_data["direction_reward_vals"] or [0.0])
        )
        episode.custom_metrics["ball_touch_reward_mean"] = float(
            np.mean(episode.user_data["ball_touch_reward_vals"] or [0.0])
        )
        # New stage-2 defense metrics
        episode.custom_metrics["pressure_reward_mean"] = float(
            np.mean(episode.user_data["pressure_reward_vals"] or [0.0])
        )
        episode.custom_metrics["idle_defense_penalty_mean"] = float(
            np.mean(episode.user_data["idle_defense_penalty_vals"] or [0.0])
        )
        episode.custom_metrics["support_reward_mean"] = float(
            np.mean(episode.user_data["support_reward_vals"] or [0.0])
        )
        episode.custom_metrics["off_ball_penalty_mean"] = float(
            np.mean(episode.user_data["off_ball_penalty_vals"] or [0.0])
        )

    def on_train_result(self, *, result: dict, **kwargs):
        cm = result.get("custom_metrics", {})

        learner_info = result.get("info", {}).get("learner", {})
        policy_key = "default_policy" if "default_policy" in learner_info else "default"
        policy_stats = learner_info.get(policy_key, {}).get("learner_stats", {})

        entropy = policy_stats.get("entropy")
        vf_loss = policy_stats.get("vf_loss")
        policy_loss = policy_stats.get("policy_loss")
        print(
            "[metrics] "
            f"iter={result.get('training_iteration')} "
            f"ts={result.get('timesteps_total')} "
            f"reward={result.get('episode_reward_mean')} "
            f"W/L/D=({cm.get('win_mean')}, {cm.get('loss_mean')}, {cm.get('draw_mean')}) "
            f"episode_length={cm.get('episode_length_mean')} "
            f"env_reward_mean={cm.get('env_reward_mean_mean')} "
            f"shaping_reward_mean={cm.get('shaping_reward_mean_mean')} "
            f"proximity_reward_mean={cm.get('proximity_reward_mean_mean')} "
            f"goal_progress_reward_mean={cm.get('goal_progress_reward_mean_mean')} "
            f"direction_reward_mean={cm.get('direction_reward_mean_mean')} "
            f"ball_touch_reward_mean={cm.get('ball_touch_reward_mean_mean')} "
            f"pressure_reward_mean={cm.get('pressure_reward_mean_mean')} "
            f"idle_defense_penalty_mean={cm.get('idle_defense_penalty_mean_mean')} "
            f"possession_pct={cm.get('possession_pct_mean')} "
            f"field_tilt={cm.get('field_tilt_mean')} "
            f"player_distance_travelled={cm.get('player_distance_travelled_mean')} "
            f"player_step_distance_mean={cm.get('player_step_distance_mean_mean')} "
            f"defensive_zone_pct={cm.get('defensive_zone_pct_mean')} "
            f"goal_side_pct={cm.get('goal_side_pct_mean')} "
            f"clearance_reward={cm.get('clearance_reward_mean')} "
            f"danger_touch_penalty={cm.get('danger_touch_penalty_mean')} "
            f"support_reward_mean={cm.get('support_reward_mean_mean')} "
            f"off_ball_penalty_mean={cm.get('off_ball_penalty_mean_mean')} "
            f"entropy={entropy} "
            f"vf_loss={vf_loss} "
            f"policy_loss={policy_loss}"
        )


def create_curriculum_env(env_config: Dict[str, Any]):
    base_env = create_rllib_env(env_config)
    return CurriculumRewardWrapper(base_env, stage=int(env_config.get("stage", 1)))


def _switch_stage_on_all_workers(trainer: PPOTrainer, stage: int) -> None:
    def _set_stage(env):
        if hasattr(env, "set_stage") and callable(getattr(env, "set_stage")):
            env.set_stage(stage)

    trainer.workers.foreach_worker(lambda w: w.foreach_env(_set_stage))


class CurriculumStageSwitchCallback(CurriculumMetricsCallback):
    def __init__(self):
        super().__init__()
        self._switched = False

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        super().on_train_result(trainer=trainer, result=result, **kwargs)

        phase1_target = trainer.config.get("env_config", {}).get("phase1_timesteps")
        timesteps = result.get("timesteps_total", 0)

        if (not self._switched) and phase1_target is not None and timesteps >= phase1_target:
            # Switch environments to stage 2
            trainer.workers.foreach_worker(
                lambda w: w.foreach_env(
                    lambda env: env.set_stage(2) if hasattr(env, "set_stage") else None
                )
            )

            # Switch PPO hyperparameters to phase-2 values
            trainer.config["clip_param"] = 0.15
            trainer.config["num_sgd_iter"] = 10
            trainer.config["vf_loss_coeff"] = 1.5

            self._switched = True
            print(f"[curriculum] entering phase 2 at timestep {timesteps} — updated clip_param, num_sgd_iter, vf_loss_coeff")


# --------------------------------------------------------------------------- #
# Config                                                                        #
# --------------------------------------------------------------------------- #

def build_config(args) -> Dict[str, Any]:
    """
    Learning-rate schedule:
      Phase 1 (0 → phase1_timesteps):        lr = 1e-4
      Phase 2 (phase1_timesteps → end):      lr = 5e-5

    The schedule uses a near-zero gap at the boundary so RLlib's linear
    interpolation produces an essentially instant step-down.

    Phase 2 hyperparameter rationale
    ---------------------------------
    * lr 5e-5  — half of phase-1; the policy is already mature, smaller
                 updates reduce catastrophic forgetting and tame VF loss.
    * clip_param 0.15 — tighter PPO clip limits large policy steps when the
                 value function is still catching up to the harder opponent.
    * vf_loss_coeff 1.5 — prioritise value-function accuracy during phase 2
                 to bring VF loss back down from the shock spike seen at 2M.
    * num_sgd_iter 10 — fewer passes per batch; combined with lower LR this
                 prevents over-fitting to a single batch during the noisy
                 post-switch period.
    * phase2_timesteps 2_000_000 — 1M was not enough for VF loss to converge
                 after the curriculum switch; 2M gives the defense shaping
                 time to show returns without over-training.
    """
    p1 = args.phase1_timesteps

    return {
        "num_gpus": 0,
        "num_workers": args.num_workers,
        "num_envs_per_worker": args.num_envs_per_worker,
        "log_level": "WARN",
        "framework": "torch",
        "callbacks": CurriculumStageSwitchCallback,
        "env": "SoccerCurriculum",
        "env_config": {
            "num_envs_per_worker": args.num_envs_per_worker,
            "variation": EnvType.team_vs_policy,
            "multiagent": False,
            "single_player": False,
            "flatten_branched": True,
            "opponent_policy": lambda *_: 0,
            "stage": 1,
            "phase1_timesteps": p1,
        },
        "model": {
            "vf_share_layers": True,
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
        },
        # Step-down LR schedule: 1e-4 in phase 1, 5e-5 in phase 2
        "lr": 1e-4,
        "lr_schedule": [
            [0,          1e-4],
            [p1 - 1,     1e-4],   # hold until one step before the switch
            [p1,         5e-5],   # drop at phase boundary
        ],
        "gamma": 0.99,
        "lambda": 0.95,
        # Tighter clip reduces large policy jumps when the VF is catching up
        "clip_param": 0.2,
        "entropy_coeff": 0.005,
        # Higher VF coeff forces the value function to converge faster in phase 2
        "vf_loss_coeff": 1.0,
        "rollout_fragment_length": 400,
        "train_batch_size": 4000,
        "sgd_minibatch_size": 256,
        # Fewer SGD passes per batch — avoids over-fitting during noisy phase 2
        "num_sgd_iter": 15,
        "batch_mode": "truncate_episodes",
    }


# --------------------------------------------------------------------------- #
# Argument parsing                                                              #
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description="Two-phase curriculum PPO run without restore.")
    parser.add_argument("--phase1-timesteps", type=int, default=2_000_000)
    # Increased from 1M to 2M — gives phase-2 defense shaping time to converge
    parser.add_argument("--phase2-timesteps", type=int, default=2_000_000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-envs-per-worker", type=int, default=1)
    parser.add_argument("--experiment-name", type=str, default="ppo_curriculum_v2_active_defense_fix")
    parser.add_argument("--local-dir", type=str, default="./ray_results")
    parser.add_argument("--debug-print-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Logging helpers                                                               #
# --------------------------------------------------------------------------- #

def _result_row(result: Dict[str, Any], stage: int) -> Dict[str, Any]:
    cm = result.get("custom_metrics", {})

    learner_info = result.get("info", {}).get("learner", {})
    policy_key = "default_policy" if "default_policy" in learner_info else "default"
    policy_stats = learner_info.get(policy_key, {}).get("learner_stats", {})

    return {
        "training_iteration": result.get("training_iteration"),
        "timesteps_total": result.get("timesteps_total"),
        "episodes_total": result.get("episodes_total"),
        "episode_reward_mean": result.get("episode_reward_mean"),
        "episode_reward_min": result.get("episode_reward_min"),
        "episode_reward_max": result.get("episode_reward_max"),
        "stage": stage,
        "episode_length": cm.get("episode_length_mean"),
        "env_reward_mean": cm.get("env_reward_mean_mean"),
        "shaping_reward_mean": cm.get("shaping_reward_mean_mean"),
        "proximity_reward_mean": cm.get("proximity_reward_mean_mean"),
        "goal_progress_reward_mean": cm.get("goal_progress_reward_mean_mean"),
        "direction_reward_mean": cm.get("direction_reward_mean_mean"),
        "ball_touch_reward_mean": cm.get("ball_touch_reward_mean_mean"),
        "pressure_reward_mean": cm.get("pressure_reward_mean_mean"),
        "idle_defense_penalty_mean": cm.get("idle_defense_penalty_mean_mean"),
        "win": cm.get("win_mean"),
        "loss": cm.get("loss_mean"),
        "draw": cm.get("draw_mean"),
        "possession_pct": cm.get("possession_pct_mean"),
        "field_tilt": cm.get("field_tilt_mean"),
        "player_distance_travelled": cm.get("player_distance_travelled_mean"),
        "player_step_distance_mean": cm.get("player_step_distance_mean_mean"),
        "defensive_zone_pct": cm.get("defensive_zone_pct_mean"),
        "goal_side_pct": cm.get("goal_side_pct_mean"),
        "goal_side_reward": cm.get("goal_side_reward_mean"),
        "clearance_reward": cm.get("clearance_reward_mean"),
        "danger_touch_penalty": cm.get("danger_touch_penalty_mean"),
        "support_reward_mean": cm.get("support_reward_mean_mean"),
        "off_ball_penalty_mean": cm.get("off_ball_penalty_mean_mean"),
        "entropy": policy_stats.get("entropy"),
        "vf_loss": policy_stats.get("vf_loss"),
        "policy_loss": policy_stats.get("policy_loss"),
    }


def _save_csv(rows, path: Path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_training_outputs(analysis: tune.ExperimentAnalysis, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    tracked_columns = [
        "training_iteration",
        "timesteps_total",
        "episodes_total",
        "time_total_s",
        "episode_reward_mean",
        "episode_reward_min",
        "episode_reward_max",

        "custom_metrics/win_mean",
        "custom_metrics/draw_mean",
        "custom_metrics/loss_mean",
        "custom_metrics/episode_length_mean",

        "custom_metrics/env_reward_mean_mean",
        "custom_metrics/shaping_reward_mean_mean",
        "custom_metrics/proximity_reward_mean_mean",
        "custom_metrics/goal_progress_reward_mean_mean",
        "custom_metrics/direction_reward_mean_mean",
        "custom_metrics/ball_touch_reward_mean_mean",

        # New stage-2 defense columns
        "custom_metrics/pressure_reward_mean_mean",
        "custom_metrics/idle_defense_penalty_mean_mean",
        "custom_metrics/support_reward_mean_mean",
        "custom_metrics/off_ball_penalty_mean_mean",

        "custom_metrics/possession_pct_mean",
        "custom_metrics/field_tilt_mean",
        "custom_metrics/player_distance_travelled_mean",
        "custom_metrics/player_step_distance_mean_mean",
        "custom_metrics/defensive_zone_pct_mean",
        "custom_metrics/goal_side_pct_mean",
        "custom_metrics/goal_side_reward_mean",
        "custom_metrics/clearance_reward_mean",
        "custom_metrics/danger_touch_penalty_mean",

        "info/learner/default_policy/learner_stats/entropy",
        "info/learner/default_policy/learner_stats/vf_loss",
        "info/learner/default_policy/learner_stats/policy_loss",
        "info/learner/default/learner_stats/entropy",
        "info/learner/default/learner_stats/vf_loss",
        "info/learner/default/learner_stats/policy_loss",
    ]

    if analysis.trials:
        trial = analysis.trials[0]
        trial_df = analysis.trial_dataframes.get(trial.logdir)
        if trial_df is not None:
            present_columns = [c for c in tracked_columns if c in trial_df.columns]
            training_log = trial_df[present_columns].copy().sort_values("training_iteration")

            csv_path = output_dir / "training_log.csv"
            json_path = output_dir / "training_log.json"
            training_log.to_csv(csv_path, index=False)
            training_log.to_json(json_path, orient="records", indent=2)

            print(f"Saved training logs: {csv_path}")
            print(f"Saved training logs: {json_path}")
        else:
            print("WARNING: No trial dataframe found; training history not saved.")


# --------------------------------------------------------------------------- #
# Entry point                                                                   #
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()

    run_dir = Path(args.local_dir) / args.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    tune.registry.register_env("SoccerCurriculum", create_curriculum_env)

    analysis = tune.run(
        "PPO",
        name=args.experiment_name,
        config=build_config(args),
        stop={"timesteps_total": args.phase1_timesteps + args.phase2_timesteps},
        checkpoint_freq=args.checkpoint_every,
        checkpoint_at_end=True,
        local_dir=args.local_dir,
        verbose=1,
    )
    output_dir = Path(args.local_dir) / args.experiment_name
    _save_training_outputs(analysis, output_dir)

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(f"Best trial: {best_trial}")

    best_checkpoint = None
    if best_trial is not None:
        best_checkpoint = analysis.get_best_checkpoint(
            trial=best_trial,
            metric="episode_reward_mean",
            mode="max",
        )

    print(f"Best checkpoint: {best_checkpoint}")

    if best_checkpoint is not None:
        best_checkpoint_path = output_dir / "best_checkpoint.txt"
        best_checkpoint_path.write_text(str(best_checkpoint) + "\n", encoding="utf-8")
        print(f"Wrote best checkpoint pointer: {best_checkpoint_path}")

    ray.shutdown()


if __name__ == "__main__":
    main()