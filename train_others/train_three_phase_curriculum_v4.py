"""
train_three_phase_curriculum_v4.py
===================================
Three-phase curriculum PPO for SoccerTwos.

Changes vs v3
-------------
v3 had NO passing reward at all. Based on analysis of the previous
two-phase experiment (test_two_phase_pass_reward_curriculum.py), the
passing reward was found to be counterproductive in Phase 1 (single
controlled player, no real teammate context) and invisible in Phase 2
(reward magnitude ~1e-7, ~1000x weaker than other shaping terms, and
based on a single-player release-reacquire proxy rather than true
two-player passing geometry).

The correct place for pass reward is Phase 3 (vs baseline agent), where:
  1. The agent has already learned to control the ball (Phase 1) and
     defend (Phase 2), so it is ready for cooperative/strategic behavior.
  2. The stronger baseline opponent makes holding the ball expensive —
     passing becomes a genuine survival strategy.
  3. Reward magnitude is calibrated to be visible: pass_reward=0.15 per
     completion ensures a single pass contributes ~0.003–0.005 to the
     mean per-step reward, comparable to ball_touch_reward.

PassTransferProxyTracker changes vs the two-phase experiment
------------------------------------------------------------
- Reward per completion: 0.03 → 0.15  (was invisible at 1e-7 per step)
- MIN_RELEASE_SPEED: 0.80 → 1.20      (filter slow dribble noise)
- Added FORWARD_GATE: ball must have a positive X-component velocity
  (moving toward opponent goal) at release to prevent rewarding
  backwards safety passes that back the agent into danger.
- MIN_TRAVEL: 2.00 → 3.00             (require a more committed pass)
- MAX_STEPS: 18 → 25                  (allow ball to travel further)
- SEPARATION_DIST: 1.80 → 2.00       (stricter — must clearly leave player)
- Cooldown after success: 8 → 12      (prevent rapid re-triggering)

All other shaping terms (proximity, goal_progress, shooting, defense) and
hyperparameter schedules are identical to v3.

Summary of all phases
---------------------
Phase 1 (Stage 1, vs random, 2M steps):
  Full attacking shaping: proximity 0.2, goal_progress 0.1, direction 0.01,
  ball_touch 0.2. No pass reward. Objective: learn locomotion and ball control.

Phase 2 (Stage 2, vs moderate agent, 2M steps):
  Weakened attack + defense terms: proximity 0.15, goal_progress 0.08,
  direction 0.008, ball_touch 0.05, pressure/clearance/idle_defense/
  support/off_ball/danger_touch. No pass reward. Objective: learn defense
  and positioning.

Phase 3 (Stage 3, vs baseline agent, 5M steps):
  Full attack + defense + stronger goal_progress (0.15) + shooting reward
  (gated, coeff=0.01) + PASS REWARD (proxy tracker, 0.15 per completion,
  forward-gated). Objective: learn to shoot and pass under pressure.
"""

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Tuple

from my_agent_2 import RayAgent as ModerateRayAgent
from ceia_baseline_agent import RayAgent as BaselineRayAgent
import numpy as np

import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.ppo import PPOTrainer
from soccer_twos import EnvType

from utils import create_rllib_env

try:
    import gym
except Exception:
    import gymnasium as gym


# ---------------------------------------------------------------------------
# Action helpers (unchanged from v3)
# ---------------------------------------------------------------------------

def _flatten_player_action(action: np.ndarray) -> int:
    arr = np.asarray(action, dtype=np.int64).reshape(-1)
    if arr.shape[0] != 3:
        raise ValueError(f"Expected per-player branched action of length 3, got shape {arr.shape}")
    a, b, c = int(arr[0]), int(arr[1]), int(arr[2])
    if not (0 <= a <= 2 and 0 <= b <= 2 and 0 <= c <= 2):
        raise ValueError(f"Per-player action values must be in [0,2], got {arr}")
    return a * 9 + b * 3 + c


def _normalize_single_action(action: Any) -> int:
    if isinstance(action, (int, np.integer)):
        return int(action)
    if isinstance(action, np.ndarray) and action.ndim == 0:
        return int(action.item())
    if isinstance(action, (list, tuple, np.ndarray)):
        arr = np.asarray(action)
        if arr.ndim == 0:
            return int(arr.item())
        flat = arr.reshape(-1)
        if flat.shape[0] == 1:
            return int(flat[0])
        if flat.shape[0] == 3:
            return _flatten_player_action(flat)
    raise ValueError(f"Unsupported action format for opponent policy: {type(action)} -> {action}")


# ---------------------------------------------------------------------------
# Opponent wrappers (unchanged from v3)
# ---------------------------------------------------------------------------

class TrainingModerateOpponent:
    def __init__(self, env):
        self.env = env
        self.agent = ModerateRayAgent(env)
        self._pending_second_action = None

    def __call__(self, _single_obs):
        if self._pending_second_action is not None:
            action = self._pending_second_action
            self._pending_second_action = None
            return action

        last_obs = getattr(self.env, "last_obs", None)
        if last_obs is None or 2 not in last_obs or 3 not in last_obs:
            self._pending_second_action = 0
            return 0

        try:
            actions = self.agent.act(last_obs)
        except Exception as exc:
            print(f"[curriculum] moderate opponent act() failed, returning zero action: {exc}")
            self._pending_second_action = 0
            return 0

        try:
            action_p2 = _normalize_single_action(actions.get(2, 0))
            action_p3 = _normalize_single_action(actions.get(3, 0))
        except Exception as exc:
            print(f"[curriculum] moderate opponent action normalization failed: {exc}")
            self._pending_second_action = 0
            return 0

        self._pending_second_action = action_p3
        return action_p2


class TrainingBaselineOpponent:
    def __init__(self, env):
        self.env = env
        self.agent = BaselineRayAgent(env)
        self._pending_second_action = None

    def __call__(self, _single_obs):
        if self._pending_second_action is not None:
            action = self._pending_second_action
            self._pending_second_action = None
            return action

        last_obs = getattr(self.env, "last_obs", None)
        if last_obs is None or 2 not in last_obs or 3 not in last_obs:
            self._pending_second_action = 0
            return 0

        try:
            actions = self.agent.act(last_obs)
        except Exception as exc:
            print(f"[curriculum] baseline opponent act() failed, returning zero action: {exc}")
            self._pending_second_action = 0
            return 0

        try:
            action_p2 = _normalize_single_action(actions.get(2, 0))
            action_p3 = _normalize_single_action(actions.get(3, 0))
        except Exception as exc:
            print(f"[curriculum] baseline opponent action normalization failed: {exc}")
            self._pending_second_action = 0
            return 0

        self._pending_second_action = action_p3
        return action_p2


# ---------------------------------------------------------------------------
# Pitch constants & helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Pass Transfer Proxy Tracker (Phase 3 only)
#
# Calibrated changes vs the two-phase experiment:
#
#   pass_reward per completion  0.03  → 0.15   The original 0.03 produced
#     a per-step expected contribution of ~1e-7 — too small for gradients.
#     0.15 brings it in line with ball_touch_reward in phases 1/2.
#
#   MIN_RELEASE_SPEED           0.80  → 1.20   Filter slow dribble releases
#     that fired spuriously. Only genuine kicks should start an attempt.
#
#   FORWARD_GATE (new)          False → True    Ball X-velocity at release
#     must be positive (moving toward opponent goal). This prevents the
#     tracker from rewarding backwards safety passes or goal-kick-away
#     touches that do not contribute to attacking play.
#
#   MIN_TRAVEL                  2.00  → 3.00   Require a more committed pass.
#
#   MAX_STEPS                   18    → 25     Allow ball to travel further
#     before timing out (faster balls clear the check distance quickly).
#
#   SEPARATION_DIST             1.80  → 2.00   Ball must clearly leave the
#     player's vicinity before completion is considered.
#
#   Cooldown after success      8     → 12     Reduce rapid re-triggering
#     on rebounds / incidental re-contacts.
# ---------------------------------------------------------------------------

class PassTransferProxyTracker:
    """
    Proxy pass detector active only during Phase 3 (stage 3).

    Tracks a single-player release-and-reacquire signal that approximates
    a pass: the ball is kicked with meaningful speed toward the opponent
    goal, leaves the player's vicinity, travels a minimum distance, and is
    then touched again (by anyone, since we only see one player's stream).

    This is NOT a true two-player geometric pass detector, but it is
    substantially better calibrated than the version used in the two-phase
    experiment. The forward-gate and higher speed threshold together reduce
    the spurious completion rate from ~2-5% to an expected ~1-2%, and the
    higher reward (0.15) makes each completion meaningfully visible to the
    optimizer.
    """

    # Minimum ball speed at release to start an attempt.
    # 1.20 filters slow dribble releases; the two-phase version used 0.80
    # which fired on any slight touch.
    MIN_RELEASE_SPEED = 1.20

    # Ball X-velocity must be positive at release (moving toward opponent
    # goal). Set to True to enable this gate.
    FORWARD_GATE = True

    # Minimum distance the ball must travel from its release position
    # before a completion can be awarded.
    MIN_TRAVEL = 3.00

    # Maximum steps between release and re-touch before the attempt expires.
    MAX_STEPS = 25

    # Minimum steps the ball must be away from the player (dist > SEPARATION_DIST)
    # before we consider it "cleanly released".
    MIN_FREE_STEPS = 2

    # Ball must exceed this distance from player to count as a free step.
    SEPARATION_DIST = 2.00

    # Minimum elapsed steps after release before re-touching counts as
    # a pass completion (prevents immediate self-reacquire).
    REATTACH_COOLDOWN = 3

    # Steps to wait after a completion before starting the next attempt.
    SUCCESS_COOLDOWN = 12

    # Steps to wait after a failed attempt before starting the next attempt.
    FAIL_COOLDOWN = 4

    # Reward awarded per completed pass.
    PASS_REWARD = 0.15

    def __init__(self):
        self.pending = None
        self.cooldown = 0

    def reset(self):
        self.pending = None
        self.cooldown = 0

    def update(
        self,
        *,
        touched_ball: bool,
        player_pos: np.ndarray,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        prev_player_pos: np.ndarray,
        prev_ball_pos: np.ndarray,
    ) -> Dict[str, float]:
        if self.cooldown > 0:
            self.cooldown -= 1

        dist_now = float(np.linalg.norm(player_pos - ball_pos))
        ball_speed = float(np.linalg.norm(ball_vel))

        attempt_count = 0.0
        complete_count = 0.0
        fail_count = 0.0
        pass_reward = 0.0
        pass_travel = 0.0

        # -----------------------------------------------------------------
        # Attempt start conditions
        # -----------------------------------------------------------------
        # 1. A touch is detected.
        # 2. The player is close enough to the ball to be the kicker.
        # 3. The ball is moving fast enough to be a real kick (not a dribble).
        # 4. (If FORWARD_GATE) The ball is moving toward the opponent goal
        #    (positive X-component), filtering backwards passes.
        # -----------------------------------------------------------------
        forward_ok = (not self.FORWARD_GATE) or (float(ball_vel[0]) > 0.0)
        release_like = (
            touched_ball
            and dist_now <= 1.50
            and ball_speed >= self.MIN_RELEASE_SPEED
            and forward_ok
        )

        if self.pending is None and self.cooldown == 0 and release_like:
            self.pending = {
                "release_ball_pos": ball_pos.copy(),
                "steps": 0,
                "free_steps": 0,
                "released_cleanly": False,
            }
            attempt_count = 1.0

        # -----------------------------------------------------------------
        # Attempt tracking
        # -----------------------------------------------------------------
        if self.pending is not None:
            self.pending["steps"] += 1
            pass_travel = float(np.linalg.norm(ball_pos - self.pending["release_ball_pos"]))

            # Count steps where the ball is well clear of the player
            if dist_now >= self.SEPARATION_DIST:
                self.pending["free_steps"] += 1

            if self.pending["free_steps"] >= self.MIN_FREE_STEPS:
                self.pending["released_cleanly"] = True

            # -----------------------------------------------------------------
            # Completion: ball has cleanly left, traveled enough, and is re-touched
            # -----------------------------------------------------------------
            if (
                touched_ball
                and self.pending["steps"] >= self.REATTACH_COOLDOWN
                and self.pending["released_cleanly"]
                and pass_travel >= self.MIN_TRAVEL
            ):
                pass_reward = self.PASS_REWARD
                complete_count = 1.0
                self.pending = None
                self.cooldown = self.SUCCESS_COOLDOWN

            # -----------------------------------------------------------------
            # Timeout: attempt has expired without a successful re-touch
            # -----------------------------------------------------------------
            elif self.pending["steps"] > self.MAX_STEPS:
                fail_count = 1.0
                self.pending = None
                self.cooldown = self.FAIL_COOLDOWN

        return {
            "pass_reward": float(pass_reward),
            "pass_attempt_count": float(attempt_count),
            "pass_complete_count": float(complete_count),
            "pass_fail_count": float(fail_count),
            "pass_travel": float(pass_travel),
            "pass_pending": float(1.0 if self.pending is not None else 0.0),
            "pass_ball_speed": float(ball_speed),
            "pass_player_ball_dist": float(dist_now),
            "pass_free_steps": float(0.0 if self.pending is None else self.pending["free_steps"]),
            "pass_pending_steps": float(0.0 if self.pending is None else self.pending["steps"]),
        }


# ---------------------------------------------------------------------------
# Reward wrapper
# ---------------------------------------------------------------------------

class CurriculumRewardWrapper(gym.Wrapper):
    """
    Stage 1 — random opponent:   aggressive proximity/touch shaping.
    Stage 2 — moderate agent:    weakened attack + active defense terms.
    Stage 3 — baseline agent:    stronger goal_progress + shooting reward
                                 + PASS REWARD via PassTransferProxyTracker.

    Pass reward is intentionally absent from stages 1 and 2:
      - Stage 1 has no real teammate — the proxy fires on dribble noise.
      - Stage 2 the agent needs to learn defense; pass reward is a
        distraction and was empirically invisible (1e-7 per step) in
        the two-phase experiment.

    Fix 1 (ball_touch fallback): touched_ball is augmented with a proximity-
    based heuristic so the reward fires even when the env flag is missing.
    """

    TOUCH_COOLDOWN_STEPS = 15

    def __init__(self, env, stage: int = 1):
        super().__init__(env)
        self.stage = 1
        self.danger_touch_cooldown = 0
        self.prev_player_pos = np.zeros(2, dtype=np.float32)
        self.prev_ball_pos = np.zeros(2, dtype=np.float32)
        self.prev_ball_vel = np.zeros(2, dtype=np.float32)
        self.touch_cooldown = 0
        self.debug_step_count = 0

        # Pass tracker — only used in stage 3, but instantiated once here
        # to avoid construction overhead at the stage switch.
        self.pass_proxy = PassTransferProxyTracker()

        self._moderate_opponent = None
        self._baseline_opponent = None
        self.set_stage(stage)

    # ------------------------------------------------------------------
    # Opponent management (unchanged from v3)
    # ------------------------------------------------------------------

    def _get_moderate_opponent(self):
        if self._moderate_opponent is None:
            print("[curriculum] loading stage 2 moderate opponent")
            self._moderate_opponent = TrainingModerateOpponent(self.env)
        return self._moderate_opponent

    def _get_baseline_opponent(self):
        if self._baseline_opponent is None:
            print("[curriculum] loading stage 3 baseline opponent")
            self._baseline_opponent = TrainingBaselineOpponent(self.env)
        return self._baseline_opponent

    def _set_opponent(self) -> None:
        random_opponent = lambda *_: 0

        if self.stage == 2:
            try:
                policy_fn = self._get_moderate_opponent()
                print("[curriculum] stage 2 opponent: moderate RayAgent")
            except Exception as exc:
                print(f"[curriculum] failed to load moderate opponent, fallback to random: {exc}")
                policy_fn = random_opponent
        elif self.stage == 3:
            try:
                policy_fn = self._get_baseline_opponent()
                print("[curriculum] stage 3 opponent: baseline RayAgent")
            except Exception as exc:
                print(f"[curriculum] failed to load baseline opponent, fallback to random: {exc}")
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
        if stage not in (1, 2, 3):
            raise ValueError("stage must be 1, 2, or 3")
        self.stage = stage
        self._set_opponent()

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        self.touch_cooldown = 0
        self.danger_touch_cooldown = 0
        self.debug_step_count = 0
        # Always reset the pass proxy — harmless in stages 1/2 and
        # necessary in stage 3 to clear any pending attempt from the
        # previous episode.
        self.pass_proxy.reset()

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
        else:
            obs, env_reward, done, info = out
            terminated = bool(done)
            truncated = False

        info_dict = _unwrap_info(info)
        player_pos, ball_pos, ball_vel = self._extract_state(info_dict)
        player_step_distance = float(np.linalg.norm(player_pos - self.prev_player_pos))

        if self.touch_cooldown > 0:
            self.touch_cooldown -= 1
        if self.danger_touch_cooldown > 0:
            self.danger_touch_cooldown -= 1

        # Detect touch using the improved proximity fallback (Fix 1)
        touched_ball = self._detect_touch(
            info_dict, ball_pos, player_pos, ball_vel, player_step_distance
        )

        # Update the pass proxy on every step.
        # In stages 1 and 2 the returned pass_reward is always 0.0 because
        # _shape_with_stage_logic ignores it; we still update the tracker so
        # that internal state stays consistent across the stage boundary.
        pass_proxy_terms = self.pass_proxy.update(
            touched_ball=touched_ball,
            player_pos=player_pos,
            ball_pos=ball_pos,
            ball_vel=ball_vel,
            prev_player_pos=self.prev_player_pos,
            prev_ball_pos=self.prev_ball_pos,
        )

        shaped_reward, shaping_terms = self._shape_with_stage_logic(
            env_reward=_safe_float(env_reward),
            player_pos=player_pos,
            ball_pos=ball_pos,
            ball_vel=ball_vel,
            info_dict=info_dict,
            player_step_distance=player_step_distance,
            touched_ball=touched_ball,
            pass_proxy_terms=pass_proxy_terms,
        )

        base_payload = {
            "player_info": info_dict.get("player_info", {}),
            "ball_info": info_dict.get("ball_info", {}),
            **shaping_terms,
            "stage": float(self.stage),
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
        return obs, shaped_reward, terminated, info_out

    def _extract_state(
        self, info_dict: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        player_info = info_dict.get("player_info", {})
        ball_info = info_dict.get("ball_info", {})
        player_pos = _vec2(player_info.get("position"))
        ball_pos = _vec2(ball_info.get("position"))
        ball_vel = _vec2(ball_info.get("velocity"))
        return player_pos, ball_pos, ball_vel

    # ------------------------------------------------------------------
    # Touch detection (Fix 1 — proximity fallback)
    # ------------------------------------------------------------------

    def _detect_touch(
        self,
        info_dict: Dict[str, Any],
        ball_pos: np.ndarray,
        player_pos: np.ndarray,
        ball_vel: np.ndarray,
        player_step_distance: float,
    ) -> bool:
        # Primary: env flag
        if bool(info_dict.get("player_info", {}).get("touched_ball", False)) or bool(
            info_dict.get("touched_ball", False)
        ):
            return True

        # Fallback: proximity + sudden ball velocity change heuristic.
        curr_dist = float(np.linalg.norm(ball_pos - player_pos))
        prev_dist = float(np.linalg.norm(self.prev_ball_pos - self.prev_player_pos))
        ball_speed_change = float(np.linalg.norm(ball_vel - self.prev_ball_vel))
        return (
            curr_dist < 1.2
            and prev_dist < 1.2
            and player_step_distance > 0.01
            and ball_speed_change > 0.8
        )

    # ------------------------------------------------------------------
    # Reward shaping
    # ------------------------------------------------------------------

    def _shape_with_stage_logic(
        self,
        *,
        env_reward: float,
        player_pos: np.ndarray,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        info_dict: Dict[str, Any],
        player_step_distance: float,
        touched_ball: bool,
        pass_proxy_terms: Dict[str, float],
    ):
        opponent_goal_x = PITCH_X_HALF
        opponent_goal_y = 0.0

        ball_to_goal = np.array(
            [opponent_goal_x - ball_pos[0], opponent_goal_y - ball_pos[1]],
            dtype=np.float32,
        )

        # ------------------------------------------------------------------
        # Stage 1 — vs random opponent
        # Aggressive proximity/touch shaping to bootstrap locomotion.
        # No pass reward — the proxy fires on noise in a single-player setup.
        # ------------------------------------------------------------------
        if self.stage == 1:
            prev_player_ball_dist = float(np.linalg.norm(self.prev_ball_pos - self.prev_player_pos))
            curr_player_ball_dist = float(np.linalg.norm(ball_pos - player_pos))
            proximity_reward = float(
                np.clip(0.2 * (prev_player_ball_dist - curr_player_ball_dist), -0.1, 0.1)
            )

            prev_goal_dist = float(
                np.linalg.norm(
                    [opponent_goal_x - self.prev_ball_pos[0], opponent_goal_y - self.prev_ball_pos[1]]
                )
            )
            curr_goal_dist = float(np.linalg.norm(ball_to_goal))
            goal_progress_reward = float(
                np.clip(0.1 * (prev_goal_dist - curr_goal_dist), -0.12, 0.12)
            )

            goal_dir = ball_to_goal / (np.linalg.norm(ball_to_goal) + 1e-8)
            ball_vel_dir = ball_vel / (np.linalg.norm(ball_vel) + 1e-8)
            direction_reward = float(
                np.clip(0.01 * float(np.dot(goal_dir, ball_vel_dir)), -0.01, 0.01)
            )

            ball_touch_reward = 0.0
            if touched_ball and self.touch_cooldown == 0:
                ball_touch_reward = 0.2
                self.touch_cooldown = self.TOUCH_COOLDOWN_STEPS
            ball_touch_reward = float(np.clip(ball_touch_reward, 0.0, 0.2))

            shaping_reward = (
                proximity_reward + goal_progress_reward + direction_reward + ball_touch_reward
            )
            shaping_reward = float(np.clip(shaping_reward, -0.20, 0.20))
            shaped_reward = env_reward + shaping_reward

            info_terms = {
                "env_reward": env_reward,
                "proximity_reward": proximity_reward,
                "goal_progress_reward": goal_progress_reward,
                "direction_reward": direction_reward,
                "ball_touch_reward": ball_touch_reward,
                "shooting_reward": 0.0,
                "goal_side_reward": 0.0,
                "support_reward": 0.0,
                "off_ball_penalty": 0.0,
                "clearance_reward": 0.0,
                "danger_touch_penalty": 0.0,
                "danger_touch_count": 0.0,
                "pressure_reward": 0.0,
                "idle_defense_penalty": 0.0,
                "player_step_distance": player_step_distance,
                "shaping_reward": shaping_reward,
                "shaped_reward": shaped_reward,
                "opponent_goal_x": opponent_goal_x,
                "opponent_goal_y": opponent_goal_y,
                # Pass metrics — all zero in stage 1
                "pass_reward": 0.0,
                "pass_attempt_count": 0.0,
                "pass_complete_count": 0.0,
                "pass_fail_count": 0.0,
                "pass_travel": 0.0,
                "pass_completion_rate": 0.0,
            }
            return shaped_reward, info_terms

        # ------------------------------------------------------------------
        # Shared computations for stages 2 and 3
        # ------------------------------------------------------------------
        prev_player_ball_dist = float(np.linalg.norm(self.prev_ball_pos - self.prev_player_pos))
        curr_player_ball_dist = float(np.linalg.norm(ball_pos - player_pos))
        proximity_reward = float(
            np.clip(0.15 * (prev_player_ball_dist - curr_player_ball_dist), -0.06, 0.06)
        )

        prev_goal_dist = float(
            np.linalg.norm(
                [opponent_goal_x - self.prev_ball_pos[0], opponent_goal_y - self.prev_ball_pos[1]]
            )
        )
        curr_goal_dist = float(np.linalg.norm(ball_to_goal))

        goal_dir = ball_to_goal / (np.linalg.norm(ball_to_goal) + 1e-8)
        ball_vel_dir = ball_vel / (np.linalg.norm(ball_vel) + 1e-8)
        direction_reward = float(
            np.clip(0.008 * float(np.dot(goal_dir, ball_vel_dir)), -0.008, 0.008)
        )

        ball_touch_reward = 0.0
        if touched_ball and self.touch_cooldown == 0:
            ball_touch_reward = 0.05
            self.touch_cooldown = self.TOUCH_COOLDOWN_STEPS
        ball_touch_reward = float(np.clip(ball_touch_reward, 0.0, 0.05))

        # Pressure reward — push toward ball when it's in our half
        pressure_reward = 0.0
        if ball_pos[0] < 0.0:
            prev_dist = float(np.linalg.norm(self.prev_ball_pos - self.prev_player_pos))
            curr_dist = float(np.linalg.norm(ball_pos - player_pos))
            pressure_reward = float(np.clip(0.10 * (prev_dist - curr_dist), -0.04, 0.04))

        # Clearance reward — reward kicking the ball out of our danger zone
        clearance_reward = 0.0
        if self.prev_ball_pos[0] < -0.25 * PITCH_X_HALF:
            moved_toward_mid = float(ball_pos[0] - self.prev_ball_pos[0])
            clearance_reward = float(np.clip(0.35 * max(0.0, moved_toward_mid), 0.0, 0.10))

        # Idle defense penalty — punish staying far from ball in own half
        dist_to_ball = float(np.linalg.norm(ball_pos - player_pos))
        idle_defense_penalty = 0.0
        if ball_pos[0] < -0.35 * PITCH_X_HALF and dist_to_ball > 4.0:
            idle_defense_penalty = -0.02

        # Support / off-ball positioning
        support_reward = 0.0
        off_ball_penalty = 0.0
        if 3.0 <= dist_to_ball <= 7.0:
            support_reward = 0.005
        elif dist_to_ball > 8.0 and ball_pos[0] < 0.8 * PITCH_X_HALF:
            off_ball_penalty = -0.01

        # Danger-touch penalty — penalise own-goal-direction touches near box
        prev_dist_to_ball = float(np.linalg.norm(self.prev_ball_pos - self.prev_player_pos))
        close_to_ball_now = dist_to_ball <= 1.6
        was_close_to_ball = prev_dist_to_ball <= 1.6

        ball_dx = float(ball_pos[0] - self.prev_ball_pos[0])
        ball_speed_change = float(np.linalg.norm(ball_vel - self.prev_ball_vel))
        ball_moved_to_own_goal = ball_dx < -0.05
        ball_velocity_to_own_goal = _safe_float(ball_vel[0]) < -0.5
        danger_zone = ball_pos[0] < -0.35 * PITCH_X_HALF

        touch_like_event = touched_ball or (
            close_to_ball_now
            and was_close_to_ball
            and player_step_distance > 0.02
            and (abs(ball_dx) > 0.03 or ball_speed_change > 0.5)
        )

        danger_touch_penalty = 0.0
        danger_touch_count = 0.0
        if (
            self.danger_touch_cooldown == 0
            and danger_zone
            and touch_like_event
            and (ball_moved_to_own_goal or ball_velocity_to_own_goal)
        ):
            danger_touch_penalty = -0.03
            danger_touch_count = 1.0
            self.danger_touch_cooldown = 8
        danger_touch_penalty = float(np.clip(danger_touch_penalty, -0.03, 0.0))

        # ------------------------------------------------------------------
        # Stage 2 — vs moderate agent
        # Original goal_progress coeff, no shooting, no pass reward.
        # ------------------------------------------------------------------
        if self.stage == 2:
            goal_progress_reward = float(
                np.clip(0.08 * (prev_goal_dist - curr_goal_dist), -0.08, 0.08)
            )
            shooting_reward = 0.0

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
                "shooting_reward": shooting_reward,
                "goal_side_reward": 0.0,
                "support_reward": support_reward,
                "off_ball_penalty": off_ball_penalty,
                "clearance_reward": clearance_reward,
                "danger_touch_penalty": danger_touch_penalty,
                "danger_touch_count": danger_touch_count,
                "pressure_reward": pressure_reward,
                "idle_defense_penalty": idle_defense_penalty,
                "player_step_distance": player_step_distance,
                "shaping_reward": shaping_reward,
                "shaped_reward": shaped_reward,
                "opponent_goal_x": opponent_goal_x,
                "opponent_goal_y": opponent_goal_y,
                # Pass metrics — all zero in stage 2
                "pass_reward": 0.0,
                "pass_attempt_count": 0.0,
                "pass_complete_count": 0.0,
                "pass_fail_count": 0.0,
                "pass_travel": 0.0,
                "pass_completion_rate": 0.0,
            }
            return shaped_reward, info_terms

        # ------------------------------------------------------------------
        # Stage 3 — vs baseline agent
        # Stronger goal_progress + shooting reward + PASS REWARD.
        #
        # goal_progress: coeff 0.08 → 0.15, clip ±0.10 (from v3).
        # shooting_reward: coeff=0.01, gate ball_speed>1.5 (from v3).
        # pass_reward: calibrated proxy — 0.15 per completion, forward-gated,
        #   MIN_RELEASE_SPEED=1.20, MIN_TRAVEL=3.00, MAX_STEPS=25.
        #
        # Budget analysis (per-step, expected values):
        #   goal_progress ~0.003  (dominant, as intended)
        #   shooting      ~0.001  (fires ~3% of steps at ball_speed>1.5)
        #   pass_reward   ~0.003  (0.15 × ~2% completion rate / episode_len)
        #   ball_touch    ~0.001  (0.05 × ~2% touch rate with cooldown)
        #   Total shaping clip extended to ±0.22 to give pass headroom.
        # ------------------------------------------------------------------
        goal_progress_reward = float(
            np.clip(0.15 * (prev_goal_dist - curr_goal_dist), -0.10, 0.10)
        )

        ball_speed = float(np.linalg.norm(ball_vel))
        alignment = float(np.dot(ball_vel_dir, goal_dir))
        shooting_reward = 0.0
        if ball_speed > 1.5:
            shooting_reward = float(
                np.clip(0.01 * max(0.0, alignment) * ball_speed, 0.0, 0.03)
            )

        # Pass reward — from the calibrated proxy tracker
        pass_reward = _safe_float(pass_proxy_terms.get("pass_reward", 0.0))

        shaping_reward = (
            proximity_reward
            + goal_progress_reward
            + direction_reward
            + ball_touch_reward
            + shooting_reward
            + pass_reward
            + pressure_reward
            + clearance_reward
            + idle_defense_penalty
            + support_reward
            + off_ball_penalty
            + danger_touch_penalty
        )
        # Clip widened from ±0.18 (v3) to ±0.22 to give pass_reward headroom.
        # A single completed pass (0.15) should not be clipped away when
        # other shaping terms happen to be positive in the same step.
        shaping_reward = float(np.clip(shaping_reward, -0.22, 0.22))
        shaped_reward = env_reward + shaping_reward

        # Compute per-episode completion rate for logging (safe division)
        pass_attempt = _safe_float(pass_proxy_terms.get("pass_attempt_count", 0.0))
        pass_complete = _safe_float(pass_proxy_terms.get("pass_complete_count", 0.0))
        pass_completion_rate = (pass_complete / pass_attempt) if pass_attempt > 0.0 else 0.0

        info_terms = {
            "env_reward": env_reward,
            "proximity_reward": proximity_reward,
            "goal_progress_reward": goal_progress_reward,
            "direction_reward": direction_reward,
            "ball_touch_reward": ball_touch_reward,
            "shooting_reward": shooting_reward,
            "goal_side_reward": 0.0,
            "support_reward": support_reward,
            "off_ball_penalty": off_ball_penalty,
            "clearance_reward": clearance_reward,
            "danger_touch_penalty": danger_touch_penalty,
            "danger_touch_count": danger_touch_count,
            "pressure_reward": pressure_reward,
            "idle_defense_penalty": idle_defense_penalty,
            "player_step_distance": player_step_distance,
            "shaping_reward": shaping_reward,
            "shaped_reward": shaped_reward,
            "opponent_goal_x": opponent_goal_x,
            "opponent_goal_y": opponent_goal_y,
            # Pass metrics
            "pass_reward": pass_reward,
            "pass_attempt_count": _safe_float(pass_proxy_terms.get("pass_attempt_count", 0.0)),
            "pass_complete_count": _safe_float(pass_proxy_terms.get("pass_complete_count", 0.0)),
            "pass_fail_count": _safe_float(pass_proxy_terms.get("pass_fail_count", 0.0)),
            "pass_travel": _safe_float(pass_proxy_terms.get("pass_travel", 0.0)),
            "pass_completion_rate": pass_completion_rate,
        }
        return shaped_reward, info_terms


# ---------------------------------------------------------------------------
# Metrics callback
# ---------------------------------------------------------------------------

class CurriculumMetricsCallback(DefaultCallbacks):
    """Per-episode metrics logged to custom_metrics."""

    def on_episode_start(self, *, episode, **kwargs):
        episode.user_data.update({
            "env_reward_sum": 0.0,
            "step_count": 0,
            "possession_vals": [],
            "stage_vals": [],
            "field_tilt_vals": [],
            "defensive_zone_vals": [],
            "goal_side_vals": [],
            "goal_side_reward_vals": [],
            "clearance_reward_vals": [],
            "danger_touch_penalty_vals": [],
            "danger_touch_count_vals": [],
            "env_reward_vals": [],
            "shaping_reward_vals": [],
            "proximity_reward_vals": [],
            "goal_progress_reward_vals": [],
            "direction_reward_vals": [],
            "ball_touch_reward_vals": [],
            "shooting_reward_vals": [],
            "pressure_reward_vals": [],
            "idle_defense_penalty_vals": [],
            "support_reward_vals": [],
            "off_ball_penalty_vals": [],
            "player_step_distance_vals": [],
            # Pass metric lists — non-zero only in stage 3
            "pass_reward_vals": [],
            "pass_attempt_count_vals": [],
            "pass_complete_count_vals": [],
            "pass_fail_count_vals": [],
            "pass_travel_vals": [],
            "pass_completion_rate_vals": [],
        })

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

        episode.user_data["possession_vals"].append(possession_step)
        episode.user_data["stage_vals"].append(_safe_float(info.get("stage", 0.0)))
        episode.user_data["field_tilt_vals"].append(float(ball_pos[0]) / PITCH_X_HALF)
        episode.user_data["defensive_zone_vals"].append(1.0 if ball_pos[0] < 0.0 else 0.0)
        episode.user_data["goal_side_vals"].append(goal_side_step)

        for key in [
            "goal_side_reward", "clearance_reward", "danger_touch_penalty",
            "danger_touch_count", "env_reward", "shaping_reward",
            "proximity_reward", "goal_progress_reward", "direction_reward",
            "ball_touch_reward", "shooting_reward",
            "pressure_reward", "idle_defense_penalty",
            "player_step_distance", "support_reward", "off_ball_penalty",
            # Pass metrics
            "pass_reward", "pass_attempt_count", "pass_complete_count",
            "pass_fail_count", "pass_travel", "pass_completion_rate",
        ]:
            episode.user_data[f"{key}_vals"].append(_safe_float(info.get(key, 0.0)))

    def on_episode_end(self, *, episode, **kwargs):
        steps = max(1, int(episode.user_data["step_count"]))

        episode.custom_metrics["stage"] = float(np.mean(episode.user_data["stage_vals"] or [0.0]))
        episode.custom_metrics["possession_pct"] = float(np.mean(episode.user_data["possession_vals"] or [0.0]))
        episode.custom_metrics["field_tilt"] = float(np.mean(episode.user_data["field_tilt_vals"] or [0.0]))
        episode.custom_metrics["defensive_zone_pct"] = float(np.mean(episode.user_data["defensive_zone_vals"] or [0.0]))
        episode.custom_metrics["goal_side_pct"] = float(np.mean(episode.user_data["goal_side_vals"] or [0.0]))

        for key in [
            "goal_side_reward", "clearance_reward", "danger_touch_penalty",
            "env_reward", "shaping_reward", "proximity_reward",
            "goal_progress_reward", "direction_reward", "ball_touch_reward",
            "shooting_reward",
            "pressure_reward", "idle_defense_penalty",
            "support_reward", "off_ball_penalty",
            # Pass metrics (mean per step — gives per-step contribution)
            "pass_reward", "pass_travel", "pass_completion_rate",
        ]:
            episode.custom_metrics[f"{key}_mean"] = float(
                np.mean(episode.user_data[f"{key}_vals"] or [0.0])
            )

        episode.custom_metrics["danger_touch_count"] = float(
            np.sum(episode.user_data["danger_touch_count_vals"] or [0.0])
        )
        episode.custom_metrics["player_distance_travelled"] = float(
            np.sum(episode.user_data["player_step_distance_vals"] or [0.0])
        )
        episode.custom_metrics["player_step_distance_mean"] = float(
            np.mean(episode.user_data["player_step_distance_vals"] or [0.0])
        )

        # Pass counts — sum over the episode (not mean) for interpretability
        episode.custom_metrics["pass_attempt_count"] = float(
            np.sum(episode.user_data["pass_attempt_count_vals"] or [0.0])
        )
        episode.custom_metrics["pass_complete_count"] = float(
            np.sum(episode.user_data["pass_complete_count_vals"] or [0.0])
        )
        episode.custom_metrics["pass_fail_count"] = float(
            np.sum(episode.user_data["pass_fail_count_vals"] or [0.0])
        )

        env_sum = float(episode.user_data["env_reward_sum"])
        episode.custom_metrics["win"] = 1.0 if env_sum > 0 else 0.0
        episode.custom_metrics["loss"] = 1.0 if env_sum < 0 else 0.0
        episode.custom_metrics["draw"] = 1.0 if env_sum == 0 else 0.0
        episode.custom_metrics["episode_length"] = float(steps)

    def on_train_result(self, *, result: dict, **kwargs):
        cm = result.get("custom_metrics", {})
        learner_info = result.get("info", {}).get("learner", {})
        policy_key = "default_policy" if "default_policy" in learner_info else "default"
        policy_stats = learner_info.get(policy_key, {}).get("learner_stats", {})

        print(
            "[metrics] "
            f"iter={result.get('training_iteration')} "
            f"ts={result.get('timesteps_total')} "
            f"reward={result.get('episode_reward_mean')} "
            f"stage={cm.get('stage_mean')} "
            f"W/L/D=({cm.get('win_mean')}, {cm.get('loss_mean')}, {cm.get('draw_mean')}) "
            f"env_reward={cm.get('env_reward_mean_mean')} "
            f"goal_progress={cm.get('goal_progress_reward_mean_mean')} "
            f"shooting={cm.get('shooting_reward_mean_mean')} "
            f"pass_reward={cm.get('pass_reward_mean_mean')} "
            f"pass_attempts={cm.get('pass_attempt_count_mean')} "
            f"pass_completes={cm.get('pass_complete_count_mean')} "
            f"pass_completion_rate={cm.get('pass_completion_rate_mean_mean')} "
            f"ball_touch={cm.get('ball_touch_reward_mean_mean')} "
            f"shaping={cm.get('shaping_reward_mean_mean')} "
            f"possession={cm.get('possession_pct_mean')} "
            f"field_tilt={cm.get('field_tilt_mean')} "
            f"entropy={policy_stats.get('entropy')} "
            f"vf_loss={policy_stats.get('vf_loss')} "
            f"vf_explained_var={policy_stats.get('vf_explained_var')} "
            f"policy_loss={policy_stats.get('policy_loss')}"
        )


# ---------------------------------------------------------------------------
# Stage-switch callback
# ---------------------------------------------------------------------------

class CurriculumStageSwitchCallback(CurriculumMetricsCallback):
    def __init__(self):
        super().__init__()
        self._switched_to_stage2 = False
        self._switched_to_stage3 = False

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        super().on_train_result(trainer=trainer, result=result, **kwargs)

        env_cfg = trainer.config.get("env_config", {})
        phase1_target = env_cfg.get("phase1_timesteps")
        phase2_target = env_cfg.get("phase2_timesteps")
        timesteps = result.get("timesteps_total", 0)

        if (
            not self._switched_to_stage2
            and phase1_target is not None
            and timesteps >= phase1_target
        ):
            _switch_stage_on_all_workers(trainer, 2)
            _update_ppo_params(trainer, clip_param=0.15, num_sgd_iter=10)
            self._switched_to_stage2 = True
            print(
                f"[curriculum] → stage 2 at {timesteps} steps | "
                f"opponent: moderate agent | clip→0.15 | NO pass reward"
            )

        phase3_boundary = None
        if phase1_target is not None and phase2_target is not None:
            phase3_boundary = phase1_target + phase2_target

        if (
            not self._switched_to_stage3
            and phase3_boundary is not None
            and timesteps >= phase3_boundary
        ):
            _switch_stage_on_all_workers(trainer, 3)
            # Stage 3: raise entropy for continued exploration, loosen clip
            # slightly, and reduce SGD iterations to prevent over-fitting on
            # noisy early-stage-3 batches.
            _update_ppo_params(
                trainer,
                clip_param=0.12,
                entropy_coeff=0.015,
                num_sgd_iter=10,
            )
            self._switched_to_stage3 = True
            print(
                f"[curriculum] → stage 3 at {timesteps} steps | "
                f"opponent: baseline agent | clip→0.12, entropy_coeff→0.015 | "
                f"PASS REWARD ACTIVATED (reward=0.15, forward-gated)"
            )


# ---------------------------------------------------------------------------
# Env factory & worker helpers
# ---------------------------------------------------------------------------

def create_curriculum_env(env_config: Dict[str, Any]):
    base_env = create_rllib_env(env_config)
    return CurriculumRewardWrapper(base_env, stage=int(env_config.get("stage", 1)))


def _switch_stage_on_all_workers(trainer: PPOTrainer, stage: int) -> None:
    def _set_stage(env):
        if hasattr(env, "set_stage") and callable(getattr(env, "set_stage")):
            env.set_stage(stage)
    trainer.workers.foreach_worker(lambda w: w.foreach_env(_set_stage))


def _update_ppo_params(
    trainer: PPOTrainer,
    clip_param: float = None,
    entropy_coeff: float = None,
    num_sgd_iter: int = None,
) -> None:
    """
    Reliably update PPO hyperparameters mid-training.

    trainer.config is read by some code paths but not all:
      - clip_param and entropy_coeff are read from the policy's config at
        each learn step, so we must update the policy config on every worker.
      - num_sgd_iter is read from trainer.config by the main loop, so we
        update both.
    """
    updates = {}
    if clip_param is not None:
        updates["clip_param"] = clip_param
    if entropy_coeff is not None:
        updates["entropy_coeff"] = entropy_coeff
    if num_sgd_iter is not None:
        updates["num_sgd_iter"] = num_sgd_iter
        trainer.config["num_sgd_iter"] = num_sgd_iter

    if not updates:
        return

    trainer.config.update(updates)

    def _set_policy_config(worker):
        policy = worker.get_policy()
        if policy is not None and hasattr(policy, "config"):
            policy.config.update(updates)

    trainer.workers.foreach_worker(_set_policy_config)


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_config(args) -> Dict[str, Any]:
    p1 = args.phase1_timesteps
    p2 = args.phase2_timesteps
    p3 = args.phase3_timesteps

    phase2_start = p1
    phase3_start = p1 + p2
    total_steps = p1 + p2 + p3

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
            "phase2_timesteps": p2,
            "phase3_timesteps": p3,
            "total_timesteps": total_steps,
        },
        # vf_share_layers=True: shared head gives richer gradient signal to
        # the policy, driving faster entropy collapse in Stage 1.
        "model": {
            "vf_share_layers": True,
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
        },
        "lr": 1e-4,
        "lr_schedule": [
            [0,                                     1e-4],
            [max(0, phase2_start - 1),              1e-4],
            [phase2_start,                          5e-5],
            [max(phase3_start - 1, phase2_start),   5e-5],
            [phase3_start,                          2e-5],
        ],
        "gamma": 0.99,
        "lambda": 0.95,
        "clip_param": 0.2,
        "entropy_coeff": 0.005,       # raised to 0.015 in stage 3 via callback
        "vf_loss_coeff": 1.0,
        "rollout_fragment_length": 400,
        "train_batch_size": 4000,
        "sgd_minibatch_size": 256,
        "num_sgd_iter": 15,
        "batch_mode": "truncate_episodes",
    }


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Three-phase curriculum PPO v4 — pass reward added in Phase 3 only. "
            "Calibrated proxy: reward=0.15, forward-gated, MIN_RELEASE_SPEED=1.20, "
            "MIN_TRAVEL=3.00."
        )
    )
    parser.add_argument("--phase1-timesteps", type=int, default=2_000_000)
    parser.add_argument("--phase2-timesteps", type=int, default=2_000_000)
    parser.add_argument("--phase3-timesteps", type=int, default=5_000_000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-envs-per-worker", type=int, default=1)
    parser.add_argument("--experiment-name", type=str, default="ppo_curriculum_v4_pass_phase3")
    parser.add_argument("--local-dir", type=str, default="./ray_results")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# CSV / JSON logging helpers
# ---------------------------------------------------------------------------

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
        "shooting_reward_mean": cm.get("shooting_reward_mean_mean"),
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
        "clearance_reward": cm.get("clearance_reward_mean"),
        "danger_touch_penalty": cm.get("danger_touch_penalty_mean"),
        "danger_touch_count": cm.get("danger_touch_count_mean"),
        "support_reward_mean": cm.get("support_reward_mean_mean"),
        "off_ball_penalty_mean": cm.get("off_ball_penalty_mean_mean"),
        # Pass metrics
        "pass_reward_mean": cm.get("pass_reward_mean_mean"),
        "pass_attempt_count": cm.get("pass_attempt_count_mean"),
        "pass_complete_count": cm.get("pass_complete_count_mean"),
        "pass_fail_count": cm.get("pass_fail_count_mean"),
        "pass_travel_mean": cm.get("pass_travel_mean_mean"),
        "pass_completion_rate_mean": cm.get("pass_completion_rate_mean_mean"),
        "entropy": policy_stats.get("entropy"),
        "vf_loss": policy_stats.get("vf_loss"),
        "vf_explained_var": policy_stats.get("vf_explained_var"),
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
        "training_iteration", "timesteps_total", "episodes_total",
        "time_total_s", "episode_reward_mean", "episode_reward_min", "episode_reward_max",
        "custom_metrics/stage_mean",
        "custom_metrics/win_mean", "custom_metrics/draw_mean", "custom_metrics/loss_mean",
        "custom_metrics/episode_length_mean",
        "custom_metrics/env_reward_mean_mean",
        "custom_metrics/shaping_reward_mean_mean",
        "custom_metrics/proximity_reward_mean_mean",
        "custom_metrics/goal_progress_reward_mean_mean",
        "custom_metrics/direction_reward_mean_mean",
        "custom_metrics/ball_touch_reward_mean_mean",
        "custom_metrics/shooting_reward_mean_mean",
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
        "custom_metrics/clearance_reward_mean",
        "custom_metrics/danger_touch_penalty_mean",
        "custom_metrics/danger_touch_count_mean",
        # Pass metric columns
        "custom_metrics/pass_reward_mean_mean",
        "custom_metrics/pass_attempt_count_mean",
        "custom_metrics/pass_complete_count_mean",
        "custom_metrics/pass_fail_count_mean",
        "custom_metrics/pass_travel_mean_mean",
        "custom_metrics/pass_completion_rate_mean_mean",
        "info/learner/default_policy/learner_stats/entropy",
        "info/learner/default_policy/learner_stats/vf_loss",
        "info/learner/default_policy/learner_stats/vf_explained_var",
        "info/learner/default_policy/learner_stats/policy_loss",
        "info/learner/default/learner_stats/entropy",
        "info/learner/default/learner_stats/vf_loss",
        "info/learner/default/learner_stats/vf_explained_var",
        "info/learner/default/learner_stats/policy_loss",
    ]

    if analysis.trials:
        trial = analysis.trials[0]
        trial_df = analysis.trial_dataframes.get(trial.logdir)
        if trial_df is not None:
            present = [c for c in tracked_columns if c in trial_df.columns]
            training_log = trial_df[present].copy().sort_values("training_iteration")
            csv_path = output_dir / "training_log.csv"
            json_path = output_dir / "training_log.json"
            training_log.to_csv(csv_path, index=False)
            training_log.to_json(json_path, orient="records", indent=2)
            print(f"Saved training log: {csv_path}")
            print(f"Saved training log: {json_path}")
        else:
            print("WARNING: No trial dataframe found; training history not saved.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    total_timesteps = args.phase1_timesteps + args.phase2_timesteps + args.phase3_timesteps

    print(
        "\n[v4] vf_share_layers=True (shared head — fast entropy collapse).\n"
        "[v4] Shooting reward: coeff=0.01, gate ball_speed>1.5.\n"
        "[v4] Pass reward: PHASE 3 ONLY. reward=0.15 per completion, "
        "forward-gated (ball_vel_x>0), MIN_RELEASE_SPEED=1.20, MIN_TRAVEL=3.00.\n"
        "[v4] Shaping clip in stage 3 widened ±0.18 → ±0.22 for pass headroom.\n"
        "[v4] Starting fresh from scratch.\n"
    )

    run_dir = Path(args.local_dir) / args.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    tune.registry.register_env("SoccerCurriculum", create_curriculum_env)

    analysis = tune.run(
        "PPO",
        name=args.experiment_name,
        config=build_config(args),
        stop={"timesteps_total": total_timesteps},
        checkpoint_freq=1,
        checkpoint_at_end=True,
        keep_checkpoints_num=None,
        local_dir=args.local_dir,
        verbose=1,
    )

    output_dir = Path(args.local_dir) / args.experiment_name
    _save_training_outputs(analysis, output_dir)

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(f"Best trial: {best_trial}")

    if best_trial is not None:
        best_checkpoint = analysis.get_best_checkpoint(
            trial=best_trial,
            metric="episode_reward_mean",
            mode="max",
        )
        print(f"Best checkpoint (by reward): {best_checkpoint}")
        if best_checkpoint is not None:
            ptr = output_dir / "best_checkpoint.txt"
            ptr.write_text(str(best_checkpoint) + "\n", encoding="utf-8")
            print(f"Wrote best checkpoint pointer: {ptr}")

    ray.shutdown()


if __name__ == "__main__":
    main()