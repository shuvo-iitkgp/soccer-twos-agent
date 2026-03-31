import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Tuple

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
        self.set_stage(stage)

    def _set_opponent(self) -> None:
        random_opponent = lambda *_: 0
        if self.stage == 2 and self._moderate_opponent is None:
            print("[curriculum] stage 2 opponent: fallback-to-random (moderate not implemented)")

        policy_fn = self._moderate_opponent if (self.stage == 2 and self._moderate_opponent) else random_opponent
        if hasattr(self.env, "set_policies") and callable(getattr(self.env, "set_policies")):
            try:
                self.env.set_policies(policy_fn)
                return
            except Exception:
                pass
        if hasattr(self.env, "set_opponent") and callable(getattr(self.env, "set_opponent")):
            try:
                self.env.set_opponent(policy_fn)
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

        return out

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

        if self.touch_cooldown > 0:
            self.touch_cooldown -= 1

        shaped_reward, shaping_terms = self._shape_with_stage_logic(
            env_reward=_safe_float(env_reward),
            player_pos=player_pos,
            ball_pos=ball_pos,
            ball_vel=ball_vel,
            info_dict=info_dict,
        )

        # payload = {
        #     "player_info": info_dict.get("player_info", {}),
        #     "ball_info": info_dict.get("ball_info", {}),
        #     "env_reward": shaping_terms["env_reward"],
        #     "proximity_reward": shaping_terms["proximity_reward"],
        #     "goal_progress_reward": shaping_terms["goal_progress_reward"],
        #     "direction_reward": shaping_terms["direction_reward"],
        #     "ball_touch_reward": shaping_terms["ball_touch_reward"],
        #     "shaping_reward": shaping_terms["shaping_reward"],
        #     "shaped_reward": shaping_terms["shaped_reward"],
        #     "opponent_goal_x": shaping_terms["opponent_goal_x"],
        #     "opponent_goal_y": shaping_terms["opponent_goal_y"],
        #     "goal_side_reward": shaping_terms["goal_side_reward"],
        #     "clearance_reward": shaping_terms["clearance_reward"],
        #     "danger_touch_penalty": shaping_terms["danger_touch_penalty"],
        #     "stage": self.stage,
        # }

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
    ):
        # Stage-1/Stage-2 formulas adapted to the original reward-shaping wrappers,
        # keeping per-component clipping and total shaping clipping behavior.
        opponent_goal_x = PITCH_X_HALF
        opponent_goal_y = 0.0

        ball_to_goal = np.array([opponent_goal_x - ball_pos[0], opponent_goal_y - ball_pos[1]], dtype=np.float32)
        player_to_ball = ball_pos - player_pos

        ball_dist = float(np.linalg.norm(player_to_ball))
        proximity_reward = float(np.clip(0.06 * (1.0 / (1.0 + ball_dist)), 0.0, 0.06))

        prev_goal_dist = float(
            np.linalg.norm(np.array([opponent_goal_x - self.prev_ball_pos[0], opponent_goal_y - self.prev_ball_pos[1]]))
        )
        curr_goal_dist = float(np.linalg.norm(ball_to_goal))
        goal_progress_reward = float(np.clip(0.30 * (prev_goal_dist - curr_goal_dist), -0.12, 0.12))

        goal_dir = ball_to_goal / (np.linalg.norm(ball_to_goal) + 1e-8)
        ball_vel_dir = ball_vel / (np.linalg.norm(ball_vel) + 1e-8)
        direction_reward = float(np.clip(0.03 * float(np.dot(goal_dir, ball_vel_dir)), -0.03, 0.03))

        touched_ball = bool(info_dict.get("player_info", {}).get("touched_ball", False)) or bool(
            info_dict.get("touched_ball", False)
        )
        ball_touch_reward = 0.0
        if touched_ball and self.touch_cooldown == 0:
            ball_touch_reward = 0.03
            self.touch_cooldown = self.TOUCH_COOLDOWN_STEPS
        ball_touch_reward = float(np.clip(ball_touch_reward, 0.0, 0.03))

        goal_side_reward = 0.0
        clearance_reward = 0.0
        danger_touch_penalty = 0.0

        if self.stage == 2:
            # Additional defensive terms from stage-2 shaping.
            goal_side_reward = float(np.clip(0.08 * (float(ball_pos[0]) / PITCH_X_HALF), -0.08, 0.08))

            moved_toward_mid = float(ball_pos[0] - self.prev_ball_pos[0])
            in_defensive_half_prev = self.prev_ball_pos[0] < 0.0
            clearance_reward = 0.25 * max(0.0, moved_toward_mid) if in_defensive_half_prev else 0.0
            clearance_reward = float(np.clip(clearance_reward, 0.0, 0.08))

            dangerous_zone = ball_pos[0] < -0.35 * PITCH_X_HALF
            moving_toward_own_goal = _safe_float(ball_vel[0]) < 0.0
            if touched_ball and dangerous_zone and moving_toward_own_goal:
                danger_touch_penalty = -0.06
            danger_touch_penalty = float(np.clip(danger_touch_penalty, -0.06, 0.0))

        shaping_reward = (
            proximity_reward
            + goal_progress_reward
            + direction_reward
            + ball_touch_reward
            + goal_side_reward
            + clearance_reward
            + danger_touch_penalty
        )
        shaping_reward = float(np.clip(shaping_reward, -0.20, 0.20))
        shaped_reward = env_reward + shaping_reward

        info_terms = {
            "env_reward": env_reward,
            "proximity_reward": proximity_reward,
            "goal_progress_reward": goal_progress_reward,
            "direction_reward": direction_reward,
            "ball_touch_reward": ball_touch_reward,
            "shaping_reward": shaping_reward,
            "shaped_reward": shaped_reward,
            "opponent_goal_x": opponent_goal_x,
            "opponent_goal_y": opponent_goal_y,
            "goal_side_reward": goal_side_reward,
            "clearance_reward": clearance_reward,
            "danger_touch_penalty": danger_touch_penalty,
        }
        return shaped_reward, info_terms



class CurriculumMetricsCallback(DefaultCallbacks):
    """Compute metrics from info schema and episode env rewards (no dependence on precomputed env metrics)."""

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

        # Distance-based possession (same definition style as earlier scripts).
        player_ball_dist = float(np.linalg.norm(player_pos - ball_pos))
        possession_step = 1.0 if player_ball_dist <= 1.5 else 0.0

        own_goal = np.array([-PITCH_X_HALF, 0.0], dtype=np.float32)
        player_to_own_goal = float(np.linalg.norm(player_pos - own_goal))
        ball_to_own_goal = float(np.linalg.norm(ball_pos - own_goal))
        goal_side_step = 1.0 if player_to_own_goal < ball_to_own_goal else 0.0

        # Same-style derived metrics computed from state each step.
        episode.user_data["field_tilt_vals"].append(float(ball_pos[0]) / PITCH_X_HALF)
        episode.user_data["defensive_zone_vals"].append(1.0 if ball_pos[0] < 0.0 else 0.0)
        episode.user_data["goal_side_vals"].append(goal_side_step)
        episode.user_data.setdefault("possession_vals", []).append(possession_step)

        episode.user_data["goal_side_reward_vals"].append(_safe_float(info.get("goal_side_reward", 0.0)))
        episode.user_data["clearance_reward_vals"].append(_safe_float(info.get("clearance_reward", 0.0)))
        episode.user_data["danger_touch_penalty_vals"].append(_safe_float(info.get("danger_touch_penalty", 0.0)))
        episode.user_data["env_reward_vals"].append(_safe_float(info.get("env_reward", 0.0)))
        episode.user_data["shaping_reward_vals"].append(_safe_float(info.get("shaping_reward", 0.0)))
        episode.user_data["proximity_reward_vals"].append(_safe_float(info.get("proximity_reward", 0.0)))
        episode.user_data["goal_progress_reward_vals"].append(_safe_float(info.get("goal_progress_reward", 0.0)))
        episode.user_data["direction_reward_vals"].append(_safe_float(info.get("direction_reward", 0.0)))
        episode.user_data["ball_touch_reward_vals"].append(_safe_float(info.get("ball_touch_reward", 0.0)))
    
    def on_episode_end(self, *, episode, **kwargs):
        steps = max(1, int(episode.user_data["step_count"]))
        possession_pct = float(np.mean(episode.user_data.get("possession_vals", [0.0])))

        episode.custom_metrics["possession_pct"] = possession_pct
        episode.custom_metrics["field_tilt"] = float(np.mean(episode.user_data["field_tilt_vals"] or [0.0]))
        episode.custom_metrics["defensive_zone_pct"] = float(
            np.mean(episode.user_data["defensive_zone_vals"] or [0.0])
        )
        episode.custom_metrics["goal_side_pct"] = float(np.mean(episode.user_data["goal_side_vals"] or [0.0]))

        episode.custom_metrics["goal_side_reward"] = float(
            np.mean(episode.user_data["goal_side_reward_vals"] or [0.0])
        )
        episode.custom_metrics["clearance_reward"] = float(
            np.mean(episode.user_data["clearance_reward_vals"] or [0.0])
        )
        episode.custom_metrics["danger_touch_penalty"] = float(
            np.mean(episode.user_data["danger_touch_penalty_vals"] or [0.0])
        )

        # W/L/D from summed environment rewards across the episode.
        env_sum = float(episode.user_data["env_reward_sum"])
        episode.custom_metrics["win"] = 1.0 if env_sum > 0 else 0.0
        episode.custom_metrics["loss"] = 1.0 if env_sum < 0 else 0.0
        episode.custom_metrics["draw"] = 1.0 if env_sum == 0 else 0.0
        episode.custom_metrics["episode_length"] = float(steps)

        episode.custom_metrics["env_reward_mean"] = float(np.mean(episode.user_data["env_reward_vals"] or [0.0]))
        episode.custom_metrics["shaping_reward_mean"] = float(np.mean(episode.user_data["shaping_reward_vals"] or [0.0]))
        episode.custom_metrics["proximity_reward_mean"] = float(np.mean(episode.user_data["proximity_reward_vals"] or [0.0]))
        episode.custom_metrics["goal_progress_reward_mean"] = float(np.mean(episode.user_data["goal_progress_reward_vals"] or [0.0]))
        episode.custom_metrics["direction_reward_mean"] = float(np.mean(episode.user_data["direction_reward_vals"] or [0.0]))
        episode.custom_metrics["ball_touch_reward_mean"] = float(np.mean(episode.user_data["ball_touch_reward_vals"] or [0.0]))

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
            f"possession_pct={cm.get('possession_pct_mean')} "
            f"field_tilt={cm.get('field_tilt_mean')} "
            f"defensive_zone_pct={cm.get('defensive_zone_pct_mean')} "
            f"goal_side_pct={cm.get('goal_side_pct_mean')} "
            f"goal_side_reward={cm.get('goal_side_reward_mean')} "
            f"clearance_reward={cm.get('clearance_reward_mean')} "
            f"danger_touch_penalty={cm.get('danger_touch_penalty_mean')} "
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


def build_config(args) -> Dict[str, Any]:
    return {
        "num_gpus": 0,
        "num_workers": args.num_workers,
        "num_envs_per_worker": args.num_envs_per_worker,
        "log_level": "WARN",
        "framework": "torch",
        "callbacks": CurriculumMetricsCallback,
        "env": "SoccerCurriculum",
        "env_config": {
            "num_envs_per_worker": args.num_envs_per_worker,
            "variation": EnvType.team_vs_policy,
            "multiagent": False,
            "single_player": False,
            "flatten_branched": True,
            "opponent_policy": lambda *_: 0,
            "stage": 1,
        },
        "model": {
            "vf_share_layers": True,
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
        },
        "lr": 1e-4,
        "gamma": 0.99,
        "lambda": 0.95,
        "clip_param": 0.2,
        "entropy_coeff": 0.005,
        "vf_loss_coeff": 1.0,
        "rollout_fragment_length": 400,
        "train_batch_size": 4000,
        "sgd_minibatch_size": 256,
        "num_sgd_iter": 15,
        "batch_mode": "truncate_episodes",
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Two-phase curriculum PPO run without restore.")
    parser.add_argument("--phase1-timesteps", type=int, default=2_000_000)
    parser.add_argument("--phase2-timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-envs-per-worker", type=int, default=1)
    parser.add_argument("--experiment-name", type=str, default="ppo_curriculum_no_restore")
    parser.add_argument("--local-dir", type=str, default="./ray_results")
    parser.add_argument("--debug-print-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    return parser.parse_args()


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
        "win": cm.get("win_mean"),
        "loss": cm.get("loss_mean"),
        "draw": cm.get("draw_mean"),
        "possession_pct": cm.get("possession_pct_mean"),
        "field_tilt": cm.get("field_tilt_mean"),
        "defensive_zone_pct": cm.get("defensive_zone_pct_mean"),
        "goal_side_pct": cm.get("goal_side_pct_mean"),
        "goal_side_reward": cm.get("goal_side_reward_mean"),
        "clearance_reward": cm.get("clearance_reward_mean"),
        "danger_touch_penalty": cm.get("danger_touch_penalty_mean"),
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


def main():
    args = parse_args()

    run_dir = Path(args.local_dir) / args.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "training_log.csv"

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    tune.registry.register_env("SoccerCurriculum", create_curriculum_env)

    trainer = PPOTrainer(config=build_config(args))

    phase1_target = int(args.phase1_timesteps)
    total_target = int(args.phase1_timesteps + args.phase2_timesteps)
    stage = 1

    print(
        "[curriculum] starting single continuous run "
        f"phase1={phase1_target} phase2={args.phase2_timesteps} total={total_target}"
    )

    rows = []
    while True:
        result = trainer.train()
        it = int(result.get("training_iteration", 0))
        ts = int(result.get("timesteps_total", 0))

        if stage == 1 and ts >= phase1_target:
            _switch_stage_on_all_workers(trainer, stage=2)
            stage = 2
            print(f"[curriculum] entering phase 2 at timestep {ts}")

        rows.append(_result_row(result, stage))

        if it % args.debug_print_every == 0:
            print(
                f"[curriculum] iter={it} stage={stage} timesteps={ts} "
                f"reward_mean={result.get('episode_reward_mean')}"
            )

        if it % args.checkpoint_every == 0:
            ckpt = trainer.save(str(run_dir))
            print(f"[curriculum] checkpoint saved: {ckpt}")

        if ts >= total_target:
            print(f"[curriculum] finished at timestep {ts} (target={total_target})")
            break

    final_ckpt = trainer.save(str(run_dir))
    print(f"[curriculum] final checkpoint saved: {final_ckpt}")

    _save_csv(rows, csv_path)
    print(f"[curriculum] metrics csv saved: {csv_path}")

    trainer.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()