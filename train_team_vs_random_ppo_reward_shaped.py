import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from soccer_twos import EnvType

from reward_shaping import RewardShapingConfig, RewardShapingWrapper
from utils import create_rllib_env
import warnings
warnings.filterwarnings('ignore')

class RewardShapedSanityCallback(DefaultCallbacks):
    """Baseline-style progress callback + shaping diagnostics."""

    POSSESSION_RADIUS = 2.0
    FIELD_TILT_PROGRESS_THRESHOLD = 1.0 / 3.0
    FIELD_TILT_DEBUG_PRINT_EVERY = 0

    def __init__(self):
        super().__init__()
        self._previous_rewards = []

    def on_episode_start(self, *, episode, **kwargs):
        episode.user_data["env_rewards"] = []
        episode.user_data["proximity_rewards"] = []
        episode.user_data["goal_progress_rewards"] = []
        episode.user_data["direction_rewards"] = []
        episode.user_data["ball_touch_rewards"] = []
        episode.user_data["shaping_rewards"] = []

        episode.user_data["possession_steps"] = 0
        episode.user_data["final_third_steps"] = 0
        episode.user_data["total_steps"] = 0

    def on_episode_step(self, *, episode, **kwargs):
        info = None
        try:
            info = episode.last_info_for("agent0")
        except Exception:
            info = None

        if not isinstance(info, dict):
            try:
                info = episode.last_info_for()
            except Exception:
                info = None

        if not isinstance(info, dict):
            return

        episode.user_data["env_rewards"].append(float(info.get("env_reward", 0.0)))
        episode.user_data["proximity_rewards"].append(float(info.get("proximity_reward", 0.0)))
        episode.user_data["goal_progress_rewards"].append(float(info.get("goal_progress_reward", 0.0)))
        episode.user_data["direction_rewards"].append(float(info.get("direction_reward", 0.0)))
        episode.user_data["ball_touch_rewards"].append(float(info.get("ball_touch_reward", 0.0)))
        episode.user_data["shaping_rewards"].append(float(info.get("shaping_reward", 0.0)))

        player_info = info.get("player_info", {})
        ball_info = info.get("ball_info", {})
        
        episode.user_data["total_steps"] += 1
        if "position" not in player_info or "position" not in ball_info:
            return

        player_pos = np.asarray(player_info["position"], dtype=np.float32)
        ball_pos = np.asarray(ball_info["position"], dtype=np.float32)


        dist = float(np.linalg.norm(player_pos - ball_pos))
        if dist < self.POSSESSION_RADIUS:
            episode.user_data["possession_steps"] += 1

        ball_x = float(ball_pos[0])
        goal_x = float(info.get("opponent_goal_x", 16.0))
        attack_sign = 1.0 if goal_x >= 0.0 else -1.0
        signed_ball_x = attack_sign * ball_x
        signed_goal_x = max(abs(goal_x), 1e-6)
        territorial_progress = signed_ball_x / signed_goal_x

        if territorial_progress > self.FIELD_TILT_PROGRESS_THRESHOLD:
            episode.user_data["final_third_steps"] += 1

        step_idx = episode.user_data["total_steps"]
        if self.FIELD_TILT_DEBUG_PRINT_EVERY > 0 and step_idx % self.FIELD_TILT_DEBUG_PRINT_EVERY == 0:
            print(
                f"[field-tilt-debug] ep_step={step_idx} "
                f"ball_x={ball_x:+.3f} "
                f"goal_x={goal_x:+.3f} "
                f"signed_ball_x={signed_ball_x:+.3f} "
                f"territorial_progress={territorial_progress:+.3f} "
                f"in_attacking_third={territorial_progress > self.FIELD_TILT_PROGRESS_THRESHOLD}"
            )

    def on_episode_end(self, *, episode, **kwargs):
        env_r = episode.user_data.get("env_rewards", [])
        prox = episode.user_data.get("proximity_rewards", [])
        goal = episode.user_data.get("goal_progress_rewards", [])
        direction = episode.user_data.get("direction_rewards", [])
        touch = episode.user_data.get("ball_touch_rewards", [])
        shp = episode.user_data.get("shaping_rewards", [])

        possession_steps = episode.user_data.get("possession_steps", 0)
        final_third_steps = episode.user_data.get("final_third_steps", 0)
        total_steps = episode.user_data.get("total_steps", 0)

        if env_r:
            episode.custom_metrics["env_reward_mean"] = sum(env_r) / len(env_r)
            total_env_reward = sum(env_r)
            episode.custom_metrics["win"]  = 1.0 if total_env_reward > 0 else 0.0
            episode.custom_metrics["loss"] = 1.0 if total_env_reward < 0 else 0.0
            episode.custom_metrics["draw"] = 1.0 if total_env_reward == 0 else 0.0

        if prox:
            episode.custom_metrics["proximity_reward_mean"] = sum(prox) / len(prox)
        if goal:
            episode.custom_metrics["goal_progress_reward_mean"] = sum(goal) / len(goal)
        if direction:
            episode.custom_metrics["direction_reward_mean"] = sum(direction) / len(direction)
        if touch:
            episode.custom_metrics["ball_touch_reward_mean"] = sum(touch) / len(touch)
        if shp:
            episode.custom_metrics["shaping_reward_mean"] = sum(shp) / len(shp)

        if total_steps > 0:
            episode.custom_metrics["possession_pct"] = possession_steps / total_steps
            episode.custom_metrics["field_tilt"] = final_third_steps / total_steps

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        iteration = result.get("training_iteration")
        timesteps = result.get("timesteps_total")
        reward_mean = result.get("episode_reward_mean")
        reward_min = result.get("episode_reward_min")
        reward_max = result.get("episode_reward_max")
        learner_info = result.get("info", {}).get("learner", {})
        custom_metrics = result.get("custom_metrics", {})
        # print(custom_metrics)
        env_mean = custom_metrics.get("env_reward_mean_mean")
        terminal_env_reward_mean = custom_metrics.get("terminal_env_reward_mean")
        shape_mean = custom_metrics.get("shaping_reward_mean_mean")
        prox_mean = custom_metrics.get("proximity_reward_mean_mean")
        goal_mean = custom_metrics.get("goal_progress_reward_mean_mean")
        dir_mean = custom_metrics.get("direction_reward_mean_mean")
        touch_mean = custom_metrics.get("ball_touch_reward_mean_mean")
        win_rate = custom_metrics.get("win_mean")
        draw_rate = custom_metrics.get("draw_mean")
        loss_rate = custom_metrics.get("loss_mean")
        possession_pct = custom_metrics.get("possession_pct_mean")
        field_tilt = custom_metrics.get("field_tilt_mean")

        print(
            f"[iter={iteration:04d}] "
            f"timesteps={timesteps} "
            f"reward_mean={reward_mean} "
            f"reward_min={reward_min} "
            f"reward_max={reward_max} "
            f"env_mean={env_mean} "
            f"terminal_env_reward_mean={terminal_env_reward_mean} "
            f"shaping_mean={shape_mean} "
            f"proximity_mean={prox_mean} "
            f"goal_progress_mean={goal_mean} "
            f"direction_mean={dir_mean} "
            f"touch_mean={touch_mean} "
            f"win_rate={win_rate} "
            f"draw_rate={draw_rate} "
            f"loss_rate={loss_rate} "
            f"possession_pct={possession_pct} "
            f"field_tilt={field_tilt}"
        )

        if _is_nan_or_inf(reward_mean) or _is_nan_or_inf(reward_min) or _is_nan_or_inf(reward_max):
            print("WARNING: Reward contains NaN/Inf values.")

        policy_key = "default_policy" if "default_policy" in learner_info else "default"
        policy_stats = learner_info.get(policy_key, {}).get("learner_stats", {})
        for metric_name in ["policy_loss", "vf_loss", "total_loss", "kl", "entropy"]:
            metric_value = policy_stats.get(metric_name)
            if _is_nan_or_inf(metric_value):
                print(f"WARNING: Learner metric '{metric_name}' is NaN/Inf: {metric_value}")

        if reward_mean is not None and not _is_nan_or_inf(reward_mean):
            self._previous_rewards.append(float(reward_mean))
            if len(self._previous_rewards) >= 5:
                tail = self._previous_rewards[-5:]
                if len(set(round(v, 6) for v in tail)) == 1:
                    print(
                        "WARNING: Mean reward has been constant for 5 iterations; "
                        "verify environment rollout/logging."
                    )


def _is_nan_or_inf(value):
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return math.isnan(value) or math.isinf(value)
    return False


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
        "custom_metrics/env_reward_mean",
        "custom_metrics/terminal_env_reward",
        "custom_metrics/win_mean",
        "custom_metrics/draw_mean",
        "custom_metrics/loss_mean",
        "custom_metrics/proximity_reward_mean",
        "custom_metrics/goal_progress_reward_mean",
        "custom_metrics/direction_reward_mean",
        "custom_metrics/ball_touch_reward_mean",
        "custom_metrics/shaping_reward_mean",
        "custom_metrics/possession_pct_mean",
        "custom_metrics/field_tilt_mean",
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


def build_config(num_workers: int, num_envs_per_worker: int, shaping_config: RewardShapingConfig):
    def _env_creator(env_config):
        env = create_rllib_env(env_config)
        return RewardShapingWrapper(env, config=shaping_config)

    tune.registry.register_env("SoccerRewardShaped", _env_creator)

    return {
        "num_gpus": 0,
        "num_workers": num_workers,
        "num_envs_per_worker": num_envs_per_worker,
        "log_level": "INFO",
        "framework": "torch",
        "callbacks": RewardShapedSanityCallback,
        "env": "SoccerRewardShaped",
        "env_config": {
            "num_envs_per_worker": num_envs_per_worker,
            "variation": EnvType.team_vs_policy,
            "multiagent": False,
            "single_player": False,
            "flatten_branched": True,
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


def main():
    parser = argparse.ArgumentParser(
        description="Train reward-shaped SoccerTwos PPO (team-vs-random baseline variant)."
    )
    parser.add_argument("--timesteps-total", type=int, default=500000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-envs-per-worker", type=int, default=1)
    parser.add_argument("--experiment-name", type=str, default="team_vs_random_ppo_reward_shaped")
    parser.add_argument("--local-dir", type=str, default="./ray_results")
    parser.add_argument("--opponent-goal-x", type=float, default=16.0)
    parser.add_argument("--opponent-goal-y", type=float, default=0.0)
    parser.add_argument("--debug-print-every", type=int, default=0)
    parser.add_argument("--restore", type=str, default=None)

    args = parser.parse_args()

    shaping_config = RewardShapingConfig(
        alpha=0.05,
        beta=0.1,
        direction_coeff=0.05,
        ball_touch_bonus=0.05,
        max_dist=30.0,
        proximity_clip=(-0.05, 0.05),
        goal_progress_clip=(-0.05, 0.05),
        direction_clip=(-0.05, 0.05),
        shaping_clip=(-0.1, 0.1),
        opponent_goal_x=args.opponent_goal_x,
        opponent_goal_y=args.opponent_goal_y,
        debug_print_every=args.debug_print_every,
    )

    ray.init(ignore_reinit_error=True)

    analysis = tune.run(
        "PPO",
        name=args.experiment_name,
        config=build_config(args.num_workers, args.num_envs_per_worker, shaping_config),
        stop={"timesteps_total": args.timesteps_total},
        checkpoint_freq=3,
        checkpoint_at_end=True,
        local_dir=args.local_dir,
        restore=args.restore,
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