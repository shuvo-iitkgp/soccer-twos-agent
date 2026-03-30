"""
train_stage2_defensive.py
=========================
Stage 2 of curriculum learning.

Stage 1: train_team_vs_random_ppo_reward_shaped.py — 2M steps, offensive shaping
Stage 2: THIS SCRIPT — 1M steps fine-tune from Stage 1 checkpoint, defensive shaping added

Key differences from Stage 1:
  - Offensive shaping weights reduced ~40%
  - Three defensive terms added: goal_side, clearance, danger_touch_penalty
  - Restores from best flatten=True checkpoint (agent 2)
  - Still trains vs random opponent (CPU-safe)
  - Logs new defensive metrics in callback

Run command:
  python train_stage2_defensive.py \
    --restore "C:/ckpt/agent2/checkpoint_000XXX/checkpoint-XXX" \
    --timesteps-total 1000000 \
    --experiment-name stage2_defensive_curriculum
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from soccer_twos import EnvType

from reward_shaping_stage2 import RewardShapingConfig, RewardShapingWrapper
from utils import create_rllib_env
import warnings
warnings.filterwarnings("ignore")


class Stage2Callback(DefaultCallbacks):
    """
    Extended callback that tracks both offensive and defensive shaping metrics.
    Mirrors Stage 1 callback structure — adds goal_side, clearance, danger_touch.
    """

    POSSESSION_RADIUS = 2.0
    FIELD_TILT_PROGRESS_THRESHOLD = 1.0 / 3.0

    def __init__(self):
        super().__init__()
        self._previous_rewards = []

    def on_episode_start(self, *, episode, **kwargs):
        # Offensive
        episode.user_data["env_rewards"] = []
        episode.user_data["proximity_rewards"] = []
        episode.user_data["goal_progress_rewards"] = []
        episode.user_data["direction_rewards"] = []
        episode.user_data["ball_touch_rewards"] = []
        episode.user_data["shaping_rewards"] = []
        # Defensive (new)
        episode.user_data["goal_side_rewards"] = []
        episode.user_data["clearance_rewards"] = []
        episode.user_data["danger_touch_penalties"] = []
        # Diagnostics
        episode.user_data["possession_steps"] = 0
        episode.user_data["defensive_zone_steps"] = 0
        episode.user_data["goal_side_steps"] = 0
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

        # Log reward terms
        episode.user_data["env_rewards"].append(float(info.get("env_reward", 0.0)))
        episode.user_data["proximity_rewards"].append(float(info.get("proximity_reward", 0.0)))
        episode.user_data["goal_progress_rewards"].append(float(info.get("goal_progress_reward", 0.0)))
        episode.user_data["direction_rewards"].append(float(info.get("direction_reward", 0.0)))
        episode.user_data["ball_touch_rewards"].append(float(info.get("ball_touch_reward", 0.0)))
        episode.user_data["shaping_rewards"].append(float(info.get("shaping_reward", 0.0)))
        episode.user_data["goal_side_rewards"].append(float(info.get("goal_side_reward", 0.0)))
        episode.user_data["clearance_rewards"].append(float(info.get("clearance_reward", 0.0)))
        episode.user_data["danger_touch_penalties"].append(float(info.get("danger_touch_penalty", 0.0)))

        player_info = info.get("player_info", {})
        ball_info = info.get("ball_info", {})

        episode.user_data["total_steps"] += 1
        if "position" not in player_info or "position" not in ball_info:
            return

        player_pos = np.asarray(player_info["position"], dtype=np.float32)
        ball_pos = np.asarray(ball_info["position"], dtype=np.float32)

        # Possession
        if float(np.linalg.norm(player_pos - ball_pos)) < self.POSSESSION_RADIUS:
            episode.user_data["possession_steps"] += 1

        goal_x = float(info.get("opponent_goal_x", 16.0))
        own_goal_x = -goal_x
        attack_sign = 1.0 if goal_x >= 0.0 else -1.0
        signed_ball_x = attack_sign * float(ball_pos[0])
        signed_goal_x = max(abs(goal_x), 1e-6)

        # Field tilt (attacking third)
        territorial_progress = signed_ball_x / signed_goal_x
        if territorial_progress > self.FIELD_TILT_PROGRESS_THRESHOLD:
            episode.user_data["final_third_steps"] += 1

        # Defensive zone (ball in own half)
        if signed_ball_x < 0:
            episode.user_data["defensive_zone_steps"] += 1

        # Goal-side steps (agent closer to own goal than ball)
        own_goal = np.array([own_goal_x, 0.0], dtype=np.float32)
        if np.linalg.norm(player_pos - own_goal) < np.linalg.norm(ball_pos - own_goal):
            episode.user_data["goal_side_steps"] += 1

    def on_episode_end(self, *, episode, **kwargs):
        def _mean(lst):
            return sum(lst) / len(lst) if lst else 0.0

        env_r = episode.user_data.get("env_rewards", [])
        if env_r:
            episode.custom_metrics["env_reward_mean"] = _mean(env_r)
            total_env = sum(env_r)
            episode.custom_metrics["win"] = 1.0 if total_env > 0 else 0.0
            episode.custom_metrics["loss"] = 1.0 if total_env < 0 else 0.0
            episode.custom_metrics["draw"] = 1.0 if total_env == 0 else 0.0

        # Offensive metrics
        episode.custom_metrics["proximity_reward_mean"] = _mean(episode.user_data.get("proximity_rewards", []))
        episode.custom_metrics["goal_progress_reward_mean"] = _mean(episode.user_data.get("goal_progress_rewards", []))
        episode.custom_metrics["direction_reward_mean"] = _mean(episode.user_data.get("direction_rewards", []))
        episode.custom_metrics["ball_touch_reward_mean"] = _mean(episode.user_data.get("ball_touch_rewards", []))
        episode.custom_metrics["shaping_reward_mean"] = _mean(episode.user_data.get("shaping_rewards", []))

        # Defensive metrics (new)
        episode.custom_metrics["goal_side_reward_mean"] = _mean(episode.user_data.get("goal_side_rewards", []))
        episode.custom_metrics["clearance_reward_mean"] = _mean(episode.user_data.get("clearance_rewards", []))
        episode.custom_metrics["danger_touch_penalty_mean"] = _mean(episode.user_data.get("danger_touch_penalties", []))

        total = episode.user_data.get("total_steps", 0)
        if total > 0:
            episode.custom_metrics["possession_pct"] = episode.user_data["possession_steps"] / total
            episode.custom_metrics["field_tilt"] = episode.user_data["final_third_steps"] / total
            episode.custom_metrics["defensive_zone_pct"] = episode.user_data["defensive_zone_steps"] / total
            episode.custom_metrics["goal_side_pct"] = episode.user_data["goal_side_steps"] / total

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        iteration = result.get("training_iteration")
        timesteps = result.get("timesteps_total")
        reward_mean = result.get("episode_reward_mean")
        custom = result.get("custom_metrics", {})

        print(
            f"[iter={iteration:04d}] "
            f"steps={timesteps} "
            f"reward={reward_mean} "
            f"win={custom.get('win_mean')} "
            f"loss={custom.get('loss_mean')} "
            f"poss={custom.get('possession_pct_mean')} "
            f"tilt={custom.get('field_tilt_mean')} "
            f"def_zone={custom.get('defensive_zone_pct_mean')} "
            f"goal_side={custom.get('goal_side_pct_mean')} "
            f"goal_side_rew={custom.get('goal_side_reward_mean_mean')} "
            f"clearance={custom.get('clearance_reward_mean_mean')} "
            f"danger_pen={custom.get('danger_touch_penalty_mean_mean')}"
        )

        if reward_mean is not None and not _is_nan_or_inf(reward_mean):
            self._previous_rewards.append(float(reward_mean))
            if len(self._previous_rewards) >= 5:
                tail = self._previous_rewards[-5:]
                if len(set(round(v, 6) for v in tail)) == 1:
                    print("WARNING: Mean reward constant for 5 iters — check env.")


def _is_nan_or_inf(value):
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return math.isnan(value) or math.isinf(value)
    return False


def _save_training_outputs(trainer, output_dir: Path, log_rows: list):
    output_dir.mkdir(parents=True, exist_ok=True)
    if log_rows:
        df = pd.DataFrame(log_rows)
        df.to_csv(output_dir / "training_log.csv", index=False)
        df.to_json(output_dir / "training_log.json", orient="records", indent=2)
        print(f"Saved logs to {output_dir}")


def build_config(
    num_workers: int,
    num_envs_per_worker: int,
    shaping_config: RewardShapingConfig,
):
    def _env_creator(env_config):
        env = create_rllib_env(env_config)
        return RewardShapingWrapper(env, config=shaping_config)

    tune.registry.register_env("SoccerStage2", _env_creator)

    return {
        "num_gpus": 0,
        "num_workers": num_workers,
        "num_envs_per_worker": num_envs_per_worker,
        "log_level": "INFO",
        "framework": "torch",
        "callbacks": Stage2Callback,
        "env": "SoccerStage2",
        "env_config": {
            "num_envs_per_worker": num_envs_per_worker,
            "variation": EnvType.team_vs_policy,
            "multiagent": False,
            "single_player": False,
            # Stage 1 used flatten_branched=True (Discrete 27)
            # Keep consistent so checkpoint weights load correctly
            "flatten_branched": True,
        },
        "model": {
            "vf_share_layers": True,
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
        },
        "lr": 5e-5,          # Lower LR for fine-tuning (was 1e-4 in Stage 1)
        "gamma": 0.99,
        "lambda": 0.95,
        "clip_param": 0.1,   # Tighter clipping for fine-tune stability (was 0.2)
        "entropy_coeff": 0.003,  # Lower entropy — policy already converged
        "vf_loss_coeff": 1.0,
        "rollout_fragment_length": 400,
        "train_batch_size": 4000,
        "sgd_minibatch_size": 256,
        "num_sgd_iter": 10,
        "batch_mode": "truncate_episodes",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2 curriculum: fine-tune with defensive shaping."
    )
    parser.add_argument("--timesteps-total", type=int, default=1_000_000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-envs-per-worker", type=int, default=1)
    parser.add_argument(
        "--experiment-name", type=str, default="stage2_defensive_curriculum"
    )
    parser.add_argument("--local-dir", type=str, default="./ray_results")
    parser.add_argument("--opponent-goal-x", type=float, default=16.0)
    parser.add_argument("--opponent-goal-y", type=float, default=0.0)
    parser.add_argument("--debug-print-every", type=int, default=0)
    parser.add_argument(
        "--restore",
        type=str,
        default=None,
        required=True,
        help="Path to Stage 1 checkpoint FILE (e.g. .../checkpoint_000500/checkpoint-500)",
    )
    args = parser.parse_args()

    # Stage 2 shaping config — offensive reduced, defensive added
    shaping_config = RewardShapingConfig(
        # Offensive (reduced ~40% from Stage 1)
        alpha=0.12,
        beta=0.06,
        direction_coeff=0.006,
        ball_touch_bonus=0.12,
        max_dist=30.0,
        proximity_clip=(-0.06, 0.06),
        goal_progress_clip=(-0.06, 0.06),
        direction_clip=(-0.006, 0.006),
        shaping_clip=(-0.3, 0.3),
        # Defensive (new)
        goal_side_coeff=0.02,
        goal_side_clip=(-0.02, 0.02),
        defensive_zone_x=0.0,
        clearance_coeff=0.03,
        clearance_clip=(-0.03, 0.03),
        danger_radius=8.0,
        danger_touch_penalty=-0.03,
        danger_touch_clip=(-0.03, 0.0),
        opponent_goal_x=args.opponent_goal_x,
        opponent_goal_y=args.opponent_goal_y,
        debug_print_every=args.debug_print_every,
    )

    ray.init(
        ignore_reinit_error=True,
        object_store_memory=500 * 1024 * 1024,
    )

    config = build_config(args.num_workers, args.num_envs_per_worker, shaping_config)
    output_dir = Path(args.local_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Fine-tune from Stage 1 checkpoint using direct PPOTrainer restore
    # (bypasses tune.run restore which has Windows path bugs)
    # ------------------------------------------------------------------ #
    import pickle, shutil
    from ray.rllib.agents.ppo import PPOTrainer

    print(f"[stage2] Restoring from: {args.restore}")

    # Strip optimizer state to avoid numpy.object_ dtype error
    checkpoint_path = args.restore
    fixed_path = checkpoint_path + ".stage2fixed"
    meta_path = checkpoint_path + ".tune_metadata"

    with open(checkpoint_path, "rb") as f:
        checkpoint_data = pickle.load(f)
    if "worker" in checkpoint_data:
        worker_state = pickle.loads(checkpoint_data["worker"])
        for pid, policy_state in worker_state.get("state", {}).items():
            policy_state.pop("optimizer_state", None)
        checkpoint_data["worker"] = pickle.dumps(worker_state)
    with open(fixed_path, "wb") as f:
        pickle.dump(checkpoint_data, f)
    if Path(meta_path).exists():
        shutil.copy(meta_path, fixed_path + ".tune_metadata")

    trainer = PPOTrainer(config=config)
    trainer.restore(fixed_path)
    print("[stage2] Restored successfully. Starting Stage 2 training...")

    log_rows = []
    best_reward = float("-inf")
    best_ckpt = None
    iteration = 0

    while True:
        result = trainer.train()
        iteration += 1
        timesteps = result.get("timesteps_total", 0)
        reward_mean = result.get("episode_reward_mean", float("nan"))
        custom = result.get("custom_metrics", {})

        log_rows.append({
            "training_iteration": result.get("training_iteration"),
            "timesteps_total": timesteps,
            "episode_reward_mean": reward_mean,
            "win_mean": custom.get("win_mean"),
            "loss_mean": custom.get("loss_mean"),
            "draw_mean": custom.get("draw_mean"),
            "possession_pct_mean": custom.get("possession_pct_mean"),
            "field_tilt_mean": custom.get("field_tilt_mean"),
            "defensive_zone_pct_mean": custom.get("defensive_zone_pct_mean"),
            "goal_side_pct_mean": custom.get("goal_side_pct_mean"),
            "goal_side_reward_mean": custom.get("goal_side_reward_mean_mean"),
            "clearance_reward_mean": custom.get("clearance_reward_mean_mean"),
            "danger_touch_penalty_mean": custom.get("danger_touch_penalty_mean_mean"),
            "shaping_reward_mean": custom.get("shaping_reward_mean_mean"),
        })

        if iteration % 3 == 0:
            ckpt = trainer.save(str(output_dir))
            print(f"[ckpt] {ckpt}")
            if not _is_nan_or_inf(reward_mean) and reward_mean > best_reward:
                best_reward = reward_mean
                best_ckpt = ckpt

        if timesteps >= args.timesteps_total:
            print(f"[stage2] Done at {timesteps} steps.")
            break

    final_ckpt = trainer.save(str(output_dir))
    print(f"[stage2] Final checkpoint: {final_ckpt}")

    _save_training_outputs(trainer, output_dir, log_rows)

    if best_ckpt:
        (output_dir / "best_checkpoint.txt").write_text(best_ckpt + "\n")
        print(f"[stage2] Best checkpoint: {best_ckpt}")

    trainer.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
