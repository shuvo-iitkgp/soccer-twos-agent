import argparse
from pathlib import Path
from typing import Dict, Optional

import csv
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from soccer_twos import EnvType

from utils import create_rllib_env


class TeamVsRandomProgressCallback(DefaultCallbacks):
    """Simple callback to print concise, human-readable training progress."""

    def on_train_result(self, *, trainer, result: Dict, **kwargs):
        print(
            "[iter={iter:04d}] steps={steps} episodes={episodes} "
            "reward_mean={r_mean:.4f} min={r_min:.4f} max={r_max:.4f}".format(
                iter=result.get("training_iteration", 0),
                steps=result.get("timesteps_total", 0),
                episodes=result.get("episodes_total", 0),
                r_mean=float(result.get("episode_reward_mean") or 0.0),
                r_min=float(result.get("episode_reward_min") or 0.0),
                r_max=float(result.get("episode_reward_max") or 0.0),
            )
        )


def build_config(num_workers: int, num_envs_per_worker: int) -> Dict:
    """Build a CPU-friendly PPO config for SoccerTwos team-vs-random training."""
    return {
        # system
        "num_gpus": 0,
        "num_workers": num_workers,
        "num_envs_per_worker": num_envs_per_worker,
        "framework": "torch",
        "log_level": "INFO",
        "callbacks": TeamVsRandomProgressCallback,
        # environment (future reward shaping can be inserted here)
        "env": "Soccer",
        "env_config": {
            "num_envs_per_worker": num_envs_per_worker,
            "variation": EnvType.team_vs_policy,
            "multiagent": False,
            "flatten_branched": True,
        },
        # PPO defaults tuned for stable, CPU-friendly debugging
        "model": {
            "vf_share_layers": True,
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
        },
        "lr": 3e-4,
        "gamma": 0.99,
        "lambda": 0.95,
        "clip_param": 0.2,
        "entropy_coeff": 0.01,
        "vf_loss_coeff": 0.5,
        "rollout_fragment_length": 200,
        "train_batch_size": 4000,
        "sgd_minibatch_size": 256,
        "num_sgd_iter": 10,
        "batch_mode": "truncate_episodes",
    }


def _latest_checkpoint_in_trial_dir(trial_dir: Path) -> Optional[Path]:
    checkpoint_dirs = sorted(
        [p for p in trial_dir.glob("checkpoint_*") if p.is_dir()],
        key=lambda p: p.name,
    )
    if not checkpoint_dirs:
        return None

    last_dir = checkpoint_dirs[-1]
    checkpoint_files = sorted(last_dir.glob("checkpoint-*"))
    if checkpoint_files:
        return checkpoint_files[-1]
    return last_dir


def save_training_outputs(analysis: tune.ExperimentAnalysis, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    tracked_columns = [
        "training_iteration",
        "timesteps_total",
        "episode_reward_mean",
        "episode_reward_min",
        "episode_reward_max",
        "episodes_total",
        "time_total_s",
    ]

    rows = []

    if analysis.trials:
        trial = analysis.trials[0]
        trial_df = analysis.trial_dataframes.get(trial.logdir)

        if trial_df is not None:
            present_columns = [c for c in tracked_columns if c in trial_df.columns]
            filtered_df = trial_df[present_columns].copy()
            csv_path = output_dir / "training_log.csv"
            filtered_df.to_csv(csv_path, index=False)
            print(f"Saved training CSV: {csv_path}")
        else:
            print("WARNING: No trial dataframe found; CSV history not saved.")
    else:
        print("WARNING: No trials found; CSV history not saved.")

    best_checkpoint = None
    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    if best_trial is not None:
        best_checkpoint = analysis.get_best_checkpoint(
            trial=best_trial,
            metric="episode_reward_mean",
            mode="max",
        )

    if best_checkpoint is not None:
        best_ckpt_path = output_dir / "best_checkpoint.txt"
        best_ckpt_path.write_text(str(best_checkpoint) + "\n", encoding="utf-8")
        print(f"Saved best checkpoint pointer: {best_ckpt_path}")

    final_checkpoint = None
    if analysis.trials:
        training_trial = analysis.trials[0]
        trial_dir = Path(getattr(training_trial, "logdir", getattr(training_trial, "local_path", "")))
        if trial_dir.exists():
            final_checkpoint = _latest_checkpoint_in_trial_dir(trial_dir)

    if final_checkpoint is not None:
        final_ckpt_path = output_dir / "final_checkpoint.txt"
        final_ckpt_path.write_text(str(final_checkpoint) + "\n", encoding="utf-8")
        print(f"Saved final checkpoint pointer: {final_ckpt_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train SoccerTwos PPO baseline in team-vs-random mode (CPU-friendly defaults)."
    )
    parser.add_argument("--timesteps-total", type=int, default=10000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-envs-per-worker", type=int, default=1)
    parser.add_argument("--checkpoint-freq", type=int, default=5)
    parser.add_argument("--experiment-name", type=str, default="team_vs_random_ppo")
    parser.add_argument("--local-dir", type=str, default="./ray_results") 
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True)
    tune.registry.register_env("Soccer", create_rllib_env)

    config = build_config(
        num_workers=args.num_workers,
        num_envs_per_worker=args.num_envs_per_worker,
    )

    analysis = tune.run(
        "PPO",
        name=args.experiment_name,
        config=config,
        stop={"timesteps_total": args.timesteps_total},
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_at_end=True,
        local_dir=args.local_dir,
        verbose=1,
    )

    output_dir = Path(args.local_dir) / args.experiment_name
    save_training_outputs(analysis, output_dir)

    ray.shutdown()


if __name__ == "__main__":
    main()