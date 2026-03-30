import argparse
import math
from pathlib import Path

# import matplotlib.pyplot as plt
import pandas as pd
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from soccer_twos import EnvType

from utils import create_rllib_env


class BaselineSanityCallback(DefaultCallbacks):
    """Minimal callback for readable console progress and sanity checks."""

    def __init__(self):
        super().__init__()
        self._previous_rewards = []

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        iteration = result.get("training_iteration")
        timesteps = result.get("timesteps_total")
        reward_mean = result.get("episode_reward_mean")
        reward_min = result.get("episode_reward_min")
        reward_max = result.get("episode_reward_max")
        learner_info = result.get("info", {}).get("learner", {})

        print(
            f"[iter={iteration:04d}] "
            f"timesteps={timesteps} "
            f"reward_mean={reward_mean} "
            f"reward_min={reward_min} "
            f"reward_max={reward_max}"
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

    df = analysis.results_df.copy()
    tracked_columns = [
        "training_iteration",
        "timesteps_total",
        "episodes_total",
        "time_total_s",
        "episode_reward_mean",
        "episode_reward_min",
        "episode_reward_max",
    ]
    present_columns = [col for col in tracked_columns if col in df.columns]
    training_log = df[present_columns].sort_values("training_iteration")

    csv_path = output_dir / "training_log.csv"
    json_path = output_dir / "training_log.json"
    training_log.to_csv(csv_path, index=False)
    training_log.to_json(json_path, orient="records", indent=2)

    # if "episode_reward_mean" in training_log.columns and "training_iteration" in training_log.columns:
    #     plt.figure(figsize=(8, 5))
    #     plt.plot(training_log["timesteps_total"], training_log["episode_reward_mean"], marker="o")
    #     plt.title("Baseline PPO Reward Curve")
    #     plt.xlabel("Training Steps")
    #     plt.ylabel("Episode reward mean")
    #     plt.grid(True, alpha=0.3)
    #     plt.tight_layout()
    #     plt.savefig(output_dir / "reward_curve.png", dpi=150)
    #     plt.close()

    print(f"Saved training logs: {csv_path}")
    print(f"Saved training logs: {json_path}")
    # print(f"Saved reward curve: {output_dir / 'reward_curve.png'}")


def build_config(num_workers: int, num_envs_per_worker: int):
    return {
        "num_gpus": 0,
        "num_workers": num_workers,
        "num_envs_per_worker": num_envs_per_worker,
        "log_level": "INFO",
        "framework": "torch",
        "callbacks": BaselineSanityCallback,
        "env": "Soccer",
        "env_config": {
            "num_envs_per_worker": num_envs_per_worker,
            "variation": EnvType.team_vs_policy,
            "multiagent": False,
            "single_player": True,
            "flatten_branched": True,
            "opponent_policy": lambda *_: 0,
        },
        "model": {
            "vf_share_layers": True,
            "fcnet_hiddens": [128, 128],
            "fcnet_activation": "relu",
        },
        "lr": 3e-4,
        "gamma": 0.99,
        "lambda": 0.95,
        "clip_param": 0.2,
        "entropy_coeff": 0.01,
        "vf_loss_coeff": 0.5,
        "rollout_fragment_length": 200,
        "train_batch_size": 2000,
        "sgd_minibatch_size": 128,
        "num_sgd_iter": 10,
        "batch_mode": "truncate_episodes",
    }


def main():
    parser = argparse.ArgumentParser(description="Train a minimal SoccerTwos PPO baseline with RLlib.")
    parser.add_argument("--timesteps-total", type=int, default=50000)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-envs-per-worker", type=int, default=1)
    parser.add_argument("--experiment-name", type=str, default="baseline_ppo")
    parser.add_argument("--local-dir", type=str, default="./ray_results")
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True)
    tune.registry.register_env("Soccer", create_rllib_env)

    analysis = tune.run(
        "PPO",
        name=args.experiment_name,
        config=build_config(args.num_workers, args.num_envs_per_worker),
        stop={"timesteps_total": args.timesteps_total},
        checkpoint_freq=3,
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