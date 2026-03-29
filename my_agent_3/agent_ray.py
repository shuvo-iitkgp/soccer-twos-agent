import os
import pickle
from typing import Dict, List

import gym
import numpy as np
import ray
from ray import tune
from ray.tune.registry import get_trainable_cls

from soccer_twos import AgentInterface


ALGORITHM = "PPO"
POLICY_NAME = "default_policy"

# Update this path before submission if you move the checkpoint.
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ray_results/flatten_false_experiment_3/PPO_SoccerRewardShaped_046ed_00000_0_2026-03-29_03-36-37/checkpoint_000474/checkpoint-474",
)

TEAM_OBS_DIM = 672
PER_PLAYER_OBS_DIM = 336
PLAYERS_PER_TEAM = 2
NUM_ACTION_BRANCHES_PER_PLAYER = 3
NUM_ACTIONS_PER_BRANCH = 3
JOINT_ACTION_SIZE = NUM_ACTIONS_PER_BRANCH ** (PLAYERS_PER_TEAM * NUM_ACTION_BRANCHES_PER_PLAYER)


class SpaceOnlyEnv(gym.Env):
    """Dummy env used only so RLlib can rebuild the policy for inference."""

    def __init__(self, env_config):
        super().__init__()
        self.observation_space = env_config["observation_space"]
        self.action_space = env_config["action_space"]

    def reset(self):
        raise NotImplementedError("SpaceOnlyEnv is only for restoring policy weights.")

    def step(self, action):
        raise NotImplementedError("SpaceOnlyEnv is only for restoring policy weights.")


class RayAgent(AgentInterface):
    def __init__(self, env: gym.Env):
        super().__init__()

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, include_dashboard=False)

        config = self._load_config(CHECKPOINT_PATH)
        config["num_workers"] = 0
        config["num_gpus"] = 0
        config["explore"] = False

        # Training was done with flatten_branched=False.
        # During packaging we rebuild a compatible inference-only policy shell.
        obs_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(TEAM_OBS_DIM,),
            dtype=np.float32,
        )
        act_space = gym.spaces.MultiDiscrete([NUM_ACTIONS_PER_BRANCH] * 6)

        try:
            tune.registry.register_env(
                "SpaceOnlyEnv",
                lambda env_config: SpaceOnlyEnv(env_config),
            )
        except Exception:
            pass

        config["env"] = "SpaceOnlyEnv"
        config["env_config"] = {
            "observation_space": obs_space,
            "action_space": act_space,
        }

        trainer_cls = get_trainable_cls(ALGORITHM)
        self.agent = trainer_cls(config=config)
        self._restore_policy_only(CHECKPOINT_PATH)
        self.policy = self.agent.get_policy(POLICY_NAME)

    def _load_config(self, checkpoint_path: str) -> dict:
        config_dir = os.path.dirname(checkpoint_path)
        config_path = os.path.join(config_dir, "params.pkl")
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, "../params.pkl")
        if not os.path.exists(config_path):
            raise ValueError("Could not find params.pkl next to the checkpoint.")

        with open(config_path, "rb") as f:
            return pickle.load(f)

    def _remove_optimizer_state(self, obj):
        if isinstance(obj, dict):
            keys_to_delete = []
            for key, value in obj.items():
                if "optimizer" in str(key).lower():
                    keys_to_delete.append(key)
                else:
                    self._remove_optimizer_state(value)
            for key in keys_to_delete:
                del obj[key]
        elif isinstance(obj, list):
            for item in obj:
                self._remove_optimizer_state(item)
        elif isinstance(obj, tuple):
            for item in obj:
                self._remove_optimizer_state(item)

    def _restore_policy_only(self, checkpoint_path: str):
        with open(checkpoint_path, "rb") as f:
            trainer_state = pickle.load(f)

        worker_blob = trainer_state["worker"]
        worker_state = pickle.loads(worker_blob) if isinstance(worker_blob, bytes) else worker_blob
        self._remove_optimizer_state(worker_state["state"])

        if isinstance(worker_blob, bytes):
            self.agent.workers.local_worker().restore(pickle.dumps(worker_state))
        else:
            self.agent.workers.local_worker().restore(worker_state)

    def _build_team_observation(self, observation: Dict[int, np.ndarray], player_ids: List[int]) -> np.ndarray:
        parts = []
        # Reverse order preserved because this matched your successful evaluation setup.
        for pid in player_ids[::-1]:
            if pid not in observation:
                raise KeyError(f"Missing observation for player id {pid}")
            part = np.asarray(observation[pid], dtype=np.float32).ravel()
            if part.shape[0] != PER_PLAYER_OBS_DIM:
                raise ValueError(
                    f"Expected per-player obs dim {PER_PLAYER_OBS_DIM}, got {part.shape[0]} for player {pid}"
                )
            parts.append(part)

        team_obs = np.concatenate(parts, axis=0).astype(np.float32)
        if team_obs.shape[0] != TEAM_OBS_DIM:
            raise ValueError(f"Expected team obs dim {TEAM_OBS_DIM}, got {team_obs.shape[0]}")
        return team_obs

    def _decode_joint_discrete_action(self, action: int) -> np.ndarray:
        if not (0 <= int(action) < JOINT_ACTION_SIZE):
            raise ValueError(f"Joint action out of range: {action}")

        digits = [0] * 6
        value = int(action)
        for i in range(5, -1, -1):
            digits[i] = value % NUM_ACTIONS_PER_BRANCH
            value //= NUM_ACTIONS_PER_BRANCH
        return np.asarray(digits, dtype=np.int64)

    def _normalize_joint_action(self, action) -> np.ndarray:
        # Some restores produce a flattened discrete id. Others produce a 6-branch action.
        if np.isscalar(action):
            return self._decode_joint_discrete_action(int(action))

        joint = np.asarray(action, dtype=np.int64).reshape(-1)
        if joint.shape[0] != 6:
            raise ValueError(f"Expected 6 joint action branches, got shape {joint.shape}")
        return joint

    def _split_joint_action(self, joint_action: np.ndarray) -> Dict[int, np.ndarray]:
        return {
            0: joint_action[:3].astype(np.int64),
            1: joint_action[3:].astype(np.int64),
        }

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions: Dict[int, np.ndarray] = {}
        visible_ids = sorted(observation.keys())

        # Usually only one team is visible during evaluation. If more are visible,
        # handle them in sorted pairs: [0,1], [2,3].
        for i in range(0, len(visible_ids), PLAYERS_PER_TEAM):
            pair = visible_ids[i : i + PLAYERS_PER_TEAM]
            if len(pair) != PLAYERS_PER_TEAM:
                continue

            team_obs = self._build_team_observation(observation, pair)
            raw_action, *_ = self.policy.compute_single_action(team_obs, explore=False)
            joint_action = self._normalize_joint_action(raw_action)
            decoded = self._split_joint_action(joint_action)

            # Keep the swap because this matched your prior watch-game results.
            actions[pair[0]] = decoded[1]
            actions[pair[1]] = decoded[0]

        return actions
