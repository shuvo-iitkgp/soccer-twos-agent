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
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "PPO_agent/exp3_2m_refreshed/PPO_SoccerRewardShaped_5dbb1_00000_0_2026-03-28_14-46-01/checkpoint_000500/checkpoint-500",
)

POLICY_NAME = "default_policy"

# Training-time spaces inferred from checkpoint mismatch:
# obs: 672 = 2 * 336
# action: 729 = 3^6 = joint action for 2 players, each with 3 branches
TEAM_OBS_DIM = 672
JOINT_ACTION_SIZE = 729
PER_PLAYER_ACTION_DIM = 3
PLAYERS_PER_TEAM = 2


class SpaceOnlyEnv(gym.Env):
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
            ray.init(ignore_reinit_error=True)

        config_dir = os.path.dirname(CHECKPOINT_PATH)
        config_path = os.path.join(config_dir, "params.pkl")
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, "../params.pkl")

        if not os.path.exists(config_path):
            raise ValueError("Could not find params.pkl")

        with open(config_path, "rb") as f:
            config = pickle.load(f)

        config["num_workers"] = 0
        config["num_gpus"] = 0
        config["explore"] = False

        obs_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(TEAM_OBS_DIM,),
            dtype=np.float32,
        )
        act_space = gym.spaces.Discrete(JOINT_ACTION_SIZE)

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

        cls = get_trainable_cls(ALGORITHM)
        self.agent = cls(config=config)

        self._restore_policy_only(CHECKPOINT_PATH)
        self.policy = self.agent.get_policy(POLICY_NAME)

    def _remove_optimizer_state(self, obj):
        if isinstance(obj, dict):
            keys_to_delete = []
            for k, v in obj.items():
                if "optimizer" in str(k).lower():
                    keys_to_delete.append(k)
                else:
                    self._remove_optimizer_state(v)
            for k in keys_to_delete:
                del obj[k]
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

        if isinstance(worker_blob, bytes):
            worker_state = pickle.loads(worker_blob)
        else:
            worker_state = worker_blob

        self._remove_optimizer_state(worker_state["state"])

        if isinstance(worker_blob, bytes):
            self.agent.workers.local_worker().restore(pickle.dumps(worker_state))
        else:
            self.agent.workers.local_worker().restore(worker_state)

    def _build_team_observation(
        self,
        observation: Dict[int, np.ndarray],
        player_ids: List[int],
    ) -> np.ndarray:
        """
        Build the 672-dim team observation expected by the checkpoint by
        concatenating two 336-dim player observations.
        """
        team_obs = []
        for pid in player_ids[::-1]:
            if pid not in observation:
                raise KeyError(f"Missing observation for player id {pid}")
            team_obs.append(np.asarray(observation[pid], dtype=np.float32).ravel())

        team_obs = np.concatenate(team_obs, axis=0).astype(np.float32)

        if team_obs.shape[0] != TEAM_OBS_DIM:
            raise ValueError(
                f"Expected concatenated team obs dim {TEAM_OBS_DIM}, got {team_obs.shape[0]}"
            )

        return team_obs

    def _decode_joint_action(self, action: int) -> Dict[int, np.ndarray]:
        """
        Decode a flattened Discrete(729) action into two 3-branch player actions.

        729 = 3^6, so we interpret the integer in base-3 and split it into:
        player A -> first 3 branches
        player B -> next 3 branches
        """
        if not (0 <= int(action) < JOINT_ACTION_SIZE):
            raise ValueError(f"Joint action out of range: {action}")

        digits = [0] * 6
        value = int(action)
        for i in range(5, -1, -1):
            digits[i] = value % 3
            value //= 3

        player_0_action = np.array(digits[:3], dtype=np.int64)
        player_1_action = np.array(digits[3:], dtype=np.int64)

        return {
            0: player_0_action,
            1: player_1_action,
        }

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Build team-level observations for (0,1) and (2,3), run the policy once per team,
        decode the joint discrete action back into per-player branched actions, and return
        actions for all visible player ids.
        """
        actions: Dict[int, np.ndarray] = {}

        team_pairs = [
            [0, 1],
            [2, 3],
        ]

        visible_ids = set(observation.keys())

        for pair in team_pairs:
            if not all(pid in visible_ids for pid in pair):
                continue

            team_obs = self._build_team_observation(observation, pair)
            joint_action, *_ = self.policy.compute_single_action(
                team_obs,
                explore=False,
            )

            decoded = self._decode_joint_action(joint_action)

            # actions[pair[0]] = decoded[0]
            # actions[pair[1]] = decoded[1]

            actions[pair[0]] = decoded[1]
            actions[pair[1]] = decoded[0]

        return actions