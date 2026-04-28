import numpy as np
import pprint
from soccer_twos import EnvType
from utils import create_rllib_env


def summarize(obj, max_len=10):
    if isinstance(obj, np.ndarray):
        flat = obj.reshape(-1)
        return {
            "shape": obj.shape,
            "dtype": str(obj.dtype),
            "preview": flat[:max_len].tolist()
        }
    if isinstance(obj, dict):
        return {k: summarize(v, max_len) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [summarize(x, max_len) for x in obj[:max_len]]
    return obj


def print_block(title, obj):
    print("\n" + "="*80)
    print(title)
    print("="*80)
    pprint.pprint(summarize(obj), width=120)
    print("="*80)


def main():
    env_config = {
        "variation": EnvType.team_vs_policy,
        "multiagent": False,
        "single_player": False,
        "flatten_branched": True,
        "opponent_policy": lambda *_: 0,
    }

    print("[INFO] Creating raw env...")
    env = create_rllib_env(env_config)

    print("[INFO] Observation space:", env.observation_space)
    print("[INFO] Action space:", env.action_space)

    # RESET
    print("\n[STEP] RESET")
    out = env.reset()

    if isinstance(out, tuple):
        obs, info = out
    else:
        obs, info = out, None

    print_block("RAW OBS (reset)", obs)
    print_block("RAW INFO (reset)", info)

    # STEP
    print("\n[STEP] ONE RANDOM STEP")
    try:
        action = env.action_space.sample()
    except:
        action = 0

    print_block("ACTION SENT", action)

    out = env.step(action)

    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = terminated or truncated
    else:
        obs, reward, done, info = out

    print_block("RAW OBS (step)", obs)
    print_block("REWARD", reward)
    print_block("DONE", done)
    print_block("RAW INFO (step)", info)

    # IMPORTANT: check hidden state
    if hasattr(env, "last_obs"):
        print_block("ENV.last_obs (VERY IMPORTANT)", env.last_obs)

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()