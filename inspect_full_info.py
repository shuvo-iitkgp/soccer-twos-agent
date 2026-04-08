import pprint
import numpy as np
from soccer_twos import EnvType
from utils import create_rllib_env


def summarize(obj, max_len=8):
    if isinstance(obj, np.ndarray):
        flat = obj.reshape(-1)
        return {
            "type": "ndarray",
            "shape": obj.shape,
            "dtype": str(obj.dtype),
            "preview": flat[:max_len].tolist(),
        }
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[k] = summarize(v, max_len)
        return out
    if isinstance(obj, (list, tuple)):
        return [summarize(x, max_len) for x in obj[:max_len]]
    return repr(obj)


def inspect_obj(name, obj):
    print("\n" + "=" * 100)
    print(name)
    print("=" * 100)
    print("type:", type(obj))

    if hasattr(obj, "__dict__"):
        print("\nATTRIBUTES:")
        for k in sorted(obj.__dict__.keys()):
            v = getattr(obj, k)
            print(f"  {k}: {type(v)}")

    interesting = [
        "players", "player", "state", "states", "agents", "agent",
        "team", "teams", "last_obs", "obs", "info", "behavior_specs",
        "decision_steps", "terminal_steps"
    ]

    if hasattr(obj, "__dict__"):
        print("\nPOSSIBLY INTERESTING VALUES:")
        for k in sorted(obj.__dict__.keys()):
            lowered = k.lower()
            if any(tok in lowered for tok in interesting):
                try:
                    v = getattr(obj, k)
                    print(f"\n--- {k} ---")
                    pprint.pprint(summarize(v), width=140, sort_dicts=False)
                except Exception as e:
                    print(f"{k}: <error reading: {e}>")


def main():
    env_config = {
        "variation": EnvType.team_vs_policy,
        "multiagent": False,
        "single_player": False,
        "flatten_branched": True,
        "opponent_policy": lambda *_: 0,
    }

    env = create_rllib_env(env_config)

    out = env.reset()
    if isinstance(out, tuple):
        obs, info = out
    else:
        obs, info = out, {}

    action = env.action_space.sample()
    out = env.step(action)

    # Walk down nested env chain
    current = env
    depth = 0
    seen = set()

    while current is not None and id(current) not in seen and depth < 10:
        seen.add(id(current))
        inspect_obj(f"ENV LAYER {depth}", current)
        current = getattr(current, "env", None)
        depth += 1


if __name__ == "__main__":
    main()