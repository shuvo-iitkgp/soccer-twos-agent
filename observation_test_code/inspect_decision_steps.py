import numpy as np
import pprint
from soccer_twos import EnvType
from utils import create_rllib_env


def summarize_array(x, max_len=12):
    arr = np.asarray(x)
    flat = arr.reshape(-1)
    return {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "preview": flat[:max_len].tolist(),
    }


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

    # Layer 0 = EnvChannelWrapper
    # Layer 1 = TeamVsPolicyWrapper
    # Layer 2 = MultiAgentUnityWrapper
    base = env.env.env

    ds = getattr(base, "_previous_decision_step", None)

    print("\n" + "=" * 100)
    print("TYPE OF _previous_decision_step")
    print("=" * 100)
    print(type(ds))

    if ds is None:
        print("No _previous_decision_step found.")
        return

    print("\n" + "=" * 100)
    print("ATTRIBUTES OF _previous_decision_step")
    print("=" * 100)
    for k in sorted(ds.__dict__.keys()):
        v = getattr(ds, k)
        print(f"{k}: {type(v)}")

    print("\n" + "=" * 100)
    print("AGENT IDS")
    print("=" * 100)
    try:
        print(ds.agent_id)
        print("agent_id shape:", np.asarray(ds.agent_id).shape)
    except Exception as e:
        print("Could not read agent_id:", e)

    print("\n" + "=" * 100)
    print("REWARD")
    print("=" * 100)
    try:
        print(summarize_array(ds.reward))
    except Exception as e:
        print("Could not read reward:", e)

    print("\n" + "=" * 100)
    print("OBS")
    print("=" * 100)
    try:
        obs_list = ds.obs
        print("number of obs tensors:", len(obs_list))
        for i, o in enumerate(obs_list):
            print(f"\nobs[{i}] summary:")
            pprint.pprint(summarize_array(o), width=120, sort_dicts=False)
    except Exception as e:
        print("Could not read obs:", e)

    print("\n" + "=" * 100)
    print("INDIVIDUAL AGENT ROWS FROM obs[0]")
    print("=" * 100)
    try:
        obs0 = np.asarray(ds.obs[0])
        print("obs[0] full shape:", obs0.shape)
        for i in range(min(obs0.shape[0], 10)):
            row = obs0[i]
            print(f"agent_row {i}: shape={row.shape}, first12={row[:12].tolist()}")
    except Exception as e:
        print("Could not inspect per-agent obs rows:", e)


if __name__ == "__main__":
    main()