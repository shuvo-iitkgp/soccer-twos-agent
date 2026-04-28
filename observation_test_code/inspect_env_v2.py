import numpy as np
from soccer_twos import EnvType
from utils import create_rllib_env


def top_changed_indices(a, b, k=30):
    diff = np.abs(np.asarray(a) - np.asarray(b))
    idx = np.argsort(-diff)[:k]
    return [(int(i), float(a[i]), float(b[i]), float(diff[i])) for i in idx if diff[i] > 1e-6]


def print_changes(title, changes):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    if not changes:
        print("No meaningful changes.")
        return
    for i, old, new, d in changes:
        print(f"idx={i:3d} | old={old: .6f} | new={new: .6f} | absdiff={d: .6f}")


def main():
    env_config = {
        "variation": EnvType.team_vs_policy,
        "multiagent": False,
        "single_player": False,
        "flatten_branched": True,
        "opponent_policy": lambda *_: 0,
    }

    env = create_rllib_env(env_config)
    env.reset()

    obs0_before = env.last_obs[0].copy()
    obs1_before = env.last_obs[1].copy()
    obs2_before = env.last_obs[2].copy()
    obs3_before = env.last_obs[3].copy()

    print(f"obs0 shape = {obs0_before.shape}")
    print(f"obs1 shape = {obs1_before.shape}")
    print(f"obs2 shape = {obs2_before.shape}")
    print(f"obs3 shape = {obs3_before.shape}")

    # take a few random steps
    for t in range(5):
        a = env.action_space.sample()
        out = env.step(a)
        done = out[2] if len(out) == 4 else (out[2] or out[3])
        if done:
            env.reset()

    obs0_after = env.last_obs[0].copy()
    obs1_after = env.last_obs[1].copy()
    obs2_after = env.last_obs[2].copy()
    obs3_after = env.last_obs[3].copy()

    print_changes("Changes in agent 0 obs", top_changed_indices(obs0_before, obs0_after, k=40))
    print_changes("Changes in agent 1 obs", top_changed_indices(obs1_before, obs1_after, k=40))
    print_changes("Changes in agent 2 obs", top_changed_indices(obs2_before, obs2_after, k=40))
    print_changes("Changes in agent 3 obs", top_changed_indices(obs3_before, obs3_after, k=40))


if __name__ == "__main__":
    main()