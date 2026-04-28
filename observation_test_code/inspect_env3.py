import numpy as np
from soccer_twos import EnvType
from utils import create_rllib_env


def top_changed_indices(a, b, k=25, eps=1e-6):
    a = np.asarray(a)
    b = np.asarray(b)
    diff = np.abs(a - b)
    idx = np.argsort(-diff)
    out = []
    for i in idx:
        if diff[i] <= eps:
            break
        out.append((int(i), float(a[i]), float(b[i]), float(diff[i])))
        if len(out) >= k:
            break
    return out


def print_changes(title, changes):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    if not changes:
        print("No meaningful changes.")
        return
    for i, old, new, d in changes:
        print(f"idx={i:3d} | old={old: .6f} | new={new: .6f} | absdiff={d: .6f}")


def action_from_branches(a, b, c):
    # flatten branched action: each in {0,1,2}
    return a * 9 + b * 3 + c


def get_team_action(p0, p1):
    # returned raw env action for the controlled team of 2 players
    return p0 * 27 + p1


def capture_last_obs(env):
    return {
        0: env.last_obs[0].copy(),
        1: env.last_obs[1].copy(),
        2: env.last_obs[2].copy(),
        3: env.last_obs[3].copy(),
    }


def run_case(env, title, action, steps=1):
    env.reset()
    before = capture_last_obs(env)

    for _ in range(steps):
        out = env.step(action)
        done = out[2] if len(out) == 4 else (out[2] or out[3])
        if done:
            env.reset()

    after = capture_last_obs(env)

    print("\n" + "#" * 120)
    print(title)
    print("#" * 120)

    for agent_id in [0, 1, 2, 3]:
        changes = top_changed_indices(before[agent_id], after[agent_id], k=20)
        print_changes(f"Changes in obs of agent {agent_id}", changes)


def main():
    env_config = {
        "variation": EnvType.team_vs_policy,
        "multiagent": False,
        "single_player": False,
        "flatten_branched": True,
        "opponent_policy": lambda *_: 0,   # opponents stand still
    }

    env = create_rllib_env(env_config)
    env.reset()

    print("obs shapes:")
    for i in range(4):
        print(i, env.last_obs[i].shape)

    # Guessing branch semantics from SoccerTwos typical setup:
    # each player action has 3 branches in {0,1,2}
    #
    # We do several one-step perturbations:
    # - both stay still
    # - only player 0 moves
    # - only player 1 moves
    #
    # Even if exact semantics of branches are imperfect,
    # differential comparison is still informative.

    stay = action_from_branches(1, 1, 1)
    move_a = action_from_branches(2, 1, 1)
    move_b = action_from_branches(0, 1, 1)
    turn_a = action_from_branches(1, 2, 1)
    kick_a = action_from_branches(1, 1, 2)

    both_still = get_team_action(stay, stay)
    only_p0_move = get_team_action(move_a, stay)
    only_p1_move = get_team_action(stay, move_a)
    only_p0_turn = get_team_action(turn_a, stay)
    only_p1_turn = get_team_action(stay, turn_a)
    only_p0_kick = get_team_action(kick_a, stay)
    only_p1_kick = get_team_action(stay, kick_a)

    run_case(env, "CASE 1: both players still", both_still, steps=1)
    run_case(env, "CASE 2: only player 0 acts", only_p0_move, steps=1)
    run_case(env, "CASE 3: only player 1 acts", only_p1_move, steps=1)
    run_case(env, "CASE 4: only player 0 turn-like action", only_p0_turn, steps=1)
    run_case(env, "CASE 5: only player 1 turn-like action", only_p1_turn, steps=1)
    run_case(env, "CASE 6: only player 0 kick-like action", only_p0_kick, steps=1)
    run_case(env, "CASE 7: only player 1 kick-like action", only_p1_kick, steps=1)


if __name__ == "__main__":
    main()