import numpy as np
from soccer_twos import EnvType
from utils import create_rllib_env


def flatten_player_action(a, b, c):
    return a * 9 + b * 3 + c


def team_action(p0, p1):
    return p0 * 27 + p1


def print_obs2(env, label, step_num=None):
    base = env.env.env
    ds = base._previous_decision_step
    obs2 = np.asarray(ds.obs[2])

    print("\n" + "=" * 100)
    if step_num is None:
        print(label)
    else:
        print(f"{label} | inner_step={step_num}")
    print("=" * 100)
    print("agent_ids:", ds.agent_id)
    print("obs[2] shape:", obs2.shape)

    for i in range(obs2.shape[0]):
        row = obs2[i]
        print(
            f"row {i}: "
            f"px={row[0]: .3f}, py={row[1]: .3f}, rot={row[2]: .3f}, "
            f"vx={row[3]: .3f}, vy={row[4]: .3f}, "
            f"bx={row[5]: .3f}, by={row[6]: .3f}, bvx={row[7]: .3f}, bvy={row[8]: .3f}"
        )


def rollout_case(env, label, action, steps=8):
    env.reset()

    # warmup one random step so state is definitely live
    env.step(env.action_space.sample())
    print_obs2(env, f"{label} | AFTER WARMUP")

    for t in range(1, steps + 1):
        out = env.step(action)
        done = out[2] if len(out) == 4 else (out[2] or out[3])
        print_obs2(env, label, step_num=t)
        if done:
            print("[INFO] Episode ended early, resetting.")
            env.reset()
            env.step(env.action_space.sample())


def main():
    env_config = {
        "variation": EnvType.team_vs_policy,
        "multiagent": False,
        "single_player": False,
        "flatten_branched": True,
        "opponent_policy": lambda *_: 0,
    }

    env = create_rllib_env(env_config)

    # guessed branch actions
    stay = flatten_player_action(1, 1, 1)
    move_a = flatten_player_action(2, 1, 1)
    move_b = flatten_player_action(0, 1, 1)
    turn_a = flatten_player_action(1, 2, 1)
    kick_a = flatten_player_action(1, 1, 2)

    both_still = team_action(stay, stay)
    only_p0_move = team_action(move_a, stay)
    only_p1_move = team_action(stay, move_a)
    only_p0_turn = team_action(turn_a, stay)
    only_p1_turn = team_action(stay, turn_a)
    only_p0_kick = team_action(kick_a, stay)
    only_p1_kick = team_action(stay, kick_a)

    rollout_case(env, "BOTH STILL", both_still, steps=6)
    rollout_case(env, "ONLY P0 MOVE", only_p0_move, steps=6)
    rollout_case(env, "ONLY P1 MOVE", only_p1_move, steps=6)
    rollout_case(env, "ONLY P0 TURN", only_p0_turn, steps=6)
    rollout_case(env, "ONLY P1 TURN", only_p1_turn, steps=6)
    rollout_case(env, "ONLY P0 KICK", only_p0_kick, steps=6)
    rollout_case(env, "ONLY P1 KICK", only_p1_kick, steps=6)


if __name__ == "__main__":
    main()