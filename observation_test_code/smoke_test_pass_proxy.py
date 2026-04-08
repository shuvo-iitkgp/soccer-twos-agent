import numpy as np
from soccer_twos import EnvType
from utils import create_rllib_env

# import from your training file
from test_two_phase_pass_reward_curriculum import CurriculumRewardWrapper, _unwrap_info, _safe_float


def flatten_player_action(a, b, c):
    return a * 9 + b * 3 + c


def team_action(p0, p1):
    return p0 * 27 + p1


def random_team_action():
    p0 = flatten_player_action(
        np.random.randint(0, 3),
        np.random.randint(0, 3),
        np.random.randint(0, 3),
    )
    p1 = flatten_player_action(
        np.random.randint(0, 3),
        np.random.randint(0, 3),
        np.random.randint(0, 3),
    )
    return team_action(p0, p1)


def build_env(stage=1):
    env_config = {
        "variation": EnvType.team_vs_policy,
        "multiagent": False,
        "single_player": False,
        "flatten_branched": True,
        "opponent_policy": lambda *_: 0,
        "stage": stage,
    }
    base_env = create_rllib_env(env_config)
    return CurriculumRewardWrapper(base_env, stage=stage)


def print_event(step_idx, info):
    print(
        f"[step {step_idx:03d}] "
        f"pass_reward={_safe_float(info.get('pass_reward', 0.0)):.3f} | "
        f"attempt={_safe_float(info.get('pass_attempt_count', 0.0)):.0f} | "
        f"complete={_safe_float(info.get('pass_complete_count', 0.0)):.0f} | "
        f"fail={_safe_float(info.get('pass_fail_count', 0.0)):.0f} | "
        f"travel={_safe_float(info.get('pass_travel', 0.0)):.3f} | "
        f"ball_touch={_safe_float(info.get('ball_touch_reward', 0.0)):.3f} | "
        f"shaping={_safe_float(info.get('shaping_reward', 0.0)):.3f}"
    )


def main():
    np.random.seed(0)

    # test stage 1 first
    env = build_env(stage=1)

    out = env.reset()
    if isinstance(out, tuple):
        obs, info = out
    else:
        obs, info = out, {}

    total_attempts = 0.0
    total_completes = 0.0
    total_fails = 0.0
    total_pass_reward = 0.0
    nonzero_steps = 0

    STEPS = 400

    print("[INFO] Starting smoke test for pass proxy")
    print("[INFO] Stage = 1")
    print("[INFO] Opponent = random/zero policy through wrapper")
    print()

    for step_idx in range(STEPS):
        action = random_team_action()
        out = env.step(action)

        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = out

        info_dict = _unwrap_info(info)

        attempt = _safe_float(info_dict.get("pass_attempt_count", 0.0))
        complete = _safe_float(info_dict.get("pass_complete_count", 0.0))
        fail = _safe_float(info_dict.get("pass_fail_count", 0.0))
        pass_reward = _safe_float(info_dict.get("pass_reward", 0.0))

        total_attempts += attempt
        total_completes += complete
        total_fails += fail
        total_pass_reward += pass_reward

        if attempt > 0 or complete > 0 or fail > 0 or pass_reward > 0:
            nonzero_steps += 1
            print_event(step_idx, info_dict)

        if done:
            print(f"[INFO] Episode ended at step {step_idx}, resetting.")
            out = env.reset()
            if isinstance(out, tuple):
                obs, info = out
            else:
                obs, info = out, {}

    print("\n" + "=" * 80)
    print("SMOKE TEST SUMMARY")
    print("=" * 80)
    print(f"Total pass attempts   : {total_attempts:.0f}")
    print(f"Total pass completes  : {total_completes:.0f}")
    print(f"Total pass fails      : {total_fails:.0f}")
    print(f"Total pass reward     : {total_pass_reward:.3f}")
    print(f"Nonzero proxy steps   : {nonzero_steps}")
    print("=" * 80)

    # stage 2 quick check too
    print("\n[INFO] Switching to stage 2 quick smoke check\n")
    env.set_stage(2)

    total_attempts_2 = 0.0
    total_completes_2 = 0.0
    total_fails_2 = 0.0
    total_pass_reward_2 = 0.0

    out = env.reset()
    if isinstance(out, tuple):
        obs, info = out
    else:
        obs, info = out, {}

    for step_idx in range(200):
        action = random_team_action()
        out = env.step(action)

        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = out

        info_dict = _unwrap_info(info)

        total_attempts_2 += _safe_float(info_dict.get("pass_attempt_count", 0.0))
        total_completes_2 += _safe_float(info_dict.get("pass_complete_count", 0.0))
        total_fails_2 += _safe_float(info_dict.get("pass_fail_count", 0.0))
        total_pass_reward_2 += _safe_float(info_dict.get("pass_reward", 0.0))

        if done:
            out = env.reset()
            if isinstance(out, tuple):
                obs, info = out
            else:
                obs, info = out, {}

    print("\n" + "=" * 80)
    print("STAGE 2 QUICK SUMMARY")
    print("=" * 80)
    print(f"Total pass attempts   : {total_attempts_2:.0f}")
    print(f"Total pass completes  : {total_completes_2:.0f}")
    print(f"Total pass fails      : {total_fails_2:.0f}")
    print(f"Total pass reward     : {total_pass_reward_2:.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()