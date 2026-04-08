import numpy as np
from soccer_twos import EnvType
from utils import create_rllib_env


POSSESSION_DIST = 3.0
PASS_MIN_TRAVEL = 0.3
PASS_MIN_BALL_SPEED = 0.05
PASS_MAX_STEPS = 40
EPISODE_STEPS = 500
DEBUG_OWNER_EVERY = 10


def vec2(x):
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.shape[0] < 2:
        return np.zeros(2, dtype=np.float32)
    return arr[:2].copy()


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


def extract_ball_info(info):
    if not isinstance(info, dict):
        return None, None

    ball_info = info.get("ball_info", {})
    ball_pos = vec2(ball_info.get("position", [0.0, 0.0]))
    ball_vel = vec2(ball_info.get("velocity", [0.0, 0.0]))
    return ball_pos, ball_vel


def collect_all_player_positions(obj):
    """
    Recursively collect position fields that look like PLAYER positions.
    Explicitly ignores ball_info.position.
    """
    found = []

    def walk(x, path="root"):
        if isinstance(x, dict):
            # Only accept positions that belong to player-like dicts, not ball_info
            if "position" in x and "velocity" in x:
                if "rotation_y" in x or "touched_ball" in x or "player" in path.lower():
                    pos = x.get("position")
                    arr = np.asarray(pos, dtype=np.float32).reshape(-1) if pos is not None else None
                    if arr is not None and arr.shape[0] >= 2:
                        found.append(arr[:2].copy())

            for k, v in x.items():
                next_path = f"{path}.{k}"
                # hard skip ball_info.position subtree as a player source
                if str(k).lower() == "ball_info":
                    continue
                walk(v, next_path)

        elif isinstance(x, (list, tuple)):
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")

    walk(obj)
    return found


def unique_positions(positions, tol=1e-4):
    uniq = []
    for p in positions:
        keep = True
        for q in uniq:
            if float(np.linalg.norm(p - q)) < tol:
                keep = False
                break
        if keep:
            uniq.append(p)
    return uniq


def pick_two_teammates(player_positions, ball_pos):
    """
    Heuristic only.
    Pick two non-ball player positions closest to the ball.
    """
    if len(player_positions) < 2:
        return None

    cleaned = []
    for p in player_positions:
        # reject positions that are basically the ball
        if float(np.linalg.norm(p - ball_pos)) < 1e-4:
            continue
        cleaned.append(p)

    cleaned = unique_positions(cleaned)

    if len(cleaned) < 2:
        return None

    dists = [(float(np.linalg.norm(p - ball_pos)), p) for p in cleaned]
    dists.sort(key=lambda x: x[0])

    return dists[0][1], dists[1][1]


def try_extract_two_teammate_positions(env, info, ball_pos, verbose=False):
    """
    Try likely sources of richer player state.
    """
    candidates = []

    for attr in [
        "player_infos",
        "players_info",
        "all_player_info",
        "all_players_info",
        "raw_player_info",
        "latest_player_info",
        "player_states",
        "players",
    ]:
        if hasattr(env, attr):
            candidates.append((attr, getattr(env, attr)))

    candidates.append(("step_info", info))

    for source_name, source in candidates:
        positions = collect_all_player_positions(source)
        positions = unique_positions(positions)

        if verbose:
            print(f"[DEBUG] source={source_name} | raw_player_positions={positions}")

        picked = pick_two_teammates(positions, ball_pos)
        if picked is not None:
            p0, p1 = picked
            return p0, p1, source_name, positions

    return None, None, None, []


class PassTracker:
    def __init__(self):
        self.last_owner = None
        self.pending_pass = None
        self.total_attempts = 0
        self.total_successes = 0

    def reset(self):
        self.last_owner = None
        self.pending_pass = None

    def update(self, p0_pos, p1_pos, ball_pos, ball_vel, step_idx):
        d0 = float(np.linalg.norm(p0_pos - ball_pos))
        d1 = float(np.linalg.norm(p1_pos - ball_pos))
        ball_speed = float(np.linalg.norm(ball_vel))

        owner = None
        if d0 <= POSSESSION_DIST and d0 < d1:
            owner = 0
        elif d1 <= POSSESSION_DIST and d1 < d0:
            owner = 1

        pass_reward = 0.0
        event = None

        # Start pass attempt when previous owner loses possession and ball leaves
        if self.last_owner is not None and owner is None:
            print(
                f"[DEBUG] potential release | step={step_idx} | "
                f"last_owner={self.last_owner} | speed={ball_speed:.3f} | d0={d0:.3f} | d1={d1:.3f}"
            )

            if ball_speed >= PASS_MIN_BALL_SPEED and self.pending_pass is None:
                passer = self.last_owner
                passer_pos = p0_pos if passer == 0 else p1_pos
                self.pending_pass = {
                    "passer": passer,
                    "release_ball_pos": ball_pos.copy(),
                    "release_passer_pos": passer_pos.copy(),
                    "steps": 0,
                    "start_step": step_idx,
                }
                self.total_attempts += 1
                event = (
                    f"[step {step_idx}] PASS ATTEMPT by player {passer} | "
                    f"ball_speed={ball_speed:.3f}"
                )

        if self.pending_pass is not None:
            self.pending_pass["steps"] += 1
            print(
                f"[DEBUG] tracking pass | step={step_idx} | passer={self.pending_pass['passer']} | "
                f"age={self.pending_pass['steps']} | owner={owner}"
            )

            if self.pending_pass["steps"] > PASS_MAX_STEPS:
                event = (
                    f"[step {step_idx}] PASS FAILED/EXPIRED | "
                    f"passer={self.pending_pass['passer']} | age={self.pending_pass['steps']}"
                )
                self.pending_pass = None
            else:
                passer = self.pending_pass["passer"]
                receiver = owner

                if receiver is not None and receiver != passer:
                    travel = float(np.linalg.norm(ball_pos - self.pending_pass["release_ball_pos"]))
                    if travel >= PASS_MIN_TRAVEL:
                        pass_reward = 0.08
                        self.total_successes += 1
                        event = (
                            f"[step {step_idx}] PASS SUCCESS "
                            f"{passer} -> {receiver} | travel={travel:.3f} | reward={pass_reward:.3f}"
                        )
                    else:
                        event = (
                            f"[step {step_idx}] TRANSFER TOO SHORT "
                            f"{passer} -> {receiver} | travel={travel:.3f}"
                        )
                    self.pending_pass = None

        self.last_owner = owner
        return pass_reward, event, owner, d0, d1, ball_speed


def main():
    env_config = {
        "variation": EnvType.team_vs_policy,
        "multiagent": False,
        "single_player": False,
        "flatten_branched": True,
        "opponent_policy": lambda *_: 0,
    }

    env = create_rllib_env(env_config)
    tracker = PassTracker()

    out = env.reset()
    if isinstance(out, tuple):
        obs, info = out
    else:
        obs, info = out, {}

    tracker.reset()

    print("[INFO] Starting pass-reward smoke test")
    print("[INFO] Opponent policy is zero-action")
    print("[INFO] Random actions for controlled team")
    print()

    found_positions_once = False
    source_used_once = None

    for step_idx in range(EPISODE_STEPS):
        action = random_team_action()
        out = env.step(action)

        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = out

        ball_pos, ball_vel = extract_ball_info(info)

        if ball_pos is None or ball_vel is None:
            print(f"[step {step_idx}] Could not read ball_info from info. Stopping.")
            return

        verbose_extract = step_idx < 5
        p0_pos, p1_pos, source_name, all_positions = try_extract_two_teammate_positions(
            env, info, ball_pos, verbose=verbose_extract
        )

        if p0_pos is None or p1_pos is None:
            print("\n[FAIL] Could not find two non-ball player positions in env/info.")
            print(f"[FAIL] ball_pos={ball_pos}")
            print(f"[FAIL] candidate player positions={all_positions}")
            return

        if not found_positions_once:
            found_positions_once = True
            source_used_once = source_name
            print(f"[INFO] Using teammate positions from source: {source_name}")
            print(f"[INFO] Example p0_pos = {p0_pos}")
            print(f"[INFO] Example p1_pos = {p1_pos}")
            print(f"[INFO] Example ball_pos = {ball_pos}")
            print(f"[INFO] Candidate player positions = {all_positions}")
            print()

        pass_reward, event, owner, d0, d1, ball_speed = tracker.update(
            p0_pos=p0_pos,
            p1_pos=p1_pos,
            ball_pos=ball_pos,
            ball_vel=ball_vel,
            step_idx=step_idx,
        )

        if step_idx % DEBUG_OWNER_EVERY == 0:
            print(
                f"[DEBUG] step={step_idx} | owner={owner} | d0={d0:.3f} | d1={d1:.3f} | "
                f"ball_speed={ball_speed:.3f} | ball_pos={ball_pos}"
            )

        if event is not None:
            print(event)
            print(
                f"         owner={owner} | d0={d0:.3f} | d1={d1:.3f} | "
                f"ball_speed={ball_speed:.3f} | ball_pos={ball_pos}"
            )

        if done:
            print(f"\n[INFO] Episode ended at step {step_idx}. Resetting.\n")
            out = env.reset()
            if isinstance(out, tuple):
                obs, info = out
            else:
                obs, info = out, {}
            tracker.reset()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Source used                    : {source_used_once}")
    print(f"Total pass attempts detected   : {tracker.total_attempts}")
    print(f"Total successful passes        : {tracker.total_successes}")
    if tracker.total_attempts > 0:
        print(f"Pass success rate              : {tracker.total_successes / tracker.total_attempts:.3f}")
    else:
        print("Pass success rate              : N/A (no attempts detected)")
    print("=" * 80)


if __name__ == "__main__":
    main()