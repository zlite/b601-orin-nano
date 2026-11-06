"""Fit the SO-101 -> reBot B601 joint mapping, one arm and one joint at a time.

Nothing has to be matched by eye. Both arms report true joint angles, so each arm joint
maps as ``follower = sign * leader + offset``:

  1. Rest pose: follower limp in its zero ("sit-down") pose, leader folded the same way
     against its end stops. This fixes every offset.
  2. Joint by joint: the follower moves ONE joint and holds; you move the same joint on the
     leader in the same direction. Only the direction matters; the capture is automatic once
     the leader has moved far enough and is held still (ENTER also captures).
  3. Gripper: the follower opens; open the leader gripper fully (its end stop).
  4. The mapping is saved; then a check pose shows the live error of the result.
  5. Follower returns to rest and goes limp (it is resting, so it does not drop).

Ctrl-C during motion freezes the follower in place; then ENTER returns it to rest, or a
second Ctrl-C cuts torque (the arm drops: support it). Exits non-zero if nothing was saved.

Usage:
  python -m lerobot_teleoperator_so101_b601.map_joints \
      --leader-port /dev/so101_leader --leader-id so101_leader \
      --follower-port can0 --follower-id b601_rs [--verify-only]
"""

import argparse
import collections
import json
import math
import sys
import threading
import time
from pathlib import Path

from lerobot.robots.rebot_b601_follower import RebotB601Follower, RebotB601FollowerRobotConfig
from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig

from .so101_b601_leader import B601_JOINTS, MAPPED_JOINTS, default_mapping_path

# Follower targets in B601 public degrees. From the zero pose (upper arm horizontal pointing
# back, forearm folded forward on top of it), negative lift/elbow raise and unfold the arm.
REST_POSE = dict.fromkeys(B601_JOINTS, 0.0)
# One joint at a time, from rest.
JOINT_TESTS = (
    ("shoulder_pan", 30.0, "turn the base"),
    ("shoulder_lift", -40.0, "raise the upper arm"),
    ("elbow_flex", -40.0, "unfold the forearm"),
    ("wrist_flex", 30.0, "tilt the wrist"),
    ("wrist_roll", 45.0, "roll the wrist"),
)
# Follower gripper angle that a fully open leader gripper maps to (limit is -270).
GRIPPER_OPEN_DEG = -250.0
CHECK_POSE = {"shoulder_pan": -25.0, "shoulder_lift": -30.0, "elbow_flex": -45.0, "wrist_flex": -30.0,
              "wrist_yaw": 0.0, "wrist_roll": -45.0, "gripper": -60.0}
REST_TOLERANCE_DEG = 15.0

# Leader movement needed to accept a joint direction (deg) / a gripper opening (0-100 units).
MIN_LEADER_MOVE = 12.0
MIN_GRIPPER_MOVE = 30.0
# Auto-capture once the leader has moved enough and stayed within this band for STILL_S.
STILL_BAND = 1.5
STILL_S = 1.0

RATE_HZ = 30
SPEED_DEG_S = {**dict.fromkeys(B601_JOINTS, 20.0), "gripper": 60.0}
# Extra guard inside send_action: commands never lead the measured position by more than this.
MAX_RELATIVE_TARGET_DEG = 6.0


class LineReader:
    """Reads stdin lines on a thread so the live display keeps updating."""

    def __init__(self):
        self.lines: list[str] = []
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        for line in sys.stdin:
            with self.lock:
                self.lines.append(line.strip().lower())

    def pop(self) -> str | None:
        with self.lock:
            return self.lines.pop(0) if self.lines else None

    def clear(self) -> None:
        with self.lock:
            self.lines.clear()


def connect_limp_follower(port: str, follower_id: str, require_calibration: bool = True) -> RebotB601Follower:
    config = RebotB601FollowerRobotConfig(
        port=port, id=follower_id, motor_family="rs", max_relative_target=MAX_RELATIVE_TARGET_DEG
    )
    follower = RebotB601Follower(config)
    if require_calibration and not follower.is_calibrated:
        raise SystemExit(f"Follower '{follower_id}' has no calibration (zero pose). Run `robot.sh calibrate-follower`.")

    # connect() ends with configure(), which enables torque; keep the arm limp instead.
    def stay_limp():
        follower.bus.disable_all()

    follower.configure = stay_limp
    follower.connect(calibrate=False)
    return follower


class Session:
    def __init__(self, leader: SOLeader, follower: RebotB601Follower, reader: LineReader):
        self.leader, self.follower, self.reader = leader, follower, reader
        self.torque = False
        self.hold: dict[str, float] | None = None

    # ---- reading / display
    def leader_pos(self) -> dict[str, float]:
        return {k.removesuffix(".pos"): v for k, v in self.leader.get_action().items()}

    def follower_pos(self) -> dict[str, float]:
        return self.follower._present_pos()

    def status_line(self, text: str) -> None:
        print(f"\r{text}\x1b[K", end="", flush=True)

    # ---- torque
    def enable_holding(self) -> None:
        """Enable torque and immediately command the current position (no jump)."""
        present = self.follower_pos()
        RebotB601Follower.configure(self.follower)
        self.torque = True
        self.command(present)

    def release(self) -> None:
        self.follower.bus.disable_all()
        self.torque = False
        self.hold = None

    def command(self, target: dict[str, float]) -> None:
        self.hold = dict(target)
        self.follower.send_action({f"{j}.pos": v for j, v in target.items()})

    def tick(self) -> None:
        """Refresh the hold command; call regularly while torque is on."""
        if self.torque and self.hold is not None:
            self.follower.send_action({f"{j}.pos": v for j, v in self.hold.items()})

    # ---- motion
    def move_to(self, target: dict[str, float], label: str) -> None:
        if not self.torque:
            self.enable_holding()
        start = self.follower_pos()
        duration = max(abs(target[j] - start[j]) / SPEED_DEG_S[j] for j in B601_JOINTS)
        steps = max(1, math.ceil(duration * RATE_HZ))
        print(f"\nMoving follower: {label} ({duration:.1f} s). Ctrl-C freezes it.")
        for i in range(1, steps + 1):
            t0 = time.perf_counter()
            a = i / steps
            self.command({j: start[j] + (target[j] - start[j]) * a for j in B601_JOINTS})
            self.status_line(f"  {100 * a:3.0f}%")
            time.sleep(max(0.0, 1 / RATE_HZ - (time.perf_counter() - t0)))
        # Settle on the final target.
        settle_until = time.time() + 0.7
        while time.time() < settle_until:
            self.tick()
            time.sleep(1 / RATE_HZ)
        print()

    def rest_and_release(self) -> None:
        if self.torque:
            self.move_to(REST_POSE, "back to rest")
            self.release()
            print("Follower at rest; torque off.")

    def freeze(self) -> None:
        """Hold wherever the arm is now."""
        if self.torque:
            self.command(self.follower_pos())

    # ---- waiting for the user
    def wait_for_enter(self, render) -> bool:
        """Show `render()` live until ENTER (True) or 'q' (False)."""
        self.reader.clear()
        while True:
            t0 = time.perf_counter()
            self.tick()
            self.status_line(render())
            cmd = self.reader.pop()
            if cmd is not None:
                print()
                return cmd != "q"
            time.sleep(max(0.0, 1 / RATE_HZ - (time.perf_counter() - t0)))

    def wait_for_move(self, joint: str, min_move: float, start: float) -> float:
        """Wait until the leader `joint` has moved >= min_move from `start` and is held still.

        ENTER captures early if the move is already large enough; 'q' aborts. Returns the
        leader reading at capture.
        """
        self.reader.clear()
        history: collections.deque[tuple[float, float]] = collections.deque()
        while True:
            t0 = time.perf_counter()
            self.tick()
            now = time.monotonic()
            value = self.leader_pos()[joint]
            history.append((now, value))
            while history and now - history[0][0] > STILL_S:
                history.popleft()
            moved = value - start
            values = [v for _, v in history]
            still = now - history[0][0] >= STILL_S * 0.9 and max(values) - min(values) < STILL_BAND
            enough = abs(moved) >= min_move
            hint = "hold still..." if enough else "keep moving"
            self.status_line(f"  leader {joint}: moved {moved:+6.1f}   {hint}")

            cmd = self.reader.pop()
            if cmd == "q":
                print()
                raise KeyboardInterrupt
            if cmd is not None and not enough:
                print(f"\n  The leader {joint} has only moved {moved:+.1f}; move it further first.")
            if enough and (still or cmd is not None):
                print(f"\n  Captured: leader {joint} moved {moved:+.1f}.")
                return value
            time.sleep(max(0.0, 1 / RATE_HZ - (time.perf_counter() - t0)))


def fmt(values: dict[str, float], joints=MAPPED_JOINTS) -> str:
    return " ".join(f"{values[j]:7.1f}" for j in joints)


HEADER = "          " + " ".join(f"{j[:7]:>7}" for j in MAPPED_JOINTS)


def capture_rest(s: Session) -> tuple[dict, dict]:
    print("\nREST POSE")
    print("  Follower: limp, resting in its zero pose (fold it there by hand if needed).")
    print("  Leader:   fold it the same way, joints against their stops, gripper CLOSED.")
    print("  ENTER captures, q + ENTER aborts.\n" + HEADER)

    def render():
        lead, foll = s.leader_pos(), s.follower_pos()
        far = [j for j in B601_JOINTS if abs(foll[j]) > REST_TOLERANCE_DEG]
        warn = f"  follower not at zero: {', '.join(far)}" if far else ""
        return f"leader   {fmt(lead)}  | follower {fmt(foll)}{warn}"

    while True:
        if not s.wait_for_enter(render):
            raise KeyboardInterrupt
        foll = s.follower_pos()
        far = [j for j in B601_JOINTS if abs(foll[j]) > REST_TOLERANCE_DEG]
        if far:
            print(f"Follower is not resting in the zero pose ({', '.join(far)}). Move it there and press ENTER again.")
            continue
        return s.leader_pos(), foll


def test_joint(s: Session, index: int, joint: str, delta: float, action: str) -> float:
    """Move one follower joint by `delta`; return the sign linking the leader to it."""
    # Measured before the follower moves, in case the leader is moved along with it.
    leader_start = s.leader_pos()[joint]
    target = {**REST_POSE, joint: delta}
    s.move_to(target, f"{joint} {delta:+.0f} deg")
    print(f"JOINT {index}/{len(JOINT_TESTS)}: {joint}. The follower moved its {joint} ({action}).")
    print(f"  Move ONLY the leader's {joint} the same way (about {abs(delta):.0f} deg; the amount is not critical).")
    leader_end = s.wait_for_move(joint, MIN_LEADER_MOVE, leader_start)
    sign = math.copysign(1.0, (leader_end - leader_start) * delta)
    s.move_to(REST_POSE, "back to rest")
    return sign


def test_gripper(s: Session) -> float:
    leader_start = s.leader_pos()["gripper"]
    s.move_to({**REST_POSE, "gripper": GRIPPER_OPEN_DEG}, "open gripper")
    print("GRIPPER: the follower opened its gripper. Open the leader gripper FULLY and hold it there.")
    value = s.wait_for_move("gripper", MIN_GRIPPER_MOVE, leader_start)
    s.move_to(REST_POSE, "back to rest")
    return value


def live_check(s: Session, mapping: dict) -> None:
    s.move_to(CHECK_POSE, "check pose")
    print("CHECK: move the leader roughly into the follower's pose. Mapped leader - follower, in degrees.")
    print("Signs are right when each error shrinks as you approach the pose; within ~10 deg is fine")
    print("(the arms' proportions differ). ENTER finishes.\n" + "error    " + HEADER.strip())

    def render():
        lead, foll = s.leader_pos(), s.follower_pos()
        err = {j: mapping[j]["scale"] * lead[j] + mapping[j]["offset"] - foll[j] for j in MAPPED_JOINTS}
        return f"error    {fmt(err)}"

    s.wait_for_enter(render)


def run(s: Session, output: Path, verify_only: bool, ids: dict) -> None:
    if verify_only:
        mapping = json.loads(output.read_text())["joints"]
        capture_rest(s)  # make sure the follower starts from rest
        live_check(s, mapping)
        return

    lead_rest, foll_rest = capture_rest(s)
    mapping = {}
    for i, (joint, delta, action) in enumerate(JOINT_TESTS, start=1):
        sign = test_joint(s, i, joint, delta, action)
        mapping[joint] = {"scale": sign, "offset": foll_rest[joint] - sign * lead_rest[joint]}
    lead_open = test_gripper(s)
    scale = (GRIPPER_OPEN_DEG - foll_rest["gripper"]) / (lead_open - lead_rest["gripper"])
    mapping["gripper"] = {"scale": scale, "offset": foll_rest["gripper"] - scale * lead_rest["gripper"]}

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(
        {**ids, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "method": "rest offsets + per-joint signs",
         "joints": mapping, "rest": {"leader": lead_rest, "follower": foll_rest},
         "leader_gripper_open": lead_open},
        indent=2,
    ))
    print("\nMAPPING (follower = scale * leader + offset):")
    for joint, m in mapping.items():
        print(f"  {joint:14s} scale {m['scale']:+7.3f}  offset {m['offset']:+8.2f}")
    print(f"Saved {output}")
    live_check(s, mapping)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--leader-port", required=True)
    parser.add_argument("--leader-id", default="so101_leader")
    parser.add_argument("--follower-port", default="can0")
    parser.add_argument("--follower-id", default="b601_rs")
    parser.add_argument("--output", type=Path, help="Mapping file (default: next to the leader calibration).")
    parser.add_argument("--verify-only", action="store_true", help="Skip fitting; check an existing mapping.")
    args = parser.parse_args()

    leader = SOLeader(SOLeaderTeleopConfig(port=args.leader_port, id=args.leader_id, use_degrees=True))
    if not leader.calibration:
        raise SystemExit(f"Leader '{args.leader_id}' is not calibrated. Run `robot.sh calibrate-leader`.")
    output = args.output or default_mapping_path(leader.calibration_dir, args.leader_id)
    if args.verify_only and not output.is_file():
        raise SystemExit(f"No mapping at {output}; run the mapping first.")

    leader.connect()
    follower = connect_limp_follower(args.follower_port, args.follower_id)
    s = Session(leader, follower, LineReader())
    ids = {"leader_id": args.leader_id, "follower_id": args.follower_id}
    completed = False
    try:
        try:
            run(s, output, args.verify_only, ids)
            completed = True
        except KeyboardInterrupt:
            s.freeze()
            if s.torque:
                print("\nFollower frozen in place. ENTER returns it slowly to rest; Ctrl-C again cuts torque "
                      "(THE ARM DROPS - support it).")
                try:
                    s.wait_for_enter(lambda: "holding...")
                except KeyboardInterrupt:
                    s.release()
                    print("\nTorque off.")
                    raise SystemExit(1)
            print("\nAborted; mapping file unchanged.")
        s.rest_and_release()
    finally:
        follower.disconnect()
        leader.disconnect()
    if not completed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
