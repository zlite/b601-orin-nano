"""Run the B601 plate policy on the arm attached to this Jetson.

Runs inside the b601-policy image (see ../jetson.sh: `check` and `rollout`), next to the policy server
container, which it reaches at tcp/127.0.0.1:7447.

    python3 b601_rollout.py check                       read the arm limp + save a wrist frame; no torque
    python3 b601_rollout.py run [--task TEXT] [--duration S] [extra lerobot-rollout args]

The arm is driven by plain `lerobot-rollout` (remote inference), with the follower patched to the
settings the training data was recorded under (b601_transfer/reference/agent.py, NOTES.txt):
  - gripper range -325..-15 deg (stock profile: -270..0). The jaws are closed (-21) before the policy starts,
    as at the start of the training episodes, which approach closed and open only at the plate.
  - gripper torque caps 0.8 Nm moving / 1.0 Nm holding (stock 3.5 Nm: closing at full driver force on the
    plate's edge broke the gripper on 2026-10-17).
  - holding gains HOLD_KP/HOLD_KD on shoulder_lift, elbow_flex, wrist_flex.
  - gripper unwrap: the gripper motor can read +360 too high after a disable/enable when the jaws are open
    past -180 (-322 reads +38). The frame is decided from a limp reading before torque goes on and checked
    again right after enabling; any doubt aborts with torque off.
Camera: wrist OAK-D-SR, CAM_B, captured 640x400 BGR (raw) and resized 320x200 INTER_AREA -> RGB, as
record.py did.
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

# This script's folder also holds the plugin's project folder robot/lerobot_camera_oak/ (no __init__.py);
# left on sys.path it shadows the installed plugin as an empty namespace package.
_HERE = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", _HERE) and Path(p or ".").resolve() != Path(_HERE)]

PORT = os.environ.get("B601_CAN", "canb601")
ROBOT_ID = "b601_rs"
WRIST_ID = "19443010A13F095A00"
OVERVIEW_ID = "19443010A155785A00"  # OAK-1-AF scene camera: recorded for review only, not given to the policy
OVERVIEW_EVERY_S = 2.0
TASKS = {
    "table": "pick up the 96-well plate and put it down somewhere else on the table",
    "holder": "put the 96-well plate in the holder",
}
FPS = 10
# Commands go to the motors at FPS x INTERP (30 Hz), linearly interpolated between the policy's 10 Hz actions
# (10 Hz steps made the motion jerky, 2026-11-03).
INTERP = 3
# Cap on how far a command may lead the *measured* position (LeRobot max_relative_target). It is a lead limit,
# not a speed limit: under gravity the big joints sit ~6 deg behind their command (agent.py used 15 deg). At 2 deg
# elbow_flex and wrist_flex stalled and the end-of-run return never reached rest (2026-11-03).
MAX_RELATIVE_TARGET_DEG = 10.0
# Before torque goes off: drive back to the pose the arm started in (where it rested limp) and check it got there.
PARK_SPEED_DEG_S = 20.0
PARK_TOL_DEG = 4.0            # every arm joint within this of the start pose
PARK_GRIPPER_TOL_DEG = 15.0
PARK_TIMEOUT_S = 20.0
PARK_GRIP_WAIT_DEG = 20.0     # park closes the jaws only once every arm joint is this close to the start pose
RELEASE_WAIT_S = 60.0         # not parked: keep holding this long for someone to support the arm
# A run that ends with the plate in the jaws: park could not close them (stalled at -242, the plate's width), timed
# out, and torque-off dropped the plate by the rest pose (2026-11-05 trial 5). Now it is set down on the pad first:
# lowered to a scripted demo's release pose (episode 15, a pad place, camera at (355, 309, 119) mm), jaws opened,
# lifted, then the usual park. Hover poses from the calibrated kinematics (kin/model.py), straight up 50 / 15 mm.
# holding = the jaws at the plate's width: the policy carries with its command ~at the jaws (-254.8 vs -256.7 at the
# end of trial 5), so no lag to go by; open is -294..-324, closed -21. Empty jaws there: a harmless set-down.
HELD_GRIPPER_RANGE = (-275.0, -225.0)
SET_DOWN_UP50 = {"shoulder_pan": 52.2, "shoulder_lift": -103.5, "elbow_flex": -67.2, "wrist_flex": 13.2,
                 "wrist_yaw": 26.3, "wrist_roll": 31.4}
SET_DOWN_UP15 = {"shoulder_pan": 52.2, "shoulder_lift": -107.3, "elbow_flex": -63.2, "wrist_flex": 5.4,
                 "wrist_yaw": 26.3, "wrist_roll": 31.4}
SET_DOWN_POSE = {"shoulder_pan": 52.2, "shoulder_lift": -109.2, "elbow_flex": -61.8, "wrist_flex": 2.1,
                 "wrist_yaw": 26.3, "wrist_roll": 31.4}
SET_DOWN_SLOW_DEG_S = 4.0     # the last 15 mm
SET_DOWN_CONTACT_LAG_DEG = 4.0  # a big joint this far behind its setpoint on the way down = the plate is on the table
GRIP_OPEN_DEG = -324.0
# Grasp squeeze: a gripper command at the plate's width goes out GRIP_SQUEEZE_DEG further closed (never past
# GRIP_MAX_CLOSE_DEG), as agent.py does after contact; the 1.0 Nm hold cap then sets the force. v5/v6 command an
# absolute ~-250 learned from the demos (which carried only ~3 deg past the jaws), so a plate that met the jaws a
# little wider was not squeezed and stayed on the pad (v6 trials 1 and 7, 2026-11-06). B601_GRIP_SQUEEZE=0: off.
GRIP_SQUEEZE_DEG = float(os.environ.get("B601_GRIP_SQUEEZE", "8"))
GRIP_MAX_CLOSE_DEG = -240.0
STATUS_EVERY_S = 1.0          # status line (time, joints, gripper) in the log this often
FRAME_EVERY_S = 5.0           # wrist frame saved to logs/frames_<time>/ this often during a run
FRAMES_DIR = None             # set by cmd_run
# Ask for the next 5 s plan when this much of the current one is left. Training executed whole 50-step chunks
# (n_action_steps=50); with 2.0 s the last ~2 s of every plan was always replaced, and the demos' plans start
# with a pause (kpick holds still while it aligns), so the arm hovered over the plate for 2.5 min (2026-11-03
# 14:04). Inference takes 0.82-0.86 s here (was 1.04 s with refill 0.5: "observation is too old"); 1.2 s keeps margin.
REFILL_S = 1.2
MAX_OBS_AGE_S = 7.0
# When a new chunk replaces the old one, average its first BLEND_STEPS actions 50/50 with the old plan for the
# arm joints, instead of a hard cut (a ~2 deg shoulder_lift jump at a chunk switch, 2026-11-03). Not the gripper.
BLEND_STEPS = 3
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll"]

# The training demos approach with the jaws CLOSED (-20.8 deg) and open them only at the plate (83 of 123 v3
# episodes; wrist video shows the jaws nearly touching until the grasp). Starting open (-322) put the policy off its
# data (2026-11-03 runs: reached, half-closed, backed off). So: close the jaws before the policy starts, and allow
# the full range. The plate and the gripper are protected by the 0.8 / 1.0 Nm torque caps, not by this limit.
GRIPPER_LIMITS = (-325.0, -15.0)
GRIP_START_DEG = -21.0        # jaws closed, as at the start of the training episodes
GRIP_MOVE_DEG_S = 60.0        # gripper ramp speed for the start close and the park
GRIP_START_TOL_DEG = 10.0
GRIP_WRAP_DEG = 20.0
GRIP_TORQUE_LIMIT_NM = 0.8
GRIP_HOLD_TORQUE_LIMIT_NM = 1.0
HOLD_KP = {"shoulder_lift": 250.0, "elbow_flex": 250.0, "wrist_flex": 90.0}
HOLD_KD = {"shoulder_lift": 14.0, "elbow_flex": 14.0, "wrist_flex": 7.0}
# A limp reading this far outside a joint's soft limits means a calibration/zero mismatch: refuse to enable.
LIMIT_MARGIN_DEG = 15.0

WRIST_CAMERA = (
    f'{{wrist: {{type: oak, device_id: "{WRIST_ID}", socket: CAM_B, width: 320, height: 200, '
    f"capture_width: 640, capture_height: 400, fps: 15, encoding: raw}}}}"
)
ROBOT_ARGS = [
    "--robot.type=rebot_b601_follower",
    "--robot.motor_family=rs",
    f"--robot.port={PORT}",
    f"--robot.id={ROBOT_ID}",
    f"--robot.max_relative_target={MAX_RELATIVE_TARGET_DEG}",
    f"--robot.gripper_torque_limit={GRIP_TORQUE_LIMIT_NM}",
    f"--robot.gripper_hold_torque_limit={GRIP_HOLD_TORQUE_LIMIT_NM}",
    f"--robot.cameras={WRIST_CAMERA}",
]
INFERENCE_ARGS = [
    "--inference.type=remote",
    "--inference.endpoint=tcp/127.0.0.1:7447",
    "--inference.deployment=b601_plate",
    "--inference.semantics=b601-follower-units-v1",
    # Ask for the next chunk with 2 s of the current one left (default 0.5 s). The policy takes ~1.04 s per
    # call here; with 0.5 s, the tail of each 5 s chunk was dispatched from an observation older than
    # max_observation_age_s=5 s and the client revoked motion (first rollout, 2026-11-03). With 2 s, actions
    # come from observations at most ~4 s old and the policy re-plans every ~2-3 s.
    f"--inference.refill_seconds={REFILL_S}",
    # Age limit for the observation an action was planned from. Playing a 5 s plan down to 1.2 s left, plus ~0.85 s
    # inference, needs ~4.7-5.0 s; the 5.0 s default stopped the run at 5 s (2026-11-03 14:21). Training executed
    # whole plans (actions up to ~5.9 s after their observation), so 7 s still matches the training data.
    f"--inference.max_observation_age_s={MAX_OBS_AGE_S}",
    f"--inference.blend_steps={BLEND_STEPS}",
    "--inference.blend_weight=0.5",
    "--inference.blend_components=[" + ",".join(f'"{j}.pos"' for j in ARM_JOINTS) + "]",
    f"--interpolation_multiplier={INTERP}",
    '--rename_map={"observation.images.wrist": "observation.images.camera1"}',
    f"--fps={FPS}",
]


class CheckDone(Exception):
    """Raised by the patched configure() in check mode, before any torque is enabled."""


def patch_follower(check_only: bool = False) -> None:
    import lerobot_camera_oak  # noqa: F401  registers camera type "oak"
    from lerobot.robots.rebot_b601_follower import rebot_b601_follower as rb

    cls = rb.RebotB601Follower
    if getattr(cls, "_b601_patched", False):
        return
    orig_init, orig_configure, orig_disconnect = cls.__init__, cls.configure, cls.disconnect
    orig_m2p, orig_p2m = cls._motor_to_public_position, cls._public_to_motor_position

    def __init__(self, config):
        orig_init(self, config)
        self._joint_limits["gripper"] = GRIPPER_LIMITS
        self._mit_kp.update(HOLD_KP)
        self._mit_kd.update(HOLD_KD)
        self.config.gripper_torque_limit = GRIP_TORQUE_LIMIT_NM
        self.config.gripper_hold_torque_limit = GRIP_HOLD_TORQUE_LIMIT_NM
        self._grip_off = 0.0

    def _motor_to_public_position(self, name, deg):
        v = orig_m2p(self, name, deg)
        return v + self._grip_off if name == "gripper" else v

    def _public_to_motor_position(self, name, deg):
        return orig_p2m(self, name, deg - self._grip_off if name == "gripper" else deg)

    def gripper_raw(self, state) -> float:
        return orig_m2p(self, "gripper", math.degrees(state.pos))

    def set_grip_frame(self, raw: float, why: str) -> None:
        off = -360.0 if raw > GRIP_WRAP_DEG else 0.0
        if off != self._grip_off or off:
            print(f"[b601] gripper reads {raw:+.1f} deg {why}: frame {'WRAPPED' if off else 'unwrapped'}, "
                  f"jaws at {raw + off:.1f} deg (offset {off:+.0f})", flush=True)
        self._grip_off = off

    def limp_states(self):
        # Disabled RobStride motors do not answer feedback requests; a disable frame is answered.
        for _ in range(3):
            self.bus.disable_all()
            time.sleep(0.03)
            self.bus.poll_feedback_once()
        states = {k: m.get_state() for k, m in self.motors.items()}
        missing = [k for k, s in states.items() if s is None]
        if missing:
            raise RuntimeError(f"no feedback from {missing} on {PORT}: arm powered? CAN wiring?")
        return states

    def check_pose(self, q: dict) -> list[str]:
        bad = []
        for name, (lo, hi) in self._joint_limits.items():
            lo_m = lo - (GRIP_WRAP_DEG if name == "gripper" else LIMIT_MARGIN_DEG)
            hi_m = (0.0 + GRIP_WRAP_DEG) if name == "gripper" else hi + LIMIT_MARGIN_DEG
            if not lo_m <= q[name] <= hi_m:
                bad.append(f"{name}={q[name]:.1f} (allowed {lo_m:.0f}..{hi_m:.0f})")
        return bad

    def configure(self):
        """agent.py safe_enable: decide the gripper frame limp, check, enable, re-check, hold in place."""
        states = self.limp_states()
        self.set_grip_frame(self.gripper_raw(states["gripper"]), "limp")
        present = self._present_pos(states)
        print("[b601] limp joints: " + ", ".join(f"{k}={v:.1f}" for k, v in present.items()), flush=True)
        bad = self.check_pose(present)
        if bad:
            self.bus.disable_all()
            raise RuntimeError("refusing to enable torque, reading out of range: " + "; ".join(bad))
        if check_only:
            raise CheckDone(present)
        orig_configure(self)
        now = self._present_pos()
        if abs(now["gripper"] - present["gripper"]) > 20.0:
            self.set_grip_frame(self.gripper_raw(self._read_feedback()["gripper"]), "after enabling")
            now = self._present_pos()
            if abs(now["gripper"] - present["gripper"]) > 20.0:
                self.bus.disable_all()
                raise RuntimeError(f"gripper read {present['gripper']:.1f} limp but {now['gripper']:.1f} "
                                   "after enabling; torque off")
        self._b601_start = dict(present)  # the limp resting pose: where park() returns before torque off
        self._b601_start["gripper"] = GRIP_START_DEG  # park leaves the jaws closed, ready for the next run
        # Hold where the arm is (not a stale reading), close the jaws as in training, then the policy starts.
        hold = {f"{k}.pos": v for k, v in now.items()}
        self.send_action(hold)
        g, dt = now["gripper"], 1.0 / 30
        t_end = time.monotonic() + 15.0  # the 0.8 Nm cap closes slower than the ramp
        while time.monotonic() < t_end:
            g += max(-GRIP_MOVE_DEG_S * dt, min(GRIP_MOVE_DEG_S * dt, GRIP_START_DEG - g))
            self.send_action({**hold, "gripper.pos": g})
            if g == GRIP_START_DEG and abs(self._present_pos()["gripper"] - GRIP_START_DEG) <= GRIP_START_TOL_DEG:
                break
            time.sleep(dt)
        print(f"[b601] jaws closed for the start: gripper {self._present_pos()['gripper']:.1f} "
              f"(target {GRIP_START_DEG:.0f})", flush=True)
        self._b601_t0 = None  # status-line clock starts with the policy

    def park(self) -> bool:
        """Closed loop: move the setpoint toward the start pose at PARK_SPEED, until the measured joints are there."""
        target = self._b601_start
        dt = 1.0 / 30
        setpoint = self._present_pos()
        t_end = time.monotonic() + PARK_TIMEOUT_S
        while time.monotonic() < t_end:
            present = self._present_pos()
            step = PARK_SPEED_DEG_S * dt
            # jaws wait until the arm is near the start pose, i.e. lifted clear of the table: closing them where the
            # run ended shut them on a plate lying between open jaws, park stalled at -242 and the torque-off
            # lost the plate (v6sq trial 10, 2026-11-06)
            arm_far = max(abs(present[k] - target[k]) for k in target if k != "gripper") > PARK_GRIP_WAIT_DEG
            for k, goal in target.items():
                if k == "gripper" and arm_far:
                    continue
                d = goal - setpoint[k]
                st = GRIP_MOVE_DEG_S * dt if k == "gripper" else step
                setpoint[k] += max(-st, min(st, d))
            self.send_action({f"{k}.pos": v for k, v in setpoint.items()})
            err = {k: abs(present[k] - target[k]) for k in target}
            if all(e <= (PARK_GRIPPER_TOL_DEG if k == "gripper" else PARK_TOL_DEG) for k, e in err.items()):
                for _ in range(15):  # settle 0.5 s on the final setpoint
                    self.send_action({f"{k}.pos": v for k, v in target.items()})
                    time.sleep(dt)
                print("[b601] parked at the start pose", flush=True)
                return True
            time.sleep(dt)
        worst = sorted(err.items(), key=lambda kv: -kv[1])[:3]
        print("[b601] NOT parked after %.0f s; furthest: %s" % (
            PARK_TIMEOUT_S, ", ".join(f"{k} {present[k]:.1f} (start {target[k]:.1f})" for k, _ in worst)), flush=True)
        return False

    def ramp(self, target, speed, grip, timeout=20.0, stop_lag=None):
        """Setpoint ramp of the arm joints to target at speed (deg/s), gripper setpoint fixed at grip. -> "done",
        "contact" (stop_lag: a lift/elbow/wrist-flex joint fell that far behind) or "timeout"."""
        dt = 1.0 / 30
        sp = {k: v for k, v in self._present_pos().items() if k != "gripper"}
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            for k, goal in target.items():
                sp[k] += max(-speed * dt, min(speed * dt, goal - sp[k]))
            self.send_action({**{f"{k}.pos": v for k, v in sp.items()}, "gripper.pos": grip})
            present = self._present_pos()
            if stop_lag is not None and any(abs(present[k] - sp[k]) > stop_lag
                                            for k in ("shoulder_lift", "elbow_flex", "wrist_flex")):
                return "contact"
            if all(sp[k] == target[k] for k in target) and all(abs(present[k] - target[k]) <= PARK_TOL_DEG
                                                                for k in target):
                return "done"
            time.sleep(dt)
        return "timeout"

    def set_down_held(self):
        """If the run ended with the plate in the jaws, set it down on the pad and open (see SET_DOWN_POSE)."""
        # judge by the last observation the policy saw: after the run the arm is driven to the start pose with
        # 10-deg-clamped steps, and the jaws stop wherever the last clamped target was (-236 with nothing in them,
        # trials 6-8, 2026-11-05). The policy's last view: -256.7 holding (trial 5), -322 open (trials 6-8).
        cmd = getattr(self, "_b601_last_grip_cmd", float("nan"))
        obs = getattr(self, "_b601_obs", None) or {}
        g = float(obs.get("gripper.pos", float("nan")))
        holding = HELD_GRIPPER_RANGE[0] <= g <= HELD_GRIPPER_RANGE[1]
        if not holding:
            print(f"[b601] run ended with the jaws free (gripper {g:.1f}, command {cmd:.1f})", flush=True)
        if not holding:
            return
        grip = min(self._present_pos()["gripper"] + 8.0, -21.0)  # keep squeezing (the 1.0 Nm hold cap sets the force)
        print(f"[b601] run ended HOLDING the plate (gripper {g:.1f}, command {cmd:.1f}): setting it down on the pad",
              flush=True)
        r1 = self.ramp(SET_DOWN_UP50, PARK_SPEED_DEG_S, grip)
        r2 = self.ramp(SET_DOWN_UP15, PARK_SPEED_DEG_S / 2, grip, stop_lag=8.0) if r1 == "done" else "skipped"
        r3 = self.ramp(SET_DOWN_POSE, SET_DOWN_SLOW_DEG_S, grip, stop_lag=SET_DOWN_CONTACT_LAG_DEG) \
            if r2 == "done" else "skipped"
        print(f"[b601] set-down: over the pad {r1}, down to 15 mm {r2}, to the table {r3}", flush=True)
        hold = {f"{k}.pos": v for k, v in self._present_pos().items() if k != "gripper"}
        dt, gg = 1.0 / 30, grip
        for _ in range(int(3.0 / dt)):  # open fully where it is
            gg += max(-GRIP_MOVE_DEG_S * dt, min(GRIP_MOVE_DEG_S * dt, GRIP_OPEN_DEG - gg))
            self.send_action({**hold, "gripper.pos": gg})
            time.sleep(dt)
        print(f"[b601] set-down: jaws opened (gripper {self._present_pos()['gripper']:.1f})", flush=True)
        self.ramp(SET_DOWN_UP50, PARK_SPEED_DEG_S / 2, GRIP_OPEN_DEG)

    def disconnect(self):
        try:
            if self.bus is not None and getattr(self, "_b601_start", None) is not None:
                try:
                    self.set_down_held()
                except Exception as e:
                    print(f"[b601] set-down failed: {e}", flush=True)
                if not self.park():
                    hold = {f"{k}.pos": v for k, v in self._present_pos().items()}
                    print(f"[b601] HOLDING. Support the arm, then press Enter to release torque "
                          f"(auto-release in {RELEASE_WAIT_S:.0f} s).", flush=True)
                    import select
                    t_end = time.monotonic() + RELEASE_WAIT_S
                    while time.monotonic() < t_end:
                        self.send_action(hold)
                        if select.select([sys.stdin], [], [], 0.1)[0]:
                            sys.stdin.readline()
                            break
        except Exception as e:
            print(f"[b601] park failed: {e}", flush=True)
        finally:  # always shut the motors down, also on Ctrl+C during the park (the arm then drops)
            orig_disconnect(self)

    orig_get_obs, orig_send = cls.get_observation, cls.send_action

    def get_observation(self):
        obs = orig_get_obs(self)
        self._b601_obs = obs
        return obs

    def send_action(self, action):
        """Once a second: elapsed time, measured joints and the gripper command, so a log shows how far a run got."""
        g_cmd = action.get("gripper.pos")
        if GRIP_SQUEEZE_DEG and g_cmd is not None and HELD_GRIPPER_RANGE[0] <= float(g_cmd) <= HELD_GRIPPER_RANGE[1]:
            action = {**action, "gripper.pos": max(float(g_cmd), min(float(g_cmd) + GRIP_SQUEEZE_DEG,
                                                                     GRIP_MAX_CLOSE_DEG))}
        sent = orig_send(self, action)
        if "gripper.pos" in action:  # set_down_held: did the run end with the jaws closing on the plate?
            self._b601_last_grip_cmd = float(action["gripper.pos"])
        now = time.monotonic()
        t0 = getattr(self, "_b601_t0", None)
        if t0 is None:
            self._b601_t0 = t0 = now
            self._b601_next = now
        if now >= self._b601_next and getattr(self, "_b601_obs", None) is not None:
            self._b601_next = now + STATUS_EVERY_S
            o = self._b601_obs
            nan = float("nan")
            meas = " ".join(f"{o.get(f'{k}.pos', nan):6.1f}" for k in ARM_JOINTS)
            cmd = " ".join(f"{action.get(f'{k}.pos', nan):6.1f}" for k in ARM_JOINTS)
            print(f"[b601] t={now - t0:5.0f}s joints {meas} | gripper {o.get('gripper.pos', nan):6.1f}"
                  f" cmd {action.get('gripper.pos', nan):6.1f} | arm cmd {cmd}", flush=True)
            # a wrist frame every FRAME_EVERY_S: what the policy was looking at
            img = o.get("wrist")
            if FRAMES_DIR and img is not None and now - t0 >= getattr(self, "_b601_nframe", 0) * FRAME_EVERY_S:
                import cv2
                FRAMES_DIR.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(FRAMES_DIR / f"wrist_t{now - t0:04.0f}s.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                self._b601_nframe = getattr(self, "_b601_nframe", 0) + 1
        return sent

    cls.get_observation, cls.send_action = get_observation, send_action
    cls.__init__ = __init__
    cls.configure = configure
    cls._motor_to_public_position = _motor_to_public_position
    cls._public_to_motor_position = _public_to_motor_position
    cls.gripper_raw, cls.set_grip_frame = gripper_raw, set_grip_frame
    cls.limp_states, cls.check_pose = limp_states, check_pose
    cls.park, cls.disconnect, cls.ramp, cls.set_down_held = park, disconnect, ramp, set_down_held
    cls._b601_patched = True


def build_robot():
    from lerobot_camera_oak import OAKCameraConfig

    from lerobot.robots import make_robot_from_config
    from lerobot.robots.rebot_b601_follower import RebotB601FollowerRobotConfig

    wrist = OAKCameraConfig(device_id=WRIST_ID, socket="CAM_B", width=320, height=200,
                            capture_width=640, capture_height=400, fps=15, encoding="raw")
    cfg = RebotB601FollowerRobotConfig(
        port=PORT, motor_family="rs", id=ROBOT_ID, max_relative_target=MAX_RELATIVE_TARGET_DEG,
        gripper_torque_limit=GRIP_TORQUE_LIMIT_NM, gripper_hold_torque_limit=GRIP_HOLD_TORQUE_LIMIT_NM,
        cameras={"wrist": wrist},
    )
    return make_robot_from_config(cfg)


def cmd_check(out_dir: Path) -> int:
    import cv2

    patch_follower(check_only=True)
    robot = build_robot()
    try:
        robot.connect()
    except CheckDone:
        pass
    try:
        img = robot.cameras["wrist"].read_latest()
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "wrist_320x200.png"
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"[b601] wrist frame {img.shape[1]}x{img.shape[0]} saved to {path}")
        print(f"[b601] CHECK OK: torque never enabled. Gripper offset {robot._grip_off:+.0f}.")
        return 0
    finally:
        for m in robot.motors.values():
            m.disable()
            m.close()
        if robot.bus is not None:
            robot.bus.close()
        for cam in robot.cameras.values():
            if cam.is_connected:
                cam.disconnect()


class OverviewRecorder:
    """Saves an overview-camera picture every OVERVIEW_EVERY_S in a background thread, so a run can be
    reviewed afterwards without anyone describing it. Any camera failure only disables the recording."""

    def __init__(self, out_dir: Path):
        import threading
        self.out_dir, self.stop_evt = out_dir, threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        try:
            import cv2
            import depthai as dai
            with dai.Device(dai.DeviceInfo(OVERVIEW_ID)) as device:
                pipeline = dai.Pipeline(device)
                cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
                q = cam.requestOutput((640, 480), type=dai.ImgFrame.Type.BGR888i, fps=5).createOutputQueue(2, False)
                pipeline.start()
                t0, n = time.monotonic(), 0
                self.out_dir.mkdir(parents=True, exist_ok=True)
                while not self.stop_evt.is_set():
                    msg = q.tryGet()
                    if msg is None:
                        time.sleep(0.02)
                        continue
                    t = time.monotonic() - t0
                    if t >= n * OVERVIEW_EVERY_S:
                        cv2.imwrite(str(self.out_dir / f"overview_t{t:04.0f}s.jpg"), msg.getCvFrame(),
                                    [cv2.IMWRITE_JPEG_QUALITY, 80])
                        n += 1
                pipeline.stop()
        except Exception as e:
            print(f"[b601] overview camera not recorded: {e}", flush=True)

    def stop(self):
        self.stop_evt.set()
        self.thread.join(timeout=5)


def cmd_run(task: str, duration: float, extra: list[str]) -> None:
    from lerobot.scripts import lerobot_rollout

    global FRAMES_DIR
    FRAMES_DIR = Path("/opt/b601/logs") / time.strftime("frames_%Y%m%d_%H%M%S")
    print(f"[b601] wrist frames every {FRAME_EVERY_S:.0f} s, overview every {OVERVIEW_EVERY_S:.0f} s -> logs/{FRAMES_DIR.name}/", flush=True)
    patch_follower()
    sys.argv = ["lerobot-rollout", *ROBOT_ARGS, *INFERENCE_ARGS, f"--task={task}",
                f"--duration={duration}", *extra]
    print("[b601] " + " ".join(sys.argv), flush=True)
    rec = OverviewRecorder(FRAMES_DIR).start()
    try:
        lerobot_rollout.main()
    finally:
        rec.stop()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="read the arm limp and save a wrist frame (no torque)")
    c.add_argument("--out", type=Path, default=Path("/opt/b601/logs"))
    r = sub.add_parser("run", help="run the policy on the arm")
    r.add_argument("--task", default="table", help=f"{list(TASKS)} or the exact text")
    r.add_argument("--duration", type=float, default=180.0, help="seconds; 0 = until Ctrl+C")
    args, extra = p.parse_known_args()
    if args.cmd == "check":
        sys.exit(cmd_check(args.out))
    cmd_run(TASKS.get(args.task, args.task), args.duration, extra)


if __name__ == "__main__":
    main()
