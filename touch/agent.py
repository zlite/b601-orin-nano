"""Touch the AprilTag in front of the B601 with its gripper tips, using only the wrist camera.

No kinematic model is needed: the controller measures how the tag moves in the wrist
camera when each joint moves a little (an image Jacobian), then steps the joints so the tag
approaches the jaw-tip point, updating the Jacobian from every step (Broyden).

Runs as one long-lived process so the wrist OAK and the CAN bus are opened only once.
Commands arrive one per line on the FIFO touch/ctl:

  perceive   measure tag pose and jaw-tip depth (no motion)
  tip X Y Z  override the jaw-tip point in camera coordinates (metres)
  diag       hold the elbow relaxed / marginal / lifted and report tag detection and vibration
  probe      enable torque, small per-joint probe moves to measure the Jacobian
  hover      servo until the tag is HOVER_M in front of the jaw tips, then hold
  touch      creep from hover until the tips reach the tag (or torque rises), back to hover
  rest       return to the zero pose and release torque
  reload     re-read this file's code without closing the camera or CAN connection
  quit       rest, release and exit

Any anomaly (tag lost, torque jump, step too large, camera error) freezes the arm in place
and refuses further motion except `rest`. Log: touch/log.txt; state: touch/status.json.
"""

import datetime
import json
import shutil
import math
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

import lerobot_teleoperator_so101_b601.map_joints as mj
from lerobot_teleoperator_so101_b601.map_joints import REST_POSE, Session, connect_limp_follower

HERE = Path(__file__).resolve().parent
CTL = HERE / "ctl"
LOG = HERE / "log.txt"
STATUS = HERE / "status.json"
DATA = HERE / "data"
PREVIEW = Path("/dev/shm/lerobot_oak_preview/19443010A13F095A00_CAM_B.jpg")

WRIST_ID = "19443010A13F095A00"
W, H, FPS = 640, 400, 15
TAG_SIZE_M = 0.060
TAG_ID = 0
# Known tags: 0 = the touch target (upright paper), 1 = next to the 96-well plate holder.
TAG_SIZES = {0: 0.060, 1: 0.030}
ACTIVE_TAG = {"id": TAG_ID, "size": TAG_SIZE_M}  # switched with the `target` command

# Joints used for servoing (wrist_roll and gripper stay fixed).
# wrist_yaw is left out: near this pose its axis is almost parallel to shoulder_pan, and the
# solver used that redundant pair to chase noise. It stays at its current angle.
SERVO_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")
PROBE_DEG = {"shoulder_pan": 4.0, "shoulder_lift": 5.0, "elbow_flex": 5.0, "wrist_flex": 5.0, "wrist_yaw": 4.0}
# Leave the zero-pose joint stops before probing so +-probe moves are not clipped.
# Only the shoulder leaves its stop: the forearm rests on the upper arm at the zero pose, and
# commanding the elbow slightly off that contact leaves it balanced against gravity (chatter).
START_OFFSET = {"shoulder_lift": -8.0}
# Probe start pose (absolute). Lift and elbow are both well off their stops so every servo joint
# can move both ways (at the folded zero pose the elbow sits on its stop and the forward
# direction is near-singular); wrist_flex brings the tag back to the image centre.
START_POSE = {"shoulder_lift": -40.0, "elbow_flex": -40.0, "wrist_flex": 24.0}
START_POSE_FULL = {**dict.fromkeys(("shoulder_pan", "wrist_yaw", "wrist_roll"), 0.0), **START_POSE}  # sees tag 0
# The gripper fingers hang into the top-centre of the (upside-down) wrist image and end at the
# jaw-tip pixel, so a tag centred on the tips is half covered. Touch the tag this far from its
# centre instead (camera +y = image down): the tag then stays below the fingers in the image.
TAG_AIM_OFFSET = np.array([0.0, 0.018, 0.0])

ROT_WEIGHT_M = 0.01         # PnP normals are noisy; a larger weight made the solver chase tilt noise instead of advancing (MAX_TILT_CHANGE still guards the angle)
MAX_STEP_DEG = 5.0         # per joint per servo iteration
LIMIT_MARGIN_DEG = 2.0     # stay this far inside the follower's soft joint limits
MAX_POS_STEP_M = 0.015     # target advances at most this far per iteration
GAIN = 0.6                 # fraction of each correction applied per step (damps overshoot)
SETTLE_S = 0.8             # hold after each step before measuring (the arm rings briefly after moves)
REPROBE_EVERY = 6          # re-measure the Jacobian with fresh probes every N steps
REPROBE_DEG = PROBE_DEG
PROBE_FRAMES = 12          # frames averaged per probe measurement (forward distance is noisy)
SERVO_FRAMES = 8
TAG_RETRY_S = 6.0          # keep looking this long before declaring the tag lost
RELAX_S = 1.0              # after a tag loss: hold the measured pose this long, then look again
# The forearm rests on the upper arm near the zero pose. Commanding the elbow only slightly
# off that contact leaves it balanced against gravity (suspected chatter / blurred frames), so
# when the elbow is near rest and asked to unfold, command it at least this far past its pose.
ELBOW_REST_ZONE_DEG = -6.0
ELBOW_LIFTOFF_DEG = 10.0
LEAD_MAX_DEG = 10.0        # servo commands never lead the measured angle by more than this (sag ~5 deg + margin)
DLS_LAMBDA = 0.0006      # m/deg; enough to stop noise-driven motion, small next to typical columns
HOVER_M = 0.05
HOVER_TOL_M = 0.004
TOUCH_STEP_M = 0.006
TOUCH_PAST_M = 0.003       # aim this far beyond the tag plane
MAX_TILT_CHANGE = 0.75     # ~43 deg change of the tag normal aborts (tag stays readable well past this)
TORQUE_JUMP_NM = {"shoulder_pan": 2.0, "shoulder_lift": 3.0, "elbow_flex": 3.0,
                  "wrist_flex": 0.8, "wrist_yaw": 0.8, "wrist_roll": 0.8, "gripper": 1.5}
# moves at or above LIFT_M over the grasp, where the jaws cannot reach the plate: the big joints' gravity-load
# swings tripped 3.0 Nm in free air (-3.01 shoulder at 26 mm over the plate top, 2026-10-21; user OK'd 3.5 there)
TORQUE_JUMP_FREE_AIR_NM = {**TORQUE_JUMP_NM, "shoulder_lift": 3.5, "elbow_flex": 3.5}
GRIPPER_LIMITS = (-325.0, 0.0)
# The gripper motor reports its angle within +-180 deg after some disable/enable cycles, so jaws opened past
# -180 can come back reading +360 higher (-322 read as +38 at rest, 2026-11-02 13:03; kpick then held "+38"
# clipped to 0, i.e. pushed the open jaws into their stop, and never closed). A limp reading above this (the
# jaws cannot open past -325) means that frame: the agent then works with a -360 offset on the gripper.
GRIP_WRAP_DEG = 20.0
# 2026-10-17: closing straight to -3 deg on the plate's edge (force limited only by the driver,
# up to 3.5 Nm) broke the gripper; an earlier ~300 deg jump in the gripper reading with the jaws
# blocked was the transmission slipping. No jaw-closing command runs until the gripper is
# inspected and closing is rewritten to stop on measured torque / camera contact.
GRIPPER_LOCKOUT = False  # lifted 2026-10-18 (user OK) after foam contact test stopped at -35 deg by lag
GRIPPER_LOCKED_COMMANDS = ("kpick", "fastpick", "flatpick", "flatfast", "flatcal", "pick", "grasp", "touchprobe", "place")  # gripcal: free air only
# Gripper force caps for this agent (stock: 3.5 Nm moving / 1.0 Nm holding broke a printed finger).
GRIP_TORQUE_LIMIT_NM = 0.8       # driver clamp while the jaws move
GRIP_HOLD_TORQUE_LIMIT_NM = 1.0  # driver clamp once they stop (the squeeze; stock default; 0.6 let the plate slip; user OK 2026-10-18)
# Sensing close: small steps, stop on measured torque rise or the jaws falling behind the command.
GRIP_STEP_DEG = 1.0
GRIP_STEP_S = 0.08
GRIP_CONTACT_NM = 0.7            # closing torque beyond this = contact (2 reads in a row); free air peaks ~0.5
# (gripcal 2026-10-18: free-air torque wanders -0.5..+0.4 Nm and the still reading 0.1..0.5, so no baseline)
GRIP_CONTACT_LAG_DEG = 5.0       # or the jaws lag the command by this much, 2 reads (free air: 4.3 once)
GRIP_BREAKAWAY_DEG = 20.0        # ignore contact signals this far into the close (start-up stiction, 2026-10-18;
                                 # 8 was not enough with the gripper pitched 45 deg). The plate is >= 50 deg away.
GRIP_SQUEEZE_DEG = 8.0           # command this far past contact: the 0.6 Nm hold cap then sets the grip force
GRIP_HELD_MIN_SHORT_DEG = 1.0    # a held plate keeps the jaws at least this far short of the command
GRIP_RESQUEEZE_SHORT_DEG = 3.0   # after the squeeze, re-squeeze unless the jaws are pushed back this much           # after contact, command this far past the contact angle
GRIP_MAX_CLOSE_DEG = -240.0      # never command further closed than this while grasping the plate
# (free air: pad gap ~260 px at -324, ~0.86 px/deg; the plate is ~205 px -> about -256 deg)
# Motor driver (MOSFET) temperatures: the shoulder tripped its over-temperature fault (status 18)
# at ~111 C after ~25 min holding the arm stretched out over the plate.
THERMAL_WARN_C = 80.0
THERMAL_ABORT_C = 90.0

# ---- 96-well plate pick (tag 1). Geometry measured from the wrist camera:
ARM6 = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll")
PROBE_DEG6 = {"shoulder_pan": 3.0, "shoulder_lift": 3.0, "elbow_flex": 3.0, "wrist_flex": 4.0,
              "wrist_yaw": 4.0, "wrist_roll": 5.0}
ROT6_W = 0.08              # m per rad: 1 deg of orientation error weighs like 1.4 mm
# Per-axis weights (camera x, y, z) on the rotation: tilt about the jaw axis (camera x) only
# leans the fingers along the plate's long side, so it is left mostly free (wrist_flex limit).
# Per-axis rotation weights (camera x = along the plate, y = across the jaws, z). At the grasp the
# fingers are held parallel to the plate top (all axes); 6 cm higher, level needs more wrist_flex
# than its 80 deg limit, so the pregrasp tolerates tilt along the plate.
ROT6_AXIS_W = np.array([1.0, 1.0, 1.0])
PREGRASP_AXIS_W = np.array([0.25, 1.0, 1.0])
MAX_ROT_STEP_RAD = math.radians(5.0)
PLATE_CENTER_TAG = np.array([-0.087, 0.010, 0.0])  # plate centre in the tag-1 frame (deck edges, overhead view at pregrasp)
PLATE_TOP_Z = 0.015        # plate top above the table (user, 2026-10-18); black holder walls 0.005
HOLDER_WALL_Z = 0.005
GRASP_Z = 0.0115           # fingertip height at the grasp: just above the holder walls
FINGER_MID_CAM = np.array([0.010, -0.017, 0.152])  # open fingertip midpoint, camera frame (z calibrated: tips resting on the 10 mm holder walls)
GRASP_PITCH_DEG = 0.0      # camera (and fingers) square to the plate top
GRASP_MAX_TILT_DEG = 2.0   # refuse to close if the fingers are tilted more than this vs the plate top
# Flat grasp: the fingers bend 45 deg from the gripper body, so pitching the gripper 45 deg (leaning
# away from the robot, camera looking down along -tag Y) lays the bent finger sections flat along the
# plate's long sides. With the jaws open the right finger hides the tag at these heights, so the
# approach and alignment run with the jaws closed and the last few cm are moved blind on the Jacobian.
FLAT_PITCH_DEG = 45.0
FLAT_AXIS_W = np.array([3.0, 1.0, 1.0])  # the damped solver barely turns about the jaw axis at weight 1
FLAT_PREGRASP_UP_M = 0.05
FLAT_ALIGN_UP_M = 0.025
FLAT_APPROACH_GRIPPER_DEG = -20.0  # closed fingers stay clear of the tag in the image
FLAT_STEP_M = 0.002
FLAT_SECTION_M = 0.070             # length of the bent (flat) finger section (user)
# Fingertip point placed so the flat sections are centred on the plate's length (the tips point to -tag Y).
FLAT_OFFSET_TAG = np.array([0.0, -FLAT_SECTION_M / 2, 0.0])
FLAT_VIEW_OFFSET_TAG = np.array([0.0, 0.022, 0.0])  # align here: centred, the tag leaves the image
FLAT_WALL_BACKOFF_M = 0.0015
FLAT_OPEN_DEG = -295.0             # pads over the holder rim (fully open they straddle it and can drop outside)
FLAT_SLIDE_STEP_M = 0.005
FLAT_MAX_SHIFT_DEG = 3.0           # replay refuses if the aligned joints differ more than this (plate moved)
SCENE_ROW0 = 170                   # wrist image rows below the fingers, compared before/after a pick
SCENE_MIN_NCC = 0.85               # below this the plate/holder looks moved
PAD_BAND_ROWS, PAD_BAND_COLS = (60, 160), (250, 470)  # the plate between the pads at the grasp (fingers excluded)
MM_PER_PX_AT_GRASP = 0.38          # plate side ~152 mm from the camera, fx 400
ABOVE_ALIGN_ITERS = 10  # 6 on the robot PC (10 let a +50 px start go down into a contact, 2026-10-29); on the Jetson at (318, 351) 6 moves of 4 mm only got +50 -> +17 px (2026-11-03 02:10)
ABOVE_STEP_MAX_MM = 4.0
# Sideways target for the view over the grasp, vs table_above_ref.png. That reference has the plate ~5 px from the
# left jaw (image) and ~45 px from the right one; aligned exactly to it on the Jetson, the plate slid ~20 px further
# left on the way down and the left pad landed on its edge ("touched something on the way down", 2026-11-03 01:50,
# data/halt.png). +20 px puts the plate about centred between the open jaws (~10 mm clearance a side).
ABOVE_SIDE_TARGET_PX = 10.0  # Jetson: aligned to 0 (-1, -4 px) the left pad came down on the plate rim twice (halt.png 01:50, 02:13); +5..+19 cleared the descent (2026-11-03)
# along from above reads +-5-10 px between moves (the plate runs off the image top); the close view at the grasp
# accepts 15 px along anyway, and a 6 px target here halted a pick at +11 px after 6 moves (2026-10-23)
ABOVE_DONE_DY_PX = 12.0
MM_PER_PX_SIDE_ABOVE, MM_PER_PX_ALONG_ABOVE = 0.43, 0.22  # LIFT_M up (~14 % farther than at the grasp)
MM_PER_PX_ALONG_AT_GRASP = 0.19    # image-down offsets along the plate (5 mm along moved the view 28 px, 2026-10-21)
PADS_MAX_DY_TABLE_PX = 15.0  # on bare table (~3 mm along the plate)
# On the Jetson the close-view silhouette read +9..+21 px sideways on views where the plate sat between the pads
# as in table_close_ref.png within a few px (data/halt.png 02:07, brighter light here); 4 px refused all of them.
PADS_MAX_DX_TABLE_PX = 20.0
PADS_MAX_DX_PX, PADS_MAX_DY_PX = 4.0, 8.0  # accepted plate offset from the taught view before closing                # below this the plate is not between the pads as taught
SLOW_DEG_S = {**dict.fromkeys(ARM6, 4.0), "gripper": 60.0}
FLAT_EDGE_Z = 0.0065               # flat fingers' lower edge at the grasp: grips 6.5-15 mm, above the 5 mm walls
PREGRASP_UP_M = 0.06
GRIPPER_OPEN_DEG = -324.0  # jaws just wider than the plate (85.5 mm) here
# The holder's screw tabs sit at mid-length beside both long sides (a finger landed on one), so
# grasp this far along the plate's long axis (tag Y) from its centre.
GRASP_OFFSET_TAG = np.array([0.0, 0.022, 0.0])
GRASP_APPROACH_UP_M = 0.02 # precise alignment this far above the grasp height, then descend
DESCEND_STEP_M = 0.003
LIFT_M = 0.03
# The gripper's angle reading drifts relative to the jaws, so grasps are judged in the image:
# pad gap ~265 px open, ~225 px on the plate, pads merged when closed on nothing.
GRIP_HELD_MIN_GAP_PX = 150.0
GRIP_HELD_MAX_GAP_PX = 245.0   # above this the jaws did not really close
PICK_CALIB = HERE / "pick_calib.json"
GRIP_PROFILE = DATA / "grip_profile.json"  # free-air closing torque vs angle (gripcal)
PICK_LEARNED = DATA / "pick_learned.json"  # joint poses and Jacobian that worked, reused next time
VIEW_POSE = {"shoulder_pan": 41.0, "shoulder_lift": -38.0, "elbow_flex": -38.5, "wrist_flex": 51.0,
             "wrist_yaw": 0.0, "wrist_roll": 3.0}  # sees the plate and tag 1 from ~33 cm
PICK_START_MAX_C = 50.0  # was 55: hot (55-61 C) stretches came with holder-wall catches, 2026-11-05 15:53-16:14
PLATE_TASK = "pick up the 96-well plate and put it back"
KPICK_TASK_TABLE = "pick up the 96-well plate and put it down somewhere else on the table"
SHUTTLE_TASK = "move the 96-well plate to the other holder"
HOLDER_TASK = "put the 96-well plate in the holder"  # random goals (kpick rand) that are a holder
HOLDER_B_PRIOR_M = (0.372, 0.352)  # second (open-frame) holder, beyond tag 1 (user, 2026-10-19)
HOLDER_MAP_DZ_M = 0.0025      # map the holder frame halfway up its 5 mm walls (near/far edges differ)
PLACE_TABLE_MAX_DZ_M = 0.006    # on bare table a place contact up to this far above the plan is the table
# holder-A wall catch: retries as (toward the arm base, base +y) mm. 10/20/30 toward the base (user: plate >= 30 mm too far
# out on the rim) seated 0 of 6 tries (2026-11-05 21:48, 21:53); -3 mm in y seated 2 of 4 walls (2026-11-04 23:09, 23:51)
HOLDER_RETRY_OFFSETS_MM = ((0.0, -3.0), (0.0, 3.0), (10.0, 0.0))
HOLDER_SEAT_MAX_DZ_M = 0.0015  # a place contact this far above the planned height: the plate is on a rim
                               # (a 2.4 mm-high contact left it on the first holder's rim, 2026-10-19)
KPICK_B_GRASP_UP_M = 0.001    # grip this much over the measured seat height in the second holder (open pads
                              # below it reach its 5 mm walls)
# Fingertip height (calibrated model, PLANNED joints) for gripping in the second holder; seating is
# KPICK_B_GRASP_UP_M higher. The saved seat joints drifted up with every save (grips at 14.5 -> 15.9 -> 18.7 mm
# measured): gripped that high the plate hangs low and the next place in the first holder caught its rim (+2.4,
# +4.9 mm), and at 17.9 it slipped out. There the measured joints sit ~2.9 mm above the plan, so 11.6 planned is
# the 14.5 mm measured grip of 2026-10-19 16:47, which went back into the first holder cleanly; but 11.5 planned
# put the open pads down on the holder (2026-10-20 14:57), so 12.8, between that and the 17.9 measured slip.
KPICK_B_TIP_Z_M = 0.0128
# Bare-table grasps close at the measured fingertip height (joints): successes 5.5-12.3 mm (median ~8.6), but the
# two no-contact closes (pads over the 15 mm plate top, eps 93, 98, 2026-10-22/23) were at 11.3 and 13.2. Above
# the gate, step down (guarded) to the target before closing.
KPICK_TABLE_TIP_MAX_M = 0.0105
KPICK_TABLE_TIP_TARGET_M = 0.0085
HOLDER_NEAR_M = 0.03          # a plate this close to a holder's pose is in that holder
PLACE_MIN_MOVE_M = 0.05       # a random goal is at least this far from where the plate is
PLACE_YAW_JITTER_DEG = (-20.0, 35.0)  # random turn on top of the radial (wrist-friendly) yaw; -40 is out of
                                      # reach almost everywhere, +40 is fine (zonemap, 2026-10-21); +50 gave ~+80 deg
                                      # plates at the far left, where lone shoulder spikes halted picks twice (2026-10-23)
# Safe zone taped in blue by the user (2026-10-21), inner tape edges from data/zonescan.png; its front edge is
# out of reach, so reach (the IK check in place_ok) bounds it there. Keep-outs: x0, x1, y0, y1 in base (m).
ZONE_X_M, ZONE_Y_MAX_M = (0.175, 0.545), 0.450
# dark mat on the table (2026-10-21, from zonescan); bare-table goals must lie fully on it so the plate never
# straddles its edge (depth: no measurable height step vs the wood)
MAT_X_M, MAT_Y_MIN_M, MAT_MARGIN_M = (0.236, 0.532), 0.292, 0.010
KEEP_OUT_M = {"solid black holder": (0.340, 0.510, -0.20, 0.115),  # floor height unknown: not a goal yet
              "cables": (0.0, 0.200, -0.20, 0.090)}
PLATE_LW_M = (0.1278, 0.0855)
LOOK_DEPTHS_M = (0.33, 0.28, 0.24, 0.38)  # plate-view distances to try, the usual one first
PLACE_CLEAR_M = 0.010         # plate outline + this stays off holders, keep-outs and tag 1 (the outline itself
                              # stays inside the tape; the flat fingers ride 6.5 mm up, over the 5 mm holder walls)
HOLDER_RIM_M = 0.012          # open frames' walls outside the plate outline
HOLDER_CLEAR_M = 0.030        # plate outline to holder frames: the fully open fingers reach ~10 mm + pads past the
                              # plate sides (a finger met holder A's wall picking 10 mm from it, 2026-10-21)
TAG1_HALF_M = 0.030
PLATE_YAW_WINDOW_DEG = 30.0   # plate finds with an expected yaw search only this far either side of it
PLACE_HOLDER_P = 0.0          # chance a random goal is an empty holder (A or B) instead of bare table (0.3; paused 2026-10-22 after A-wall catches, eps 86, 88)
KIN_SPREAD = {"shoulder_pan": 10.0, "shoulder_lift": 8.0, "elbow_flex": 8.0, "wrist_flex": 10.0,
              "wrist_yaw": 15.0, "wrist_roll": 15.0}
KIN_SPREAD_NEAR = {"shoulder_pan": 5.0, "shoulder_lift": 5.0, "elbow_flex": 5.0, "wrist_flex": 8.0,
                   "wrist_yaw": 10.0, "wrist_roll": 10.0}  # around the taught align pose, ~20 cm from tag 1
KIN_NEAR_MIN_TIP_Z = 0.030  # predicted fingertip height over tag 1 (plate top is +15 mm); else skip the pose
PLATE_MAP_RES_M = 0.001      # top-down map resolution for finding the plate without a tag
PLATE_MIN_SCORE, PLATE_MIN_MARGIN = 0.25, 1.15  # lattice match score, and vs the best other hypothesis
PLATE_MIN_SCORE_PRIOR = 0.18  # within 5 mm of a known rough position (bare wood shows through the plate)
PLATE_WEAK_SCORE = 0.18      # a first-view fit this good (in the zone) aims the look poses, which must confirm
PLATE_ROT_MARGIN = 1.05      # without a prior: best vs the same plate turned 90 deg (see plate_find)
ZONE_FIND_MARGIN_M = 0.005   # plate finds ignore the map this far beyond the taped zone (the other arm's side)
HOLDER_A_BIAS_LEARN = False  # off: a +2.0 mm x shift seated the plate at -4.7 mm, not ~0 (2026-11-04 21:12)
# set by hand to (0, +2.5) mm: the user saw the plate ~5 mm too far from AprilTag 1 (-y) on the wall (21:23); the
# seated places read +0.7..+1.5 mm in y
HOLDER_A_BIAS_MAX_MM = 5.0    # learned shift of the holder-A place (holder_a_biased) stays within this
REANCHOR_MIN_MM = 15.0        # release read vs the arm's set-down pose beyond this (and > a 9 mm pitch): use the read
REANCHOR_MAX_SPREAD_MM = 6.0  # ...when its look reads agree this well
PLATE_SCAN_SPREAD = {"shoulder_pan": 6.0, "shoulder_lift": 5.0, "elbow_flex": 5.0, "wrist_flex": 6.0,
                     "wrist_yaw": 10.0, "wrist_roll": 10.0}
KPICK_IK_MAX_MM, KPICK_IK_MAX_DEG = 1.0, 0.5  # IK must hit the mapped poses this closely
KPICK_TIP_DZ_M = 0.003       # mapped grasp/align fingertip heights must match the taught ones this closely
KPICK_ALIGN_ITERS = 4         # close-view alignment moves before closing
KPICK_TABLE_MEASURE_ONLY = False  # True: bare-table picks stop over the plate (view saved), for tuning
KPICK_ALIGN_DEG = 0.3         # accept this much turn left between the plate and the taught view
KPICK_SIL_TURN_MIN_DEG = 2.0  # side edges disagree: turn by the silhouette angle (x KPICK_EDGE_TURN_GAIN) if at least this
GRIP_PRELIFT_MIN_NM = 0.3  # closing torque magnitude the jaws must hold before kpick lifts (held grips -0.62..-0.91)
KPICK_CLOSE_SIDE_GAIN = 0.6  # fraction of the close-view sideways offset (at MM_PER_PX_AT_GRASP) moved per step
KPICK_EDGE_TURN_GAIN = 0.5  # fraction of the side-edge angle turned per close-view move
KPICK_TURN_IN_PLACE_DEG = 1.0  # bigger close-view turns on bare table are made LIFT_M up (finger ends swing ~1 mm/deg)
CARRY_PLAN_MAX_JUMP_DEG = 7.5  # random goals whose planned carry needs a bigger IK step are drawn again (move_cart aborts at 8)
KPICK_CARRY_UP_M = 0.05       # carry the plate this high over the table (holder walls are 5 mm)
# Fast replay of a learned pick: transit at FAST_DEG_S, the last few cm onto the plate slower.
FAST_DEG_S = {**dict.fromkeys(("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw",
                                "wrist_roll"), 30.0), "gripper": 120.0}
NEAR_DEG_S = {**dict.fromkeys(FAST_DEG_S, 10.0), "gripper": 120.0}
FASTPICK_START_MAX_C = 62.0  # a fast pick spends ~20 s out there instead of minutes
REPLAY_MAX_ERR_MM = 2.0      # one-shot visual correction above this tag-pose error at pregrasp    # do not start a pick with a warmer shoulder (it heats ~8 C/min out there)
CLOSED_TIP_CAM = np.array([0.0083, -0.0176, 0.147])  # closed jaw tips, camera frame (z: stereo +9 mm like the open tips)
WRIST_FOLDED_MAX_DEG = 2.0  # wrist_flex stop near the folded pose (pushing past it trips a stall fault)
SERVO_SPEED_DEG_S = 10.0
CAN_STALL_S = 0.5             # a motor-bus call blocked this long = the CAN adapter stalled: reset the link
CAN_RESETS_MAX = 3            # the 3rd try is a USB reset of the adapter; still blocked after it = link dead
STREAM_MOTION = True          # free-air moves, carries, big joint moves and rest as one continuous streamed motion
STREAM_GUARD_DT_S = 0.1       # streamed guard: torque read this often...
STREAM_GUARD_WINDOW_S = 0.5   # ...vs the torques this long before (rolling baseline)
# The motors are position springs with no gravity feed-forward, so the stretched-out arm sags
# (~3 deg per joint at the stock kp) and visibly dips after every new command. Stiffer springs on
# the gravity-loaded joints cut the sag; MIT gains are sent with every command, so this is live.
HOLD_KP = {"shoulder_lift": 250.0, "elbow_flex": 250.0, "wrist_flex": 90.0}
HOLD_KD = {"shoulder_lift": 14.0, "elbow_flex": 14.0, "wrist_flex": 7.0}
# Commands may lead the measured position by this much (gravity sag of ~6 deg plus one step).
MAX_RELATIVE_TARGET_DEG = 15.0


def planar_pose(T, z: float):
    """T as a pure table-plane pose (yaw only) at height z."""
    a = math.atan2(T[1][0], T[0][0])
    P = np.eye(4)
    P[:3, :3] = [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
    P[:2, 3] = T[0][3], T[1][3]
    P[2, 3] = z
    return P


def yaw_deg(T) -> float:
    """Yaw of a base-frame pose about z (deg)."""
    return math.degrees(math.atan2(T[1][0], T[0][0]))


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


class Abort(Exception):
    pass


class TagLost(Abort):
    """Tag not detected in enough frames; carries diagnostics about the attempt."""

    def __init__(self, msg: str, stats: dict, frames: list):
        super().__init__(msg)
        self.stats = stats
        self.frames = frames


# ---------------------------------------------------------------- camera


class WristCamera:
    def __init__(self):
        self.device = self._open()
        cal = self.device.readCalibration()
        self.K = np.array(cal.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, W, H))
        self.dist = np.array(cal.getDistortionCoefficients(dai.CameraBoardSocket.CAM_B))
        self.pipeline = dai.Pipeline(self.device)
        left = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        right = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        self.color_q = left.requestOutput((W, H), type=dai.ImgFrame.Type.BGR888i, fps=FPS).createOutputQueue(2, False)
        stereo = self.pipeline.create(dai.node.StereoDepth).build(
            left.requestOutput((W, H), fps=FPS), right.requestOutput((W, H), fps=FPS))
        stereo.setExtendedDisparity(True)
        stereo.setLeftRightCheck(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_B)
        self.depth_q = stereo.depth.createOutputQueue(2, False)
        self.pipeline.start()
        self.lock = threading.Lock()
        self.frame = None
        self.depth = None
        self.seq = 0
        self.stamp = 0.0
        self.error: str | None = None
        self.overlay: list = []
        self.stop = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    @staticmethod
    def _open() -> dai.Device:
        for i in range(20):
            try:
                return dai.Device(dai.DeviceInfo(WRIST_ID))
            except RuntimeError:
                if i == 19:
                    raise
                time.sleep(1.0)

    def _loop(self):
        last_preview = 0.0
        while not self.stop.is_set():
            try:
                msg = self.color_q.get(datetime.timedelta(milliseconds=1000))
                if msg is None:
                    continue
                frame = msg.getCvFrame()
                dmsg = self.depth_q.tryGet()
                with self.lock:
                    self.frame = frame
                    if dmsg is not None:
                        self.depth = dmsg.getFrame()
                    self.seq += 1
                    self.stamp = time.monotonic()
                if time.monotonic() - last_preview > 0.2:
                    last_preview = time.monotonic()
                    self._write_preview(frame)
            except Exception as e:
                self.error = str(e)
                log(f"camera error: {e}")
                return

    def _write_preview(self, frame):
        img = frame.copy()
        for kind, data in list(self.overlay):
            if kind == "tag":
                cv2.polylines(img, [np.int32(data)], True, (0, 255, 0), 2)
            elif kind == "tip":
                cv2.drawMarker(img, tuple(int(v) for v in data), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
        try:
            PREVIEW.parent.mkdir(parents=True, exist_ok=True)
            tmp = PREVIEW.with_suffix(".tmp")
            tmp.write_bytes(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
            os.replace(tmp, PREVIEW)
        except OSError:
            pass

    def latest(self, after_seq: int = -1, timeout: float = 2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.error:
                raise Abort(f"wrist camera failed: {self.error}")
            with self.lock:
                if self.frame is not None and self.seq > after_seq:
                    return self.seq, self.frame.copy(), None if self.depth is None else self.depth.copy()
            time.sleep(0.01)
        raise Abort("no new wrist camera frame for 2 s")

    def close(self):
        self.stop.set()
        try:
            self.pipeline.stop()
        finally:
            self.device.close()


# ---------------------------------------------------------------- tag measurement


# Pixels covered by the gripper fingers in the wrist image (static between poses); features
# there move with the camera, so the occlusion tracker ignores them.
GRIPPER_BOXES = ((295, 0, 420, 165), (0, 360, 275, 400),  # x0, y0, x1, y1 (closed fingers)
                 (120, 0, 600, 162))                        # open fingers (gripper < -250)
TRACK_MIN_INLIERS = 12


def gripper_mask() -> np.ndarray:
    m = np.full((H, W), 255, np.uint8)
    for x0, y0, x1, y1 in GRIPPER_BOXES:
        m[y0:y1, x0:x1] = 0
    return m


def detect_corners(gray):
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
                                  cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None or ACTIVE_TAG["id"] not in ids.ravel():
        return None
    return corners[list(ids.ravel()).index(ACTIVE_TAG["id"])].reshape(4, 2)


def pose_from_corners(c, K, dist, prev_normal=None, size=None):
    h = (ACTIVE_TAG["size"] if size is None else size) / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)
    n_sol, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, np.asarray(c, np.float64), K, dist,
                                                    flags=cv2.SOLVEPNP_IPPE_SQUARE)
    best = None
    for rvec, tvec, err in zip(rvecs, tvecs, np.ravel(errs)):
        R = cv2.Rodrigues(rvec)[0]
        n = R[:, 2]
        score = err if prev_normal is None else float(np.linalg.norm(n - prev_normal))
        if best is None or score < best[0]:
            best = (score, tvec.ravel(), n, R)
    return {"p": best[1], "n": best[2], "R": best[3], "corners": np.asarray(c, np.float64)}


def detect(frame, K, dist, prev_normal=None):
    c = detect_corners(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    return None if c is None else pose_from_corners(c, K, dist, prev_normal)


def track_corners(ref_gray, ref_corners, gray):
    """Carry the tag corners from a reference frame into `gray` when the tag is partly covered
    (by the gripper, near the end of an approach): KLT-track features on the tag and the paper
    around it, fit a RANSAC homography, map the corners. Returns (corners, inliers) or None."""
    centre = ref_corners.mean(axis=0)
    region = np.zeros_like(ref_gray)
    cv2.fillConvexPoly(region, np.int32(centre + 4.0 * (ref_corners - centre)), 255)
    region &= gripper_mask()
    pts = cv2.goodFeaturesToTrack(ref_gray, 400, 0.01, 5, mask=region)
    if pts is None or len(pts) < TRACK_MIN_INLIERS:
        return None
    lk = dict(winSize=(21, 21), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(ref_gray, gray, pts, None, **lk)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, ref_gray, nxt, None, **lk)
    ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm((back - pts).reshape(-1, 2), axis=1) < 1.0)
    inside = gripper_mask()[np.clip(nxt[:, 0, 1], 0, H - 1).astype(int), np.clip(nxt[:, 0, 0], 0, W - 1).astype(int)] > 0
    ok &= inside
    if ok.sum() < TRACK_MIN_INLIERS:
        return None
    Hm, inl = cv2.findHomography(pts[ok], nxt[ok], cv2.RANSAC, 2.0)
    if Hm is None or int(inl.sum()) < TRACK_MIN_INLIERS:
        return None
    c = cv2.perspectiveTransform(ref_corners.reshape(-1, 1, 2).astype(np.float64), Hm).reshape(4, 2)
    if not cv2.isContourConvex(np.float32(c)) or cv2.contourArea(np.float32(c)) < 100:
        return None
    return c, int(inl.sum())


class Measurer:
    def __init__(self, cam: WristCamera):
        self.cam = cam
        self.prev_normal = None
        self.seq = -1
        self.ref = None  # (gray, corners) of the last good measurement, for occlusion tracking

    def measure(self, frames: int = 5, timeout_s: float | None = None) -> dict:
        ps, ns, Rs, last, detected_ref = [], [], [], None, None
        seen, sharp, recent, tracked, inliers = 0, [], [], 0, []
        ref = getattr(self, "ref", None)
        deadline = time.monotonic() + (TAG_RETRY_S if timeout_s is None else timeout_s)
        while len(ps) < frames and time.monotonic() < deadline:
            # The background hold thread is blocked while a command runs; keep the motors fed.
            if getattr(self, "tick", None):
                self.tick()
            self.seq, frame, depth = self.cam.latest(self.seq)
            seen += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharp.append(float(cv2.Laplacian(gray[H // 4:3 * H // 4, W // 4:3 * W // 4], cv2.CV_64F).var()))
            recent = (recent + [frame])[-3:]
            c = detect_corners(gray)
            was_detected = c is not None
            if c is None and ref is not None:
                t = track_corners(ref[0], ref[1], gray)
                if t is not None:
                    c = t[0]
                    tracked += 1
                    inliers.append(t[1])
            if c is not None:
                d = pose_from_corners(c, self.cam.K, self.cam.dist, self.prev_normal)
                ps.append(d["p"]); ns.append(d["n"]); Rs.append(d["R"]); last = (d, frame, depth, gray)
                if was_detected:
                    detected_ref = (gray, c)
        if len(ps) < 3:
            stats = {"frames": seen, "detections": len(ps), "tracked": tracked,
                     "sharpness_median": round(float(np.median(sharp)), 1) if sharp else None,
                     "sharpness_min": round(min(sharp), 1) if sharp else None}
            raise TagLost(f"tag not visible in the wrist camera ({len(ps)}/{seen} frames)", stats, recent)
        p = np.median(ps, axis=0)
        n = np.median(ns, axis=0)
        n /= np.linalg.norm(n)
        self.prev_normal = n
        # Track only from frames where the detector itself saw the tag: chaining tracked frames
        # drifted in scale (reported 190 mm while the tag was 240 mm away).
        if detected_ref is not None:
            self.ref = detected_ref
        self.cam.overlay = [o for o in self.cam.overlay if o[0] != "tag"] + [("tag", last[0]["corners"])]
        U, _, Vt = np.linalg.svd(np.mean(Rs, axis=0))  # mean rotation, re-orthonormalised
        R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
        return {"p": p, "n": n, "R": R, "corners": last[0]["corners"], "frame": last[1],
                "depth": last[2], "tracked": tracked, "count": len(ps),
                "inliers": int(np.median(inliers)) if inliers else None}


PAD_WIDTHS_PX = 40.0


def pad_gap_px(frame) -> float | None:
    """Pixel gap between the inner edges of the two dark finger pads, just above the jaw tips
    (each pad sits against its white finger on the outside). None when the jaws are closed
    (the pads meet in the middle) or the pads cannot be found."""
    band = np.median(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[95:140, :].astype(float), axis=0)
    gap = _pad_gap_at(band, 70)
    # In some light a pad reads 70-80 gray (2026-10-20: right pad missed on the plate -> false
    # halt); retry looser only when the strict pass found nothing.
    return gap if gap is not None else _pad_gap_at(band, 90)


def _pad_gap_at(band, dark_below: float) -> float | None:
    dark = band < dark_below
    runs, x = [], 0
    while x < W:
        if dark[x]:
            s0 = x
            while x < W and dark[x]:
                x += 1
            if x - s0 >= 6:
                runs.append((s0, x - 1))
        x += 1
    # the white finger beside the pad, relative to the pad (absolute brightness varies with room light)
    def whiter(seg, r):
        return seg.size and seg.max() > band[r[0]:r[1] + 1].mean() + 80
    left = [r for r in runs if 150 <= r[0] and r[1] < 345 and whiter(band[max(0, r[0] - 14):r[0] - 1], r)]
    right = [r for r in runs if r[0] > 345 and r[1] <= 570 and whiter(band[r[1] + 1:r[1] + 14], r)]
    if not left or not right:
        return None
    # Measured between the pads' outer (white finger -> dark pad) edges: the inner edges merge with
    # dark background behind the jaws. PAD_WIDTHS_PX makes it roughly the inner gap again.
    return float(min(r[1] for r in right) - max(r[0] for r in left) - PAD_WIDTHS_PX)


def rotvec(R: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(R, np.float64))[0].ravel()


def y_vec(m: dict) -> np.ndarray:
    """Task vector: tag position (m) and weighted tag-normal tilt."""
    return np.concatenate([m["p"], ROT_WEIGHT_M * m["n"][:2]])


# ---------------------------------------------------------------- controller


class Agent:
    def __init__(self):
        mj.SPEED_DEG_S = {**dict.fromkeys(mj.B601_JOINTS, SERVO_SPEED_DEG_S), "gripper": 30.0}
        log("opening wrist camera")
        self.cam = WristCamera()
        self.meas = Measurer(self.cam)
        log("connecting follower (limp)")
        self.follower = connect_limp_follower(os.environ.get("B601_CAN", "can0"), "b601_rs")  # Jetson: canb601
        self.follower.config.max_relative_target = MAX_RELATIVE_TARGET_DEG
        self.install_can_watchdog()
        self.apply_gains()
        self.s = Session(None, self.follower, None)
        self.meas.tick = self.s.tick
        self.apply_limits()
        self.limits = self.follower.config.joint_limits
        self.tip: np.ndarray | None = None
        self.J: np.ndarray | None = None
        self.n0: np.ndarray | None = None
        self.frozen = False
        self.stage = "idle"
        self.note = ""
        self.path: list[dict[str, float]] = []  # joint poses visited, retraced by `rest`
        self.write_status()

    # ---- helpers
    def apply_limits(self):
        self.install_recording_hooks()
        # The configured gripper range (-270) is narrower than the jaws' real travel; the plate
        # (85.5 mm) needs about -310. The user opened it by hand to -324.
        self.follower.config.joint_limits["gripper"] = GRIPPER_LIMITS
        self.follower._joint_limits["gripper"] = GRIPPER_LIMITS
        self.apply_grip_limits()
        self.install_gripper_unwrap()

    def install_gripper_unwrap(self):
        """Apply self.grip_off (0 or -360 deg) to every gripper position read and command (see GRIP_WRAP_DEG).
        The originals stay on the follower, so a `reload` re-installs without stacking."""
        f = self.follower
        if not hasattr(f, "_grip_orig"):
            f._grip_orig = (f._motor_to_public_position, f._public_to_motor_position)
        m2p, p2m = f._grip_orig
        self.grip_off = getattr(self, "grip_off", 0.0)

        def to_public(name, deg):
            v = m2p(name, deg)
            return v + self.grip_off if name == "gripper" else v

        def to_motor(name, deg):
            return p2m(name, deg - self.grip_off if name == "gripper" else deg)
        f._motor_to_public_position, f._public_to_motor_position = to_public, to_motor

    def gripper_raw(self, state=None) -> float:
        """The gripper angle in the motor's own (possibly wrapped) frame, before grip_off."""
        st = state if state is not None else self.follower._read_feedback()["gripper"]
        return self.follower._grip_orig[0]("gripper", math.degrees(st.pos))

    def set_grip_frame(self, raw: float, why: str):
        off = -360.0 if raw > GRIP_WRAP_DEG else 0.0
        if off != self.grip_off:
            log(f"gripper reads {raw:+.1f} deg {why}: motor frame {'wrapped' if off else 'unwrapped'}; "
                f"jaws at {raw + off:.1f} deg (offset {off:+.0f})")
        self.grip_off = off

    def check_gripper_reading(self, g: float):
        lo, hi = GRIPPER_LIMITS
        if not lo - 20.0 <= g <= hi + GRIP_WRAP_DEG:
            raise Abort(f"gripper reading {g:.1f} deg is outside its travel ({lo:.0f}..{hi:.0f}); its motor frame "
                        f"may have wrapped while enabled - not moving")

    # ---- dataset recording (LeRobot format, for policy training; see record.py)
    def install_recording_hooks(self):
        """Sample a frame after every motion command and hold refresh (idempotent across reloads)."""
        s, agent = self.s, self

        def command(target):
            Session.command(s, target)
            agent.record_sample()

        def tick():
            Session.tick(s)
            agent.record_sample()

        s.command, s.tick = command, tick

    def record_sample(self):
        rec = getattr(self, "rec", None)
        if rec is None or not rec.active or not self.s.hold:
            return
        try:
            states = {k: m.get_state() for k, m in self.follower.motors.items()}  # cached: no bus traffic
            if any(v is None for v in states.values()):
                return
            rec.sample(self.follower._present_pos(states), self.s.hold, {"wrist": self.cam.latest()[1]})
        except Exception as e:  # recording must never disturb control
            log(f"recording error (frame skipped): {e}")

    def rec_start(self, task: str, **info):
        if getattr(self, "rec", None) is not None:
            self.rec.start(task, **info)

    def rec_stop(self, success: bool):
        rec = getattr(self, "rec", None)
        if rec is not None and rec.active:
            n = rec.stop(success)
            log(f"recorded episode: {n} frames, {'success' if success else 'failure'} ({len(rec.outcomes)} episodes in {rec.root.name})")

    def rec_note(self, **info):
        """Add facts learned after an episode closed (e.g. the measured place error) to its outcome."""
        rec = getattr(self, "rec", None)
        if rec is not None and rec.outcomes:
            rec.outcomes[-1].update(info)
            rec.outcomes_path.write_text(json.dumps(rec.outcomes, indent=1))

    def holder_b(self):
        path = DATA / "holder_b.json"
        if not path.exists():
            raise Abort("no pose for the second holder: run `holderlook` with it empty")
        return json.loads(path.read_text())

    def other_holder(self) -> str:
        """The holder the plate is not in (from where it was last seen)."""
        now = np.array(json.loads((DATA / "plate_pose.json").read_text())["T"])[:2, 3]
        T_ref = np.array(json.loads((DATA / "plate_ref.json").read_text())["T"])
        T_b = np.array(self.holder_b()["T"])
        if np.linalg.norm(now - T_ref[:2, 3]) < HOLDER_NEAR_M:
            return "B"
        if np.linalg.norm(now - T_b[:2, 3]) < HOLDER_NEAR_M:
            return "home"
        raise Abort(f"the plate was last seen at ({now[0] * 1000:.0f}, {now[1] * 1000:.0f}) mm, in neither holder")

    def holder_find(self, tag: str, prior):
        """Find the empty second holder (open black frame) near prior (base x, y) in the current view.
        -> its pose as a plate pose (origin on the plate top at the opening's centre, z up) or None."""
        sys.path.insert(0, str(HERE / "kin"))
        import birdseye
        import holder_detect
        self.hold(SETTLE_S)
        _, frame, _ = self.cam.latest()
        q = self.joints()
        k = self.kin()
        z_tab = k.tags[1][2, 3]
        res, h = 0.0005, 0.15
        img, to_base = birdseye.plane_map(frame, self.cam.K, self.cam.dist, k.cam(q), z_tab + HOLDER_MAP_DZ_M,
                                          (prior[0] - h, prior[0] + h), (prior[1] - h, prior[1] + h), res)
        d = holder_detect.find(img, res, (h / res, h / res))
        stamp = time.strftime("%H%M%S")
        cv2.imwrite(str(DATA / f"holderfind{tag}_{stamp}.png"), frame)
        (DATA / f"holderfind{tag}_{stamp}.json").write_text(json.dumps(q))
        if d is None:
            log("holder_find: no empty holder there")
            return None
        x, y = to_base(*d["c"])
        a = math.radians(d["angle_deg"])
        T = np.eye(4)
        T[:3, :3] = [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
        T[:3, 3] = (x, y, z_tab + PLATE_TOP_Z)
        log(f"HOLDER at ({x * 1000:.1f}, {y * 1000:.1f}) mm, {d['angle_deg']:+.2f} deg, opening "
            f"{d['size_m'][0] * 1000:.1f} x {d['size_m'][1] * 1000:.1f} mm (fill {d['fill']:.2f})")
        return T

    def holderlook(self, n: int = 3):
        """Measure the empty second holder from look poses like platelook's; saves data/holder_b.json."""
        self.require_motion_ok()
        if not self.s.torque:
            self.safe_enable()
        if self.joints()["shoulder_lift"] > -30.0:
            self.move_segmented({**VIEW_POSE, "wrist_flex": 0.0}, "unfold")
        self.move_segmented(VIEW_POSE, "view")
        path = DATA / "holder_b.json"
        p = np.array([*(np.array(json.loads(path.read_text())["T"])[:2, 3] if path.exists() else HOLDER_B_PRIOR_M),
                      self.kin().tags[1][2, 3] + PLATE_TOP_Z])
        found = []
        rng = np.random.default_rng()
        for i in range(n):
            self.thermal_check()
            px = (325.0 + rng.uniform(-40, 40), 260.0 + rng.uniform(-15, 15)) if i else None
            self.look_at_plate(p, f"look {i}", px=px)
            T = self.holder_find(f"_look{i}", p[:2])
            if T is not None:
                found.append(T)
                if i == 0:
                    p = T[:3, 3]
        if not found:
            raise Abort("holderlook: no empty holder found")
        xy = np.array([T[:2, 3] for T in found]) * 1000
        ang = np.array([math.degrees(math.atan2(T[1, 0], T[0, 0])) for T in found])
        log(f"HOLDERLOOK: {len(found)}/{n} found; mean ({xy[:, 0].mean():.1f}, {xy[:, 1].mean():.1f}) mm "
            f"{ang.mean():+.2f} deg; spread {np.ptp(xy[:, 0]):.1f} x {np.ptp(xy[:, 1]):.1f} mm, {np.ptp(ang):.2f} deg")
        T = found[0].copy()
        a = math.radians(ang.mean())
        T[:3, :3] = [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
        T[:2, 3] = xy.mean(0) / 1000
        path.write_text(json.dumps({"T": T.tolist(), "source": "frame", "time": time.strftime("%Y-%m-%d %H:%M:%S")}))
        L = self.learned()
        if "holderB_grasp_q" in L:  # taught for the old pose
            L.pop("holderB_grasp_q")
            PICK_LEARNED.write_text(json.dumps(L, indent=1))
        return T

    def zonescan(self, x_range=(0.15, 0.60), y_range=(0.0, 0.55), step=0.10, res=0.002):
        """Look at a grid of table points and stitch a top-down mosaic of the table plane (base frame,
        1 px = res; each pixel from the view whose optical axis hits nearest it): data/zonescan.png/.json."""
        sys.path.insert(0, str(HERE / "kin"))
        import birdseye
        self.require_motion_ok()
        if not self.s.torque:
            self.safe_enable()
        if self.joints()["shoulder_lift"] > -30.0:
            self.move_segmented({**VIEW_POSE, "wrist_flex": 0.0}, "unfold")
        self.move_segmented(VIEW_POSE, "view")
        k = self.kin()
        z_tab = k.tags[1][2, 3]
        xs = np.arange(x_range[0], x_range[1] + 1e-9, step)
        ys = np.arange(y_range[0], y_range[1] + 1e-9, step)
        nx, ny = int(round((x_range[1] - x_range[0]) / res)), int(round((y_range[1] - y_range[0]) / res))
        mosaic = np.zeros((ny, nx, 3), np.uint8)
        best = np.full((ny, nx), np.inf)
        X, Y = np.meshgrid(x_range[0] + np.arange(nx) * res, y_range[1] - np.arange(ny) * res)
        views = []
        for i, x in enumerate(xs):
            for y in (ys if i % 2 == 0 else ys[::-1]):
                self.thermal_check()
                try:
                    # usual depth only: the closer LOOK_DEPTHS_M views of spots near the base stall the wrist
                    # (wrist_flex fault 16 at (150, 300), twice on 2026-10-21)
                    self.look_at_plate(np.array([x, y, z_tab]), f"scan ({x * 1000:.0f}, {y * 1000:.0f})",
                                       depth=LOOK_DEPTHS_M[0])
                except Abort as e:
                    if "no reachable view" not in str(e):  # a fault or contact: stop (it kept going, 2026-10-21)
                        raise
                    log(f"zonescan: skip ({x * 1000:.0f}, {y * 1000:.0f}): {e}")
                    continue
                self.hold(SETTLE_S)
                _, frame, _ = self.cam.latest()
                q = self.joints()
                C = k.cam(q)
                ray = C[:3, 2]
                hit = C[:3, 3] + ray * (z_tab - C[2, 3]) / ray[2]
                img, _ = birdseye.plane_map(frame, self.cam.K, self.cam.dist, C, z_tab, x_range, y_range, res)
                d = np.hypot(X - hit[0], Y - hit[1])
                use = (img.max(2) > 0) & (d < best)
                mosaic[use], best[use] = img[use], d[use]
                views.append({"aim": [float(x), float(y)], "q": q})
        cv2.imwrite(str(DATA / "zonescan.png"), mosaic)
        (DATA / "zonescan.json").write_text(json.dumps({"x_range": x_range, "y_range": y_range, "res": res,
                                                        "z_tab": float(z_tab), "views": views,
                                                        "time": time.strftime("%Y-%m-%d %H:%M:%S")}))
        log(f"ZONESCAN: {len(views)} views stitched into data/zonescan.png")

    @staticmethod
    def plate_outline(x, y, yaw_deg, grow=0.0):
        """Corners (4x2, m) of the plate footprint at (x, y, yaw), grown by `grow` all round."""
        a = math.radians(yaw_deg)
        u, v = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
        hl, hw = PLATE_LW_M[0] / 2 + grow, PLATE_LW_M[1] / 2 + grow
        return np.array([[x, y] + sx * hl * u + sy * hw * v for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))])

    @staticmethod
    def polys_overlap(P, Q) -> bool:
        """Convex polygons overlap (separating axis test)."""
        for poly in (P, Q):
            for i in range(len(poly)):
                e = poly[(i + 1) % len(poly)] - poly[i]
                n = np.array([-e[1], e[0]])
                if (P @ n).max() < (Q @ n).min() or (Q @ n).max() < (P @ n).min():
                    return False
        return True

    def place_ok(self, x, y, yaw, T_ref, L, k, ignore=(), geo=True, reach=True) -> str | None:
        """None if a plate at (x, y, yaw) is a safe (geo), reachable (reach) goal and pick start on bare table,
        else why not."""
        P0 = self.plate_outline(x, y, yaw)
        P = self.plate_outline(x, y, yaw, PLACE_CLEAR_M)
        if geo and (P0[:, 0].min() < ZONE_X_M[0] or P0[:, 0].max() > ZONE_X_M[1] or P0[:, 1].max() > ZONE_Y_MAX_M):
            return "outside the taped zone"
        if geo and (P0[:, 0].min() < MAT_X_M[0] + MAT_MARGIN_M or P0[:, 0].max() > MAT_X_M[1] - MAT_MARGIN_M
                    or P0[:, 1].min() < MAT_Y_MIN_M + MAT_MARGIN_M):
            return "off the mat"
        if geo and (r := self.place_clash(P, T_ref, k, ignore, Ph=self.plate_outline(x, y, yaw, HOLDER_CLEAR_M))):
            return r
        return self.place_reach(x, y, yaw, T_ref, L, k) if reach else None

    def place_clash(self, P, T_ref, k, ignore=(), Ph=None) -> str | None:
        for name, (x0, x1, y0, y1) in KEEP_OUT_M.items():
            if self.polys_overlap(P, np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])):
                return name
        holders = {"A": T_ref}
        if (DATA / "holder_b.json").exists():
            holders["B"] = np.array(self.holder_b()["T"])
        for name, T in holders.items():
            if name in ignore:
                continue
            a = math.degrees(math.atan2(T[1, 0], T[0, 0]))
            if self.polys_overlap(P if Ph is None else Ph, self.plate_outline(T[0, 3], T[1, 3], a, HOLDER_RIM_M)):
                return f"holder {name}"
        t = k.tags[1][:2, 3]
        if self.polys_overlap(P, np.array([t + [sx * TAG1_HALF_M, sy * TAG1_HALF_M]
                                           for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))])):
            return "tag 1"
        return None

    def place_reach(self, x, y, yaw, T_ref, L, k) -> str | None:
        sys.path.insert(0, str(HERE / "kin"))
        import lookdown
        T = np.eye(4)
        a = math.radians(yaw)
        T[:3, :3] = [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
        T[:3, 3] = (x, y, T_ref[2, 3])
        T = self.plate_side(T, T_ref)
        D = T @ np.linalg.inv(T_ref)
        dyaw = math.degrees(math.atan2(D[1, 0], D[0, 0]))
        try:
            for n in ("flat_pregrasp_joints", "flat_ref_align_q", "flat_lift_q", "flat_grasp_q"):
                C = D @ k.cam(L[n])
                q = self.ik_cam(C, {**L[n], "shoulder_pan": L[n]["shoulder_pan"] + dyaw}, n, k)
                if n == "flat_grasp_q":
                    C[2, 3] += KPICK_CARRY_UP_M
                    self.ik_cam(C, q, "carry", k)
        except Abort as e:
            return str(e)
        _, ep, er = self.look_solve(k, T[:3, 3], VIEW_POSE)
        if ep > 2.0 or er > 1.0:
            return "no view of it"
        return None

    def random_place(self):
        """-> "home" / "B" (an empty holder, PLACE_HOLDER_P of the time) or (x, y, yaw) [m, m, deg]: a random
        reachable spot on bare table in the taped zone, away from where the plate is now."""
        import random
        T_ref = np.array(json.loads((DATA / "plate_ref.json").read_text())["T"])
        T_now = np.array(json.loads((DATA / "plate_pose.json").read_text())["T"])
        now = T_now[:2, 3]
        holders = [("home", T_ref)] + ([("B", np.array(self.holder_b()["T"]))] if (DATA / "holder_b.json").exists() else [])
        empty = [h for h, T in holders if np.linalg.norm(now - T[:2, 3]) >= HOLDER_NEAR_M]
        if empty and random.random() < PLACE_HOLDER_P:
            return random.choice(empty)
        L, k = self.learned(), self.kin()
        why = {}
        for _ in range(2000):  # on the mat few draws fit (400 all missed once, 2026-10-21)
            xy = np.array([random.uniform(*MAT_X_M), random.uniform(MAT_Y_MIN_M, ZONE_Y_MAX_M)])  # goals are on the mat
            if np.linalg.norm(xy - now) < PLACE_MIN_MOVE_M:
                continue
            yaw = self.radial_yaw(T_ref, xy) + random.uniform(*PLACE_YAW_JITTER_DEG)
            r = self.place_ok(xy[0], xy[1], yaw, T_ref, L, k)
            if r is None:
                r = self.carry_path_ok(T_now, xy[0], xy[1], yaw, T_ref, L, k)
                if r:
                    log(f"random_place: ({xy[0] * 1000:.0f}, {xy[1] * 1000:.0f}) mm, {yaw:+.1f} deg rejected: {r}")
            if r is None:
                return float(xy[0]), float(xy[1]), float(yaw)
            r = r.split(":")[0]
            why[r] = why.get(r, 0) + 1
        raise Abort(f"random_place: no safe reachable spot found ({why})")

    def cart_path_ok(self, C0, C1, seed: dict, k, step_m: float) -> tuple[str | None, dict]:
        """Plan move_cart's IK waypoints from C0 to C1 without moving: (None, end joints) or (why, seed)."""
        sys.path.insert(0, str(HERE / "kin"))
        import model
        w = model.log_rot(C0[:3, :3].T @ C1[:3, :3])
        n = max(1, int(math.ceil(max(np.linalg.norm(C1[:3, 3] - C0[:3, 3]) / step_m, np.degrees(np.linalg.norm(w)) / 3.0))))
        for i in range(1, n + 1):
            Ci = np.eye(4)
            Ci[:3, :3] = C0[:3, :3] @ model.rot_vec(w * i / n)
            Ci[:3, 3] = C0[:3, 3] + i / n * (C1[:3, 3] - C0[:3, 3])
            try:
                qi = self.ik_cam(Ci, seed, "plan", k)
            except Abort as e:
                return str(e), seed
            jump = max(abs(qi[j] - seed[j]) for j in ARM6)
            if jump > CARRY_PLAN_MAX_JUMP_DEG:
                return f"carry plan: IK jumps {jump:.1f} deg", seed
            seed = qi
        return None, seed

    def carry_path_ok(self, T_now, x, y, yaw, T_ref, L, k) -> str | None:
        """Plan kpick's moves with the plate (up from the grasp, carry, down to the place) for a plate at T_now and
        a goal (x, y m, yaw deg). A carry that needed an 8.6 deg IK jump froze the arm holding the plate 30 mm over
        the mat (goal (385, 395) mm, 2026-10-30 15:44); such goals are drawn again instead."""
        D = T_now @ np.linalg.inv(T_ref)
        q = {**L["flat_grasp_q"], "shoulder_pan": L["flat_grasp_q"]["shoulder_pan"] + yaw_deg(D)}
        C_g = D @ k.cam(L["flat_grasp_q"])
        try:
            q = self.ik_cam(C_g, q, "plan", k)
        except Abort as e:
            return str(e)
        a = math.radians(yaw)
        T_goal = np.eye(4)
        T_goal[:3, :3] = [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
        T_goal[:3, 3] = (x, y, T_now[2, 3])
        C_place = T_goal @ np.linalg.inv(T_now) @ C_g
        C_up, C_carry, C_low = C_g.copy(), C_place.copy(), C_place.copy()
        C_up[2, 3] += KPICK_CARRY_UP_M
        C_carry[2, 3] += KPICK_CARRY_UP_M
        C_low[2, 3] += LIFT_M
        for C0, C1, step in ((C_g, C_up, 0.01), (C_up, C_carry, 0.02), (C_carry, C_low, 0.01), (C_low, C_place, 0.01)):
            why, q = self.cart_path_ok(C0, C1, q, k, step)
            if why:
                return why
        return None

    def apply_grip_limits(self):
        self.follower.config.gripper_torque_limit = GRIP_TORQUE_LIMIT_NM
        self.follower.config.gripper_hold_torque_limit = GRIP_HOLD_TORQUE_LIMIT_NM

    def apply_gains(self):
        self.follower._mit_kp.update(HOLD_KP)
        self.follower._mit_kd.update(HOLD_KD)

    # ---- CAN watchdog: the USB CAN adapter can stop draining its queue (222 frames stuck, no kernel error,
    # 2026-11-04 14:04); the next motor command then blocked forever while the motors held the last pose and
    # heated for 7 min until the user cut the arm's power. Every bus call is timed; a stalled one resets the link
    # (down/up clears the adapter, tested 14:14), and the action in progress is aborted (rest still runs).
    def install_can_watchdog(self):
        f, agent = self.follower, self
        self.bus_t0, self.bus_op, self.can_fault, self.can_dead = None, "", False, False
        for name in ("send_action", "_read_feedback", "_present_pos"):
            orig = getattr(f, name)

            def timed(*a, _orig=orig, _name=name, **kw):
                outer = agent.bus_t0 is None
                if outer:
                    agent.bus_t0, agent.bus_op = time.monotonic(), _name
                try:
                    return _orig(*a, **kw)
                finally:
                    if outer:
                        agent.bus_t0 = None
            setattr(f, name, timed)
        threading.Thread(target=self.can_watchdog_loop, name="can-watchdog", daemon=True).start()

    def can_link_reset(self) -> str:
        import subprocess
        dev = os.environ.get("B601_CAN", "can0")
        try:
            r1 = subprocess.run(["ip", "link", "set", dev, "down"], capture_output=True, text=True, timeout=5)
            r2 = subprocess.run(["ip", "link", "set", dev, "up", "type", "can", "bitrate", "1000000"],
                                capture_output=True, text=True, timeout=5)
            return "ok" if r1.returncode == 0 and r2.returncode == 0 else (r1.stderr + r2.stderr).strip()[:150]
        except Exception as e:
            return f"failed: {e}"

    def can_usb_reset(self) -> str:
        """Second step after link resets fail: a USB-level reset of the CAN adapter (as unplugging it), then the
        link brought up again (udev names it canb601 again; host network namespace)."""
        import fcntl
        import subprocess
        dev = os.environ.get("B601_CAN", "can0")
        try:
            usb = os.path.realpath(f"/sys/class/net/{dev}/device/..")
            bus, num = (int(open(f"{usb}/{f}").read()) for f in ("busnum", "devnum"))
            with open(f"/dev/bus/usb/{bus:03d}/{num:03d}", "wb") as fd:
                fcntl.ioctl(fd, ord("U") << 8 | 20)  # USBDEVFS_RESET
            for _ in range(50):  # the interface comes back under its udev name
                time.sleep(0.2)
                if os.path.exists(f"/sys/class/net/{dev}"):
                    break
            r = subprocess.run(["ip", "link", "set", dev, "up", "type", "can", "bitrate", "1000000"],
                               capture_output=True, text=True, timeout=5)
            return "ok" if r.returncode == 0 else r.stderr.strip()[:150]
        except Exception as e:
            return f"failed: {e}"

    def can_watchdog_loop(self):
        tries = 0
        while True:
            time.sleep(0.1)
            t0, op = self.bus_t0, self.bus_op
            if t0 is None or time.monotonic() - t0 < CAN_STALL_S:
                if t0 is None and tries:
                    log(f"CAN watchdog: bus calls flowing again after {tries} reset(s)")
                    tries = 0
                continue
            tries += 1
            self.can_fault = True
            if tries == CAN_RESETS_MAX:  # link resets did not free it: reset the adapter on USB (= replug)
                r = self.can_usb_reset()
                log(f"CAN STALL: {op} blocked {time.monotonic() - t0:.1f} s; USB reset of the adapter: {r}")
            else:
                r = self.can_link_reset()
                log(f"CAN STALL: {op} blocked {time.monotonic() - t0:.1f} s; link reset #{tries}: {r}")
            if tries > CAN_RESETS_MAX and not self.can_dead:
                self.can_dead = True
                self.frozen, self.note = True, ("CAN LINK DEAD: motors are holding the last pose and heating. Support "
                                                "the arm and switch its power off, then replug the CAN adapter.")
                self.write_status(can="dead")
                log("CAN DEAD: " + self.note)
            # wait for the blocked call to return before judging again (a reset needs ~0.5 s to take)
            end = time.monotonic() + (2.0 if tries < CAN_RESETS_MAX else 10.0)
            while time.monotonic() < end and self.bus_t0 == t0:
                time.sleep(0.1)

    def can_check(self, label: str):
        """Abort the action in progress after a CAN stall (rest clears the flag and still runs)."""
        if self.can_fault:
            raise Abort(f"{label}: the CAN link stalled and was reset; stopping this action")

    def write_status(self, **extra):
        status = {"stage": self.stage, "frozen": self.frozen, "torque": self.s.torque, "note": self.note,
                  "tip": None if self.tip is None else self.tip.round(4).tolist(), "time": time.time(), **extra}
        STATUS.write_text(json.dumps(status, indent=2, default=lambda o: np.asarray(o).round(4).tolist()))

    def joints(self) -> dict[str, float]:
        if not self.s.torque:
            # Disabled RobStride motors do not answer feedback requests, so the follower's reading
            # goes stale while limp (someone may move the arm by hand). A disable frame is answered
            # with a fresh state and is harmless when torque is already off.
            return self.fresh_joints_limp()
        q = self.follower._present_pos()
        self.check_gripper_reading(q["gripper"])
        return q

    def safe_enable(self):
        """Enable torque holding the arm where it really is. (Session.enable_holding commands the
        follower's reading, which is stale after the arm was moved by hand while limp -> a jump.)"""
        present = self.fresh_joints_limp()
        self.check_gripper_reading(present["gripper"])
        mj.RebotB601Follower.configure(self.follower)
        # the frame can also change on enabling: re-read before the first command
        now = self.follower._present_pos()["gripper"]
        if abs(now - present["gripper"]) > 20.0:
            self.set_grip_frame(self.gripper_raw(), "after enabling")
            now = self.follower._present_pos()["gripper"]
            if abs(now - present["gripper"]) > 20.0:
                self.s.release()
                raise Abort(f"gripper read {present['gripper']:.1f} deg limp but {now:.1f} after enabling; torque off")
        self.s.torque = True
        self.s.command(present)

    def fresh_joints_limp(self) -> dict[str, float]:
        if self.s.torque:
            raise Abort("fresh_joints_limp called with torque on")
        bus = self.follower.bus
        for _ in range(3):
            bus.disable_all()
            time.sleep(0.03)
            bus.poll_feedback_once()
        states = {k: m.get_state() for k, m in self.follower.motors.items()}
        self.set_grip_frame(self.gripper_raw(states["gripper"]), "limp")
        return self.follower._present_pos(states)

    def torques(self) -> dict[str, float]:
        return {k: s.torq for k, s in self.follower._read_feedback().items()}

    def joint_speeds(self, samples: int = 5) -> dict[str, float]:
        """RMS joint speed in deg/s over a few feedback reads (vibration indicator)."""
        acc: dict[str, list[float]] = {}
        for _ in range(samples):
            for k, st in self.follower._read_feedback().items():
                acc.setdefault(k, []).append(math.degrees(st.vel))
            time.sleep(0.02)
        return {k: round(float(np.sqrt(np.mean(np.square(v)))), 2) for k, v in acc.items()}

    def measure(self, frames: int = 5) -> dict:
        """Measure the tag; on a loss, log diagnostics, release hold tension and look once more."""
        try:
            return self.meas.measure(frames)
        except TagLost as e:
            stamp = time.strftime("%H%M%S")
            DATA.mkdir(exist_ok=True)
            for i, frame in enumerate(e.frames):
                cv2.imwrite(str(DATA / f"lost_{stamp}_{i}.png"), frame)
            present = self.joints()
            lead = {j: round(self.s.hold[j] - present[j], 1) for j in mj.B601_JOINTS} if self.s.hold else {}
            log(f"TAG LOST: {e.stats}; joint speeds deg/s {self.joint_speeds()}; command lead deg {lead}; "
                f"frames saved as data/lost_{stamp}_*.png")
            if not self.s.torque:
                raise
            # First just look again while holding the same command (a brief miss costs nothing).
            try:
                m = self.meas.measure(frames, timeout_s=2.0)
                log("tag found again on a plain retry")
                return m
            except TagLost:
                pass
            # (No "relax the hold" step any more: re-commanding the measured pose dropped the arm by
            # its whole gravity sag, and the diag run showed no elbow chatter to relieve.)
            raise TagLost("tag not visible in the wrist camera, even after a retry", e.stats, e.frames)

    def require_motion_ok(self):
        if self.frozen:
            raise Abort("frozen after an earlier anomaly; only `rest` is allowed")

    def freeze(self, why: str):
        self.frozen = True
        self.note = why
        try:
            self.s.freeze()
        except Exception as e:
            log(f"freeze command failed: {e}")
        log(f"FROZEN: {why}")

    def hold(self, seconds: float):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.s.tick()
            time.sleep(1 / 30)

    def move(self, target: dict[str, float], label: str):
        """Slow joint move; refuses jumps larger than expected."""
        present = self.joints()
        # Judge the size of the move against the previous command (what is actually being asked
        # for); the command may lead the measured pose by the sag compensation. send_action still
        # clips every command to MAX_RELATIVE_TARGET_DEG of the measured position.
        reference = self.s.hold if self.s.hold else present
        jump = max(abs(target[j] - reference[j]) for j in mj.B601_JOINTS if j != "gripper")
        if jump > 15.0:
            raise Abort(f"refusing a {jump:.1f} deg move ({label})")
        if not self.path:
            self.path.append(present)
        self.glide(target)
        self.path.append(dict(target))

    def glide(self, target: dict[str, float], speed: dict[str, float] | None = None, settle: float = 0.7):
        """Interpolate from the last *command* to target, then settle.

        Session.move_to starts from the measured pose; the command leads that by the gravity sag,
        so its first command dropped the arm by the sag and every move began with a visible dip."""
        if not self.s.torque:
            self.safe_enable()
        start = dict(self.s.hold) if self.s.hold else self.joints()
        speed = speed or mj.SPEED_DEG_S
        duration = max(abs(target[j] - start[j]) / speed[j] for j in mj.B601_JOINTS)
        steps = max(1, math.ceil(duration * mj.RATE_HZ))
        for i in range(1, steps + 1):
            t0 = time.perf_counter()
            a = i / steps
            a = a * a * (3 - 2 * a)  # smoothstep: gentle start and stop at the higher speeds
            self.s.command({j: start[j] + (target[j] - start[j]) * a for j in mj.B601_JOINTS})
            if not getattr(self, "resting", False):
                self.can_check("move")
            time.sleep(max(0.0, 1 / mj.RATE_HZ - (time.perf_counter() - t0)))
        if settle <= 0:
            return
        self.hold(settle)
        q = self.joints()
        blocked = {j: round(target[j] - q[j], 1) for j in mj.B601_JOINTS
                   if j != "gripper" and abs(target[j] - q[j]) > 12.0}
        faulted = [k for k, v in self.follower._read_feedback().items() if v.status_code]
        if blocked or faulted:
            # Stop pushing (a joint held against an obstacle trips a stall-overload fault).
            self.s.command({**self.s.hold, **{j: q[j] for j in blocked}})
            raise Abort(f"joint(s) blocked {blocked} / motor fault {faulted}; holding the measured angles")

    def tip_pixel(self):
        if self.tip is None:
            return None
        uv, _ = cv2.projectPoints(self.tip.reshape(1, 3), np.zeros(3), np.zeros(3), self.cam.K, self.cam.dist)
        return uv.ravel()

    # ---- stages
    def perceive(self):
        m = self.measure(9)
        frame, depth = m["frame"], m["depth"]
        DATA.mkdir(exist_ok=True)
        cv2.imwrite(str(DATA / "wrist.png"), frame)
        if depth is not None:
            np.save(DATA / "wrist_depth.npy", depth)
        cx, cy = m["corners"].mean(axis=0).astype(int)
        stereo = None
        if depth is not None:
            patch = depth[max(0, cy - 6):cy + 6, max(0, cx - 6):cx + 6]
            valid = patch[patch > 0]
            stereo = float(np.median(valid)) / 1000 if valid.size else None
        log(f"tag p = {np.round(m['p'] * 1000, 1).tolist()} mm, normal = {np.round(m['n'], 3).tolist()}, "
            f"stereo depth at tag centre = {None if stereo is None else round(stereo * 1000)} mm, "
            f"joints = { {k: round(v, 1) for k, v in self.joints().items()} }")
        self.write_status(tag_p=m["p"], tag_n=m["n"], stereo_tag_m=stereo)

    def set_tip(self, xyz):
        self.tip = np.array(xyz, dtype=float)
        uv = self.tip_pixel()
        self.cam.overlay = [o for o in self.cam.overlay if o[0] != "tip"] + [("tip", uv)]
        log(f"jaw-tip point set to {np.round(self.tip * 1000, 1).tolist()} mm (pixel {np.round(uv).tolist()})")

    def probe(self):
        self.require_motion_ok()
        if not self.s.torque:
            self.safe_enable()
            self.hold(0.5)
        self.move_segmented(START_POSE, "probe start pose")
        self.hold(0.5)
        self.n0 = self.measure()["n"]
        self.J = self.measure_jacobian(PROBE_DEG, verbose=True)
        np.save(DATA / "jacobian.npy", self.J)
        self.write_status(J=self.J)

    def measure_jacobian(self, deltas, verbose=False) -> np.ndarray:
        """Central differences around the current pose (one-sided next to a joint limit)."""
        # Probe around the commanded pose: re-centring on the measured pose would drop the arm by
        # its gravity sag after every probe. (The probe start pose no longer pre-lifts the elbow,
        # so this does not leave it balanced on its rest contact.)
        base = dict(self.s.hold) if self.s.hold else self.joints()
        J = np.zeros((5, len(SERVO_JOINTS)))
        m0 = self.measure(SERVO_FRAMES) if self.J is not None else None
        for k, j in enumerate(SERVO_JOINTS):
            d = deltas[j] if isinstance(deltas, dict) else deltas
            if m0 is not None:
                d = self.probe_size(m0["p"], self.J[:3, k], d, j)
            lo, hi = self.limits[j]
            ok = lambda t: lo + LIMIT_MARGIN_DEG <= t <= hi - LIMIT_MARGIN_DEG
            targets = [base[j] + d, base[j] - d]
            if not ok(targets[0]):      # next to the upper limit: probe the full span below
                targets = [base[j], base[j] - 2 * d]
            elif not ok(targets[1]):    # next to the lower limit: probe the full span above
                targets = [base[j] + 2 * d, base[j]]
            ys = []
            for target in targets:
                self.move({**base, j: target}, f"probe {j} {target - base[j]:+.1f}")
                self.hold(SETTLE_S)
                ys.append((self.joints()[j], y_vec(self.measure(PROBE_FRAMES))))
            (a1, y1), (a2, y2) = ys
            if abs(a1 - a2) < 0.4 * abs(targets[0] - targets[1]):
                raise Abort(f"{j} moved only {abs(a1 - a2):.1f} of {abs(targets[0] - targets[1]):.1f} deg while probing")
            J[:, k] = (y1 - y2) / (a1 - a2)
            self.move(base, "probe centre")
            if verbose:
                log(f"J[{j}] = {np.round(J[:, k] * 1000, 2).tolist()} (mm or mm-equiv per deg)")
        log(f"Jacobian measured; singular values {np.round(np.linalg.svd(J[:3], compute_uv=False) * 1000, 2).tolist()}")
        return J

    def tag_in_view(self, p: np.ndarray, margin_px: float = 15.0) -> bool:
        """Would a tag centred at camera point p lie fully inside the image (and in front)?"""
        if p[2] < 0.05:
            return False
        uv, _ = cv2.projectPoints(p.reshape(1, 3), np.zeros(3), np.zeros(3), self.cam.K, self.cam.dist)
        u, v = uv.ravel()
        r = 0.75 * ACTIVE_TAG["size"] * self.cam.K[0, 0] / p[2] + margin_px  # half-diagonal plus margin
        if not (r <= u <= W - r and r <= v <= H - r):
            return False
        if self.s.hold and self.s.hold.get("gripper", 0.0) < -250.0:
            x0, y0, x1, y1 = GRIPPER_BOXES[2]  # open fingers hang into the top of the image
            r_tag = r - margin_px
            if x0 - r_tag < u < x1 + r_tag and v < y1 + r_tag:
                return False
        return True

    def probe_size(self, p: np.ndarray, col: np.ndarray, d: float, j: str) -> float:
        """Shrink a probe until the predicted tag position stays in view on both sides."""
        for f in (1.0, 0.75, 0.5):
            if all(self.tag_in_view(p + s * f * d * col) for s in (2.0, -2.0)):
                if f < 1.0:
                    log(f"probe {j}: reduced to {f * d:.1f} deg to keep the tag in view")
                return f * d
        small = max(1.5, 0.5 * d)
        log(f"probe {j}: tag near the image edge; probing with {small:.1f} deg")
        return small

    def servo_to(self, target_p: np.ndarray, tol: float, label: str, max_iter: int = 60, torque_guard=False,
                 reprobe=True):
        """Iterate joint steps until the tag sits at target_p in the camera frame."""
        if self.J is None:
            raise Abort("run `probe` first")
        self.require_motion_ok()
        base_torque = self.torques() if torque_guard else None
        worse = 0
        best = float("inf")
        for it in range(max_iter):
            # (No probing next to the tag: probe swings of ~10 mm could push the tips into it.)
            if reprobe and it and (it % REPROBE_EVERY == 0 or worse >= 3):
                log(f"{label}: re-measuring the Jacobian here ({'error growing' if worse >= 3 else 'periodic'})")
                self.J = self.measure_jacobian(REPROBE_DEG)
                worse = 0
            m = self.measure(SERVO_FRAMES)
            tilt = float(np.linalg.norm(m["n"] - self.n0))
            if tilt > MAX_TILT_CHANGE:
                raise Abort(f"approach angle changed too much ({math.degrees(tilt):.0f} deg)")
            err_p = target_p - m["p"]
            dist = float(np.linalg.norm(err_p))
            q_now = self.joints()
            note = f" (tracked {m['tracked']}/{m['count']})" if m.get("tracked") else ""
            log(f"{label} it {it}: tag p {np.round(m['p'] * 1000, 1).tolist()} mm, error {dist * 1000:.1f} mm, "
                f"joints {[round(q_now[j], 1) for j in SERVO_JOINTS]}{note}")
            self.write_status(tag_p=m["p"], error_mm=dist * 1000, iteration=it)
            if dist < tol:
                return m
            worse = worse + 1 if dist > best + 0.002 else 0
            best = min(best, dist)
            # Step the target towards the goal, keep the initial approach orientation.
            # Lateral (image position) and depth are capped separately, so the tag is brought to
            # its target pixel early and stays clear of the gripper and the image edges while the
            # slower advance in depth continues.
            lat, dep = err_p[:2], err_p[2:]
            step_p = GAIN * np.concatenate([lat * min(1.0, MAX_POS_STEP_M / max(1e-9, float(np.linalg.norm(lat)))),
                                            dep * min(1.0, MAX_POS_STEP_M / max(1e-9, abs(float(dep[0]))))])
            e = np.concatenate([step_p, ROT_WEIGHT_M * (self.n0[:2] - m["n"][:2])])
            J = self.J
            before = self.joints()
            # Step from the last *commanded* pose: joints sag a few degrees under gravity, and
            # stepping from the measured pose would re-apply that sag and cancel small steps.
            cmd = dict(self.s.hold) if self.s.hold else before
            dq = self.solve_step(J, e, cmd)
            y_before = y_vec(m)
            target = {**cmd, **{j: cmd[j] + dq[k] for k, j in enumerate(SERVO_JOINTS)}}
            for j in SERVO_JOINTS:  # bounded sag compensation
                target[j] = min(before[j] + LEAD_MAX_DEG, max(before[j] - LEAD_MAX_DEG, target[j]))
            e_now, e_tgt = before["elbow_flex"], target["elbow_flex"]
            if e_now > ELBOW_REST_ZONE_DEG and e_tgt < e_now - 0.5:
                target["elbow_flex"] = min(e_tgt, e_now - ELBOW_LIFTOFF_DEG)
                log(f"{label}: elbow lift-off, command {target['elbow_flex']:.1f} (measured {e_now:.1f})")
            self.move(target, f"{label} step")
            self.hold(SETTLE_S)
            if torque_guard:
                now = self.torques()
                jumps = {j: now[j] - base_torque[j] for j in now if abs(now[j] - base_torque[j]) > TORQUE_JUMP_NM[j]}
                if jumps:
                    log(f"torque change: { {j: round(v, 2) for j, v in jumps.items()} } -> treating as contact")
                    return None
            after = self.joints()
            dq_actual = np.array([after[j] - before[j] for j in SERVO_JOINTS])
            m2 = self.measure(SERVO_FRAMES)
            dy = y_vec(m2) - y_before
            if np.linalg.norm(dq_actual) > 1.0:  # half-step Broyden update; small steps are mostly noise
                self.J = J + 0.5 * np.outer(dy - J @ dq_actual, dq_actual) / float(dq_actual @ dq_actual)
        raise Abort(f"{label}: did not converge in {max_iter} iterations")

    def solve_step(self, J: np.ndarray, e: np.ndarray, q: dict[str, float]) -> np.ndarray:
        """Damped least-squares step, re-solved without joints that would cross a limit."""
        active = np.ones(len(SERVO_JOINTS), dtype=bool)
        for _ in range(len(SERVO_JOINTS)):
            Ja = J[:, active]
            dqa = Ja.T @ np.linalg.solve(Ja @ Ja.T + DLS_LAMBDA ** 2 * np.eye(len(e)), e)
            dqa *= min(1.0, MAX_STEP_DEG / max(1e-9, np.max(np.abs(dqa))))
            dq = np.zeros(len(SERVO_JOINTS))
            dq[active] = dqa
            blocked = False
            for k, j in enumerate(SERVO_JOINTS):
                lo, hi = self.limits[j]
                if active[k] and not (lo + LIMIT_MARGIN_DEG <= q[j] + dq[k] <= hi - LIMIT_MARGIN_DEG):
                    active[k] = False
                    blocked = True
            if not blocked:
                return dq
            if not active.any():
                break
        raise Abort("every servo joint is at a limit; target unreachable from here")

    def detection_rate(self, seconds: float) -> dict:
        """Hold the current command for `seconds`; report tag detection rate, sharpness, vibration."""
        seen = hits = 0
        sharp, speeds = [], []
        end = time.monotonic() + seconds
        seq = -1
        while time.monotonic() < end:
            self.s.tick()
            seq, frame, _ = self.cam.latest(seq)
            seen += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharp.append(cv2.Laplacian(gray[H // 4:3 * H // 4, W // 4:3 * W // 4], cv2.CV_64F).var())
            hits += detect(frame, self.cam.K, self.cam.dist) is not None
            if seen % 5 == 0:
                speeds.append(self.joint_speeds(2)["elbow_flex"])
        return {"detected": f"{hits}/{seen}", "sharpness": round(float(np.median(sharp)), 0),
                "elbow_speed_rms": round(float(np.mean(speeds)), 2) if speeds else None}

    def diag(self):
        """Test the chatter hypothesis: hold the elbow relaxed, in the marginal band, then lifted."""
        self.require_motion_ok()
        if not self.s.torque:
            self.safe_enable()
        start = {**self.joints(), **{j: REST_POSE[j] + v for j, v in START_OFFSET.items()}}
        self.move(start, "diag start pose")
        self.hold(1.0)
        base = self.joints()
        results = {}
        for label, lead in (("relaxed (cmd = measured)", 0.0), ("marginal (cmd 5 deg past)", 5.0),
                            ("lifted (cmd 12 deg past)", 12.0)):
            self.s.command({**base, "elbow_flex": base["elbow_flex"] - lead})
            self.hold(1.5)
            results[label] = {**self.detection_rate(4.0), "elbow_measured": round(self.joints()["elbow_flex"], 1)}
            log(f"DIAG {label}: {results[label]}")
        self.s.command(base)
        self.hold(1.0)
        self.write_status(diag=results)
        log("DIAG done. Chatter theory confirmed if 'marginal' shows a low detection rate and/or high "
            "elbow speed while 'relaxed' and 'lifted' are clean.")

    def goto(self, args: list[str]):
        """`goto joint=deg ...`: slow absolute move of the named joints (in <=10 deg segments),
        then report where the tag is. For exploring start poses by hand."""
        self.require_motion_ok()
        want = {}
        for a in args:
            j, v = a.split("=")
            if j not in mj.B601_JOINTS:
                raise Abort(f"unknown joint {j}")
            lo, hi = self.limits[j]
            margin = 0.5 if j == "gripper" else LIMIT_MARGIN_DEG
            want[j] = min(hi - margin, max(lo + margin, float(v)))
        if not self.s.torque:
            self.safe_enable()
            self.hold(0.5)
        self.move_segmented(want, "goto")
        self.hold(SETTLE_S)
        q = self.joints()
        blocked = {j: round(want[j] - q[j], 1) for j in want if abs(want[j] - q[j]) > 8.0}
        if blocked:
            # Pressing against an obstacle or a stop: stop pushing, hold where it actually is.
            self.s.command({**self.s.hold, **{j: q[j] for j in blocked}})
            self.hold(0.5)
            log(f"goto: BLOCKED {blocked} deg short of the command; now holding the measured angle")
        try:
            m = self.meas.measure(SERVO_FRAMES, timeout_s=3.0)
            log(f"goto done: joints {[round(q[j], 1) for j in mj.B601_JOINTS]}; tag p "
                f"{np.round(m['p'] * 1000, 1).tolist()} mm n {np.round(m['n'], 2).tolist()} "
                f"(tracked {m['tracked']}/{m['count']})")
        except TagLost as e:
            log(f"goto done: joints {[round(q[j], 1) for j in mj.B601_JOINTS]}; TAG NOT VISIBLE {e.stats}")
        self.snap()

    def stream_path(self, cmds: list[dict], label: str, guard: bool = False, limits: dict | None = None,
                    speed_scale: float = 1.0) -> bool:
        """One continuous motion through joint *commands* (all joints), streamed at RATE_HZ with a single
        smoothstep over the whole path: no stop at the waypoints (the per-waypoint glide + settle made every
        move start-stop, the user's ask 2026-11-04). Peak joint speed stays at SERVO_SPEED_DEG_S x speed_scale.
        guard: every STREAM_GUARD_DT_S compare the torques with those STREAM_GUARD_WINDOW_S earlier (a rolling
        baseline: a long move's slow gravity-load change is not contact) against limits; on a jump, stop, hold
        0.5 s and re-check against the pre-jump torques: still over -> contact, False (holding where it stopped);
        settled -> carry on. Records the passed poses for rest's retrace."""
        if not self.s.torque:
            self.safe_enable()
        lim = limits or TORQUE_JUMP_FREE_AIR_NM
        start = dict(self.s.hold) if self.s.hold else self.joints()
        pts = [start] + [{**start, **c} for c in cmds]
        speed = {j: v * speed_scale for j, v in mj.SPEED_DEG_S.items()}
        seg = [max(abs(b[j] - a[j]) / speed[j] for j in mj.B601_JOINTS) for a, b in zip(pts, pts[1:])]
        total = sum(seg)
        if total < 1e-3:
            return True
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        T = 1.5 * total  # smoothstep peaks at 1.5x its mean speed
        n = max(1, math.ceil(T * mj.RATE_HZ))
        if not self.path:
            self.path.append(self.joints())
        last_kept = start
        hist = []  # (time, torques) for the rolling guard baseline
        every = max(1, round(STREAM_GUARD_DT_S * mj.RATE_HZ))
        i = 0
        while i < n:
            t0 = time.perf_counter()
            i += 1
            a = i / n
            u = a * a * (3 - 2 * a) * total  # path "time" reached
            k = min(len(seg) - 1, int(np.searchsorted(cum, u, side="right") - 1))
            f = 0.0 if seg[k] <= 0 else min(1.0, (u - cum[k]) / seg[k])
            cmd = {j: pts[k][j] + (pts[k + 1][j] - pts[k][j]) * f for j in mj.B601_JOINTS}
            self.s.command(cmd)
            if label != "rest":
                self.can_check(label)
            if max(abs(cmd[j] - last_kept[j]) for j in mj.B601_JOINTS if j != "gripper") >= 8.0:
                self.path.append(dict(cmd))
                last_kept = cmd
            if guard and i % every == 0:
                now_t, tq = time.monotonic(), self.torques()
                hist.append((now_t, tq))
                old = [h for h in hist if now_t - h[0] >= STREAM_GUARD_WINDOW_S]
                if old:
                    base = old[-1][1]
                    hist = [h for h in hist if now_t - h[0] < STREAM_GUARD_WINDOW_S + 0.2]
                    jumps = {j: round(tq[j] - base[j], 2) for j in tq if j != "gripper" and abs(tq[j] - base[j]) > lim[j]}
                    if jumps:
                        self.hold(0.5)
                        still = self.torque_jumps(base, lim)
                        if still:
                            log(f"{label}: torque change {jumps}, still {still} after 0.5 s -> contact "
                                f"({100 * a:.0f}% along)")
                            self.path.append(dict(self.s.hold))
                            return False
                        log(f"{label}: torque change {jumps} settled; not contact ({100 * a:.0f}% along)")
                        hist = []
            time.sleep(max(0.0, 1 / mj.RATE_HZ - (time.perf_counter() - t0)))
        self.path.append(dict(pts[-1]))
        return True

    def move_segmented(self, want: dict[str, float], label: str):
        """Move the named joints to absolute targets: one smooth streamed motion (STREAM_MOTION), else in
        <=10 deg glides (passes the jump guard)."""
        if STREAM_MOTION:
            self.stream_path([want], label)
            self.hold(0.3)
            q = self.joints()
            blocked = {j: round(want[j] - q[j], 1) for j in want if j != "gripper" and abs(want[j] - q[j]) > 12.0}
            if blocked:
                self.s.command({**self.s.hold, **{j: q[j] for j in blocked}})
                raise Abort(f"joint(s) blocked {blocked}; holding the measured angles")
            return
        cmd = dict(self.s.hold) if self.s.hold else self.joints()
        while True:
            gap = max(abs(want[j] - cmd[j]) for j in want)
            if gap < 0.05:
                break
            f = min(1.0, 10.0 / gap)
            cmd = {**cmd, **{j: cmd[j] + f * (want[j] - cmd[j]) for j in want}}
            self.move(cmd, label)

    def recover(self, names: list[str]):
        """Clear a faulted motor (e.g. a stall-overload trip) and re-engage it at its present angle."""
        import motorbridge
        fb = self.follower._read_feedback()
        present = self.joints()
        for j in names:
            if fb[j].status_code == 0:
                continue
            motor = self.follower.motors[j]
            try:
                report = motor.robstride_get_fault_report()
            except Exception as e:
                report = f"unreadable ({e})"
            lo, hi = self.limits[j]
            if self.s.hold:
                self.s.hold[j] = min(hi, max(lo, present[j]))
            log(f"recover {j}: status {fb[j].status_code}, fault report {report}; clearing and re-enabling "
                f"at {present[j]:.1f} deg")
            motor.clear_error()
            time.sleep(0.1)
            motor.ensure_mode(motorbridge.Mode.MIT)
            motor.enable()
            self.hold(0.5)
        log(f"status codes now { {k: v.status_code for k, v in self.follower._read_feedback().items()} }")

    def kin(self):
        """Calibrated kinematics (kin/model.py), re-read each call so a refit applies at once."""
        sys.path.insert(0, str(HERE / "kin"))
        import model
        return model.Kin()

    def plate_find(self, tag: str = "", prior=None, prior_yaw=None, weak_ok: bool = False):
        """Find the plate (no tag) in the current wrist view: top-down map of the plate-top plane
        from the calibrated camera pose, then the well-lattice match. Returns the plate pose in base
        (origin at the lattice centre on the plate top, x along the 12-well axis, z up) or None.
        prior: rough plate centre (base x, y in m); only fits within 5 mm of it count.
        prior_yaw: expected plate yaw (deg); only lattice angles within PLATE_YAW_WINDOW_DEG of it count."""
        sys.path.insert(0, str(HERE / "kin"))
        import birdseye
        import plate_detect
        self.hold(SETTLE_S)
        _, frame, _ = self.cam.latest()
        q = self.joints()
        k = self.kin()
        z = k.tags[1][2, 3] + PLATE_TOP_Z
        C = k.cam(q)
        # map a 50 cm square of table centred under the optical axis' hit point
        ray = C[:3, 2]
        if ray[2] > -0.2:
            log("plate_find: camera not looking down at the table")
            return None
        hit = C[:3, 3] + ray * (z - C[2, 3]) / ray[2]
        res = PLATE_MAP_RES_M
        img, to_base = birdseye.plane_map(frame, self.cam.K, self.cam.dist, C, z,
                                          (hit[0] - 0.25, hit[0] + 0.25), (hit[1] - 0.25, hit[1] + 0.25), res)
        pp = None if prior is None else ((prior[0] - (hit[0] - 0.25)) / res, ((hit[1] + 0.25) - prior[1]) / res)
        angles = None
        if prior_yaw is not None:  # the square well lattice also fits 90 deg turned (-65.9 vs +24.1, 2026-10-30 11:11)
            angles = (prior_yaw + np.arange(-PLATE_YAW_WINDOW_DEG, PLATE_YAW_WINDOW_DEG + 1e-6, 2.0) + 90) % 180 - 90
        # only our taped zone: the plate beyond the blue tape belongs to the other arm (user, 2026-11-04)
        rows_y = (hit[1] + 0.25) - np.arange(img.shape[0]) * res
        cols_x = (hit[0] - 0.25) + np.arange(img.shape[1]) * res
        img[rows_y > ZONE_Y_MAX_M + ZONE_FIND_MARGIN_M, :] = 0
        img[:, (cols_x < ZONE_X_M[0] - ZONE_FIND_MARGIN_M) | (cols_x > ZONE_X_M[1] + ZONE_FIND_MARGIN_M)] = 0
        d = plate_detect.detect(img, res, angles=angles, prior=pp)
        if d is not None:
            d = plate_detect.refine(img, res, d, prior=pp)
        if d is not None and prior is None and d["score"] >= PLATE_MIN_SCORE and d["score"] < PLATE_MIN_MARGIN * d["second"]:
            # the runner-up may be the same plate turned 90 deg (the 12x8 lattice nearly fits that way: 0.281 vs
            # 0.251 with no other plate in view, 2026-11-04 20:46). If nothing scores near it elsewhere once the
            # best spot is blanked, keep the best: the look reads then hold its angle, the jaw view checks it.
            blank = img.copy()
            cv2.circle(blank, (int(d["c"][0]), int(d["c"][1])), int(0.07 / res), (0, 0, 0), -1)
            d2 = plate_detect.detect(blank, res, angles=angles)
            place_rival = d2["score"] if d2 is not None else 0.0
            if d["score"] >= PLATE_MIN_MARGIN * place_rival and d["score"] >= PLATE_ROT_MARGIN * d["second"]:
                log(f"plate_find: runner-up {d['second']:.3f} is this plate turned (elsewhere at most {place_rival:.3f}); "
                    f"keeping {d['score']:.3f}")
                d["second"] = place_rival
        stamp = time.strftime("%H%M%S")
        cv2.imwrite(str(DATA / f"platefind{tag}_{stamp}.png"), frame)
        (DATA / f"platefind{tag}_{stamp}.json").write_text(json.dumps(q))
        min_score = PLATE_MIN_SCORE if prior is None else PLATE_MIN_SCORE_PRIOR
        if d is None or d["score"] < min_score or d["score"] < PLATE_MIN_MARGIN * d["second"]:
            if weak_ok and d is not None and d["score"] >= PLATE_WEAK_SCORE:
                wx, wy = to_base(*d["c"])
                if ZONE_X_M[0] <= wx <= ZONE_X_M[1] and wy <= ZONE_Y_MAX_M:
                    log(f"plate_find: weak fit {d['score']:.3f} (vs {d['second']:.3f}) at ({wx * 1000:.0f}, {wy * 1000:.0f}) mm: "
                        f"a hint only, for the look poses to confirm")
                    T = np.eye(4)
                    T[:3, 3] = (wx, wy, z)
                    return {"T": T, "q": q, "score": float(d["score"]), "second": float(d["second"]), "weak": True}
            log(f"plate_find: no plate ({d and {k_: round(float(v), 3) for k_, v in d.items() if k_ != 'c'}})")
            return None
        x, y = to_base(*d["c"])
        a = math.radians(d["angle_deg"])
        T = np.eye(4)
        T[:3, :3] = [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
        T[:3, 3] = (x, y, z)
        log(f"PLATE at ({x * 1000:.1f}, {y * 1000:.1f}) mm, {d['angle_deg']:+.2f} deg "
            f"(score {d['score']:.2f} vs {d['second']:.2f})")
        return {"T": T, "q": q, "score": float(d["score"]), "second": float(d["second"])}

    def platescan(self, n: int):
        """Detect the plate from n random poses around VIEW_POSE; the spread of the detected poses is
        the combined kinematics + detection error (written to data/platescan.json)."""
        self.require_motion_ok()
        if not self.s.torque:
            self.safe_enable()
        if self.joints()["shoulder_lift"] > -30.0:
            self.move_segmented({**VIEW_POSE, "wrist_flex": 0.0}, "unfold")
        rng = np.random.default_rng()
        out = []
        for i in range(n + 1):
            self.thermal_check()
            target = VIEW_POSE if i == 0 else {j: VIEW_POSE[j] + rng.uniform(-1, 1) * PLATE_SCAN_SPREAD[j]
                                               for j in ARM6}
            self.move_segmented(target, f"scan pose {i}")
            r = self.plate_find(tag=f"_scan{i}")
            if r:
                a = math.degrees(math.atan2(r["T"][1, 0], r["T"][0, 0]))
                out.append({"q": r["q"], "xy_mm": (r["T"][:2, 3] * 1000).tolist(), "angle_deg": a,
                            "score": r["score"]})
        (DATA / "platescan.json").write_text(json.dumps(out, indent=1))
        if out:
            xy = np.array([o["xy_mm"] for o in out])
            ang = np.array([o["angle_deg"] for o in out])
            log(f"PLATESCAN: {len(out)}/{n + 1} found; mean ({xy[:, 0].mean():.1f}, {xy[:, 1].mean():.1f}) mm "
                f"{ang.mean():+.2f} deg; spread (std) {xy[:, 0].std():.1f}, {xy[:, 1].std():.1f} mm, "
                f"{ang.std():.2f} deg; max dev {np.abs(xy - xy.mean(0)).max(0).round(1).tolist()} mm")
        self.move_segmented({**VIEW_POSE, "wrist_flex": 0.0}, "back")
        self.path = []
        self.rest()

    def look_solve(self, k, p_base, seed, depth=None):
        """look_q at `depth`, or the first of LOOK_DEPTHS_M that reaches (spots near the base have no view
        from the usual 0.33 m). -> (q, pos_err_mm, rot_err_deg)"""
        sys.path.insert(0, str(HERE / "kin"))
        import lookdown
        for d in [depth] if depth else LOOK_DEPTHS_M:
            q, ep, er = lookdown.look_q(k, self.cam.K, p_base, seed, VIEW_POSE, limits=self.limits, depth=d)
            if ep <= 2.0 and er <= 1.0:
                break
        return q, ep, er

    def look_at_plate(self, p_base, label="look", px=None, depth=None):
        """Move so the camera views p_base (plate centre) like VIEW_POSE does, centred low in the image."""
        sys.path.insert(0, str(HERE / "kin"))
        import lookdown
        k = self.kin()
        if px is not None:
            saved, lookdown.LOOK_PX = lookdown.LOOK_PX, px
        try:
            q, ep, er = self.look_solve(k, p_base, self.joints(), depth)
        finally:
            if px is not None:
                lookdown.LOOK_PX = saved
        if ep > 2.0 or er > 1.0:
            raise Abort(f"{label}: no reachable view of ({p_base[0] * 1000:.0f}, {p_base[1] * 1000:.0f}) mm "
                        f"(IK error {ep:.1f} mm, {er:.1f} deg)")
        self.move_segmented(q, label)
        self.move_exact(q, label)  # the elbow sags ~3 deg here, which moves the view ~5 cm
        return q

    def platelook(self, n: int = 3, prior=None, anchor: bool = False, prior_yaw=None):
        """Coarse plate find from VIEW_POSE, then n fine finds from look poses centred on it (slightly
        varied); logs the spread and stores the median pose as data/plate_pose.json. With anchor, every
        find keeps the 5 mm window around prior (use when prior is a measured pose, not a goal)."""
        self.require_motion_ok()
        if not self.s.torque:
            self.safe_enable()
        if self.joints()["shoulder_lift"] > -30.0:
            self.move_segmented({**VIEW_POSE, "wrist_flex": 0.0}, "unfold")
        self.move_segmented(VIEW_POSE, "view")
        if prior is not None:  # start by looking where the plate should be
            self.look_at_plate(np.array([prior[0], prior[1], self.kin().tags[1][2, 3] + PLATE_TOP_Z]), "look at the prior")
        r = self.plate_find(tag="_coarse", prior=prior, prior_yaw=prior_yaw)
        if prior is not None:
            # also search the whole view: a plate pushed by a failed grasp still fits the 5 mm window, by wells
            # whole pitches off its true place. On the Jetson the window walked with it 3 cm from the plate's
            # real position over 6 halts, and each grasp started further off (+21 -> +57 px, 2026-11-03).
            free = self.plate_find(tag="_coarse_free")  # passes only with score >= 0.25 and a 1.15 margin
            if free is not None and (r is None or (np.linalg.norm(free["T"][:2, 3] - r["T"][:2, 3]) > 0.005
                                                   and free["score"] > 1.15 * r["score"])):
                fp = free["T"][:3, 3]
                was = "none" if r is None else f"score {r['score']:.2f}"
                log(f"platelook: the whole-view fit ({fp[0] * 1000:.1f}, {fp[1] * 1000:.1f}) mm, score {free['score']:.2f} "
                    f"beats the window fit ({was}); using it (the plate was moved)")
                r, prior = free, fp[:2].copy()
        weak = False
        if r is None:
            # a far first view can score a clearly visible plate below the cutoff (0.196, plate flat on the pad,
            # 2026-11-04 18:59): its best fit, if inside the taped zone, only aims the look poses, which must confirm
            r = self.plate_find(tag="_coarse_weak", weak_ok=True)
            if r is None:
                raise Abort("platelook: no plate from the first view")
            weak = True
            prior_yaw = None  # the weak fit's angle is not trusted: the looks search all angles
        p = r["T"][:3, 3]
        window = prior if anchor and prior is not None else p[:2].copy()
        found = []
        rng = np.random.default_rng()
        for i in range(n):
            self.thermal_check()
            px = (325.0 + rng.uniform(-40, 40), 260.0 + rng.uniform(-15, 15)) if i else None
            self.look_at_plate(p, f"look {i}", px=px)
            # anchored: a window chained from read to read walked a whole well pitch (355 -> 359 -> 363 mm,
            # 2026-10-29 11:47) and put a pad on the plate edge
            # (unanchored: around the coarse find; the release check walked 319 -> 316 -> 314 mm at 13:16)
            f = self.plate_find(tag=f"_look{i}", prior=window, prior_yaw=prior_yaw)
            if f is not None:
                found.append(f)
                if i == 0:
                    p = f["T"][:3, 3]
        if not found or (weak and len(found) < 2):  # a weak hint needs two confirming looks
            raise Abort("platelook: lost the plate at the look poses")
        xy = np.array([f["T"][:2, 3] for f in found]) * 1000
        ang = np.array([math.degrees(math.atan2(f["T"][1, 0], f["T"][0, 0])) for f in found])
        # median, not mean: one of three reads was 5-6 mm off three times on 2026-10-23, and its pull on the mean
        # put a pad on the plate rim twice
        log(f"PLATELOOK: {len(found)}/{n} found; median ({np.median(xy[:, 0]):.1f}, {np.median(xy[:, 1]):.1f}) mm "
            f"{np.median(ang):+.2f} deg; spread {np.ptp(xy[:, 0]):.1f} x {np.ptp(xy[:, 1]):.1f} mm, {np.ptp(ang):.2f} deg "
            f"(coarse was ({r['T'][0, 3] * 1000:.1f}, {r['T'][1, 3] * 1000:.1f}))")
        T = found[0]["T"].copy()
        a = math.radians(np.median(ang))
        T[:3, :3] = [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
        T[:2, 3] = np.median(xy, axis=0) / 1000
        self.platelook_spread_mm = float(max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])))
        (DATA / "plate_pose.json").write_text(json.dumps({"T": T.tolist(), "time": time.strftime("%H:%M:%S"),
                                                          "n": len(found)}))
        return T

    def snap(self):
        _, frame, depth = self.cam.latest()
        DATA.mkdir(exist_ok=True)
        cv2.imwrite(str(DATA / "snap_raw.png"), frame)
        (DATA / "snap_joints.json").write_text(json.dumps(self.joints()))
        if depth is not None:
            np.save(DATA / "snap_depth.npy", depth)
        det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
                                      cv2.aruco.DetectorParameters())
        corners, ids, _ = det.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        found = {}
        for c, i in zip(corners, [] if ids is None else ids.ravel()):
            size = TAG_SIZES.get(int(i), TAG_SIZE_M)
            d = pose_from_corners(c.reshape(4, 2), self.cam.K, self.cam.dist, size=size)
            found[int(i)] = {"px": np.round(c.reshape(4, 2).mean(axis=0)).tolist(),
                             "p_mm": np.round(d["p"] * 1000, 1).tolist(), "n": np.round(d["n"], 2).tolist()}
            if depth is not None:
                # Stereo depth over the tag (it is assumed TAG_SIZE_M by PnP): implied true size.
                mask = np.zeros(depth.shape, np.uint8)
                cv2.fillConvexPoly(mask, np.int32(c.reshape(4, 2)), 1)
                vals = depth[(mask > 0) & (depth > 0)]
                if vals.size > 20:
                    z = float(np.median(vals)) / 1000
                    found[int(i)]["stereo_z_mm"] = round(z * 1000)
                    found[int(i)]["implied_size_mm"] = round(size * z / d["p"][2] * 1000, 1)
        log(f"snap: tags {found}")
        img = frame.copy()
        for kind, data in list(self.cam.overlay):
            if kind == "tag":
                cv2.polylines(img, [np.int32(data)], True, (0, 255, 0), 2)
            elif kind == "tip":
                cv2.drawMarker(img, tuple(int(v) for v in data), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
        DATA.mkdir(exist_ok=True)
        cv2.imwrite(str(DATA / "snap.png"), img)

    # ---- 6-DOF servo (tag position + orientation) for the plate pick
    def y6(self, m: dict, R_ref: np.ndarray) -> np.ndarray:
        return np.concatenate([m["p"], ROT6_W * rotvec(m["R"] @ R_ref.T)])

    def measure_jacobian6(self, joints=ARM6) -> np.ndarray:
        base = dict(self.s.hold) if self.s.hold else self.joints()
        m0 = self.measure(SERVO_FRAMES)
        R_ref = m0["R"]
        J = np.zeros((6, len(joints)))
        J_old = getattr(self, "J6", None)
        for k, j in enumerate(joints):
            d = PROBE_DEG6[j]
            if J_old is not None:
                d = self.probe_size(m0["p"], J_old[:3, k], d, j)
            lo, hi = self.limits[j]
            ok = lambda t: lo + LIMIT_MARGIN_DEG <= t <= hi - LIMIT_MARGIN_DEG
            targets = [base[j] + d, base[j] - d]
            if not ok(targets[0]):
                targets = [base[j], base[j] - 2 * d]
            elif not ok(targets[1]):
                targets = [base[j] + 2 * d, base[j]]
            ys = []
            for target in targets:
                self.move({**base, j: target}, f"probe6 {j} {target - base[j]:+.1f}")
                self.hold(SETTLE_S)
                ys.append((self.joints()[j], self.y6(self.measure(PROBE_FRAMES), R_ref)))
            (a1, y1), (a2, y2) = ys
            if abs(a1 - a2) < 0.3 * abs(targets[0] - targets[1]):
                raise Abort(f"{j} moved only {abs(a1 - a2):.1f} of {abs(targets[0] - targets[1]):.1f} deg while probing")
            J[:, k] = (y1 - y2) / (a1 - a2)
            self.move(base, "probe centre")
            log(f"J6[{j}] = {np.round(J[:, k] * 1000, 2).tolist()}")
        log(f"J6 singular values {np.round(np.linalg.svd(J, compute_uv=False) * 1000, 2).tolist()}")
        return J

    def solve6(self, J, e, q, joints) -> np.ndarray:
        active = np.ones(len(joints), dtype=bool)
        for _ in range(len(joints)):
            Ja = J[:, active]
            dqa = Ja.T @ np.linalg.solve(Ja @ Ja.T + DLS_LAMBDA ** 2 * np.eye(len(e)), e)
            dqa *= min(1.0, MAX_STEP_DEG / max(1e-9, np.max(np.abs(dqa))))
            dq = np.zeros(len(joints))
            dq[active] = dqa
            blocked = False
            for k, j in enumerate(joints):
                lo, hi = self.limits[j]
                if active[k] and not (lo + LIMIT_MARGIN_DEG <= q[j] + dq[k] <= hi - LIMIT_MARGIN_DEG):
                    active[k] = False
                    blocked = True
            if not blocked:
                return dq
            if not active.any():
                break
        raise Abort("every joint is at a limit; pose unreachable from here")

    def servo_pose(self, R_t: np.ndarray, p_t: np.ndarray, tol_mm: float, tol_deg: float, label: str,
                   max_iter: int = 80, joints=ARM6, reprobe=True, torque_guard=False, max_pos_step=MAX_POS_STEP_M,
                   periodic=True, axis_w=ROT6_AXIS_W, torque_base=None):
        """Servo the tag to pose (R_t, p_t) in the camera frame. Returns the last measurement, or
        None if the torque guard saw contact."""
        self.require_motion_ok()
        if getattr(self, "J6", None) is None:
            self.J6 = self.measure_jacobian6(joints)
        base_torque = (torque_base or self.torques()) if torque_guard else None
        best, worse, reprobes = float("inf"), 0, 0
        Wd = np.diag(np.concatenate([np.ones(3), axis_w]))
        for it in range(max_iter):
            if reprobe and it and ((periodic and it % REPROBE_EVERY == 0) or (worse >= 4 and reprobes == 0)):
                log(f"{label}: re-measuring the 6-DOF Jacobian")
                self.J6 = self.measure_jacobian6(joints)
                worse = 0
                reprobes += 1
            self.thermal_check()
            m = self.measure(SERVO_FRAMES)
            e_p = p_t - m["p"]
            e_r = rotvec(R_t @ m["R"].T)
            dist_mm = float(np.linalg.norm(e_p)) * 1000
            ang_deg = math.degrees(float(np.linalg.norm(e_r * axis_w)))  # weighted per axis
            q_now = self.joints()
            note = f" (tracked {m['tracked']}/{m['count']})" if m.get("tracked") else ""
            log(f"{label} it {it}: pos err {dist_mm:.1f} mm {np.round(e_p * 1000, 1).tolist()}, rot err {ang_deg:.1f} deg "
                f"(about cam x/y/z {np.round(np.degrees(e_r), 1).tolist()}), "
                f"joints {[round(q_now[j], 1) for j in joints]}{note}")
            self.write_status(error_mm=dist_mm, error_deg=ang_deg, iteration=it)
            if dist_mm < tol_mm and ang_deg < tol_deg:
                return m
            score = dist_mm + ang_deg * ROT6_W * 1000 * math.pi / 180
            worse = worse + 1 if score > best + 2.0 else 0
            best = min(best, score)
            step_p = GAIN * e_p * min(1.0, max_pos_step / max(1e-9, float(np.linalg.norm(e_p))))
            step_r = GAIN * e_r * min(1.0, MAX_ROT_STEP_RAD / max(1e-9, float(np.linalg.norm(e_r))))
            e = np.concatenate([step_p, ROT6_W * step_r])
            before = self.joints()
            cmd = dict(self.s.hold) if self.s.hold else before
            dq = self.solve6(Wd @ self.J6, Wd @ e, cmd, joints)
            target = {**cmd, **{j: cmd[j] + dq[k] for k, j in enumerate(joints)}}
            for j in joints:
                target[j] = min(before[j] + LEAD_MAX_DEG, max(before[j] - LEAD_MAX_DEG, target[j]))
            y_before = self.y6(m, m["R"])
            self.move(target, f"{label} step")
            self.hold(SETTLE_S)
            if torque_guard:
                now = self.torques()
                jumps = {j: now[j] - base_torque[j] for j in now
                         if j != "gripper" and abs(now[j] - base_torque[j]) > TORQUE_JUMP_NM[j]}
                if jumps:
                    log(f"torque change: { {j: round(v, 2) for j, v in jumps.items()} } -> treating as contact")
                    return None
            after = self.joints()
            dq_actual = np.array([after[j] - before[j] for j in joints])
            m2 = self.measure(SERVO_FRAMES)
            dy = self.y6(m2, m["R"]) - y_before
            if np.linalg.norm(dq_actual) > 1.0:
                J = self.J6
                self.J6 = J + 0.5 * np.outer(dy - J @ dq_actual, dq_actual) / float(dq_actual @ dq_actual)
        raise Abort(f"{label}: did not converge in {max_iter} iterations")

    def grasp_target(self, up_m: float = 0.0, offset_tag=None):
        """Tag pose in the camera frame that puts the open fingertip midpoint at the plate centre,
        GRASP_Z (+up) above the table, jaws along the tag X axis (the plate's 85.5 mm side)."""
        a = math.radians(GRASP_PITCH_DEG)
        Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
        R_t = Rx @ np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
        offset_tag = GRASP_OFFSET_TAG if offset_tag is None else offset_tag
        G = PLATE_CENTER_TAG + offset_tag + np.array([0, 0, GRASP_Z + up_m])
        return R_t, FINGER_MID_CAM - R_t @ G

    def tip_target(self, tip_cam: np.ndarray, point_tag: np.ndarray):
        """Tag pose in the camera frame that puts camera point tip_cam at tag-frame point point_tag,
        camera looking straight down with its x axis along tag X (same orientation as the grasp)."""
        a = math.radians(GRASP_PITCH_DEG)
        Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
        R_t = Rx @ np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
        return R_t, tip_cam - R_t @ point_tag

    def touch_probe(self, x_mm: float, y_mm: float):
        """Lower the closed jaw tips onto tag-frame point (x, y) until contact; report the height."""
        if ACTIVE_TAG["id"] != 1:
            raise Abort("select the plate tag first: `target 1`")
        self.require_motion_ok()
        if self.joints()["gripper"] < -20:
            self.close_gripper(max_close_deg=-3.0)
            if self.joints()["gripper"] < -20:
                raise Abort("jaws met something while closing for the touch probe; not probing")
        xy = np.array([x_mm, y_mm]) / 1000
        R_t, p_t = self.tip_target(CLOSED_TIP_CAM, np.array([*xy, 0.035]))
        self.servo_pose(R_t, p_t, 1.5, 2.5, f"probe-touch above ({x_mm:.0f},{y_mm:.0f})", max_iter=40)
        h = 0.035
        while h > -0.006:
            h -= 0.002
            R_t, p_t = self.tip_target(CLOSED_TIP_CAM, np.array([*xy, h]))
            m = self.servo_pose_soft(R_t, p_t, 1.0, 3.0, f"probe-touch h {h * 1000:.0f}", max_iter=4, reprobe=False,
                                     torque_guard=True, max_pos_step=0.003)
            if m is None:
                break
        m = self.measure(SERVO_FRAMES)
        tip = m["R"].T @ (CLOSED_TIP_CAM - m["p"])
        log(f"TOUCH-PROBE at ({x_mm:.0f},{y_mm:.0f}): contact with tip at tag-frame "
            f"{np.round(tip * 1000, 1).tolist()} mm (height {tip[2] * 1000:.1f} mm)")
        R_t, p_t = self.tip_target(CLOSED_TIP_CAM, np.array([*xy, tip[2] + 0.02]))
        self.servo_pose_soft(R_t, p_t, 3.0, 4.0, "probe-touch retreat", max_iter=6, reprobe=False)
        return tip

    def cmd_arm(self) -> dict[str, float]:
        return {j: round(self.s.hold[j], 2) for j in ARM6}

    def learned(self) -> dict:
        return json.loads(PICK_LEARNED.read_text()) if PICK_LEARNED.exists() else {}

    def save_learned(self, **kw):
        d = {**self.learned(), **kw}
        PICK_LEARNED.write_text(json.dumps(d, indent=1, default=lambda o: np.asarray(o).tolist()))

    def pregrasp(self):
        if ACTIVE_TAG["id"] != 1:
            raise Abort("select the plate tag first: `target 1`")
        self.stage = "pregrasp"
        L = self.learned()
        if "pregrasp_joints" in L and "J6" in L:
            # Go straight to where this worked before (via the viewing pose), then only refine.
            if self.joints()["shoulder_lift"] > -30.0:
                # Unfold with the wrist straight first (wrist_flex has a stop near the folded pose).
                self.move_segmented({**VIEW_POSE, "wrist_flex": 0.0}, "unfold")
                self.move_segmented({**VIEW_POSE, "gripper": GRIPPER_OPEN_DEG}, "to viewing pose")
            self.move_segmented({**L["pregrasp_joints"], "gripper": GRIPPER_OPEN_DEG}, "to learned pregrasp")
            self.J6 = np.array(L["J6"])
            periodic = False
        else:
            periodic = True
        R_t, p_t = self.grasp_target(PREGRASP_UP_M)
        log(f"pregrasp target: tag at {np.round(p_t * 1000, 1).tolist()} mm in the camera frame")
        self.servo_pose(R_t, p_t, 3.0, 3.0, "pregrasp", max_iter=120, periodic=periodic, axis_w=PREGRASP_AXIS_W)
        q = self.joints()
        m = self.measure(PROBE_FRAMES)
        self.save_learned(pregrasp_joints={j: round(q[j], 2) for j in ARM6}, J6=self.J6,
                          pregrasp_cmd=self.cmd_arm(), pregrasp_tag_p=m["p"], pregrasp_tag_R=m["R"])
        log("PREGRASP reached: open fingertips 6 cm above the plate centre.")
        self.note = "pregrasp"

    def fast_to(self, pose: dict[str, float], speed: dict[str, float], settle: float = 0.3, label: str = ""):
        target = {**self.s.hold, **pose}
        if not self.path:
            self.path.append(self.joints())
        self.glide(target, speed=speed, settle=settle)
        self.path.append(dict(target))

    def gap_now(self) -> float | None:
        self.hold(0.2)
        return pad_gap_px(self.cam.latest(self.cam.seq)[1])

    def fastpick(self):
        """Replay a learned pick quickly: ~20 s with the arm stretched out instead of minutes."""
        L = self.learned()
        need = ("pregrasp_cmd", "grasp_cmd", "lift_cmd", "pregrasp_tag_p", "pregrasp_tag_R", "J6")
        if not all(k in L for k in need):
            raise Abort(f"no learned pick yet (missing {[k for k in need if k not in L]}); run `pick` once")
        if ACTIVE_TAG["id"] != 1:
            raise Abort("select the plate tag first: `target 1`")
        temps = {k: st.t_mos for k, st in self.follower._read_feedback().items()}
        if temps["shoulder_lift"] > FASTPICK_START_MAX_C:
            raise Abort(f"shoulder at {temps['shoulder_lift']:.0f} C; let it cool below {FASTPICK_START_MAX_C:.0f} C")
        self.require_motion_ok()
        t0 = time.monotonic()
        if not self.s.torque:
            self.safe_enable()
        self.path = []
        pre, grasp, lift = L["pregrasp_cmd"], L["grasp_cmd"], L["lift_cmd"]
        # Out: unfold with the wrist straight (stop near the fold), open the jaws on the way.
        self.fast_to({**VIEW_POSE, "wrist_flex": 0.0, "gripper": GRIPPER_OPEN_DEG}, FAST_DEG_S, 0.0)
        self.fast_to(VIEW_POSE, FAST_DEG_S, 0.0)
        t_out = time.monotonic()
        self.fast_to(pre, FAST_DEG_S, 0.5)
        # One look: is the plate where it was when this was learned?
        m = self.measure(SERVO_FRAMES)
        e_p = np.array(L["pregrasp_tag_p"]) - m["p"]
        e_r = rotvec(np.array(L["pregrasp_tag_R"]) @ m["R"].T)
        log(f"fastpick: tag offset at pregrasp {np.round(e_p * 1000, 1).tolist()} mm, "
            f"{np.round(np.degrees(e_r), 1).tolist()} deg")
        shift = np.zeros(len(ARM6))
        if np.linalg.norm(e_p) * 1000 > REPLAY_MAX_ERR_MM:
            Wd = np.diag(np.concatenate([np.ones(3), PREGRASP_AXIS_W]))
            shift = self.solve6(Wd @ np.array(L["J6"]), Wd @ np.concatenate([e_p, ROT6_W * e_r]), self.s.hold, ARM6)
            log(f"fastpick: one-shot correction {np.round(shift, 2).tolist()} deg")
        add = lambda pose: {j: pose[j] + shift[k] for k, j in enumerate(ARM6)}
        # Down onto the plate (slower for the last few cm), close, lift, check.
        self.fast_to(add(grasp), NEAR_DEG_S, 0.4)
        gap = self.close_gripper()
        gripped = gap is not None
        self.fast_to(add(lift), NEAR_DEG_S, 0.3)
        gap_lift = self.gap_now()
        held = gripped and gap_lift is not None and GRIP_HELD_MIN_GAP_PX < gap_lift < GRIP_HELD_MAX_GAP_PX
        DATA.mkdir(exist_ok=True)
        cv2.imwrite(str(DATA / "fastpick_lift.png"), self.cam.latest()[1])
        log(f"fastpick: pad gap closed {gap}, lifted {gap_lift} -> {'PLATE HELD' if held else 'NOT HELD'}")
        # Put it back, release, and go home.
        self.fast_to(add(grasp), NEAR_DEG_S, 0.3)
        self.fast_to({"gripper": GRIPPER_OPEN_DEG}, FAST_DEG_S, 0.3)
        self.fast_to(add(pre), NEAR_DEG_S, 0.0)
        self.fast_to(VIEW_POSE, FAST_DEG_S, 0.3)
        self.close_gripper(max_close_deg=-3.0)
        t_back = time.monotonic()
        self.fast_to({"wrist_flex": 0.0}, FAST_DEG_S, 0.3)
        self.path = []
        self.rest()
        temps = {k: st.t_mos for k, st in self.follower._read_feedback().items()}
        log(f"FASTPICK {'SUCCEEDED' if held else 'FAILED'}: {time.monotonic() - t0:.0f} s total, "
            f"{t_back - t_out:.0f} s beyond the viewing pose; shoulder {temps['shoulder_lift']:.0f} C")

    def pick(self):
        """Whole cycle: approach, grasp, lift + check, put back, return to rest."""
        temps = self.fresh_temps()
        if temps["shoulder_lift"] > PICK_START_MAX_C:
            raise Abort(f"shoulder at {temps['shoulder_lift']:.0f} C; let it cool below {PICK_START_MAX_C:.0f} C first")
        t0 = time.monotonic()
        if not self.s.torque:
            self.safe_enable()
        self.pregrasp()
        self.grasp()
        held = self.stage == "holding"
        self.place()
        self.save_learned(J6=self.J6)
        self.move_segmented(VIEW_POSE, "to viewing pose")
        self.close_gripper(max_close_deg=-3.0)
        self.move_segmented({"wrist_flex": 0.0}, "straighten wrist")
        self.path = []  # fold straight from here; no need to retrace the servo path
        self.rest()
        temps = {k: st.t_mos for k, st in self.follower._read_feedback().items()}
        log(f"PICK CYCLE {'SUCCEEDED' if held else 'FAILED'} in {time.monotonic() - t0:.0f} s; "
            f"shoulder {temps['shoulder_lift']:.0f} C, elbow {temps['elbow_flex']:.0f} C")

    def fresh_temps(self) -> dict[str, float]:
        """Driver temperatures from replies to a fresh request: with torque off nothing refreshes the cached
        states, and _read_feedback polls before the replies arrive (the pick gate saw 56 C for 10 min at rest
        while the shoulder read 40 C, 2026-10-21). Disabled RobStride motors answer only a disable frame."""
        if not self.s.torque:
            self.fresh_joints_limp()
        else:
            for m in self.follower.motors.values():
                m.request_feedback()
            time.sleep(0.05)
        return {k: st.t_mos for k, st in self.follower._read_feedback().items()}

    def thermal_check(self):
        temps = {k: st.t_mos for k, st in self.follower._read_feedback().items()}
        hot = {k: t for k, t in temps.items() if t >= THERMAL_ABORT_C}
        if hot:
            raise Abort(f"motor driver too hot {hot}; stopping (rest the arm to cool)")
        warm = {k: t for k, t in temps.items() if t >= THERMAL_WARN_C}
        peak = max(temps.values())
        if warm and peak > getattr(self, "_warned_peak", 0.0):
            self._warned_peak = peak
            log(f"warning: motors warm {warm}")
        elif not warm:
            self._warned_peak = 0.0

    def servo_pose_soft(self, *args, **kw):
        """servo_pose that returns the latest measurement instead of failing when it runs out of iterations."""
        try:
            return self.servo_pose(*args, **kw)
        except Abort as e:
            if "did not converge" not in str(e):
                raise
            log(f"({e}; continuing)")
            return self.measure(SERVO_FRAMES)

    def gripper_state(self):
        st = self.follower._read_feedback()["gripper"]
        return self.follower._motor_to_public_position("gripper", math.degrees(st.pos)), st.torq

    def close_gripper(self, max_close_deg: float = GRIP_MAX_CLOSE_DEG) -> float | None:
        """Close in small steps and stop the moment the jaws meet resistance.

        Contact = closing torque beyond GRIP_CONTACT_NM on two consecutive reads, or the jaws fall GRIP_CONTACT_LAG_DEG behind the command, or the
        camera shows the pads stopped. The command then goes only GRIP_SQUEEZE_DEG past the contact
        angle (and the driver clamps torque to GRIP_HOLD_TORQUE_LIMIT_NM). Never commands past
        max_close_deg. Returns the camera pad gap if the jaws stopped on something plate-wide."""
        self.apply_grip_limits()
        cmd = dict(self.s.hold)
        g = cmd["gripper"]
        t_end = time.time() + 3.0  # opening lags the command a lot; start from settled jaws
        while abs(self.gripper_state()[0] - g) > 2.0 and time.time() < t_end:
            self.s.command(cmd)
            time.sleep(GRIP_STEP_S)
        g = start_g = self.gripper_state()[0]
        hits, lags, last_gap, contact = 0, 0, None, None
        while g + GRIP_STEP_DEG <= max_close_deg:
            g += GRIP_STEP_DEG  # closing = towards 0
            self.s.command({**cmd, "gripper": g})
            time.sleep(GRIP_STEP_S)
            pos, tq = self.gripper_state()
            hits = hits + 1 if -tq > GRIP_CONTACT_NM else 0  # closing pushes negative torque
            lag = g - pos
            lags = lags + 1 if lag > GRIP_CONTACT_LAG_DEG else 0
            if g - start_g < GRIP_BREAKAWAY_DEG:  # the jaws stick briefly as they start to move
                hits = lags = 0
            why = ("torque" if hits >= 2 else "lag" if lags >= 2 else None)
            if why is None and int(g) % 8 == 0:  # free air: ~0.8 px/deg
                gap = pad_gap_px(self.cam.latest(self.cam.seq)[1])
                if gap is not None and last_gap is not None and abs(gap - last_gap) < 2.0:
                    why = "camera (pads stopped)"
                last_gap = gap
            if why:
                contact = pos
                squeeze = min(pos + GRIP_SQUEEZE_DEG, max_close_deg)  # never past the close limit
                self.s.command({**cmd, "gripper": squeeze})
                log(f"jaw contact by {why}: gripper {pos:.1f} deg, torque {tq:+.2f} Nm, lag {lag:.1f} deg, "
                    f"command now {squeeze:.1f}")
                break
        if contact is None:
            log(f"no contact before the close limit {max_close_deg:.0f} deg; stopping there")
        else:
            # The camera can call contact a few degrees early; make sure the jaws really press (the
            # hold torque cap still bounds the force), re-squeezing from where they actually stopped.
            for _ in range(2):
                self.hold(0.4)
                pos, tq = self.gripper_state()
                cmd_g = self.s.hold["gripper"]
                if pos - cmd_g < -GRIP_RESQUEEZE_SHORT_DEG or cmd_g >= max_close_deg:
                    break
                new_g = min(pos + GRIP_SQUEEZE_DEG, max_close_deg)
                self.s.command({**self.s.hold, "gripper": new_g})
                log(f"re-squeeze: jaws at {pos:.1f} deg, only {cmd_g - pos:.1f} short of {cmd_g:.1f}; command now {new_g:.1f}")
        self.hold(0.6)
        gap = pad_gap_px(self.cam.latest(self.cam.seq)[1])
        log(f"jaws: pad gap {gap} px (open ~260, on the plate ~205)")
        held = contact is not None and gap is not None and GRIP_HELD_MIN_GAP_PX < gap < GRIP_HELD_MAX_GAP_PX
        return gap if held else None

    def gripcal(self, max_close_deg: float):
        """Free-air check of the sensing close (nothing between the jaws): open, then close in
        steps to max_close_deg, logging the torque/lag trace both ways so GRIP_CONTACT_NM can be set."""
        if not self.s.torque:
            self.safe_enable()
        self.apply_grip_limits()
        cmd = dict(self.s.hold)

        def sweep(start, stop, step):
            trace, g = [], start
            while (stop - g) * step > 0:
                g += step
                self.s.command({**cmd, "gripper": g})
                time.sleep(GRIP_STEP_S)
                pos, tq = self.gripper_state()
                trace.append((round(g, 1), round(pos, 1), round(tq, 3)))
            return trace

        tr = sweep(cmd["gripper"], GRIPPER_OPEN_DEG, -2.0)
        self.hold(0.5)
        tq = np.array([t[2] for t in tr])
        log(f"gripcal open: torque {tq.min():+.3f}..{tq.max():+.3f} Nm, max lag "
            f"{max(abs(t[0] - t[1]) for t in tr):.1f} deg, pad gap {pad_gap_px(self.cam.latest(self.cam.seq)[1])} px; "
            f"every 10th {tr[::10]}")
        base = []
        for _ in range(5):
            self.s.command({**cmd, "gripper": GRIPPER_OPEN_DEG})
            time.sleep(GRIP_STEP_S)
            base.append(self.gripper_state()[1])
        b = float(np.median(base))
        tr = sweep(GRIPPER_OPEN_DEG, max_close_deg, GRIP_STEP_DEG)
        dev = np.array([abs(t[2] - b) for t in tr])
        log(f"gripcal close: baseline {b:+.3f} (still: {np.round(base, 3).tolist()}), |torque-baseline| "
            f"median {np.median(dev):.3f} p95 {np.percentile(dev, 95):.3f} max {dev.max():.3f} Nm, max lag "
            f"{max(t[0] - t[1] for t in tr):.1f} deg; every 10th {tr[::10]}")
        self.hold(0.5)
        log(f"jaws at {max_close_deg:.0f}: pad gap {pad_gap_px(self.cam.latest(self.cam.seq)[1])} px")
        prof = json.loads(GRIP_PROFILE.read_text()) if GRIP_PROFILE.exists() else []
        prof.append({"cmd": [t[0] for t in tr], "pos": [t[1] for t in tr], "torque": [t[2] for t in tr]})
        GRIP_PROFILE.write_text(json.dumps(prof[-5:]))
        log(f"saved free-air close profile #{len(prof[-5:])} to {GRIP_PROFILE.name}")

    def grasp(self):
        """Pick the 96-well plate: align above it, descend, close on contact, lift and verify."""
        if ACTIVE_TAG["id"] != 1:
            raise Abort("select the plate tag first: `target 1`")
        self.require_motion_ok()
        self.stage = "grasp"
        if self.joints()["gripper"] > GRIPPER_OPEN_DEG + 5:
            self.move({**self.s.hold, "gripper": GRIPPER_OPEN_DEG}, "open gripper")
        R_t, p_t = self.grasp_target(GRASP_APPROACH_UP_M)
        # From here down the fingers are held parallel to the plate top on every axis (wrist_flex
        # is ~70 deg here, inside its 80 deg limit; only the pregrasp 6 cm up needs the relaxed axis).
        self.servo_pose(R_t, p_t, 1.5, 1.0, "align", max_iter=40, periodic=False, axis_w=ROT6_AXIS_W)
        align_cmd = self.cmd_arm()
        # Descend along the tag normal with continuous alignment and a contact guard.
        up = GRASP_APPROACH_UP_M
        contact_up = None
        while up > 0.0005:
            up = max(0.0, up - DESCEND_STEP_M)
            R_t, p_t = self.grasp_target(up)
            m = self.servo_pose_soft(R_t, p_t, 1.5, 1.5, f"descend {up * 1000:.0f}", max_iter=6, reprobe=False,
                                torque_guard=True, max_pos_step=0.004, axis_w=ROT6_AXIS_W)
            if m is None:
                contact_up = up
                break
        m = self.measure(SERVO_FRAMES)
        # Fingertip height above the table from the tag pose (camera frame -> tag frame).
        tip_tag = m["R"].T @ (FINGER_MID_CAM - m["p"])
        tilt = np.degrees(rotvec(m["R"] @ R_t.T))
        log(f"finger tilt vs plate top: {tilt[0]:+.1f} deg along the plate, {tilt[1]:+.1f} deg across the jaws")
        log(f"fingertips at {tip_tag[2] * 1000:.1f} mm above the table (plate top ~{PLATE_TOP_Z * 1000:.1f}, "
            f"walls ~10); x/y offset from the grasp point "
            f"{np.round((tip_tag[:2] - PLATE_CENTER_TAG[:2] - GRASP_OFFSET_TAG[:2]) * 1000, 1).tolist()} mm")
        if np.abs(tilt[:2]).max() > GRASP_MAX_TILT_DEG:
            raise Abort(f"fingers not parallel to the plate top (tilt {np.round(tilt[:2], 1).tolist()} deg); not closing")
        if tip_tag[2] > PLATE_TOP_Z - 0.002:
            raise Abort("fingers stopped on the plate rim (misaligned across the jaws); not closing")
        grasp_cmd = self.cmd_arm()
        contact = self.close_gripper()
        if contact is None:
            self.move({**self.s.hold, "gripper": GRIPPER_OPEN_DEG}, "re-open")
            raise Abort("grasp missed")
        # Lift and check the plate came along: jaws must stay at the plate width.
        frame_before = self.cam.latest()[1]
        R_t, p_t = self.grasp_target(LIFT_M)
        self.servo_pose_soft(R_t, p_t, 4.0, 4.0, "lift", max_iter=12, reprobe=False, max_pos_step=0.01)
        self.hold(0.5)
        frame_after = self.cam.latest()[1]
        DATA.mkdir(exist_ok=True)
        cv2.imwrite(str(DATA / "grasp_before_lift.png"), frame_before)
        cv2.imwrite(str(DATA / "grasp_after_lift.png"), frame_after)
        gap_after = pad_gap_px(frame_after)
        held = gap_after is not None and GRIP_HELD_MIN_GAP_PX < gap_after < GRIP_HELD_MAX_GAP_PX
        if held:  # a pick that worked: remember its commands for fast replay
            self.save_learned(align_cmd=align_cmd, grasp_cmd=grasp_cmd, lift_cmd=self.cmd_arm())
        log(f"LIFT: pad gap {gap_after} px (at grasp {contact:.0f}) -> {'PLATE HELD' if held else 'plate slipped/lost'}")
        calib = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "pad_gap_px": contact, "pad_gap_after_lift_px": gap_after, "held": held,
                 "fingertip_tag_mm": np.round(tip_tag * 1000, 1).tolist(), "contact_up_mm": None if contact_up is None else contact_up * 1000}
        history = json.loads(PICK_CALIB.read_text()) if PICK_CALIB.exists() else []
        PICK_CALIB.write_text(json.dumps(history + [calib], indent=1))
        self.stage = "holding" if held else "grasp"
        self.note = "holding the plate" if held else "grasp failed"

    def place(self):
        """Put the plate back into its holder and back off to the pregrasp pose."""
        self.require_motion_ok()
        R_t, p_t = self.grasp_target(0.004)
        self.servo_pose_soft(R_t, p_t, 2.0, 3.0, "place", max_iter=15, reprobe=False, max_pos_step=0.006, torque_guard=True)
        self.move({**self.s.hold, "gripper": GRIPPER_OPEN_DEG}, "release")
        self.hold(0.5)
        R_t, p_t = self.grasp_target(PREGRASP_UP_M)
        self.servo_pose_soft(R_t, p_t, 5.0, 5.0, "retreat", max_iter=15, reprobe=False)
        self.stage = "pregrasp"
        log("PLACED: plate released in the holder; back above it.")

    def flat_target(self, up_m: float, offset_tag=None):
        a = math.radians(FLAT_PITCH_DEG)
        Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
        R_t = Rx @ np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
        offset_tag = FLAT_VIEW_OFFSET_TAG if offset_tag is None else offset_tag
        G = PLATE_CENTER_TAG + offset_tag + np.array([0, 0, GRASP_Z + up_m])
        return R_t, FINGER_MID_CAM - R_t @ G

    def tip_height(self) -> float | None:
        try:
            m = self.measure(SERVO_FRAMES)
        except Exception:
            return None
        return float((m["R"].T @ (FINGER_MID_CAM - m["p"]))[2])

    def torque_jumps(self, base: dict, limits: dict | None = None) -> dict:
        lim = limits or TORQUE_JUMP_NM
        now = self.torques()
        return {j: round(now[j] - base[j], 2) for j in now
                if j != "gripper" and abs(now[j] - base[j]) > lim[j]}

    def move_exact(self, target: dict[str, float], label: str, tol_deg: float = 0.3, iters: int = 5,
                   guard_base: dict | None = None, jump_limits: dict | None = None) -> bool:
        """Move until the MEASURED joints reach target (the arm sags under gravity, so a plain command
        falls short). The J6 used for blind moves maps measured angles, so this keeps those moves true.
        With guard_base (torques at the start of this move), returns False as soon as any joint torque
        jumps (contact), before the corrections can push harder."""
        q = self.joints()
        cmd = {j: target[j] + (self.s.hold[j] - q[j]) for j in target}  # carry the current sag over
        err = {j: 0.0 for j in target}
        if max(abs(cmd[j] - self.s.hold[j]) for j in target) > 8.0:  # large: segmented first, then refine
            self.move_segmented(cmd, label)
            self.hold(0.3)
            q = self.joints()
            cmd = {j: cmd[j] + (target[j] - q[j]) for j in target}
        for _ in range(iters):
            cmd = {j: min(target[j] + LEAD_MAX_DEG, max(target[j] - LEAD_MAX_DEG, cmd[j])) for j in cmd}
            self.move({**self.s.hold, **cmd}, label)
            self.hold(0.3)
            if guard_base is not None:
                jumps = self.torque_jumps(guard_base, jump_limits)
                if jumps and set(jumps) <= {"shoulder_lift", "elbow_flex"}:
                    # big joints: hold the command 0.5 s and re-read; contact keeps pressing, the lone spikes
                    # settle (shoulder 3.25-3.76 Nm in free air four times on 2026-10-29; user's OK, option 1)
                    self.hold(0.5)
                    still = self.torque_jumps(guard_base, jump_limits)
                    if not still:
                        log(f"{label}: torque change {jumps} settled; not contact")
                        jumps = {}
                    else:
                        log(f"{label}: torque change {jumps}, still {still} after 0.5 s -> contact")
                        return False
                if jumps:
                    log(f"{label}: torque change {jumps} -> contact")
                    return False
            q = self.joints()
            err = {j: target[j] - q[j] for j in target}
            if max(abs(v) for v in err.values()) < tol_deg:
                return True
            cmd = {j: cmd[j] + err[j] for j in cmd}
        log(f"{label}: joints within {max(abs(v) for v in err.values()):.1f} deg of the plan after {iters} corrections")
        return True

    def move_exact_guarded(self, target: dict[str, float], label: str, steps: int = 6) -> bool:
        """Guarded approach in `steps` pieces. Intermediate pieces are single commands carrying the current
        sag over, each checked against the torques at its own start: the correction loop of move_exact
        raises a stuck joint's command until it breaks free, and that torque spike looks like contact.
        Only the last piece is corrected (small)."""
        self.hold(0.3)  # let the previous move settle before taking torque baselines
        q0 = self.joints()
        for i in range(1, steps):
            sub = {j: q0[j] + (target[j] - q0[j]) * i / steps for j in target}
            q = self.joints()
            base = self.torques()
            self.move({**self.s.hold, **{j: sub[j] + (self.s.hold[j] - q[j]) for j in sub}}, f"{label} {i}/{steps}")
            self.hold(0.3)
            now = self.torques()
            # the joint nearest its limit, logged on every piece to see the false-trip margin
            j_max = max((j for j in now if j != "gripper"), key=lambda j: abs(now[j] - base[j]) / TORQUE_JUMP_NM[j])
            log(f"{label} {i}/{steps}: largest torque change {j_max} {now[j_max] - base[j_max]:+.2f} Nm "
                f"(limit {TORQUE_JUMP_NM[j_max]})")
            jumps = self.torque_jumps(base)
            if jumps:  # contact keeps pressing; the gravity-load and breakaway spikes of a move settle
                self.hold(0.5)
                still = self.torque_jumps(base)
                if not still:
                    log(f"{label} {i}/{steps}: torque change {jumps} settled; not contact")
                    continue
                log(f"{label} {i}/{steps}: torque change {jumps}, still {still} after 0.5 s -> contact")
                return False
        return self.move_exact(target, f"{label} {steps}/{steps}", iters=3, guard_base=self.torques())

    def jog_tip(self, d_tag, R_t: np.ndarray, label: str, guard: bool = False,
                step_m: float = FLAT_STEP_M) -> bool:
        """Blind fingertip move by d_tag (tag frame, m) on the J6 learned at the flat pose, in
        <= step_m steps, each driven until the measured joints match. With guard, each step watches
        for a torque jump against the torques at its own start (slow gravity-load drift over a long
        move is not contact) and the move stops and returns False on contact."""
        d_tag = np.asarray(d_tag, float)
        Wd = np.diag(np.concatenate([np.ones(3), FLAT_AXIS_W]))
        n = max(1, int(math.ceil(np.linalg.norm(d_tag) / step_m)))
        q0 = self.joints()
        cum = np.zeros(len(ARM6))
        for i in range(n):
            self.thermal_check()
            e = np.concatenate([-R_t @ (d_tag / n), np.zeros(3)])  # how the tag moves in the camera frame
            q_plan = {j: q0[j] + cum[k] for k, j in enumerate(ARM6)}
            cum += self.solve6(Wd @ self.J6, Wd @ e, q_plan, ARM6)
            ok = self.move_exact({j: q0[j] + cum[k] for k, j in enumerate(ARM6)}, f"{label} step",
                                 guard_base=self.torques() if guard else None)
            if not ok:
                log(f"{label}: contact after {np.linalg.norm(d_tag) * (i + 1) / n * 1000:.0f} mm")
                return False
        return True

    def gripping(self) -> bool:
        """True if something holds the jaws open short of their command (an empty gripper reaches it)."""
        pos, tq = self.gripper_state()
        short = pos - self.s.hold["gripper"]  # closing is towards 0: blocked jaws sit below the command
        log(f"grip check: jaws {pos:.1f} deg vs command {self.s.hold['gripper']:.1f} ({short:+.1f}), torque {tq:+.2f} Nm")
        return short < -GRIP_HELD_MIN_SHORT_DEG

    def flat_approach(self, probe: bool = True):
        """Closed jaws, gripper pitched FLAT_PITCH_DEG, fingertips FLAT_ALIGN_UP_M above the grasp height,
        at the viewing offset (tips 22 mm past the plate centre, where the tag stays in the image)."""
        if ACTIVE_TAG["id"] != 1:
            raise Abort("select the plate tag first: `target 1`")
        temps = self.fresh_temps()
        if temps["shoulder_lift"] > PICK_START_MAX_C:
            raise Abort(f"shoulder at {temps['shoulder_lift']:.0f} C; let it cool below {PICK_START_MAX_C:.0f} C first")
        self.require_motion_ok()
        if not self.s.torque:
            self.safe_enable()
        L = self.learned()
        g_app = FLAT_APPROACH_GRIPPER_DEG
        if self.joints()["gripper"] < g_app - 5:
            self.close_gripper(max_close_deg=g_app)
        if self.joints()["shoulder_lift"] > -30.0:
            self.move_segmented({**VIEW_POSE, "wrist_flex": 0.0, "gripper": g_app}, "unfold")
            self.move_segmented({**VIEW_POSE, "gripper": g_app}, "to viewing pose")
        if self.joints()["shoulder_lift"] > -50.0 or getattr(self, "J6", None) is None:  # else continue from here
            start = L.get("flat_pregrasp_joints") or L["pregrasp_joints"]
            self.move_segmented({**start, "gripper": g_app}, "to pregrasp")
            self.J6 = np.array(L.get("flat_J6") or L["J6"])
        R_t, p_t = self.flat_target(FLAT_PREGRASP_UP_M)
        self.servo_pose(R_t, p_t, 3.0, 4.5, "flat pregrasp", max_iter=60, periodic=False, axis_w=FLAT_AXIS_W)
        # A full probe here (5 cm up, clear of the plate): the blind slide and descent rely on every
        # column of J6, and servo updates only correct the directions the servo happened to move.
        if probe:
            self.J6 = self.measure_jacobian6(ARM6)
        q = self.joints()
        self.save_learned(flat_pregrasp_joints={j: round(q[j], 2) for j in ARM6}, flat_J6=self.J6)
        R_t, p_t = self.flat_target(FLAT_ALIGN_UP_M)
        m = self.servo_pose(R_t, p_t, 1.5, 2.0, "flat align", max_iter=40, periodic=False, axis_w=FLAT_AXIS_W)
        tilt = np.degrees(rotvec(m["R"] @ R_t.T))
        log(f"flat align: finger tilt vs plate top {tilt[0]:+.1f} deg along the plate, {tilt[1]:+.1f} deg across the jaws"
            f"{' (tracked)' if m.get('tracked') else ''}")
        if np.abs(tilt[:2]).max() > GRASP_MAX_TILT_DEG:
            raise Abort(f"fingers not parallel to the plate top (tilt {np.round(tilt[:2], 1).tolist()} deg)")
        self.save_learned(flat_J6=self.J6, flat_align_cmd=self.cmd_arm())
        self.flat_align_tag = (m["p"], m["R"])
        self.flat_align_q = {j: round(self.joints()[j], 2) for j in ARM6}
        self.flat_align_frame = self.cam.latest(self.cam.seq)[1].copy()
        return R_t

    def flat_descend(self, tip_z_m: float, label: str):
        """Servo the fingertip point (viewing offset) down to tip_z_m above the table in 2 mm steps
        (vision + torque guard). Returns the tip height reached, and whether the guard saw contact."""
        up = FLAT_ALIGN_UP_M
        floor = tip_z_m - GRASP_Z
        base = self.torques()  # one baseline for the whole descent, so slowly building pressure still trips
        while up > floor + 1e-4:
            up = max(floor, up - FLAT_STEP_M)
            R_t, p_t = self.flat_target(up)
            m = self.servo_pose_soft(R_t, p_t, 1.0, 2.0, f"{label} {(GRASP_Z + up) * 1000:.1f}", max_iter=8,
                                     reprobe=False, torque_guard=True, max_pos_step=0.003, axis_w=FLAT_AXIS_W,
                                     torque_base=base)
            if m is None:
                return self.tip_height(), True
        return self.tip_height(), False

    def flatcal(self):
        """Find how far the fingertip point sits above the flat fingers' lower edge: lower the closed
        jaws onto the plate top until the torque guard trips, then read the tip height from the tag."""
        self.flat_approach()
        if self.joints()["gripper"] < FLAT_APPROACH_GRIPPER_DEG - 10:
            raise Abort(f"flatcal: jaws not closed (gripper {self.joints()['gripper']:.0f} deg); not touching down")
        h, met = self.flat_descend(PLATE_TOP_Z - 0.010, "flatcal touch")
        if not met or h is None:
            log(f"flatcal: no contact (tip at {h})")
        else:
            off = h - PLATE_TOP_Z
            self.save_learned(flat_tip_above_edge=round(off, 4))
            log(f"FLATCAL: closed fingers met the plate top with the tip point at {h * 1000:.1f} mm -> "
                f"tip point is {off * 1000:+.1f} mm above the fingers' lower edge (saved)")
        R_t, p_t = self.flat_target(FLAT_ALIGN_UP_M)
        self.servo_pose_soft(R_t, p_t, 3.0, 4.0, "flatcal retreat", max_iter=8, reprobe=False, axis_w=FLAT_AXIS_W)

    # ---- camera checks around the plate
    def scene_crop(self, frame):
        """Grey image below the fingers (the top of the image is finger), for before/after comparisons."""
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(g[SCENE_ROW0:, :], (7, 7), 0).astype(np.float32)

    @staticmethod
    def ncc(a, b) -> float:
        a = a - a.mean()
        b = b - b.mean()
        return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))

    def between_pads(self, frame):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(g[PAD_BAND_ROWS[0]:PAD_BAND_ROWS[1], PAD_BAND_COLS[0]:PAD_BAND_COLS[1]], (5, 5), 0).astype(np.float32)

    def plate_rot_shift(self, taught, frame):
        """(angle_deg, dx, dy, match) of the plate between the pads, current vs taught close view: the
        Euclidean image motion (ECC) over the plate between the pads. The pads move with the camera, so
        they are masked out. A plate turned CCW (seen from above) from the taught relation gives a
        negative angle."""
        A = cv2.GaussianBlur(cv2.cvtColor(taught, cv2.COLOR_BGR2GRAY), (0, 0), 1.5).astype(np.float32)
        B = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (0, 0), 1.5).astype(np.float32)
        mask = np.zeros(A.shape, np.uint8)
        mask[0:205, 235:465] = 255
        W = np.eye(2, 3, dtype=np.float32)
        try:
            cc, W = cv2.findTransformECC(A, B, W, cv2.MOTION_EUCLIDEAN,
                                         (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5), mask, 5)
        except cv2.error:
            return 0.0, 0.0, 0.0, 0.0
        return math.degrees(math.atan2(W[1, 0], W[0, 0])), float(W[0, 2]), float(W[1, 2]), float(cc)

    @staticmethod
    def turn_about_vertical(C, p, deg):
        """Pose C turned by deg (CCW seen from above) about the vertical line through point p."""
        a = math.radians(deg)
        G = np.eye(4)
        G[:3, :3] = [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
        G[:3, 3] = p - G[:3, :3] @ p
        return G @ C

    def note_holder_view(self, info: dict):
        v = getattr(self, "holder_view", None)
        if v is not None and v.with_suffix(".json").exists():
            d = json.loads(v.with_suffix(".json").read_text())
            v.with_suffix(".json").write_text(json.dumps({**d, **info}))

    def holder_a_biased(self, q, k, name):
        """Taught joints q moved by the learned holder-A place bias (base x, y): the taught place left the plate
        -2.5,+1.1 / -1.4,+0.9 / -2.0,+1.5 mm off, and the next one landed on the wall 4.3 mm high (2026-11-04)."""
        b = self.learned().get("holderA_place_bias_mm", [0.0, 0.0])
        if abs(b[0]) < 0.05 and abs(b[1]) < 0.05:
            return q
        C = k.cam(q)
        C[0, 3] += b[0] / 1000
        C[1, 3] += b[1] / 1000
        log(f"kpick: {name} shifted {b[0]:+.1f}, {b[1]:+.1f} mm (learned holder-A bias)")
        return self.ik_cam(C, q, name, k)

    def learn_holder_a_bias(self, err_mm):
        """After a seated place in holder A, err_mm (x, y, measured with the bias applied) moves the bias half way
        to cancel it; clipped to +-HOLDER_A_BIAS_MAX_MM."""
        if not HOLDER_A_BIAS_LEARN:
            log(f"kpick: holder-A place error {err_mm[0]:+.1f}, {err_mm[1]:+.1f} mm (bias learning off)")
            return
        b = np.array(self.learned().get("holderA_place_bias_mm", [0.0, 0.0]), float)
        nb = np.clip(b - 0.5 * np.asarray(err_mm[:2], float), -HOLDER_A_BIAS_MAX_MM, HOLDER_A_BIAS_MAX_MM)
        self.save_learned(holderA_place_bias_mm=[round(float(v), 2) for v in nb])
        log(f"kpick: holder-A place bias {b[0]:+.1f}, {b[1]:+.1f} -> {nb[0]:+.1f}, {nb[1]:+.1f} mm")

    def set_down_approach_fix(self, k, L, D, T_last):
        """Going by the arm's set-down pose: that pose was computed through the release reference's camera-to-plate
        relation (table_close_ref*.json, carried pick to pick), which had drifted ~40 mm from the taught one used
        to map the approach, so the view over the grasp started 50-70 px off (2026-11-04, 6 halts). When the
        set-down pose is that reference's own, aim the taught grasp camera at the release camera instead (a
        table-plane move: x, y and yaw only). -> D, corrected or unchanged."""
        pj, qj = DATA / "table_close_ref_plate.json", DATA / "table_close_ref.json"
        if not (pj.exists() and qj.exists()):
            return D
        T_at = np.array(json.loads(pj.read_text())["T"])
        if np.linalg.norm(T_at[:2, 3] - T_last[:2, 3]) > 0.001 or abs(yaw_deg(T_at) - yaw_deg(T_last)) > 0.5:
            log("kpick: set-down pose is not the release reference's; approach not corrected")
            return D
        C_want = k.cam(json.loads(qj.read_text()))  # plate unmoved since that release: the camera there
        C_have = D @ k.cam(L["flat_grasp_q"])
        yaw_of = lambda C: math.atan2(C[1, 0], C[0, 0])  # noqa: E731 (camera x axis, horizontal part)
        da = (yaw_of(C_want) - yaw_of(C_have) + math.pi) % (2 * math.pi) - math.pi
        dxy = np.linalg.norm(C_want[:2, 3] - C_have[:2, 3])
        if abs(math.degrees(da)) > 10.0 or dxy > 0.06:
            log(f"kpick: set-down approach fix too large ({dxy * 1000:.0f} mm, {math.degrees(da):+.1f} deg); not applied")
            return D
        c, s = math.cos(da), math.sin(da)
        E = np.eye(4)
        E[:2, :2] = [[c, -s], [s, c]]
        E[:2, 3] = C_want[:2, 3] - E[:2, :2] @ C_have[:2, 3]
        log(f"kpick: set-down approach aimed at the release camera: {dxy * 1000:.1f} mm, {math.degrees(da):+.1f} deg")
        return E @ D

    def held_plate_pose(self, k, C_grasp, C_taught, T_ref, ang, dx, dy):
        """Plate pose implied by the camera at C_grasp: the taught camera-to-plate relation, plus the
        residual image offset (dx, dy px; ang deg, see plate_rot_shift) mapped onto the table."""
        T = C_grasp @ np.linalg.inv(C_taught) @ T_ref
        shift = np.zeros(3)
        for axis, d in ((0, dx), (1, dy)):
            v = C_grasp[:3, axis].copy()
            v[2] = 0.0  # image motion along a camera axis = table motion along its horizontal part
            shift += d * MM_PER_PX_AT_GRASP / 1000 * v / (v @ v)
        T = self.turn_about_vertical(T, T[:3, 3], -ang)
        T[:2, 3] += shift[:2]
        T[2, 3] = T_ref[2, 3]
        R = T[:3, :3]  # keep it a pure table-plane pose
        a = math.atan2(R[1, 0], R[0, 0])
        T[:3, :3] = [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
        return T

    def plate_shift(self, taught, frame):
        """Image shift (px) of the plate between the pads, current vs taught (phase correlation)."""
        r0, r1, c0, c1 = 40, 180, 240, 480
        A = cv2.cvtColor(taught, cv2.COLOR_BGR2GRAY)[r0:r1, c0:c1].astype(np.float32)
        B = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[r0:r1, c0:c1].astype(np.float32)
        win = cv2.createHanningWindow((c1 - c0, r1 - r0), cv2.CV_32F)
        return cv2.phaseCorrelate(A, B, win)

    def lifted_with_gripper(self, before_close, name: str) -> bool:
        """Held = the jaws are pushed back by something AND, in the wrist camera, the plate stayed put
        between the pads while the gripper rose (a dropped plate stays behind with the table)."""
        self.hold(0.4)
        after = self.cam.latest(self.cam.seq)[1]
        cv2.imwrite(str(DATA / name), after)
        (dx, dy), resp = self.plate_shift(before_close, after)
        # a dropped plate moves 50+ px with the table; the clear plate's match drops to ~0.25 in evening light
        stayed = resp > 0.2 and abs(dx) <= 10 and abs(dy) <= 10
        if not stayed:
            # phase correlation can lock a well pitch off on the clear plate (-22.9 px, match 0.28, where the well-area
            # silhouette read +4,+1 px, IoU 0.94, plate held, 2026-10-22): let the silhouette overrule it
            sys.path.insert(0, str(HERE / "kin"))
            import plate_edges
            m = plate_edges.mask_shift(before_close, after)
            if m is not None:
                log(f"lift check: well-area silhouette moved {m['dx']:+.0f},{m['dy']:+.0f} px (IoU {m['iou']:.2f})")
                # the jaws square up a plate sitting a few deg turned between the pads: -12..-14 px, IoU 0.95, held
                # (2026-10-23, twice); a dropped plate moves 50+ px
                stayed = m["iou"] >= 0.9 and abs(m["dx"]) <= 16 and abs(m["dy"]) <= 16
        gripping = self.gripping()
        held = gripping and stayed
        log(f"LIFT CHECK: plate moved {dx:+.1f},{dy:+.1f} px in the wrist image (match {resp:.2f}), jaws "
            f"{'pushed back' if gripping else 'not pushed back'} -> {'PLATE HELD' if held else 'NOT HELD'}")
        return held

    def halt(self, why: str):
        """Stop everything around the plate: jaws fully open, straight up, home with the jaws open,
        and refuse further plate commands until the user has looked (`clear`)."""
        self.halted = why
        log(f"HALT: {why}")
        cv2.imwrite(str(DATA / "halt.png"), self.cam.latest()[1])
        if self.s.hold is None:  # already at rest with torque off: nothing to move
            raise Abort(f"halted: {why} (already at rest, torque off)")
        try:
            self.move({**self.s.hold, "gripper": GRIPPER_OPEN_DEG}, "open fully")
            self.hold(0.3)
            if self.stage == "kpick":
                self.cam_up(LIFT_M, "straight up")
            else:
                self.jog_tip([0, 0, LIFT_M], self.flat_target(0.0)[0], "straight up")
            self.move_segmented({**VIEW_POSE, "gripper": GRIPPER_OPEN_DEG}, "to viewing pose (jaws open)")
            self.move_segmented({"wrist_flex": 0.0}, "straighten wrist")
            self.path = []
            self.rest()
        finally:
            self.halted = why
        raise Abort(f"halted: {why} (jaws left open; check the plate, then `clear`)")

    def align_above_table(self, k, grasp, ref):
        """From LIFT_M straight over the planned grasp, shift the jaws sideways / along the plate until the view
        matches the one taken LIFT_M over a plate just released on bare table (jaws fully open, as held).
        Far from the taught pose the plate find + kinematics put the jaws ~10 mm off (a pad came down on the
        plate edge at (246, 372), 2026-10-21). -> the grasp joints under the aligned camera."""
        sys.path.insert(0, str(HERE / "kin"))
        import plate_edges
        C = k.cam(grasp)
        C[2, 3] += LIFT_M
        if not self.move_cart(C, "straight over the grasp", guard=True, free_air=True):
            self.halt("touched something moving over the grasp")
        # corrections add up on the commanded pose: re-basing each step on the measured joints (each within
        # 0.3 deg of its plan) random-walked the camera's tilt and the view jumped 22 px on a 4 mm step (2026-10-21)
        C_cmd = C.copy()
        # image px per mm of camera move, learned from each move: at (472, 370) the nominal sideways sign moved
        # the plate the wrong way (4 -> 10 px over 4 moves, 2026-10-21)
        gx, gy = 1 / MM_PER_PX_SIDE_ABOVE, 1 / MM_PER_PX_ALONG_ABOVE
        last = None
        for i in range(ABOVE_ALIGN_ITERS + 1):
            self.hold(0.3)
            frame = self.cam.latest(self.cam.seq)[1]
            stamp = time.strftime("%H%M%S")
            cv2.imwrite(str(DATA / f"kpick_above_{stamp}_{i}.png"), frame)
            (DATA / f"kpick_above_{stamp}_{i}.json").write_text(json.dumps(self.joints()))
            # the plate's well-area silhouette (grey + textured) slid over the reference one: whole-image matching
            # on the clear, repeating plate disagreed with itself by up to 16 px (2026-10-21)
            r = plate_edges.mask_shift(ref, frame)
            if r is None:
                self.halt("over the grasp: no plate silhouette in view; not going down")
            dx, dy, iou = r["dx"] - ABOVE_SIDE_TARGET_PX, r["dy"], r["iou"]  # dx: error from the centred target
            log(f"kpick above: plate vs the bare-table reference {r['dx']:+.0f} px sideways (target "
                f"{ABOVE_SIDE_TARGET_PX:+.0f}), {dy:+.0f} px along, {r['dang']:+.1f} deg (overlap {iou:.2f})")
            if last is not None:
                (ldx, ldy, lsx, lsy) = last
                # learn an axis' gain only from a move along it alone (a combined move couples them)
                # and only from a change well above the +-5 px read noise: 2-3 px changes flipped the sideways sign and
                # walked the view from +15 to +33 px (11:24, 15:14, 2026-10-23)
                if abs(lsx) >= 1.0 and lsy == 0.0 and abs(dx - ldx) >= 6.0:
                    gx = float(np.clip(-(dx - ldx) / lsx, -5.0, 5.0))
                if abs(lsy) >= 1.0 and lsx == 0.0 and abs(dy - ldy) >= 6.0:
                    gy = float(np.clip(-(dy - ldy) / lsy, -8.0, 8.0))
                # a noise jump taught 4.26 px/mm, then 1.2 mm steps never moved the view (2026-10-21)
                gx = math.copysign(min(max(abs(gx), 1.5), 3.5), gx)
                gy = gy if abs(gy) >= 1.5 else math.copysign(1.5, gy)
            done = abs(dx) <= 5.0 and abs(dy) <= ABOVE_DONE_DY_PX  # ~2 mm of ~10 mm per side; the close view checks 4 px
            # while still lining up, allow up to 80 px / overlap 0.45: every -60..-70 px halt came after the plate find
            # missed and the pick went by the set-down pose (6 of 6, 2026-11-03/04); a -64 px start read overlap 0.54
            # (18:02, 2026-11-04), -70 px lined up from 0.61 to +10 px / 0.89 in 10 moves; going down still needs 0.72
            if iou < (0.72 if done else 0.45) or abs(r["dx"]) > 80 or abs(dy) > 55:
                self.halt(f"over the grasp: the plate does not look as in the bare-table reference ({dx:+.0f},{dy:+.0f} px, "
                          f"overlap {iou:.2f}); not going down")
            if done:
                break
            if i == ABOVE_ALIGN_ITERS:
                self.halt(f"over the grasp: still {dx:+.0f},{dy:+.0f} px off after {i} moves; not going down")
            C = C_cmd
            yh = C[:3, 1].copy()
            yh[2] = 0.0
            yh /= np.linalg.norm(yh)
            sx_mm = float(np.clip(dx / gx, -ABOVE_STEP_MAX_MM, ABOVE_STEP_MAX_MM)) if abs(dx) > 5.0 else 0.0
            sy_mm = float(np.clip(dy / gy, -ABOVE_STEP_MAX_MM, ABOVE_STEP_MAX_MM)) if abs(dy) > ABOVE_DONE_DY_PX else 0.0
            last = (dx, dy, sx_mm, sy_mm)
            log(f"kpick above: move {sx_mm:+.1f} mm sideways, {sy_mm:+.1f} mm along ({gx:+.2f}, {gy:+.2f} px/mm)")
            sx, sy = sx_mm / 1000, sy_mm / 1000
            xh = C[:3, 0].copy()
            xh[2] = 0.0
            xh /= np.linalg.norm(xh)
            C[:3, 3] += xh * sx + yh * sy
            if not self.move_cart(C, "line the jaws up from above", guard=True, free_air=True):
                self.halt("touched something lining the jaws up from above")
        C = k.cam(self.joints())  # the measured pose is the one the view confirmed (move_exact leaves ~0.3 deg per joint)
        C[2, 3] -= LIFT_M
        return self.ik_cam(C, self.joints(), "grasp under the aligned view", k)

    def leave_plate(self, R_t, snap_to: Path | None = None, snap_up: Path | None = None):
        """Let go of a placed plate without dislodging it: jaws all the way open, then straight up.
        snap_to / snap_up: save the open-jaw view there before / after going up."""
        self.move({**self.s.hold, "gripper": GRIPPER_OPEN_DEG}, "release (fully open)")
        self.hold(0.4)
        if snap_to is not None:
            cv2.imwrite(str(snap_to), self.cam.latest(self.cam.seq)[1])
            snap_to.with_suffix(".json").write_text(json.dumps(self.joints()))
        if R_t is None:  # away from the taught pose: the calibrated kinematics instead of the learned J6
            self.cam_up(LIFT_M, "straight up off the plate")
        else:
            self.jog_tip([0, 0, LIFT_M], R_t, "straight up off the plate")
        if snap_up is not None:
            self.hold(0.4)
            cv2.imwrite(str(snap_up), self.cam.latest(self.cam.seq)[1])
            snap_up.with_suffix(".json").write_text(json.dumps(self.joints()))

    def check_plate_undisturbed(self, ref_crop, label: str):
        """At the align pose, compare the scene with the picture taken there before the grasp."""
        self.hold(0.3)
        score = self.ncc(ref_crop, self.scene_crop(self.cam.latest(self.cam.seq)[1]))
        log(f"{label}: scene vs before the grasp, similarity {score:.2f} (need {SCENE_MIN_NCC})")
        if score < SCENE_MIN_NCC:
            self.halt(f"{label}: the plate or holder looks moved (similarity {score:.2f})")

    def flatpick(self, align_only: bool = False):
        """Pick the plate with the bent finger sections flat along its long sides (gripper pitched 45 deg),
        centred on the plate. Aligned on the tag at the viewing offset; the slide to the centre and the
        last few cm down are blind on the J6 learned there (the tag leaves the image when centred)."""
        L = self.learned()
        if "flat_tip_above_edge" not in L and not align_only:
            raise Abort("run `flatcal` first (finger lower-edge height in the flat pose is unknown)")
        t0 = time.monotonic()
        self.stage = "flatpick"
        R_t = self.flat_approach()
        if align_only:
            log("FLAT ALIGNED: jaws closed, fingertips 2 cm above the grasp height; holding (send `rest`)")
            return
        h0 = self.tip_height()
        if h0 is None:
            self.halt("tag lost at the align pose")
        scene_ref = self.scene_crop(self.cam.latest(self.cam.seq)[1])
        tip_z = FLAT_EDGE_Z + L["flat_tip_above_edge"]
        slide = FLAT_OFFSET_TAG - FLAT_VIEW_OFFSET_TAG
        self.move({**self.s.hold, "gripper": FLAT_OPEN_DEG}, "open gripper")
        self.hold(0.5)
        if not self.jog_tip(slide, R_t, "flat slide to centre", guard=True, step_m=FLAT_SLIDE_STEP_M):
            self.halt("contact while sliding to the centre")
        slide_cmd = self.cmd_arm()
        met = not self.jog_tip([0, 0, tip_z - h0], R_t, "flat descend", guard=True)
        if met:
            self.halt("touched something on the way down")
        grasp_cmd = self.cmd_arm()
        grasp_q = {j: round(self.joints()[j], 2) for j in ARM6}
        frame = self.cam.latest(self.cam.seq)[1]
        cv2.imwrite(str(DATA / "flat_before_close.png"), frame)
        gap = self.close_gripper()
        if gap is None:
            self.halt("jaws did not stop on the plate")
        self.jog_tip([0, 0, LIFT_M], R_t, "flat lift")
        held = self.lifted_with_gripper(frame, "flat_lift.png")
        if held:  # remember what worked, for the replay (`flatfast`)
            cv2.imwrite(str(DATA / "flat_taught_close.png"), frame)
            cv2.imwrite(str(DATA / "flat_taught_align.png"), self.flat_align_frame)
            p_al, R_al = self.flat_align_tag
            self.save_learned(flat_slide_cmd=slide_cmd, flat_grasp_cmd=grasp_cmd, flat_lift_cmd=self.cmd_arm(),
                              flat_align_tag_p=p_al, flat_align_tag_R=R_al, flat_grasp_q=grasp_q,
                              flat_lift_q={j: round(self.joints()[j], 2) for j in ARM6},
                              flat_ref_align_q=self.flat_align_q)
        if not self.jog_tip([0, 0, -LIFT_M - 0.002], R_t, "flat place", guard=True):
            log("flat place: plate sits in the holder (contact)")
        self.leave_plate(R_t)
        if not held:
            self.halt("plate was not held on the lift")
        self.jog_tip(-slide, R_t, "flat slide back", step_m=FLAT_SLIDE_STEP_M)
        self.move_exact(self.flat_align_q, "back to the align pose")
        self.check_plate_undisturbed(scene_ref, "after placing")
        self.go_home_from_plate()
        temps = {k: st.t_mos for k, st in self.follower._read_feedback().items()}
        log(f"FLATPICK SUCCEEDED in {time.monotonic() - t0:.0f} s; shoulder {temps['shoulder_lift']:.0f} C")

    def flatfast(self):
        """Repeat the taught flat grasp: vision-align on the tag (in view), then drive the MEASURED joints
        to the taught grasp and lift angles (commands do not repeat: sag and friction depend on the path).
        Camera checks: the plate must look as taught before the jaws close, and undisturbed afterwards."""
        L = self.learned()
        need = ("flat_ref_align_q", "flat_grasp_q", "flat_lift_q", "flat_J6")
        if not all(k in L for k in need) or not (DATA / "flat_taught_close.png").exists():
            raise Abort("no taught flat pick yet; run `flatpick` until one succeeds")
        t0 = time.monotonic()
        self.stage = "flatfast"
        ref = L["flat_ref_align_q"]
        R_t = self.flat_approach(probe=False)
        align_q = dict(self.flat_align_q)
        diff = max(abs(align_q[j] - ref[j]) for j in ARM6)
        log(f"flatfast: aligned joints within {diff:.2f} deg of the taught ones")
        if diff > FLAT_MAX_SHIFT_DEG:
            self.halt(f"aligned joints {diff:.1f} deg from the taught ones: plate or tag moved? re-teach with `flatpick`")
        scene_ref = self.scene_crop(self.cam.latest(self.cam.seq)[1])
        taught_align = cv2.imread(str(DATA / "flat_taught_align.png"))
        if taught_align is not None:
            score = self.ncc(self.scene_crop(taught_align), scene_ref)
            log(f"flatfast: scene vs when the grasp was taught, similarity {score:.2f}")
            if score < SCENE_MIN_NCC:
                self.halt(f"the plate does not look where it was when the grasp was taught (similarity {score:.2f})")
        grasp, lift = L["flat_grasp_q"], L["flat_lift_q"]
        self.move({**self.s.hold, "gripper": FLAT_OPEN_DEG}, "open gripper (as taught)")
        self.move_exact(lift, "to above the grasp")  # the taught lift pose: 3 cm above the grasp, centred
        if not self.move_exact_guarded(grasp, "down to the grasp"):
            self.halt("touched something on the way down")
        taught = cv2.imread(str(DATA / "flat_taught_close.png"))
        # Line the plate up between the pads with the camera: measure how far it sits sideways from
        # where it was when the grasp was taught, and move the jaws by that much (camera x = tag X).
        for k in range(3):
            self.hold(0.3)
            frame = self.cam.latest(self.cam.seq)[1]
            (dx, dy), resp = self.plate_shift(taught, frame)
            log(f"flatfast: plate offset from the taught view {dx:+.1f} px sideways, {dy:+.1f} px along (match {resp:.2f})")
            if resp < 0.3 or abs(dx) > 25 or abs(dy) > 25:
                self.halt(f"the plate is not between the pads as taught (offset {dx:+.0f},{dy:+.0f} px, match {resp:.2f}); not closing")
            if abs(dx) <= 1.5:
                break
            if not self.jog_tip([dx * MM_PER_PX_AT_GRASP / 1000, 0, 0], R_t, "centre the jaws on the plate", guard=True):
                self.halt("touched something while centring the jaws on the plate")
        cv2.imwrite(str(DATA / "flatfast_before_close.png"), frame)
        # Final camera check, on the measured offset (image similarity scores collapse on the plate's
        # repeating wells even at 1 mm): the jaws have ~4 mm (~10 px) of clearance each side.
        if abs(dx) > PADS_MAX_DX_PX or abs(dy) > PADS_MAX_DY_PX or resp < 0.5:
            self.halt(f"the plate is not between the pads as taught (offset {dx:+.1f},{dy:+.1f} px, match {resp:.2f}); not closing")
        gap = self.close_gripper()
        if gap is None:
            self.halt("jaws did not stop on the plate")
        self.move_exact(lift, "lift")
        held = self.lifted_with_gripper(frame, "flatfast_lift.png")
        if not self.move_exact_guarded(grasp, "place"):
            log("place: plate sits in the holder (contact)")
        self.leave_plate(R_t)
        if not held:
            self.halt("plate was not held on the lift")
        self.move_exact(align_q, "back to the align pose")
        self.check_plate_undisturbed(scene_ref, "after placing")
        self.go_home_from_plate()
        temps = {k: st.t_mos for k, st in self.follower._read_feedback().items()}
        log(f"FLATFAST SUCCEEDED in {time.monotonic() - t0:.0f} s; shoulder {temps['shoulder_lift']:.0f} C")

    # ---- kinematic pick / place anywhere on the table (calibrated URDF + tagless plate finding)

    def ik_cam(self, C_goal, seed: dict, label: str, k=None) -> dict:
        """Joints putting the camera at C_goal (base frame); Abort if out of reach."""
        k = k or self.kin()
        q, ep, er = k.ik(C_goal, seed, limits={j: (self.limits[j][0] + LIMIT_MARGIN_DEG,
                                                    self.limits[j][1] - LIMIT_MARGIN_DEG) for j in ARM6})
        if ep > KPICK_IK_MAX_MM or er > KPICK_IK_MAX_DEG:
            raise Abort(f"{label}: out of reach (IK error {ep:.1f} mm, {er:.1f} deg)")
        return q

    def move_cart(self, C_goal, label: str, step_m: float = 0.01, guard: bool = False, free_air: bool = False,
                  smooth: bool | None = None) -> bool:
        """Straight-line camera move to C_goal through IK waypoints (<= step_m apart), each driven
        until the measured joints match. With guard, stops at a torque jump and returns False."""
        sys.path.insert(0, str(HERE / "kin"))
        import model
        k = self.kin()
        q = self.joints()
        C0 = k.cam(q)
        w = model.log_rot(C0[:3, :3].T @ C_goal[:3, :3])
        dist = np.linalg.norm(C_goal[:3, 3] - C0[:3, 3])
        n = max(1, int(math.ceil(max(dist / step_m, np.degrees(np.linalg.norm(w)) / 3.0))))
        seed = {j: q[j] for j in ARM6}
        if smooth is None:  # free air, unguarded and the carry stream; guarded moves near contact keep the steps
            smooth = STREAM_MOTION and (free_air or not guard)
        if smooth and n > 1:
            qs = []
            for i in range(1, n + 1):
                f = i / n
                Ci = np.eye(4)
                Ci[:3, :3] = C0[:3, :3] @ model.rot_vec(w * f)
                Ci[:3, 3] = C0[:3, 3] + f * (C_goal[:3, 3] - C0[:3, 3])
                qi = self.ik_cam(Ci, seed, label, k)
                if max(abs(qi[j] - seed[j]) for j in ARM6) > 8.0:
                    raise Abort(f"{label}: IK jumped {max(abs(qi[j] - seed[j]) for j in ARM6):.1f} deg between waypoints")
                qs.append(qi)
                seed = qi
            # commands carry the present gravity sag over (as move_exact does), then one exact settle at the end
            sag = {j: self.s.hold[j] - q[j] for j in ARM6} if self.s.hold else dict.fromkeys(ARM6, 0.0)
            lim = TORQUE_JUMP_FREE_AIR_NM if free_air else TORQUE_JUMP_NM
            if not self.stream_path([{j: qi[j] + sag[j] for j in ARM6} for qi in qs], label, guard=guard, limits=lim):
                log(f"{label}: contact while streaming")
                return False
            self.hold(0.3)
            return self.move_exact(qs[-1], f"{label} (settle)", iters=3,
                                   guard_base=self.torques() if guard else None, jump_limits=lim if guard else None)
        for i in range(1, n + 1):
            f = i / n
            Ci = np.eye(4)
            Ci[:3, :3] = C0[:3, :3] @ model.rot_vec(w * f)
            Ci[:3, 3] = C0[:3, 3] + f * (C_goal[:3, 3] - C0[:3, 3])
            qi = self.ik_cam(Ci, seed, label, k)
            if max(abs(qi[j] - seed[j]) for j in ARM6) > 8.0:
                raise Abort(f"{label}: IK jumped {max(abs(qi[j] - seed[j]) for j in ARM6):.1f} deg between waypoints")
            if not self.move_exact(qi, f"{label} {i}/{n}", guard_base=self.torques() if guard else None,
                                   jump_limits=TORQUE_JUMP_FREE_AIR_NM if free_air else None):
                log(f"{label}: contact at waypoint {i}/{n}")
                return False
            seed = qi
        return True

    def cam_up(self, dz: float, label: str, guard: bool = False) -> bool:
        k = self.kin()
        C = k.cam(self.joints())
        C[2, 3] += dz
        return self.move_cart(C, label, guard=guard)

    @staticmethod
    def radial_yaw(T_from, xy_to):
        """Plate yaw at xy_to that keeps its orientation relative to the arm (turned by the bearing change)."""
        a0 = math.atan2(T_from[1, 3], T_from[0, 3])
        return math.degrees(math.atan2(T_from[1, 0], T_from[0, 0]) + math.atan2(xy_to[1], xy_to[0]) - a0)

    def plate_side(self, T, T_ref):
        """The plate looks the same turned 180 deg: pick the frame whose grasp keeps the wrist nearest
        the taught one (relative yaw vs the bearing change closest to zero)."""
        want = self.radial_yaw(T_ref, T[:2, 3])
        have = math.degrees(math.atan2(T[1, 0], T[0, 0]))
        if abs((have - want + 180) % 360 - 180) > 90:
            T = T.copy()
            T[:3, :2] *= -1  # rotate 180 deg about z
        return T

    def kpick(self, place_xy_yaw=None):
        """Pick the plate wherever it lies, by the calibrated kinematics: find it (no tag), map the
        taught grasp (holder pose -> plate pose), grasp with the usual camera and torque checks, lift.
        Then place it at (x, y, yaw) [m, m, deg] if given (measuring where it really lands), else put
        it back where it was."""
        ref_path = DATA / "plate_ref.json"
        if not ref_path.exists():
            raise Abort("no plate reference pose: with the plate in the holder (as taught) run `plateref`")
        temps = self.fresh_temps()
        if temps["shoulder_lift"] > PICK_START_MAX_C:
            raise Abort(f"shoulder at {temps['shoulder_lift']:.0f} C; let it cool below {PICK_START_MAX_C:.0f} C first")
        L = self.learned()
        T_ref = np.array(json.loads(ref_path.read_text())["T"])
        t0 = time.monotonic()
        self.stage = "kpick"
        if self.s.torque and self.joints()["gripper"] < FLAT_APPROACH_GRIPPER_DEG - 5:
            self.move_segmented({**VIEW_POSE, "wrist_flex": 0.0}, "unfold")
            self.close_gripper(max_close_deg=FLAT_APPROACH_GRIPPER_DEG)
        last = DATA / "plate_pose.json"  # where the plate was last seen: look there first
        last_d = json.loads(last.read_text()) if last.exists() else {}
        T_last = np.array(last_d["T"]) if last_d else None
        prior = T_last[:2, 3] if T_last is not None else None
        try:
            T = self.platelook(3, prior=prior, anchor=True, prior_yaw=None if T_last is None else yaw_deg(T_last))
        except Abort as e:
            # set down by this arm, the plate is where the arm left it (checked by overlay, 2026-10-30 14:59, where the
            # lattice scored 0.158 < 0.18 on it); the from-above alignment still corrects a few mm
            if last_d.get("source") != "arm at release" or self.frozen \
                    or not str(e).startswith(("platelook: no plate", "platelook: lost the plate")):
                raise
            log(f"kpick: {e}; going by the arm's set-down pose ({T_last[0, 3] * 1000:.1f}, {T_last[1, 3] * 1000:.1f}) mm")
            T = T_last.copy()
            by_set_down = True
        else:
            by_set_down = False
        k = self.kin()
        T = self.plate_side(T, T_ref)
        D = T @ np.linalg.inv(T_ref)  # taught plate pose -> plate pose now (a table-plane motion)
        if by_set_down:
            D = self.set_down_approach_fix(k, L, D, T_last)
        yaw = math.degrees(math.atan2(D[1, 0], D[0, 0]))
        log(f"kpick: plate {np.linalg.norm(T[:2, 3] - T_ref[:2, 3]) * 1000:.0f} mm and {yaw:+.1f} deg from the taught pose")

        def mapped(name, M=D):
            return self.ik_cam(M @ k.cam(L[name]), {**L[name], "shoulder_pan": L[name]["shoulder_pan"] + yaw}, name, k)
        pre, align, lift, grasp = (mapped(n) for n in ("flat_pregrasp_joints", "flat_ref_align_q", "flat_lift_q",
                                                       "flat_grasp_q"))
        tip_z = self.kin_tip_height_fn()  # same model both sides: absolute errors cancel
        hb = json.loads((DATA / "holder_b.json").read_text()) if (DATA / "holder_b.json").exists() else None
        at_b = bool(hb and np.linalg.norm(T[:2, 3] - np.array(hb["T"])[:2, 3]) < HOLDER_NEAR_M)
        # reference for the close-view alignment: the taught grasp in the first holder; in the second, the
        # first pick there (its frame holds the plate to ~1 mm, and the view through the clear plate
        # differs from the first holder's, so the taught view does not match there)
        C_ref, T_at_ref, ref_img = k.cam(L["flat_grasp_q"]), T_ref, DATA / "flat_taught_close.png"
        on_table = not at_b and np.linalg.norm(T[:2, 3] - T_ref[:2, 3]) >= HOLDER_NEAR_M
        # along is set from LIFT_M up when there is a bare-table reference there; correcting it again at the
        # grasp made the sideways reading swing +-8 px (2026-10-21)
        along_fix = on_table and not (DATA / "table_above_ref.png").exists()
        if on_table and (DATA / "table_close_ref.png").exists():
            # the open-jaw view at release over a plate set down on bare table as held (taught relation);
            # the holder-A view differs around the plate edges (frame vs wood) and matched only 0.47 (2026-10-21)
            ref_img = DATA / "table_close_ref.png"
            pj, qj = DATA / "table_close_ref_plate.json", DATA / "table_close_ref.json"
            if pj.exists() and qj.exists():
                # the reference's own relation: the plate as measured after that release, from the release joints.
                # Assuming the taught one carried that place's in-jaw error (~3.7 mm) into the next: the places
                # into A caught its wall after such refs (eps 86, 88, 2026-10-22)
                C_ref, T_at_ref = k.cam(json.loads(qj.read_text())), np.array(json.loads(pj.read_text())["T"])
        if at_b and "holderB_grasp_q" in L and "holderB_P" in L:
            # the joints that seated it there (they cleared its walls), a little higher; the lattice fit
            # wanders by +-4 mm in this holder, so it only says which holder the plate is in
            C_g = k.cam(L["holderB_grasp_q"])
            T_at_ref = C_g @ np.array(L["holderB_P"])
            T_at_ref[2, 3] = T_ref[2, 3]
            C_seat = C_g.copy()
            C_g[2, 3] += KPICK_B_TIP_Z_M - tip_z(L["holderB_grasp_q"])  # its place, not its drifting height
            grasp = self.ik_cam(C_g, L["holderB_grasp_q"], "grasp at the second holder", k)
            C_l = C_g.copy()
            C_l[2, 3] += LIFT_M
            lift = self.ik_cam(C_l, grasp, "above the grasp at the second holder", k)
            C_ref, ref_img = C_seat, DATA / "holderB_close_ref.png"
            T = T_at_ref.copy()  # the pivot for the alignment turns
            log(f"kpick: using the seated joints at the second holder, fingertips at {tip_z(grasp) * 1000:.1f} mm "
                f"(seat saved at {tip_z(L['holderB_grasp_q']) * 1000:.1f})")
        elif at_b:
            C_g = k.cam(grasp)
            C_g[2, 3] += KPICK_B_GRASP_UP_M
            grasp = self.ik_cam(C_g, grasp, "grasp at the second holder", k)
        grip_dz = KPICK_B_GRASP_UP_M if at_b else 0.0  # the plate will sit this much higher in the jaws
        seat = at_b and "holderB_grasp_q" in L and "holderB_P" in L  # measured joints there: no mapping to check
        dz = {n: tip_z(q) - tip_z(L[n]) - (grip_dz if n == "flat_grasp_q" else 0.0)
              for n, q in (("flat_grasp_q", grasp), ("flat_ref_align_q", align)) if not (seat and n == "flat_grasp_q")}
        if max(abs(v) for v in dz.values()) > KPICK_TIP_DZ_M:
            raise Abort(f"kpick: planned fingertip heights differ from the taught ones by "
                        f"{ {n: round(v * 1000, 1) for n, v in dz.items()} } mm; not moving")
        self.move_segmented(pre, "to the pregrasp")
        self.move_exact(pre, "to the pregrasp")
        self.move_exact(align, "to the align pose")
        scene_ref = self.scene_crop(self.cam.latest(self.cam.seq)[1])
        at_a = np.linalg.norm(T[:2, 3] - T_ref[:2, 3]) < HOLDER_NEAR_M
        self.move({**self.s.hold, "gripper": FLAT_OPEN_DEG}, "open gripper (as taught)")
        self.move_exact(lift, "to above the grasp")
        # From LIFT_M over the grasp, compare the view with one over the plate in holder A (taught relation):
        # on bare table the jaws came down a few mm sideways and probably low (the wells looked 1.5-2.4 % larger
        # at the 12 mm hover, 2026-10-21); the 12 mm hover left the pads only ~3.5 mm over the plate.
        above_ref = DATA / "kpick_above_ref.png"
        if at_a and not above_ref.exists():
            self.hold(0.3)
            cv2.imwrite(str(above_ref), self.cam.latest(self.cam.seq)[1])
            C_a = k.cam(self.joints())
            (DATA / "kpick_above_ref.json").write_text(json.dumps(
                {"plate_in_cam": (np.linalg.inv(C_a) @ np.append(T[:3, 3], 1))[:3].tolist(),
                 "time": time.strftime("%Y-%m-%d %H:%M:%S")}))
            log(f"kpick: saved {above_ref.name} (view over the plate in holder A, {LIFT_M * 1000:.0f} mm up)")
        elif not at_a and not at_b:
            self.hold(0.3)
            cv2.imwrite(str(DATA / f"kpick_above_{time.strftime('%H%M%S')}.png"), self.cam.latest(self.cam.seq)[1])
            if KPICK_TABLE_MEASURE_ONLY:
                self.halt("bare table: view over the plate saved; stopping before the descent (measure-only mode)")
            self.move({**self.s.hold, "gripper": GRIPPER_OPEN_DEG}, "open gripper fully (bare table)")
            if (DATA / "table_above_ref.png").exists():
                grasp = self.align_above_table(k, grasp, cv2.imread(str(DATA / "table_above_ref.png")))
        if not self.move_exact_guarded(grasp, "down to the grasp"):
            self.halt("touched something on the way down")
        taught = cv2.imread(str(ref_img)) if ref_img.exists() else None
        if taught is None:  # first pick at the second holder: this view becomes its reference
            self.hold(0.3)
            cv2.imwrite(str(ref_img), self.cam.latest(self.cam.seq)[1])
            taught = cv2.imread(str(ref_img))
            log(f"kpick: saved {ref_img.name} as the close-view reference for this holder")
        # line the jaws up with the plate as in the reference: turn about the vertical through the plate
        # centre and shift sideways until the close view matches (the plate's angle from the lattice fit
        # is only good to ~1.5 deg)
        C_cmd = None  # corrections add up on the commanded pose (see align_above_table)
        for i in range(KPICK_ALIGN_ITERS + 1):
            self.hold(0.3)
            frame = self.cam.latest(self.cam.seq)[1]
            if ref_img.name == "table_close_ref.png":
                # the well-area silhouette, as from above: ECC on the clear plate read +10,-12 px (match 0.51)
                # off a view the silhouettes put 6,8 px off (IoU 0.97), twice on the mat (2026-10-21)
                sys.path.insert(0, str(HERE / "kin"))
                import plate_edges
                r = plate_edges.mask_shift(taught, frame)
                if r is None:
                    self.halt("no plate silhouette between the pads; not closing")
                ang, dx, dy, cc = -r["dang"], r["dx"], r["dy"], r["iou"]  # dang: CCW in the image = from above
                e = plate_edges.edge_turn(taught, frame)
                if e is not None:  # the turn from the straight side edges
                    log(f"kpick: side edges turned {e[0]:+.2f} / {e[1]:+.2f} deg (silhouette {r['dang']:+.1f})")
                    # edges that disagree (a skewed reference edge read +6.5 / -2.6, 2026-10-22) say nothing: keep the
                    # yaw planned from the lattice fit (good to ~0.4 deg) instead of picking one
                    ang = -sum(e) / 2 if abs(e[0] - e[1]) <= 2.0 else 0.0
                    # unless the silhouette sees a clear turn: with no turn the pads met the corners of a plate turned
                    # ~3 deg (silhouette -3.0, edges +1.1 / -2.1) and lost it on the lift (2026-10-30 13:35)
                    # only on the first look: later ones flipped +3 / -3 deg (the silhouette's search limit) while one edge
                    # read +6 -> +22 deg, and the turns went back and forth until the gate halted (15:41)
                    if i == 0 and abs(e[0] - e[1]) > 2.0 and abs(r["dang"]) >= KPICK_SIL_TURN_MIN_DEG:
                        ang = -r["dang"]
            else:
                ang, dx, dy, cc = self.plate_rot_shift(taught, frame)
            log(f"kpick: plate vs the taught view {ang:+.2f} deg, {dx:+.1f} px sideways, {dy:+.1f} px along (match {cc:.2f})")
            # at a holder, along is only corrected past the closing gate: a hand-seated plate sat 2.5 mm short in A
            # (+8.6 px vs the 8 px gate, 2026-10-22)
            # on table too once past its gate: lined up from above at +10..+12 px along (inside ABOVE_DONE_DY_PX), a plate
            # turned +34 deg read +17..+21 px here and the gate halted twice (2026-11-05 18:35, 18:37)
            fix_along = along_fix or abs(dy) > (PADS_MAX_DY_TABLE_PX if on_table else PADS_MAX_DY_PX)
            if cc < (0.5 if on_table and ref_img.name == "flat_taught_close.png" else 0.6) or abs(ang) > 6 \
                    or abs(dx) > 40 or abs(dy) > 40:
                self.halt(f"the plate is not between the pads as taught ({ang:+.1f} deg, {dx:+.0f},{dy:+.0f} px, "
                          f"match {cc:.2f}); not closing")
            # on table, stop once the view passes the closing gate below: further moves chase the ~1 deg / few px
            # repeatability (a pass at -1.0 deg, +1 px went on to -1.5 deg, -7 px and halted, 2026-10-21)
            table_ok = on_table and ref_img.name == "table_close_ref.png" and abs(ang) <= 1.0 \
                and abs(dx) <= PADS_MAX_DX_TABLE_PX and abs(dy) <= PADS_MAX_DY_TABLE_PX and cc >= 0.7
            if (abs(ang) <= KPICK_ALIGN_DEG and abs(dx) <= 1.5 and not (fix_along and abs(dy) > 4)) \
                    or table_ok or i == KPICK_ALIGN_ITERS:
                break
            # the side-edge angle moved ~2x per commanded turn (+2.4 -> -2.1 -> +1.5 -> -3.9 -> +4.1 deg, 2026-10-21)
            gain = KPICK_EDGE_TURN_GAIN if ref_img.name == "table_close_ref.png" else 1.0
            # re-basing each step on the measured joints lost the small sideways steps: 2.3-2.7 mm moves left the
            # view at +7,+7,+6,+6,+6 px (each within move_exact's 0.3 deg) and the gate halted (2026-10-29 17:48)
            C = self.turn_about_vertical(k.cam(self.joints()) if C_cmd is None else C_cmd, T[:3, 3], -ang * gain)
            # once the steps really moved (C_cmd), 0.38 mm/px overshot ~2x: -9 -> +9 px for 3.4 mm (2026-10-30 11:20)
            side = dx * MM_PER_PX_AT_GRASP * KPICK_CLOSE_SIDE_GAIN / 1000
            C = C @ np.array([[1, 0, 0, side], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            if fix_along and abs(dy) > 4:  # along the plate, level with the table (image down = camera y)
                yh = C[:3, 1].copy()
                yh[2] = 0.0
                yh /= np.linalg.norm(yh)
                step = float(np.clip(dy * MM_PER_PX_ALONG_AT_GRASP, -5.0, 5.0)) / 1000
                C[:3, 3] += yh * step
            if on_table and abs(ang) > KPICK_TURN_IN_PLACE_DEG:
                # turn LIFT_M up, not with the fingers beside the plate: a 2 deg turn there tripped the elbow
                # guard (2026-10-21). The angle from above is not usable (oblique view; disagreed in sign).
                C_up = k.cam(self.joints())
                C_up[2, 3] += LIFT_M
                C_turn = C.copy()
                C_turn[2, 3] += LIFT_M
                # 3.5 Nm on shoulder/elbow here too (user's OK 2026-10-29 after a lone +3.25 shoulder trip lifting
                # open jaws, plate unmoved); the wrist keeps its 0.8 Nm limit
                if not (self.move_cart(C_up, "up to turn the jaws", guard=True, free_air=True)  # starts beside the plate
                        and self.move_cart(C_turn, "turn the jaws over the plate", guard=True, free_air=True)):
                    self.halt("touched something lifting to turn the jaws")
                if not self.move_cart(C, "back down to the grasp", guard=True):
                    self.halt("touched something going back down to the grasp")
            elif not self.move_cart(C, "line the jaws up with the plate", guard=True):
                self.halt("touched something while lining the jaws up with the plate")
            C_cmd = C
        if on_table:
            z_now = tip_z({j: self.joints()[j] for j in ARM6})
            if z_now > KPICK_TABLE_TIP_MAX_M:
                down = z_now - KPICK_TABLE_TIP_TARGET_M
                if down > 0.006:
                    self.halt(f"fingertips {z_now * 1000:.1f} mm over the table at the grasp; not closing")
                C = k.cam(self.joints())
                C[2, 3] -= down
                log(f"kpick: fingertips at {z_now * 1000:.1f} mm (measured), stepping down {down * 1000:.1f} mm")
                if not self.move_cart(C, "down to the grip height", guard=True):
                    self.halt("touched something stepping down to the grip height")
                self.hold(0.3)
                frame = self.cam.latest(self.cam.seq)[1]
                log(f"kpick: fingertips now at {tip_z({j: self.joints()[j] for j in ARM6}) * 1000:.1f} mm")
        cv2.imwrite(str(DATA / "kpick_before_close.png"), frame)
        # on bare table against the holder-A view the frame-vs-wood edges cap the match at ~0.6-0.78 (2026-10-21)
        # in holder A against its taught view (robot PC, 2026-10-18: other light, no blue tape) the Jetson matched
        # 0.70-0.78 with the plate centred (-0.1, -0.5 px, 2026-11-04 20:03): same 0.6 floor there
        min_cc = 0.6 if ref_img.name == "flat_taught_close.png" else 0.7
        # along does not matter much for the grip (the fingers lie along the long sides)
        if abs(dx) > (PADS_MAX_DX_TABLE_PX if on_table else PADS_MAX_DX_PX) \
                or abs(dy) > (PADS_MAX_DY_TABLE_PX if on_table else PADS_MAX_DY_PX) or abs(ang) > 1.0 or cc < min_cc:
            self.halt(f"the plate is not between the pads as taught ({ang:+.2f} deg, {dx:+.1f},{dy:+.1f} px, "
                      f"match {cc:.2f}); not closing")
        grasp_now = {j: self.joints()[j] for j in ARM6}
        C_grasp = k.cam(grasp_now)
        # the plate's pose from the grasp: the taught grasp-to-plate relation, corrected by the residual
        # image offset (the camera moves with the jaws, so this is also where the held plate is)
        T = self.held_plate_pose(k, C_grasp, C_ref, T_at_ref, ang, dx, dy)
        if self.close_gripper() is None:
            self.halt("jaws did not stop on the plate")
        # pads on the plate's corners (turned ~3 deg in the jaws) stop pressing: -0.11 Nm vs -0.62..-0.91 on every
        # held grip, and the lift then lost the plate (2026-10-30 11:36, 13:35). Let go before lifting instead.
        self.hold(0.3)
        g_pos, g_tq = self.gripper_state()
        if -g_tq < GRIP_PRELIFT_MIN_NM:
            log(f"kpick: jaws not pressing the plate (torque {g_tq:+.2f} Nm at {g_pos:.1f} deg, need "
                f"-{GRIP_PRELIFT_MIN_NM:.2f}); letting go without lifting")
            self.leave_plate(None)
            self.halt("jaws not pressing the plate after closing; not lifting")
        self.cam_up(LIFT_M, "lift")
        held = self.lifted_with_gripper(frame, "kpick_lift.png")
        if not held:
            self.move_exact_guarded(grasp_now, "put it down")
            self.leave_plate(None)
            self.halt("plate was not held on the lift")
        self.kpick_place(place_xy_yaw, T, C_grasp, L, k, T_ref, hb, tip_z, t0, scene_ref, align, grip_dz=grip_dz)

    def carry(self, C_goal, guard: bool) -> bool:
        """Carry move with the plate; a guard trip is retried from where it stopped if the torques settle
        back (the correction loop's breakaway spikes look like contact), up to twice."""
        for attempt in range(3):
            if self.move_cart(C_goal, "carry", step_m=0.02, guard=guard, smooth=STREAM_MOTION):
                return True
            self.hold(1.0)
            log(f"carry: stopped (attempt {attempt + 1}); camera view saved, retrying from here")
            cv2.imwrite(str(DATA / f"carry_stop{attempt}.png"), self.cam.latest()[1])
        return False

    def kresume(self, place_xy_yaw, ang, dx, dy):
        """Finish a kpick that stopped while holding the plate (e.g. a false contact on the carry): the
        held plate's pose follows from the current camera pose and the logged grasp alignment."""
        k = self.kin()
        L = self.learned()
        T_ref = np.array(json.loads((DATA / "plate_ref.json").read_text())["T"])
        # the held plate is rigid with the camera: express it for the camera at the taught grasp (a
        # virtual grasp pose; placing only needs the camera-to-plate relation, whatever the arm's tilt now)
        C_grasp = k.cam(L["flat_grasp_q"])
        T = self.held_plate_pose(k, C_grasp, C_grasp, T_ref, ang, dx, dy)
        hb = json.loads((DATA / "holder_b.json").read_text()) if (DATA / "holder_b.json").exists() else None
        self.stage = "kpick"
        self.kpick_place(place_xy_yaw, T, C_grasp, L, k, T_ref, hb, self.kin_tip_height_fn(), time.monotonic(),
                         None, None, carry_guard=False)

    def kpick_place(self, place_xy_yaw, T, C_grasp, L, k, T_ref, hb, tip_z, t0, scene_ref, align,
                    carry_guard=True, grip_dz=0.0):
        """kpick's second half: carry the held plate (pose T when the camera was at C_grasp) and set it
        down: back where it was (None), in the taught holder ("home"), the second holder ("B") or at
        (x, y, yaw); then release fully and check where it landed."""
        home, to_b = place_xy_yaw == "home", place_xy_yaw == "B"
        q_place = None
        Tn = k.cam(self.joints()) @ np.linalg.inv(k.cam(L["flat_grasp_q"])) @ T_ref
        log(f"kpick: held plate tilted {math.degrees(math.acos(min(1.0, Tn[2, 2]))):.1f} deg (taught relation, now)")
        if place_xy_yaw is None:
            T_goal, C_place = T, C_grasp
        elif home:  # back into the holder: the taught joints (the camera centred the plate in the jaws as taught)
            T_goal, C_place = T_ref, k.cam(L["flat_grasp_q"])
            self.cam_up(KPICK_CARRY_UP_M - LIFT_M, "up to carry height")
            C_carry = C_place.copy()
            C_carry[2, 3] += KPICK_CARRY_UP_M
            if not self.carry(C_carry, carry_guard):
                raise Abort("touched something while carrying the plate; holding it here (check, then decide)")
            C_low = C_place.copy()
            C_low[2, 3] += LIFT_M
            self.move_cart(C_low, "down to above the holder")
            self.move_exact(self.holder_a_biased(L["flat_lift_q"], k, "lift pose over holder A"),
                            "to the taught lift pose")
            # the view over the empty holder with the plate in the jaws, for a camera-aimed set-down (collected
            # with each place's outcome, 2026-11-04: the open-loop place caught the wall ~1 in 3)
            self.hold(0.3)
            stamp = time.strftime("%H%M%S")
            self.holder_view = DATA / f"holderA_above_{stamp}.png"
            cv2.imwrite(str(self.holder_view), self.cam.latest(self.cam.seq)[1])
            self.holder_view.with_suffix(".json").write_text(json.dumps({"q": self.joints(), "outcome": None}))
        else:
            if to_b:  # the holder's pose, turned by 180 deg if that keeps the wrist nearer its taught turn
                T_goal = np.array(hb["T"]) if hb else np.array(self.holder_b()["T"])
                x, y = T_goal[:2, 3]
                a = math.degrees(math.atan2(T_goal[1, 0], T_goal[0, 0]))
                a += 180 * (abs((a - self.radial_yaw(T, (x, y)) + 180) % 360 - 180) > 90)
            else:
                x, y, a = place_xy_yaw
            T_goal = np.eye(4)
            T_goal[:3, :3] = [[math.cos(math.radians(a)), -math.sin(math.radians(a)), 0],
                              [math.sin(math.radians(a)), math.cos(math.radians(a)), 0], [0, 0, 1]]
            T_goal[:3, 3] = (x, y, T[2, 3])
            M = T_goal @ np.linalg.inv(T)  # the held plate moves rigidly with the camera
            C_place = M @ C_grasp
            if to_b and "holderB_grasp_q" in L:  # the saved seat's place, at the fixed seating height
                C_place = k.cam(L["holderB_grasp_q"])
                # gripped as taught, the plate hangs grip_dz lower than when gripped there: seat that much higher
                C_place[2, 3] += KPICK_B_TIP_Z_M + KPICK_B_GRASP_UP_M - tip_z(L["holderB_grasp_q"])
                q_place = self.ik_cam(C_place, L["holderB_grasp_q"], "seat in the second holder", k)
            self.cam_up(KPICK_CARRY_UP_M - LIFT_M, "up to carry height")
            C_carry = C_place.copy()
            C_carry[2, 3] += KPICK_CARRY_UP_M
            if not self.carry(C_carry, carry_guard):
                # a halt would open the jaws and drop the plate from carry height: hold it and wait
                raise Abort("touched something while carrying the plate; holding it here (check, then decide)")
            C_low = C_place.copy()
            C_low[2, 3] += LIFT_M
            self.move_cart(C_low, "down to above the place")
        if q_place is None and home:
            q_place = self.holder_a_biased(L["flat_grasp_q"], k, "place in holder A")
        if grip_dz and q_place is not None:  # joints for a grip at the usual height: place that much higher
            C_place = k.cam(q_place)  # (a C_place from the plate pose already carries the grip height)
            C_place[2, 3] += grip_dz
            q_place = None
        if q_place is None:
            q_place = self.ik_cam(C_place, self.joints(), "place", k)
        q_seat = None
        if self.move_exact_guarded(q_place, "place"):
            q_seat = {j: self.joints()[j] for j in ARM6}
        else:
            high = tip_z(self.joints()) - tip_z(q_place)
            log(f"place: contact {high * 1000:+.1f} mm from the planned height")
            # bare-table goals keep PLACE_CLEAR_M off every holder, so a touch a little high there is the table
            # (+2.6 mm at (274, 290), 2026-10-21: the model's height is a few mm off); never let go on a rim
            table_goal = isinstance(place_xy_yaw, tuple) and self.place_clash(
                self.plate_outline(x, y, a, PLACE_CLEAR_M), T_ref, k) is None
            if high > (PLACE_TABLE_MAX_DZ_M if table_goal else HOLDER_SEAT_MAX_DZ_M):
                if home:
                    self.note_holder_view({"outcome": "wall", "high_mm": round(high * 1000, 1)})
                self.cam_up(LIFT_M, "lift off the holder wall")
                seated = False
                if home:
                    # unattended (user, 2026-11-04): retry stepping toward the arm's base (the user saw the plate on the rim
                    # >= 30 mm too far out, 15:40), then fall back to a pad spot instead of waiting
                    radial = T_ref[:2, 3] / np.linalg.norm(T_ref[:2, 3])  # base -> holder A, level
                    for base_mm, y_mm in HOLDER_RETRY_OFFSETS_MM:
                        off = f"{base_mm:.0f} mm toward the base, {y_mm:+.0f} mm y"
                        C_try = k.cam(q_place)
                        C_try[:2, 3] -= radial * base_mm / 1000  # toward the arm's base
                        C_try[1, 3] += y_mm / 1000
                        C_up = C_try.copy()
                        C_up[2, 3] += LIFT_M
                        if not self.move_cart(C_up, f"over holder A, retry {off}", guard=True, free_air=True):
                            log("place: touched something moving over the retry; giving up on holder A")
                            break
                        q_try = self.ik_cam(C_try, q_place, f"place retry {off}", k)
                        if self.move_exact_guarded(q_try, f"place retry {off}"):
                            log(f"place: seated on the retry {off}")
                            self.note_holder_view({"seated_on_retry_mm": [base_mm, y_mm]})
                            q_place, seated = q_try, True
                            q_seat = {j: self.joints()[j] for j in ARM6}
                            break
                        h2 = tip_z(self.joints()) - tip_z(q_try)
                        log(f"place: retry {off} stopped {h2 * 1000:+.1f} mm high")
                        self.cam_up(LIFT_M, "lift off the holder wall")
                    if not seated:
                        goal = self.random_place()
                        if isinstance(goal, tuple):
                            log(f"HOLDER FALLBACK: holder A caught the plate on every try; setting it down on the pad at "
                                f"({goal[0] * 1000:.0f}, {goal[1] * 1000:.0f}) mm instead (episode = failure)")
                            self.rec_stop(False)
                            self.kpick_place(goal, T, C_grasp, L, k, T_ref, hb, tip_z, t0, scene_ref, align,
                                             carry_guard=carry_guard, grip_dz=grip_dz)
                            raise Abort("HOLDER FALLBACK: plate set down on the pad after holder A caught it")
                if not seated:
                    raise Abort(f"place: the plate stopped {high * 1000:.1f} mm high, on the holder's wall; "
                                "holding it above the holder (check, then decide)")
        # where the plate is set down, from the arm: the held pose carried with the camera. Not a lattice read: release
        # reads a well pitch off became the next pick's reference, and that place missed by 13-16 mm (2026-10-30
        # 13:22, 14:11, 14:31)
        T_rel = planar_pose(k.cam(self.joints()) @ np.linalg.inv(C_grasp) @ T, T[2, 3])
        # at the second holder, the open-jaw view over the seated plate is the next pick's close-view reference
        b_ref_new = DATA / "holderB_close_ref_new.png"
        table_ref_new = DATA / "table_close_ref_new.png"
        on_goal = isinstance(place_xy_yaw, tuple)  # reached here: set down on bare table, resting as held
        table_up_new = DATA / "table_above_ref_new.png"
        self.leave_plate(None, b_ref_new if to_b and q_seat is not None else table_ref_new if on_goal else None,
                         table_up_new if on_goal else None)
        if place_xy_yaw is None or home:
            if home:
                self.cam_up(0.03, "clear of the plate")
                self.rec_stop(True)
                self.move_segmented({**VIEW_POSE, "gripper": GRIPPER_OPEN_DEG}, "to viewing pose")
                self.close_gripper(max_close_deg=FLAT_APPROACH_GRIPPER_DEG)
                T_got = self.platelook(3, prior=T_ref[:2, 3], prior_yaw=yaw_deg(T_ref))
                err = (T_got[:2, 3] - T_ref[:2, 3]) * 1000
                self.rec_note(place_err_mm=[round(float(v), 1) for v in err])
                self.learn_holder_a_bias(err)
                self.note_holder_view({"outcome": "seated", "err_mm": [round(float(v), 2) for v in err]})
                log(f"KPICK SUCCEEDED (back in the holder): {err[0]:+.1f}, {err[1]:+.1f} mm from the reference, "
                    f"in {time.monotonic() - t0:.0f} s")
                self.move_segmented({**VIEW_POSE, "wrist_flex": 0.0}, "back")
                self.path = []
                self.rest()
                return
            self.move_exact(align, "back to the align pose")
            self.check_plate_undisturbed(scene_ref, "after placing")
            self.go_home_from_plate()
            log(f"KPICK SUCCEEDED (put back) in {time.monotonic() - t0:.0f} s")
            return
        self.cam_up(0.03, "clear of the plate")
        self.rec_stop(True)
        self.move_segmented({**VIEW_POSE, "gripper": GRIPPER_OPEN_DEG}, "to viewing pose")
        self.close_gripper(max_close_deg=FLAT_APPROACH_GRIPPER_DEG)
        # the read is only a check now (it can lock a well pitch off, or miss the plate: 'no plate' halts at 13:26, 14:15)
        try:
            T_read = self.platelook(3, prior=T_rel[:2, 3], prior_yaw=yaw_deg(T_rel), anchor=True)
        except Abort as e:
            if not str(e).startswith(("platelook: no plate", "platelook: lost the plate")) or self.frozen:
                raise  # a halt (guard trip, heat) still stops
            log(f"kpick: no plate read at release ({e}); going by the arm's set-down pose")
            T_read = None
        # re-anchor on a clear read: the arm's pose comes through the reference relation, carried pick to pick, and
        # that walked ~3-5 mm per pick until reads disagreed by 42 -> 61 mm (2026-11-04 18:12-18:54); far beyond a
        # well pitch (9 mm), so the read is the better pose then. Small disagreements keep the arm's pose (pitch locks).
        read_err = None if T_read is None else (T_read[:2, 3] - T_rel[:2, 3]) * 1000  # vs the arm's own pose
        reanchored = False
        if T_read is not None:
            gap = float(np.hypot(*(T_read[:2, 3] - T_rel[:2, 3]))) * 1000
            dyaw = (yaw_deg(T_read) - yaw_deg(T_rel) + 90) % 180 - 90
            if gap > REANCHOR_MIN_MM and getattr(self, "platelook_spread_mm", 99.0) <= REANCHOR_MAX_SPREAD_MM \
                    and abs(dyaw) < 6.0:
                log(f"kpick: re-anchored the set-down pose on the release read ({gap:.1f} mm, {dyaw:+.2f} deg off)")
                T_rel = planar_pose(T_read, T_rel[2, 3])
                reanchored = True
        (DATA / "plate_pose.json").write_text(json.dumps({"T": T_rel.tolist(), "time": time.strftime("%H:%M:%S"),
                                                          "n": 0, "source": "arm at release",
                                                          "reanchored": reanchored}))
        T_got = T_rel
        err = (T_got[:2, 3] - T_goal[:2, 3]) * 1000
        a_err = (yaw_deg(T_got) - a + 90) % 180 - 90
        if read_err is not None:
            log(f"kpick: release read vs the arm's set-down pose {read_err[0]:+.1f}, {read_err[1]:+.1f} mm, "
                f"{(yaw_deg(T_read) - yaw_deg(T_rel) + 90) % 180 - 90:+.2f} deg")
        self.rec_note(goal_mm=[round(float(v) * 1000, 1) for v in (x, y)], goal_yaw_deg=round(float(a), 1),
                      place_err_mm=[round(float(v), 1) for v in err], place_err_deg=round(float(a_err), 2),
                      read_vs_arm_mm=None if read_err is None else [round(float(v), 1) for v in read_err])
        if to_b and np.hypot(*err) < 4.0:  # seated: the plate's own pose is the better holder reference
            # unless the empty frame was measured: plate-pose goals crept ~8 mm deeper into it (2026-10-20)
            if not (hb and hb.get("source") == "frame"):
                (DATA / "holder_b.json").write_text(json.dumps({"T": T_got.tolist(), "source": "plate",
                                                                "time": time.strftime("%Y-%m-%d %H:%M:%S")}))
            if q_seat is not None:  # reached the planned depth without touching: the pick there reuses it
                # with the plate's pose in the camera frame as carried (rigid in the jaws), for the next pick
                self.save_learned(holderB_grasp_q=q_seat, holderB_P=np.linalg.inv(C_grasp) @ T)
                # taken with the jaws open, as the next pick sees it (a closed-jaw view never matched: match 0.8,
                # halted 2026-10-20; with no reference the plate's place in the jaws went unchecked)
                b_ref_new.replace(DATA / "holderB_close_ref.png")
                log("kpick: seated joints and the open-jaw close view at the second holder saved")
        if to_b and not (np.hypot(*err) < 4.0 and q_seat is not None):
            # not seated as planned: no trusted reference; the next pick there saves its own
            (DATA / "holderB_close_ref.png").unlink(missing_ok=True)
            b_ref_new.unlink(missing_ok=True)
        if on_goal and table_ref_new.exists():
            # set down without touching (q_seat): the view shows the plate as held, wherever it landed; the goal
            # error is mostly placing accuracy (a 6 mm one threw the first away, 2026-10-21)
            keep = DATA / f"table_close_ref_{time.strftime('%H%M%S')}.png"
            keep_up = DATA / f"table_above_ref_{time.strftime('%H%M%S')}.png"
            for src, dst in ((table_ref_new, keep), (table_up_new, keep_up)):  # each view with its joints
                for ext in (".png", ".json"):
                    if src.with_suffix(ext).exists():
                        src.with_suffix(ext).replace(dst.with_suffix(ext))
            # and only from reads that agree: the 10:23 ref came from reads 4.8 mm apart, and the next picks started
            # 14-35 px off over the grasp (2026-10-23)
            # (the pose stored with it is the arm's, so read spreads no longer matter)
            # (a re-anchored pose is off the goal by the walk it removed, but its relation is the measured one)
            if reanchored or (np.hypot(*err) < 10.0 and abs(a_err) < 3.0):
                for src, dst in ((keep, "table_close_ref"), (keep_up, "table_above_ref")):
                    for ext in (".png", ".json"):
                        if src.with_suffix(ext).exists():
                            shutil.copyfile(src.with_suffix(ext), (DATA / dst).with_suffix(ext))
                (DATA / "table_close_ref_plate.json").write_text(json.dumps({"T": T_got.tolist(), "ref": keep.name}))
                log(f"kpick: the open-jaw view at release is the new bare-table close-view reference ({keep.name})")
        log(f"KPICK SUCCEEDED: placed at ({T_got[0, 3] * 1000:.1f}, {T_got[1, 3] * 1000:.1f}) mm, "
            f"{err[0]:+.1f}, {err[1]:+.1f} mm and {a_err:+.2f} deg from the goal, in {time.monotonic() - t0:.0f} s")
        self.move_segmented({**VIEW_POSE, "wrist_flex": 0.0}, "back")
        self.path = []
        self.rest()

    def table_height_from_depth(self, tag: str):
        """Camera height over the table plane measured by stereo depth (RANSAC plane through the wood; the
        pads are nearer and the plate top is a smaller plane 15 mm up) vs the kinematic prediction.
        -> (measured_m, predicted_m) or None."""
        self.hold(0.5)
        depth = None
        for _ in range(10):
            with self.cam.lock:
                depth = None if self.cam.depth is None else self.cam.depth.copy()
            if depth is not None:
                break
            self.hold(0.2)
        if depth is None:
            log("heightcheck: no depth frame")
            return None
        k = self.kin()
        C = k.cam(self.joints())
        np.save(DATA / f"heightcheck_{tag}_depth.npy", depth)
        cv2.imwrite(str(DATA / f"heightcheck_{tag}.png"), self.cam.latest(self.cam.seq)[1])
        v, u = np.mgrid[0:depth.shape[0]:4, 0:depth.shape[1]:4]
        z = depth[v, u].astype(float) / 1000
        ok = (z > 0.10) & (z < 0.60)
        Kc = self.cam.K
        P = np.stack([(u[ok] - Kc[0][2]) / Kc[0][0] * z[ok], (v[ok] - Kc[1][2]) / Kc[1][1] * z[ok], z[ok]], -1)
        if len(P) < 200:
            log(f"heightcheck {tag}: only {len(P)} depth points")
            return None
        rng = np.random.default_rng(0)
        best = None
        for _ in range(300):
            A = P[rng.choice(len(P), 3, replace=False)]
            n = np.cross(A[1] - A[0], A[2] - A[0])
            if np.linalg.norm(n) < 1e-9:
                continue
            n /= np.linalg.norm(n)
            inl = np.abs((P - A[0]) @ n) < 0.003
            if best is None or inl.sum() > best.sum():
                best = inl
        Q = P[best]
        c = Q.mean(0)
        n = np.linalg.svd(Q - c)[2][2]
        h_meas = abs(c @ n)  # camera origin to the plane
        n_base = C[:3, :3] @ n
        h_pred = C[2, 3] - k.tags[1][2, 3]
        log(f"heightcheck {tag}: camera {h_meas * 1000:.1f} mm from the table plane by depth, {h_pred * 1000:.1f} mm "
            f"by the model (diff {(h_meas - h_pred) * 1000:+.1f}); {best.sum()}/{len(P)} points, plane tilt "
            f"{np.degrees(np.arccos(min(1.0, abs(n_base[2])))):.1f} deg in the model's base frame")
        return h_meas, h_pred

    def heightcheck(self):
        """Over the empty holder A (taught lift pose) and over the plate on bare table (mapped lift pose):
        compare the depth-measured camera height with the model's. Never goes below the lift poses."""
        L = self.learned()
        T_ref = np.array(json.loads((DATA / "plate_ref.json").read_text())["T"])
        prior = np.array(json.loads((DATA / "plate_pose.json").read_text())["T"])[:2, 3]
        if np.linalg.norm(prior - T_ref[:2, 3]) < HOLDER_NEAR_M:
            raise Abort("heightcheck: the plate is in holder A; it needs A empty and the plate on bare table")
        if self.s.torque and self.joints()["gripper"] < FLAT_APPROACH_GRIPPER_DEG - 5:
            self.move_segmented({**VIEW_POSE, "wrist_flex": 0.0}, "unfold")
            self.close_gripper(max_close_deg=FLAT_APPROACH_GRIPPER_DEG)
        T = self.plate_side(self.platelook(3, prior=prior), T_ref)
        k = self.kin()
        D = T @ np.linalg.inv(T_ref)
        yaw = math.degrees(math.atan2(D[1, 0], D[0, 0]))
        out = {}
        for tag, M, dyaw in (("A", np.eye(4), 0.0), ("table", D, yaw)):
            pre, align, lift = (self.ik_cam(M @ k.cam(L[n]), {**L[n], "shoulder_pan": L[n]["shoulder_pan"] + dyaw}, n, k)
                                for n in ("flat_pregrasp_joints", "flat_ref_align_q", "flat_lift_q"))
            self.thermal_check()
            self.move_segmented(pre, f"to the pregrasp ({tag})")
            self.move_exact(pre, f"to the pregrasp ({tag})")
            self.move_exact(align, "to the align pose")
            self.move({**self.s.hold, "gripper": FLAT_OPEN_DEG}, "open gripper (as taught)")
            self.move_exact(lift, f"to above the grasp ({tag})")
            out[tag] = self.table_height_from_depth(tag)
            if tag == "table":  # the plate measured from right over it, vs from the look poses
                r = self.plate_find(tag="_lift", prior=T[:2, 3])
                if r is not None:
                    Tl = self.plate_side(r["T"], T_ref)
                    d = np.linalg.inv(T) @ Tl  # in the plate frame: x along the 128 mm side, y across (jaw opening)
                    log(f"HEIGHTCHECK: plate from over it vs from the look poses: {d[0, 3] * 1000:+.1f} mm along, "
                        f"{d[1, 3] * 1000:+.1f} mm across, {math.degrees(math.atan2(d[1, 0], d[0, 0])):+.2f} deg")
            self.move_exact(align, "back up to the align pose")
            self.move_exact(pre, "back to the pregrasp")
        if out.get("A") and out.get("table"):
            e = (out["table"][0] - out["table"][1]) - (out["A"][0] - out["A"][1])
            log(f"HEIGHTCHECK: over bare table the camera is {e * 1000:+.1f} mm from the model's height, relative to holder A "
                f"(negative: lower than planned)")
        self.go_home_from_plate()

    def go_home_from_plate(self):
        """Up to the viewing pose with the jaws open, close them there (sensing, well clear of the plate), fold."""
        self.move_segmented({**VIEW_POSE, "gripper": GRIPPER_OPEN_DEG}, "to viewing pose")
        self.close_gripper(max_close_deg=FLAT_APPROACH_GRIPPER_DEG)
        self.move_segmented({"wrist_flex": 0.0}, "straighten wrist")
        self.path = []
        self.rest()

    def kin_tip_height_fn(self):
        """Fingertip height over tag 1 (m) as a function of the arm joints, from the calibrated URDF."""
        sys.path.insert(0, str(HERE / "kin"))
        import fk as fkmod
        K = json.loads((HERE / "kin" / "kin_calib.json").read_text())
        X, Y1 = np.array(K["X_cam_in_end"]), np.array(K["tags_in_base"]["1"])
        signs, offs = K["signs"], K["offsets_deg"]
        tip_end = X[:3, :3] @ FINGER_MID_CAM + X[:3, 3]

        def tip_z(q):
            qq = {j: -(signs[k] * q[j] + offs[k]) for k, j in enumerate(fkmod.PUBLIC)}
            T = fkmod.fk(qq)
            return float((T[:3, :3] @ tip_end + T[:3, 3])[2] - Y1[2, 3])
        return tip_z

    def kincollect(self, n: int, center: dict, spread: dict, near: bool = False):
        """Kinematic calibration data: visit n random poses around `center` (deg, +-spread) and record
        the measured joints with the tag pose (detected frames only) to data/kin_samples.json."""
        self.require_motion_ok()
        if not self.s.torque:
            self.safe_enable()
        if self.joints()["gripper"] < FLAT_APPROACH_GRIPPER_DEG - 5:
            self.close_gripper(max_close_deg=FLAT_APPROACH_GRIPPER_DEG)
        if self.joints()["shoulder_lift"] > -30.0:
            self.move_segmented({**center, "wrist_flex": 0.0}, "unfold")
        out = DATA / "kin_samples.json"
        samples = json.loads(out.read_text()) if out.exists() else []
        rng = np.random.default_rng()
        got = 0
        tip_z = self.kin_tip_height_fn() if near else None
        if near:
            self.move_segmented(VIEW_POSE, "view")
            self.move_segmented(self.learned()["flat_pregrasp_joints"], "pregrasp")
        prev = self.joints()
        for i in range(n):
            self.thermal_check()
            for _ in range(50):
                target = {j: center[j] + rng.uniform(-1, 1) * spread.get(j, 0.0) for j in ARM6}
                if not near:
                    break
                mid = {j: (target[j] + prev[j]) / 2 for j in ARM6}
                if min(tip_z(target), tip_z(mid)) >= KIN_NEAR_MIN_TIP_Z:
                    break
            else:
                raise Abort("kincollect: no pose clear of the table found")
            self.move_segmented(target, f"kin pose {i}")
            prev = target
            self.hold(SETTLE_S + 0.4)
            try:
                m = self.meas.measure(SERVO_FRAMES, timeout_s=2.0)
            except Exception as e:
                log(f"kin pose {i}: no tag ({e})")
                continue
            if m.get("tracked"):
                log(f"kin pose {i}: tag only tracked, skipped")
                continue
            q = self.joints()
            samples.append({"tag": ACTIVE_TAG["id"], "q": {j: round(q[j], 3) for j in ARM6},
                            "p": np.asarray(m["p"]).tolist(), "R": np.asarray(m["R"]).tolist()})
            got += 1
            out.write_text(json.dumps(samples))
        log(f"KINCOLLECT: {got}/{n} samples recorded (total {len(samples)} in {out.name})")
        if near:
            self.move_segmented(center, "back to centre")
            self.move_segmented(self.learned()["flat_pregrasp_joints"], "pregrasp")
            self.move_segmented(VIEW_POSE, "view")
        else:
            self.move_segmented({**center, "wrist_flex": 0.0}, "back to centre")
        self.path = []
        self.rest()

    def hover(self):
        if self.tip is None:
            raise Abort("jaw-tip point unknown; run perceive / set `tip`")
        goal = self.tip + TAG_AIM_OFFSET + np.array([0, 0, HOVER_M])
        self.stage = "hover"
        self.servo_to(goal, HOVER_TOL_M, "hover", max_iter=150)
        log(f"HOVERING {HOVER_M * 100:.0f} cm in front of the tag. Holding.")
        self.note = "hovering"

    def touch(self):
        if self.stage != "hover":
            raise Abort("run `hover` first")
        self.stage = "touch"
        goal = self.tip + TAG_AIM_OFFSET + np.array([0, 0, -TOUCH_PAST_M])
        saved_step = globals()["MAX_POS_STEP_M"]
        globals()["MAX_POS_STEP_M"] = TOUCH_STEP_M
        contact = False
        try:
            m = self.servo_to(goal, 0.002, "touch", max_iter=40, torque_guard=True, reprobe=False)
            contact = m is None  # servo_to returns None when the torque guard saw contact
        except Abort as e:
            if "did not converge" not in str(e):
                raise
            m = None
        finally:
            globals()["MAX_POS_STEP_M"] = saved_step
        m = m or self.measure()
        gap = (m["p"][2] - self.tip[2]) * 1000
        if contact or gap <= 3.0:
            log(f"TOUCH reached: tag plane {gap:+.1f} mm from the jaw-tip depth"
                f"{' (contact torque)' if contact else ''}. Holding 2 s.")
        else:
            log(f"TOUCH NOT reached: still {gap:.1f} mm short after 40 steps. Backing off.")
        self.write_status(touch_gap_mm=gap)
        self.hold(2.0)
        self.stage = "hover"
        self.servo_to(self.tip + TAG_AIM_OFFSET + np.array([0, 0, HOVER_M]), 0.01, "retreat")
        log("Back at hover.")

    def rest(self):
        """Retrace the visited poses in reverse, then settle into the zero pose and go limp."""
        if self.can_fault and not self.can_dead:
            log("rest: after a CAN stall and reset; going to rest")
        self.can_fault = False
        self.resting = True
        try:
            self._rest()
        finally:
            self.resting = False

    def _rest(self):
        if self.s.torque:
            self.recover(list(mj.B601_JOINTS))
            waypoints, last = [], self.joints()
            for pose in reversed(self.path):
                pose = {**dict(pose), "gripper": last["gripper"]}  # the jaws never close blind
                if pose["shoulder_lift"] > -25.0:
                    # Near the folded pose wrist_flex meets a mechanical stop at about +3 deg.
                    pose["wrist_flex"] = min(pose["wrist_flex"], WRIST_FOLDED_MAX_DEG)
                if max(abs(pose[j] - last[j]) for j in mj.B601_JOINTS) >= 8.0:
                    waypoints.append(pose)
                    last = pose
            waypoints.append({**REST_POSE, "gripper": last["gripper"]})
            log(f"returning to rest along {len(waypoints)} waypoints")
            if STREAM_MOTION:
                self.stream_path(waypoints, "rest")
                self.hold(0.7)
            else:
                for pose in waypoints:
                    self.glide(pose)
            self.s.release()
        self.path = []
        self.frozen = False
        self.stage = "idle"
        self.note = "at rest, torque off"
        log("At rest; torque off.")

    # ---- command loop
    def handle(self, line: str) -> bool:
        parts = line.split()
        if not parts:
            return True
        cmd = parts[0]
        log(f"> {line}")
        try:
            if cmd == "perceive":
                self.perceive()
            elif cmd == "tip":
                self.set_tip([float(v) for v in parts[1:4]])
            elif cmd == "probe":
                self.probe()
            elif cmd == "diag":
                self.diag()
            elif cmd == "goto":
                self.goto(parts[1:])
            elif cmd == "recover":
                self.recover(parts[1:] or [j for j in mj.B601_JOINTS])
            elif cmd == "info":
                (DATA / "intrinsics.json").write_text(json.dumps({"K": self.cam.K.tolist(), "dist": self.cam.dist.tolist()}))
                log(f"K = {np.round(self.cam.K, 1).tolist()}")
            elif cmd == "target":
                tag_id = int(parts[1])
                size = float(parts[2]) / 1000 if len(parts) > 2 else TAG_SIZES[tag_id]
                ACTIVE_TAG.update(id=tag_id, size=size)
                self.active_tag = dict(ACTIVE_TAG)
                self.meas.prev_normal, self.meas.ref = None, None
                self.J = self.J6 = None
                log(f"target tag {tag_id} ({size * 1000:.0f} mm)")
            elif cmd == "where":
                log(f"true joints (torque off) { {k: round(v, 1) for k, v in self.fresh_joints_limp().items()} }")
            elif cmd == "limits":
                log(f"config {self.follower.config.joint_limits.get('gripper')}, follower {self.follower._joint_limits.get('gripper')}, "
                    f"same dict {self.follower.config.joint_limits is self.follower._joint_limits}")
            elif cmd == "motors":
                for m in self.follower.motors.values():  # fresh states even while torque is off
                    try:
                        m.request_feedback()
                    except Exception as e:
                        log(f"feedback request failed: {e}")
                time.sleep(0.05)
                log(f"joints { {k: round(v, 1) for k, v in self.joints().items()} }, hold "
                    f"{ {k: round(v, 1) for k, v in self.s.hold.items()} if self.s.hold else None}")
                for k, st in self.follower._read_feedback().items():
                    log(f"motor {k}: {st}")
            elif cmd == "snap":
                self.snap()
            elif cmd == "unfreeze":  # keep holding where it is and allow motion again
                self.frozen = False
                log("unfrozen; holding the current command")
            elif cmd == "align":  # servo to the grasp pose this many mm higher, then snapshot
                R_t, p_t = self.grasp_target(float(parts[1]) / 1000)
                self.servo_pose(R_t, p_t, 1.5, 2.0, f"align {parts[1]}", max_iter=40)
                self.snap()
            elif cmd == "touchprobe":
                self.touch_probe(float(parts[1]), float(parts[2]))
            elif cmd == "learn":  # save the current pose as the pregrasp pose, with the current J6
                q = self.joints()
                self.save_learned(pregrasp_joints={j: round(q[j], 2) for j in ARM6}, J6=self.J6)
                log(f"saved learned pregrasp {self.learned()['pregrasp_joints']}")
            elif cmd == "savej6":
                self.save_learned(J6=self.J6)
                log(f"saved J6 (singular values {np.round(np.linalg.svd(self.J6, compute_uv=False) * 1000, 2).tolist()})")
            elif cmd in GRIPPER_LOCKED_COMMANDS and getattr(self, "halted", None):
                log(f"refused `{cmd}`: halted after an anomaly ({self.halted}); check the plate, then send `clear`")
            elif cmd == "clear":
                log(f"cleared halt ({getattr(self, 'halted', None)})")
                self.halted = None
            elif cmd in GRIPPER_LOCKED_COMMANDS and GRIPPER_LOCKOUT:
                log(f"refused `{cmd}`: gripper lockout (the jaws were broken by force-blind closing on 2026-10-17)")
            elif cmd == "gripclose":  # test the sensing close with empty jaws: open, then close to the limit
                if not self.s.torque:
                    self.safe_enable()
                self.move({**self.s.hold, "gripper": GRIPPER_OPEN_DEG}, "open gripper")
                self.close_gripper(float(parts[1]) if len(parts) > 1 else -150.0)
            elif cmd == "gripcal":
                self.gripcal(float(parts[1]) if len(parts) > 1 else -150.0)
            elif cmd == "flatdown":  # from the flat align pose, servo the fingertip point down to this height (mm)
                h, met = self.flat_descend(float(parts[1]) / 1000, "flatdown")
                log(f"flatdown: tip point at {h * 1000 if h is not None else float('nan'):.1f} mm, contact {met}")
                self.snap()
            elif cmd == "record":  # record on [name] | record off
                if parts[1:2] == ["on"]:
                    import record as recmod
                    self.rec = recmod.EpisodeRecorder(parts[2] if len(parts) > 2 else "plate_pick")
                    log(f"recording episodes to {self.rec.root} ({len(self.rec.outcomes)} so far)")
                else:
                    if getattr(self, "rec", None) is not None:
                        self.rec.close()
                    self.rec = None
                    log("recording off")
            elif cmd in ("flatfast", "flatpick") and getattr(self, "rec", None) is not None:
                self.rec_start(PLATE_TASK, command=cmd)
                ok = False
                try:
                    (self.flatfast if cmd == "flatfast" else self.flatpick)()
                    ok = True
                finally:
                    self.rec_stop(ok)
            elif cmd == "flatfast":
                self.flatfast()
            elif cmd == "platefind":  # at the current pose
                self.plate_find()
            elif cmd == "platelook":  # platelook [n [x_mm y_mm]]: x y = a rough plate centre to look at first
                prior = None
                if len(parts) > 3:  # aim there and find it unrestricted (a rough guess may be a well or more off)
                    self.require_motion_ok()
                    if not self.s.torque:
                        self.safe_enable()
                    self.move_segmented({**VIEW_POSE, "wrist_flex": 0.0}, "unfold")
                    self.move_segmented(VIEW_POSE, "view")
                    xy = np.array([float(parts[2]), float(parts[3])]) / 1000
                    self.look_at_plate(np.array([*xy, self.kin().tags[1][2, 3] + PLATE_TOP_Z]), "look at the guess")
                    r = self.plate_find(tag="_aim")
                    if r is None:
                        raise Abort("platelook: no plate where it was guessed")
                    prior = r["T"][:2, 3]
                self.platelook(int(parts[1]) if len(parts) > 1 else 3, prior=prior)
                self.move_segmented({**VIEW_POSE, "wrist_flex": 0.0}, "back")
                self.path = []
                self.rest()
            elif cmd == "halt":  # manual: jaws fully open, straight up, home; then `clear`
                self.halt("manual halt")
            elif cmd == "kresume":  # kresume B|home|x y yaw  ang dx dy (from the log's alignment line)
                dest = {"home": "home", "B": "B"}.get(parts[1])
                if dest is None:
                    dest, rest_ = (float(parts[1]) / 1000, float(parts[2]) / 1000, float(parts[3])), parts[4:]
                else:
                    rest_ = parts[2:]
                self.kresume(dest, *(float(v) for v in rest_[:3]))
            elif cmd == "holderlook":  # holderlook [n]: measure the empty second holder
                self.holderlook(int(parts[1]) if len(parts) > 1 else 3)
                self.move_segmented({**VIEW_POSE, "wrist_flex": 0.0}, "back")
                self.path = []
                self.rest()
            elif cmd == "randtest":  # randtest N: sample N random goals (no motion), drawn on data/zonescan_goals.png
                meta = json.loads((DATA / "zonescan.json").read_text())
                img = cv2.imread(str(DATA / "zonescan.png"))
                (x0, _), (_, y1), res = meta["x_range"], meta["y_range"], meta["res"]
                px = lambda P: np.round([[(x - x0) / res, (y1 - y) / res] for x, y in P]).astype(np.int32)
                for _ in range(int(parts[1]) if len(parts) > 1 else 20):
                    g = self.random_place()
                    if isinstance(g, str):
                        log(f"randtest: holder {g}")
                        continue
                    log(f"randtest: ({g[0] * 1000:.0f}, {g[1] * 1000:.0f}) mm, {g[2]:+.1f} deg")
                    cv2.polylines(img, [px(self.plate_outline(*g))], True, (0, 0, 255), 1)
                cv2.imwrite(str(DATA / "zonescan_goals.png"), img)
            elif cmd == "zonemap":  # why each table spot (20 mm grid, radial yaw + yaw arg) is or is not a goal
                T_ref = np.array(json.loads((DATA / "plate_ref.json").read_text())["T"])
                L, k, dyaw = self.learned(), self.kin(), float(parts[1]) if len(parts) > 1 else 0.0
                rows = []
                for y in np.arange(ZONE_Y_MAX_M, -0.001, -0.02):
                    row = ""
                    for x in np.arange(ZONE_X_M[0], ZONE_X_M[1] + 0.001, 0.02):
                        a = self.radial_yaw(T_ref, (x, y)) + dyaw
                        r = self.place_ok(x, y, a, T_ref, L, k, ignore=("B",), geo=False)
                        r = r or self.place_ok(x, y, a, T_ref, L, k, ignore=("B",), reach=False)
                        row += "." if r is None else "Z" if r.startswith("outside") else "H" if r.startswith("holder") \
                            else "T" if r == "tag 1" else "K" if r in KEEP_OUT_M else "V" if r.startswith("no view") \
                            else r.split(":")[0][5:7] if r.startswith("flat_") else "c" if r.startswith("carry") else "?"
                    rows.append(f"{y * 1000:4.0f} {row}")
                log("zonemap x from %.0f step 20 mm (. ok, Z zone, H holder, T tag, K keep-out, V view, pr/re/li/gr IK)\n" %
                    (ZONE_X_M[0] * 1000) + "\n".join(rows))
            elif cmd == "heightcheck":  # depth-measured camera height over A and over the plate (never descends)
                self.heightcheck()
            elif cmd == "zonescan":  # stitch a top-down mosaic of the table (data/zonescan.png)
                self.zonescan()
                self.move_segmented({**VIEW_POSE, "wrist_flex": 0.0}, "back")
                self.path = []
                self.rest()
            elif cmd == "plateref":  # the plate pose now (in the holder, as taught) anchors kpick
                T = self.platelook(5)
                (DATA / "plate_ref.json").write_text(json.dumps({"T": T.tolist(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}))
                log("plate reference pose saved")
                self.move_segmented({**VIEW_POSE, "wrist_flex": 0.0}, "back")
                self.path = []
                self.rest()
            elif cmd == "kpick":  # kpick [x_mm y_mm yaw_deg | rand | home]
                arg = parts[1] if len(parts) == 2 else None
                place = (float(parts[1]) / 1000, float(parts[2]) / 1000, float(parts[3])) if len(parts) >= 4 else \
                    self.other_holder() if arg == "other" else self.random_place() if arg == "rand" else \
                    {"home": "home", "B": "B", "b": "B"}.get(arg)
                if isinstance(place, tuple):
                    log(f"kpick: goal ({place[0] * 1000:.0f}, {place[1] * 1000:.0f}) mm, yaw {place[2]:+.1f} deg")
                elif arg == "rand":
                    log(f"kpick: goal holder {place}")
                # an episode runs from the first motion to the jaws clearing the placed plate (kpick closes
                # it there, as a success); anything that ends kpick earlier is recorded as a failure
                # pad -> holder A ("kpick home") is the holder task too; only holder -> holder is the shuttle
                self.rec_start((HOLDER_TASK if arg in ("rand", "home") else SHUTTLE_TASK) if place in ("home", "B")
                               else KPICK_TASK_TABLE if place else PLATE_TASK,
                               command=line.strip())
                try:
                    self.kpick(place)
                finally:
                    self.rec_stop(False)
            elif cmd == "platescan":  # platescan N
                self.platescan(int(parts[1]) if len(parts) > 1 else 8)
            elif cmd == "kincollect":  # kincollect N [view|start|near]
                n = int(parts[1]) if len(parts) > 1 else 30
                if parts[2:] == ["near"]:
                    self.kincollect(n, self.learned()["flat_ref_align_q"], KIN_SPREAD_NEAR, near=True)
                else:
                    centre = START_POSE_FULL if parts[2:] == ["start"] else VIEW_POSE
                    self.kincollect(n, centre, KIN_SPREAD)
            elif cmd == "flatcal":
                self.flatcal()
            elif cmd == "flatpick":
                self.flatpick(align_only=parts[1:] == ["align"])
            elif cmd == "fastpick":
                self.fastpick()
            elif cmd == "pick":
                self.pick()
            elif cmd == "grasp":
                self.grasp()
            elif cmd == "place":
                self.place()
            elif cmd == "pregrasp":
                self.pregrasp()
            elif cmd == "hover":
                self.hover()
            elif cmd == "touch":
                self.touch()
            elif cmd == "rest":
                if parts[1:] == ["direct"]:  # straight to the zero pose, no retrace
                    self.path = []
                self.rest()
            elif cmd == "reload":
                for m in ("fk", "model", "birdseye", "plate_detect", "lookdown", "holder_detect", "plate_edges"):  # kin/ modules too
                    sys.modules.pop(m, None)
                ns = {"__name__": "agent_reloaded", "__file__": str(HERE / "agent.py")}
                exec(compile((HERE / "agent.py").read_text(), str(HERE / "agent.py"), "exec"), ns)
                ns["ACTIVE_TAG"].update(getattr(self, "active_tag", {}))
                self.__class__ = ns["Agent"]
                self.meas.__class__ = ns["Measurer"]
                self.meas.tick = self.s.tick
                self.follower.config.max_relative_target = ns["MAX_RELATIVE_TARGET_DEG"]
                self.apply_gains()
                self.apply_limits()
                log("reloaded agent code (camera and follower connections kept)")
            elif cmd == "quit":
                return False
            else:
                log(f"unknown command {cmd}")
        except Abort as e:
            self.freeze(str(e))
        except Exception as e:
            log(traceback.format_exc())
            # with the jaws open (nothing held) leave the plate as a halt does: a code error froze the arm stretched
            # at the grasp and the shoulder reached 80 C before anyone looked (2026-10-21)
            try:
                empty = self.s.torque and self.stage == "kpick" and self.joints()["gripper"] < GRIPPER_OPEN_DEG + 20
            except Exception:
                empty = False
            if empty:
                try:
                    self.halt(f"error: {e}")
                except Abort as a:
                    self.freeze(str(a))
                except Exception as e2:
                    log(traceback.format_exc())
                    self.freeze(f"error: {e}; halting failed too: {e2}")
            else:
                self.freeze(f"error: {e}")
        self.write_status()
        log(f"< done {cmd}")
        return True

    def run(self):
        if not CTL.exists():
            os.mkfifo(CTL)
        log(f"ready; send commands to {CTL}")
        keep_going = True
        while keep_going:
            try:
                with CTL.open() as ctl:  # blocks until a writer opens it; the hold thread keeps ticking
                    for line in ctl:
                        keep_going = self.handle(line.strip())
                        if not keep_going:
                            break
            except OSError as e:
                log(f"command pipe error ({e}); reopening")
                time.sleep(0.5)

    def close(self):
        try:
            self.rest()
        except Exception as e:
            log(f"rest on exit failed: {e}")
        try:
            self.follower.disconnect()
        finally:
            self.cam.close()


def detach_output() -> None:
    """Send all output (including DepthAI's C++ logging) to agent.out and read nothing from the
    terminal, so a backgrounded agent is never stopped for terminal I/O."""
    out = os.open(HERE / "agent.out", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.dup2(out, 1)
    os.dup2(out, 2)
    sys.stdout = os.fdopen(1, "w", buffering=1)
    sys.stderr = os.fdopen(2, "w", buffering=1)


def main():
    detach_output()
    agent = Agent()
    holder = threading.Event()

    # Keep refreshing the hold command while waiting for the next FIFO command.
    def keep_holding():
        while not holder.is_set():
            if bus_lock.acquire(blocking=False):
                try:
                    if agent.s.torque:
                        agent.s.tick()
                except Exception as e:
                    log(f"hold refresh failed: {e}")
                finally:
                    bus_lock.release()
            time.sleep(1 / 30)

    bus_lock = threading.Lock()

    def handle(line):
        with bus_lock:
            # Look the method up on the current class so `reload` takes effect here too.
            return type(agent).handle(agent, line)

    agent.handle = handle
    threading.Thread(target=keep_holding, daemon=True).start()
    try:
        agent.run()
    finally:
        holder.set()
        agent.close()


if __name__ == "__main__":
    main()
