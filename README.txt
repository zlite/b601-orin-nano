B601 plate policy server for a Jetson Orin
==========================================

What this is
  A Docker image that runs the LeRobot policy server with the plate policy (SmolVLA fine-tune
  chr1sa/smolvla_plate_random_v3, included in ./checkpoint, so no Hugging Face login is needed).
  The robot PC sends camera frames and joint positions over the LAN (port 7447) and gets back
  5-second action plans. The Jetson never talks to the robot; the robot PC keeps control and its
  safety checks.

Requirements
  - Jetson Orin (Nano / NX / AGX) with JetPack 7 (L4T R39, driver supporting CUDA 13.2+). Check: cat /etc/nv_tegra_release
  - Docker with the NVIDIA runtime (installed with JetPack's nvidia-container packages)
  - Internet during the build (base image ~6 GB + Python packages), ~15 GB free disk
  - Same network as the robot PC

Steps (in this folder on the Jetson)
  1. ./jetson.sh check     JetPack, Docker, NVIDIA runtime, power mode, IP address
  2. ./jetson.sh build     builds image "b601-policy" (15-40 min the first time)
  3. ./jetson.sh bench     times the model; a chunk should take well under 1.5 s
  4. ./jetson.sh start     starts the server; prints READY and the address for the robot PC
     ./jetson.sh logs / ./jetson.sh stop

  For full speed set the maximum power mode first: sudo nvpmodel -m 0 (MAXN / MAXN SUPER) and
  sudo jetson_clocks. ./jetson.sh check shows the current mode.

  If docker says permission denied: sudo usermod -aG docker $USER, then log out and back in.

Running the policy on the arm (the B601 and its wrist OAK are plugged into this Jetson)
  0. sudo can/setup_can.sh       once: installs the gs_usb CAN driver (JetPack 7 lacks it) and brings the
                                 adapter up as canb601 at 1 Mbit/s on every plug-in (can0 is the Orin's own CAN)
  1. ./jetson.sh build           rebuild after pulling these changes (adds motorbridge, depthai, OAK plugin)
  2. ./jetson.sh start           policy server
  3. ./jetson.sh arm-check       torque stays OFF: prints the joints, the gripper frame, saves logs/wrist_320x200.png
  4. ./jetson.sh rollout table 180    (or: holder) the policy drives the arm for 180 s (training demos take ~2 min); Ctrl+C stops.
                                 At the end it parks at the start pose (closed loop, checked) before torque goes OFF;
                                 if it cannot get there it keeps holding and asks you to support the arm.
                                 Start each run from the rest (sit-down) pose: that is where it parks.
                                 Each run saves wrist frames (every 5 s) and overview-camera frames (every 2 s)
                                 to logs/frames_<time>/, next to logs/rollout_<time>.log.
  Settings (robot/b601_rollout.py): calibration/robots/rebot_b601_follower/b601_rs.json (from the robot PC),
  policy 10 Hz, commands 30 Hz (interpolated), commands lead the measured joints by at most 10 deg, chunk switches blended, gripper -325..-15 deg with 0.8/1.0 Nm caps, jaws closed before the policy starts (as in training) and the +360 unwrap from agent.py,
  wrist camera CAM_B 640x400 raw -> 320x200 INTER_AREA RGB (as record.py made the training images).

Security
  The server listens on all interfaces without authentication. Use it only on the trusted lab network.

Contents
  Dockerfile     nvcr.io/nvidia/pytorch:26.05-py3 (PyTorch 2.12, CUDA 13.2, Python 3.12) + LeRobot server
  lerobot/       LeRobot source at the robot PC's version (commit ae2b57f)
  checkpoint/    the policy; config.json state shape corrected 6 -> 7 (the training data has 7 joints)
  server.yaml    deployment b601_plate: 7 joint values + 320x200 wrist image -> 7 joint targets, 10 Hz
  bench.py       GPU timing (float32 and bfloat16)
  jetson.sh      check / build / bench / start / logs / stop
