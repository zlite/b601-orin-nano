#!/usr/bin/env bash
# SO-101 leader -> reBot B601-RS follower workflow with two OAK cameras.
# Settings live in robot.env. Run `bash robot.sh help` for the steps.
set -Eeuo pipefail
robot_setup_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
robot_action="${1:-help}"
shift || true

if [[ "$robot_action" == help || "$robot_action" == --help ]]; then
    cat <<'HELP'
Usage: bash robot.sh ACTION [extra lerobot args]

One-time setup (in order):
  status              Show devices, CAN state, calibration and mapping files.
  cameras             List OAK devices and save a test frame per sensor to logs/cameras/.
  calibrate-leader    SO-101 calibration (middle pose, then sweep every joint).
  calibrate-follower  B601 zero calibration. Arm limp; move it to the zero ("sit-down") pose.
  map                 Fit SO-101 -> B601 mapping: follower rests, then moves itself to preset poses; you match the leader.
  map-verify          Follower moves to a check pose; match the leader and read the live error.

Data collection:
  teleop              Leader drives the follower (motors ENABLED) with camera view.
  record              Record NUM_EPISODES episodes into DATASET_ROOT/HF_USER/DATASET_NAME.
  resume              Add more episodes to the existing dataset.
  push                Upload the dataset to the Hugging Face Hub (needs `hf auth login`).
  train-cmd           Print the training command to run on a GPU machine.

Other:
  app                 Start the web app at http://127.0.0.1:8601 (all of the above, with buttons).
  torque-off          Disable all follower motors. THE ARM WILL DROP: support it first.
  hf-login            Log in to Hugging Face (needed for push).
HELP
    exit 0
fi

source "$robot_setup_dir/robot.env"
source "$robot_setup_dir/miniforge/etc/profile.d/conda.sh"
conda activate "$robot_setup_dir/env"

[[ -e "$LEADER_PORT" ]] || LEADER_PORT=/dev/ttyACM0
robot_repo_id="$HF_USER/$DATASET_NAME"
robot_dataset_dir="$DATASET_ROOT/$robot_repo_id"
robot_calib_dir="$HOME/.cache/huggingface/lerobot/calibration"

robot_cameras() {
    if [[ -z "$WRIST_OAK_ID" || -z "$OVERVIEW_OAK_ID" ]]; then
        echo 'Set WRIST_OAK_ID and OVERVIEW_OAK_ID in robot.env (see: bash robot.sh cameras).' >&2
        exit 1
    fi
    printf '{wrist: {type: oak, device_id: "%s", socket: %s, %s, fps: %s, encoding: %s}, ' \
        "$WRIST_OAK_ID" "$WRIST_SOCKET" "$WRIST_SIZE" "$CAMERA_FPS" "$CAMERA_ENCODING"
    printf 'overview: {type: oak, device_id: "%s", socket: CAM_A, %s, rotation: %s, fps: %s, encoding: %s}}' \
        "$OVERVIEW_OAK_ID" "$OVERVIEW_SIZE" "$OVERVIEW_ROTATION" "$CAMERA_FPS" "$CAMERA_ENCODING"
}

robot_require_can() {
    if (( ($(cat /sys/class/net/"$FOLLOWER_PORT"/flags 2>/dev/null || echo 0) & 1) == 0 )); then
        echo "$FOLLOWER_PORT is down or missing. Run: sudo bash $robot_setup_dir/root-setup.sh" >&2
        exit 1
    fi
}

robot_args=(
    --robot.type=rebot_b601_follower --robot.motor_family=rs
    --robot.port="$FOLLOWER_PORT" --robot.id="$FOLLOWER_ID"
    --robot.max_relative_target="$MAX_RELATIVE_TARGET"
)
teleop_args=(
    --teleop.type=so101_b601_leader --teleop.port="$LEADER_PORT" --teleop.id="$LEADER_ID"
    --teleop.wrist_yaw_deg="$WRIST_YAW_DEG"
)

robot_record() {
    robot_require_can
    lerobot-record "${robot_args[@]}" --robot.cameras="$(robot_cameras)" "${teleop_args[@]}" \
        --dataset.repo_id="$robot_repo_id" --dataset.root="$robot_dataset_dir" \
        --dataset.single_task="$TASK" --dataset.fps="$CAMERA_FPS" \
        --dataset.num_episodes="$NUM_EPISODES" --dataset.episode_time_s="$EPISODE_TIME_S" \
        --dataset.reset_time_s="$RESET_TIME_S" --dataset.rgb_encoder.vcodec="$VCODEC" \
        --dataset.push_to_hub=false --display_data="${DISPLAY_DATA:-true}" "$@"
}

case "$robot_action" in
    status)
        printf 'Leader:   %s -> %s\n' "$LEADER_PORT" "$(readlink -f "$LEADER_PORT")"
        ls -l "$(readlink -f "$LEADER_PORT")"
        printf 'Follower: %s, flags=%s, ' "$FOLLOWER_PORT" "$(cat /sys/class/net/"$FOLLOWER_PORT"/flags 2>/dev/null || echo missing)"
        ip -details link show "$FOLLOWER_PORT" 2>/dev/null | grep -o 'can state [A-Z-]*\|bitrate [0-9]*' | tr '\n' ' '; echo
        python -m lerobot_camera_oak.find 2>&1 | grep -v '\[warning\]' || true
        for robot_f in "teleoperators/so_leader/$LEADER_ID.json" \
                       "robots/rebot_b601_follower/$FOLLOWER_ID.json" \
                       "teleoperators/so_leader/${LEADER_ID}_b601_mapping.json"; do
            if [[ -f "$robot_calib_dir/$robot_f" ]]; then echo "ok       $robot_f"; else echo "MISSING  $robot_f"; fi
        done
        if [[ -d "$robot_dataset_dir" ]]; then echo "Dataset: $robot_dataset_dir"; fi
        ;;
    cameras)
        python -m lerobot_camera_oak.find --snapshot "$robot_setup_dir/logs/cameras" --encoding "$CAMERA_ENCODING" "$@"
        ;;
    calibrate-leader)
        lerobot-calibrate --teleop.type=so101_leader --teleop.port="$LEADER_PORT" --teleop.id="$LEADER_ID" "$@"
        ;;
    calibrate-follower)
        robot_require_can
        lerobot-calibrate --robot.type=rebot_b601_follower --robot.motor_family=rs \
            --robot.port="$FOLLOWER_PORT" --robot.id="$FOLLOWER_ID" "$@"
        ;;
    map|map-verify)
        robot_require_can
        robot_extra=()
        [[ "$robot_action" == map-verify ]] && robot_extra=(--verify-only)
        python -m lerobot_teleoperator_so101_b601.map_joints --leader-port "$LEADER_PORT" --leader-id "$LEADER_ID" \
            --follower-port "$FOLLOWER_PORT" --follower-id "$FOLLOWER_ID" "${robot_extra[@]}" "$@"
        ;;
    teleop)
        robot_require_can
        robot_cam_args=()
        [[ -n "$WRIST_OAK_ID" && -n "$OVERVIEW_OAK_ID" ]] && robot_cam_args=(--robot.cameras="$(robot_cameras)" --display_data="${DISPLAY_DATA:-true}")
        lerobot-teleoperate "${robot_args[@]}" "${robot_cam_args[@]}" "${teleop_args[@]}" --fps="$CAMERA_FPS" "$@"
        ;;
    record)
        if [[ -d "$robot_dataset_dir" ]]; then
            echo "$robot_dataset_dir exists. Use 'resume' to add episodes or change DATASET_NAME." >&2
            exit 1
        fi
        robot_record "$@"
        ;;
    resume)
        robot_record --resume=true "$@"
        ;;
    push)
        if [[ "$HF_USER" == local ]]; then
            echo 'Set HF_USER in robot.env and rename the dataset folder to match before pushing.' >&2
            exit 1
        fi
        python - "$robot_repo_id" "$robot_dataset_dir" <<'PY'
import sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset
LeRobotDataset(sys.argv[1], root=sys.argv[2]).push_to_hub(private=True)
print(f"Pushed https://huggingface.co/datasets/{sys.argv[1]}")
PY
        ;;
    train-cmd)
        cat <<CMD
# On a machine with an NVIDIA GPU and LeRobot installed (this PC has no GPU):
lerobot-train \\
  --dataset.repo_id=$robot_repo_id \\
  --policy.type=act --policy.device=cuda \\
  --policy.repo_id=$HF_USER/act_$DATASET_NAME --policy.push_to_hub=true \\
  --output_dir=outputs/train/act_$DATASET_NAME --job_name=act_$DATASET_NAME \\
  --batch_size=8 --steps=100000 --wandb.enable=false
CMD
        ;;
    torque-off)
        robot_require_can
        python - "$FOLLOWER_PORT" "$FOLLOWER_ID" <<'PY'
import sys
from lerobot_teleoperator_so101_b601.map_joints import connect_limp_follower
connect_limp_follower(sys.argv[1], sys.argv[2], require_calibration=False).disconnect()
print("Follower torque disabled.")
PY
        ;;
    hf-login)
        hf auth login "$@"
        ;;
    app)
        exec python "$robot_setup_dir/app/server.py" "$@"
        ;;
    *)
        printf 'Unknown action: %s. Run: bash robot.sh help\n' "$robot_action" >&2
        exit 2
        ;;
esac
