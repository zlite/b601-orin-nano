#!/usr/bin/env bash
# B601 plate policy server on a Jetson Orin. Run from the unpacked folder.
#   ./jetson.sh check    check JetPack, Docker and the NVIDIA runtime (changes nothing)
#   ./jetson.sh build    build the image b601-policy (needs internet; 15-40 min the first time)
#   ./jetson.sh bench    time the model on this GPU
#   ./jetson.sh start    start the policy server in the background (port 7447 on the LAN)
#   ./jetson.sh logs     follow the server log (Ctrl+C stops following, not the server)
#   ./jetson.sh stop     stop the server
#   ./jetson.sh cameras            test both OAK cameras: USB link speed, frame rate, snapshots in logs/
#   ./jetson.sh arm-check          read the arm with torque OFF and save a wrist frame to logs/ (moves nothing)
#   ./jetson.sh rollout [table|holder] [seconds]   run the policy on the arm, default 180 s (server must be started)
#   ./jetson.sh agent-start        start the scripted controller (touch/agent.py, arm stays limp until commanded)
#   ./jetson.sh agent CMD [ARGS]   send it one command (e.g. rest, platefind, kpick rand) and print its log
#   ./jetson.sh agent-log | agent-stop
#   ./jetson.sh stats [SECONDS]     collection + training summary in the terminal, refreshed (default 30 s; Ctrl+C quits)
#   ./jetson.sh train-progress [JOB_ID]   follow a SmolVLA training job on HF Jobs (default: the last one launched)
# The arm (CAN adapter canb601, wrist OAK) is on this Jetson: one-time `sudo can/setup_can.sh` first.
set -euo pipefail
cd "$(dirname "$0")"
IMAGE=b601-policy
NAME=b601-policy-server
# --memory-swap = --memory: no swap for the policy container. Idle for hours, ~700 MB of the server was swapped out
# and the first calls of the next rollout came back too late ("observation is too old", 2026-11-03). 6g is a
# ceiling well above its ~3.2 GB; the rest of the system keeps using /swapfile.
RUN=(docker run --runtime nvidia --network host --ipc host --memory=6g --memory-swap=6g
     -v "$PWD/${CHECKPOINT:-checkpoint}:/opt/b601/checkpoint:ro"  # CHECKPOINT=checkpoint_v4 ./jetson.sh start
     -v b601-hf-cache:/root/.cache/huggingface)
# Arm container: host network (sees canb601 and the server on 127.0.0.1), USB for the OAK (depthai reboots it,
# so it re-enumerates: needs the whole bus + privileged), calibration and live robot/ code mounted.
# No --runtime nvidia: the arm side needs no GPU, and on the 8 GB Orin Nano a second CUDA context costs memory.
ARM_RUN=(docker run --rm $([ -t 0 ] && echo -it || true) --network host --ipc host --privileged  # -it only from a terminal (trials run unattended)
     -v /dev/bus/usb:/dev/bus/usb
     -v "$PWD/robot:/opt/b601/robot"
     -v "$PWD/calibration:/opt/b601/calibration:ro"
     -v "$PWD/logs:/opt/b601/logs"
     -e HF_LEROBOT_CALIBRATION=/opt/b601/calibration
     -e B601_CAN=canb601)
AGENT=b601-agent
# Scripted pick/place controller from the robot PC (touch/agent.py + kin/ + map_joints), its data in touch/data.
AGENT_RUN=(docker run -d --name "$AGENT" --network host --ipc host --privileged
     -v /dev/bus/usb:/dev/bus/usb
     -v "$PWD/touch:/opt/b601/touch"
     -v "$PWD/calibration:/opt/b601/calibration:ro"
     -e HF_LEROBOT_CALIBRATION=/opt/b601/calibration
     -e B601_CAN=canb601
     -e PYTHONPATH=/opt/b601/touch/lerobot_teleoperator_so101_b601
     # the host's ip (+ its libraries): the controller's CAN watchdog resets a stalled adapter (down/up)
     -v /usr/sbin/ip:/usr/local/bin/ip:ro
     -v /lib/aarch64-linux-gnu/libbpf.so.1:/lib/aarch64-linux-gnu/libbpf.so.1:ro
     -v /lib/aarch64-linux-gnu/libmnl.so.0:/lib/aarch64-linux-gnu/libmnl.so.0:ro
     -v /lib/aarch64-linux-gnu/libelf.so.1:/lib/aarch64-linux-gnu/libelf.so.1:ro)
agent_up() { docker ps -q -f name="^$AGENT\$" | grep -q .; }
can_ok() { [[ $(( $(cat /sys/class/net/canb601/flags 2>/dev/null || echo 0) & 1 )) == 1 ]] || { echo "canb601 is down or missing: run 'sudo can/setup_can.sh' (once), check the adapter is plugged in" >&2; exit 1; }; }

docker_ok() { docker info >/dev/null 2>&1 || { echo "docker needs sudo here: run 'sudo usermod -aG docker \$USER', log out and in, or prefix with sudo" >&2; exit 1; }; }

case "${1:-}" in
check)
    echo "== JetPack / L4T"; cat /etc/nv_tegra_release 2>/dev/null || echo "not found (is this a Jetson?)"
    grep -q "R39" /etc/nv_tegra_release 2>/dev/null && echo "OK: L4T R39 (JetPack 7)" \
        || echo "WARNING: this image is built for L4T R39 (JetPack 7, CUDA 13.2); other versions need a different base image"
    nvidia-smi 2>/dev/null | grep -o "CUDA Version: [0-9.]*" \
        | awk '{split($3,v,"."); if (v[1]<13 || (v[1]==13 && v[2]<2)) print "WARNING: driver supports " $0 "; the image needs CUDA 13.2"; else print "OK: driver " $0}'
    echo "== model"; tr -d '\0' < /proc/device-tree/model 2>/dev/null; echo
    echo "== memory"; free -h | head -2
    echo "== disk (needs ~15 GB free)"; df -h "$PWD" | tail -1
    echo "== power mode"; (nvpmodel -q 2>/dev/null || sudo -n nvpmodel -q 2>/dev/null || echo "nvpmodel not readable") | head -3
    echo "== docker"; docker --version; docker_ok
    docker info 2>/dev/null | grep -i -E "runtimes|default runtime"
    docker info 2>/dev/null | grep -qi nvidia && echo "OK: NVIDIA runtime present" \
        || echo "WARNING: no nvidia runtime: sudo apt install nvidia-container-toolkit && sudo systemctl restart docker"
    echo "== network (give this address to the robot PC)"; hostname -I
    ;;
build)
    docker_ok
    docker build --network=host -t "$IMAGE" .
    ;;
bench)
    docker_ok
    "${RUN[@]}" --rm "$IMAGE" python3 /opt/b601/bench.py "${2:-10}"
    ;;
start)
    docker_ok
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    "${RUN[@]}" -d --name "$NAME" "$IMAGE"
    echo "started; waiting for 'Policy server ready' (model load + warmup, ~1-2 min)..."
    for _ in $(seq 120); do
        docker logs "$NAME" 2>&1 | grep -q "Policy server ready" && { echo "READY on $(hostname -I | awk '{print $1}'):7447"; exit 0; }
        docker ps -q -f name="$NAME" | grep -q . || { docker logs --tail 30 "$NAME"; echo "server exited" >&2; exit 1; }
        sleep 2
    done
    echo "not ready after 4 min; see ./jetson.sh logs" >&2; exit 1
    ;;
cameras)
    docker_ok; mkdir -p logs
    "${ARM_RUN[@]}" "$IMAGE" python3 /opt/b601/robot/test_cameras.py
    ;;
arm-check)
    docker_ok; can_ok; mkdir -p logs
    "${ARM_RUN[@]}" "$IMAGE" python3 /opt/b601/robot/b601_rollout.py check
    ;;
rollout)
    docker_ok; can_ok
    agent_up && { echo "the scripted agent has the arm and camera: ./jetson.sh agent-stop first" >&2; exit 1; }
    docker ps -q -f name="$NAME" | grep -q . || { echo "policy server not running: ./jetson.sh start first" >&2; exit 1; }
    mkdir -p logs; chmod a+rx logs 2>/dev/null || true
    stamp=$(date +%Y%m%d_%H%M%S); since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    log="logs/rollout_${stamp}.log"
    echo "saving this run to $log (+ ${log%.log}_server.log)"
    # tee -i: keeps writing through Ctrl+C, so the park/shutdown lines are saved too
    rc=0
    "${ARM_RUN[@]}" "$IMAGE" python3 /opt/b601/robot/b601_rollout.py run --task "${2:-table}" --duration "${3:-180}" "${@:4}" \
        2>&1 | tee -i "$log" || rc=$?
    docker logs --since "$since" "$NAME" > "${log%.log}_server.log" 2>&1 || true
    chmod a+r "$log" "${log%.log}_server.log" 2>/dev/null || true
    exit "$rc"
    ;;
agent-start)
    docker_ok; can_ok
    agent_up && { echo "agent already running"; exit 0; }
    docker rm -f "$AGENT" >/dev/null 2>&1 || true
    [ -e touch/log.txt ] || touch touch/log.txt; n=$(($(wc -l < touch/log.txt) + 1))
    "${AGENT_RUN[@]}" "$IMAGE" python3 /opt/b601/touch/agent.py >/dev/null
    echo "starting the agent (camera + CAN, arm limp)..."
    for _ in $(seq 60); do
        tail -n +"$n" touch/log.txt | grep -q "ready; send commands" && { tail -n +"$n" touch/log.txt; echo "READY: ./jetson.sh agent CMD"; exit 0; }
        agent_up || { tail -n +"$n" touch/log.txt; tail -30 touch/agent.out 2>/dev/null; echo "agent exited" >&2; exit 1; }
        sleep 1
    done
    tail -n +"$n" touch/log.txt; echo "not ready after 60 s; see ./jetson.sh agent-log" >&2; exit 1
    ;;
agent)
    docker_ok; agent_up || { echo "agent not running: ./jetson.sh agent-start" >&2; exit 1; }
    [[ $# -ge 2 ]] || { echo "usage: ./jetson.sh agent CMD [ARGS]" >&2; exit 1; }
    shift; cmd="$*"; n=$(($(wc -l < touch/log.txt) + 1))
    docker exec "$AGENT" sh -c 'printf "%s\n" "$1" > /opt/b601/touch/ctl' _ "$cmd"
    # print the command's log lines as they come, until "< done <first word>" (Ctrl+C stops watching only;
    # the command keeps running; the agent reads the next command only when this one ends, so the emergency
    # stop is the arm's power switch)
    while :; do
        m=$(wc -l < touch/log.txt)
        if (( m >= n )); then
            new=$(sed -n "${n},${m}p" touch/log.txt); printf '%s\n' "$new"; n=$((m + 1))
            grep -q "< done ${cmd%% *}\$" <<< "$new" && break
        fi
        agent_up || { echo "agent exited; see touch/agent.out" >&2; exit 1; }
        sleep 0.5
    done
    ;;
agent-log)
    tail -n 40 -f touch/log.txt
    ;;
agent-stop)
    docker_ok
    if agent_up; then
        n=$(($(wc -l < touch/log.txt) + 1))
        docker exec "$AGENT" sh -c 'echo quit > /opt/b601/touch/ctl'
        for _ in $(seq 60); do agent_up || break; sleep 1; done
        tail -n +"$n" touch/log.txt
    fi
    docker rm -f "$AGENT" >/dev/null 2>&1 || true
    echo "agent stopped"
    ;;
logs)
    docker logs -f "$NAME"
    ;;
stop)
    docker rm -f "$NAME"
    ;;
stats)
    # plain text, no browser: touch/stats.py reads only files; the training rows come from HF Jobs about once a
    # minute while the last launched job runs (a short container), then never again for that job
    every=${2:-30}; n=0
    while true; do
        if (( n % (60 / every > 0 ? 60 / every : 1) == 0 )) && [[ -f training/last_job.json ]]; then
            jid=$(python3 -c "import json;print(json.load(open('training/last_job.json'))['job_id'])")
            if [[ ! -f "training/.done_$jid" ]] && docker info >/dev/null 2>&1; then
                docker run --rm -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -v "$PWD:/w" "$IMAGE" \
                    python3 /w/training/progress.py "$jid" 2>/dev/null | grep -E "^job |^ +step|^ *[0-9]+/|^summary|^  INFO|^[0-9:]+ job " \
                    > training/progress_cache.txt.new && mv training/progress_cache.txt.new training/progress_cache.txt
                grep -qE "^summary: (COMPLETED|ERROR|CANCEL)" training/progress_cache.txt && touch "training/.done_$jid"
            fi
        fi
        out=$(python3 touch/stats.py)
        clear; echo "$out"; echo; echo "(refreshes every ${every} s; Ctrl+C quits)"
        n=$((n + 1)); sleep "$every"
    done
    ;;
train-progress)
    docker_ok
    docker run --rm -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -v "$PWD:/w" "$IMAGE" \
        python3 /w/training/progress.py ${2:-} --watch 60 2>&1 \
        | grep -vE "NVIDIA|Copyright|governed|agreements|Container image|Various files|All rights|SHMEM|docker run|CUDA|driver|_pytree|PyTorch Version|^=+$|^\s*$"
    ;;
*)
    sed -n 2,18p "$0"; exit 1
    ;;
esac
