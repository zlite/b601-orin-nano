#!/usr/bin/env bash
# More pad <-> holder demos with the scripted cycle, then train the next SmolVLA on HF Jobs.
#   bash touch/collect_then_train.sh [N=20] [POLICY_REPO=chr1sa/smolvla_plate_jetson_v6]
# Trains only if at least MIN_NEW (10) new holder demos succeeded. Log: touch/cycle.log + touch/train_v6.log
cd "$(dirname "$0")/.." || exit 1
N=${1:-20}; REPO=${2:-chr1sa/smolvla_plate_jetson_v6}; MIN_NEW=${MIN_NEW:-10}
say() { echo "$(date '+%F %T') $*" | tee -a touch/cycle.log; }
sg docker -c "./jetson.sh stop" >/dev/null 2>&1  # the policy server: frees ~2 GB for the controller
CYCLE_NEW=${CYCLE_NEW-1} bash touch/cycle.sh "$N"; rc=$?  # CYCLE_NEW= (empty): keep the goal file's start
nh=$(python3 - <<'P'
import json
g = json.load(open("touch/collect_goal.json")); e = json.load(open("touch/data/lerobot/plate_jetson/episode_outcomes.json"))
print(sum(x["time"] >= g["since"] and x["success"] and x["task"] == "put the 96-well plate in the holder" for x in e))
P
)
say "collection ended (cycle exit $rc): $nh new holder demos"
if (( nh < MIN_NEW )); then say "NOT training: fewer than $MIN_NEW new holder demos"; exit 1; fi
say "training $REPO"
sg docker -c "docker run --rm -e POLICY_REPO=$REPO -e FLAVOR=${FLAVOR:-a100-large} -e STEPS=${STEPS:-10000} \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface -v $PWD:/w b601-policy python3 /w/training/launch.py" 2>&1 \
    | grep -vE "NVIDIA|CUDA|PyTorch|Copyright|license|SHMEM|docker run|agreement|^$|=====|driver|Terms|Various|reserved|insufficient|\[\[" \
    | tee touch/train_v6.log | tail -5
