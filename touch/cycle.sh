#!/usr/bin/env bash
# Scripted demonstrations for the VLA, recorded to touch/data/lerobot/plate_jetson:
#   plate on the pad   -> `kpick home`  (into holder A, the black holder by AprilTag 1)  "put the 96-well plate in the holder"
#   plate in holder A  -> `kpick rand`  (to a random pad spot in reach)    "pick up the 96-well plate and put it down
#                                                                           somewhere else on the table"
# Runs unattended: a failure rests and retries; stops at N successes of each kind (default 20), MAX_FAILS (6)
# failures in a row, darkness, 6 h, a silent controller or a CAN fault; then parks, recording off.
# A shoulder over 55 C waits 5 min. Log: touch/cycle.log (./jetson.sh stats shows the progress).
#   bash touch/cycle.sh [N]
cd "$(dirname "$0")/.." || exit 1
N=${1:-20}
MAX_FAILS=${MAX_FAILS:-6}  # unattended: rest and retry after a failure; stop only after this many in a row
LOG=touch/cycle.log
A() { sg docker -c "./jetson.sh agent $*" 2>&1; }
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
bright() { python3 -c "from PIL import Image,ImageStat;import sys;print(int(ImageStat.Stat(Image.open(sys.argv[1]).convert('L')).mean[0]))" "$1" 2>/dev/null || echo 999; }
in_holder() {  # plate within 30 mm of the taught holder-A pose (data/plate_ref.json) per data/plate_pose.json
    python3 - <<'EOF'
import json, math
r = json.load(open("touch/data/plate_ref.json"))["T"]; p = json.load(open("touch/data/plate_pose.json"))["T"]
print("yes" if math.hypot(p[0][3] - r[0][3], p[1][3] - r[1][3]) < 0.03 else "no")
EOF
}
# a goal file from an earlier run of the same cycle keeps its start, so its successes count (CYCLE_NEW=1: fresh)
if [[ -n ${CYCLE_NEW:-} ]] || ! grep -q '"holder"' touch/collect_goal.json 2>/dev/null; then
    echo "{\"holder\": $N, \"pad\": $N, \"since\": \"$(date -u '+%F %T')\"}" > touch/collect_goal.json
fi
counts() {  # successes since the goal's start: "holder pad"
    python3 - <<'EOF2'
import json
g = json.load(open("touch/collect_goal.json")); e = json.load(open("touch/data/lerobot/plate_jetson/episode_outcomes.json"))
ok = [x for x in e if x["time"] >= g["since"] and x["success"]]
print(sum(x["task"] == "put the 96-well plate in the holder" for x in ok),
      sum(x["task"].startswith("pick up the 96-well plate and put it down") for x in ok))
EOF2
}
sg docker -c "./jetson.sh agent-start" 2>&1 | tail -1; sleep 15
A rest >/dev/null; A clear >/dev/null
A record on plate_jetson | grep -E "recording episodes" | cut -c1-200
say "cycle start: goal $N into the holder + $N back to the pad"
end=$(( $(date +%s) + 6 * 3600 )); fails=0; read -r nh np < <(counts); why="6 h cap"
say "  so far: $nh/$N into the holder, $np/$N back to the pad"
while (( $(date +%s) < end )); do
    if [[ $(in_holder) == yes ]]; then kind=pad; cmd="kpick rand"; else kind=holder; cmd="kpick home"; fi
    # run the command, watching that the controller keeps logging: a hang (CAN adapter stall, 2026-11-04 14:04)
    # left the arm holding and heating for 7 min unnoticed. Silent > SILENT_S -> stop and alert (the controller
    # stays up: killing it would not release the motors, they keep holding until power-off or a disable frame).
    tmp=$(mktemp); A $cmd > "$tmp" & apid=$!
    while kill -0 $apid 2>/dev/null; do
        sleep 5
        idle=$(( $(date +%s) - $(stat -c %Y touch/log.txt) ))
        if (( idle > ${SILENT_S:-120} )); then
            say "STOP: the controller has logged nothing for ${idle} s during '$cmd': it may be hung with the motors"
            say "  holding. Support the arm and switch its power off if the motors are warm; see touch/log.txt"
            exit 3
        fi
    done
    out=$(cat "$tmp"); rm -f "$tmp"
    if echo "$out" | grep -qE "CAN (STALL|DEAD)|CAN link stalled"; then
        say "STOP: CAN link trouble: $(echo "$out" | grep -E "CAN (STALL|DEAD)" | tail -1 | cut -c1-150)"
        A rest >/dev/null; A record off | tail -1
        exit 4
    fi
    snap=logs/cycle_$(date +%H%M%S).jpg
    sg docker -c "docker run --rm --network host --privileged -v /dev/bus/usb:/dev/bus/usb -v $PWD/robot:/opt/b601/robot -v $PWD/logs:/opt/b601/logs b601-policy python3 /opt/b601/robot/overview_snap.py /opt/b601/$snap" >/dev/null 2>&1
    ob=$(bright "$snap")
    res=$(echo "$out" | grep -E "KPICK SUCCEEDED|HALT:|FROZEN:|let it cool" | tail -1 | cut -c1-170)
    say "$cmd -> ${res:-no result} (overview brightness $ob)"
    if (( ob < 95 )); then why="too dark (overview $ob)"; break; fi
    if echo "$out" | grep -q "HOLDER FALLBACK"; then
        # holder A caught the plate on every try; the controller set it on the pad instead (not a success)
        fails=$((fails + 1)); say "  holder fallback: plate on the pad; failures in a row: $fails"
        if (( fails >= MAX_FAILS )); then why="$MAX_FAILS failures in a row"; break; fi
        # the retries + the carry back heat the shoulder (walls came in its hot stretches): rest limp 3 min
        A rest >/dev/null; A clear >/dev/null; say "  resting 3 min"; sleep 180; continue
    fi
    if echo "$out" | grep -q "KPICK SUCCEEDED"; then
        fails=0
        if [[ $kind == holder ]]; then nh=$((nh + 1)); else np=$((np + 1)); fi
        say "  successes: $nh/$N into the holder, $np/$N back to the pad"
        if (( nh >= N && np >= N )); then why="goal reached"; break; fi
        continue
    fi
    if echo "$out" | grep -q "holding it"; then
        # the plate is still in the jaws (on a holder wall, or a touch while carrying): `rest` would carry it back
        # and drop it when the motors go limp (2026-11-04 20:34). Leave the arm holding it for a person.
        say "STOP: the arm is HOLDING THE PLATE: $(echo "$out" | grep -oE "FROZEN: .*" | tail -1 | cut -c1-150)"
        say "  the controller stays on with the plate held; record off. Hold the plate (rest opens the jaws only at the end, motors off), then: ./jetson.sh agent rest; ./jetson.sh agent-stop"
        A record off | tail -1
        exit 2
    fi
    if echo "$out" | grep -q "let it cool"; then say "  shoulder warm: cooling 10 min"; A rest >/dev/null; sleep 600; A clear >/dev/null; continue; fi
    fails=$((fails + 1))
    if (( fails >= MAX_FAILS )); then why="$MAX_FAILS failures in a row"; break; fi
    say "  failures in a row: $fails/$MAX_FAILS; resting and retrying"
    A rest >/dev/null; A clear >/dev/null
    (( fails >= 2 )) && { say "  pausing 2 min before the retry"; sleep 120; }
done
say "STOP: $why ($nh into the holder, $np back to the pad)"
A rest | tail -1; A record off | tail -1
sg docker -c "./jetson.sh agent-stop" 2>&1 | tail -1
say "done"
