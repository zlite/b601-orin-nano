#!/usr/bin/env bash
# Unattended VLA trials, alternating the two tasks:
#   holder: plate on the pad  -> policy "put the 96-well plate in the holder"          success: plate in holder A
#   table:  plate in holder A -> policy "pick up the 96-well plate and put it down somewhere else on the table"
#                                                                                     success: plate on the pad, out of A
# Per trial: the scripted controller finds the plate (platelook) and, if the plate is not where the next task starts,
# moves it there (kpick home / kpick rand); then the controller is stopped and the policy server drives the arm
# (./jetson.sh rollout TASK SECONDS) with a CAN watchdog alongside; then the controller finds the plate again to score.
# Results: touch/trials.log (text) and touch/trials.jsonl. Policy server must be started with the model to test.
#   bash touch/trials.sh [N_PER_TASK=5] [SECONDS=240] [LABEL=v5]
cd "$(dirname "$0")/.." || exit 1
N=${1:-5}; SECS=${2:-240}; LABEL=${3:-v5}
LOG=touch/trials.log; OUT=touch/trials.jsonl
A() { sg docker -c "./jetson.sh agent $*" 2>&1; }
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
agent_on() {
    sg docker -c "./jetson.sh agent-start" >/dev/null 2>&1
    for _ in $(seq 60); do tail -n 3 touch/log.txt | grep -q "ready; send commands" && return 0; sleep 1; done
    say "controller did not start"; return 1
}
agent_off() { sg docker -c "./jetson.sh agent-stop" >/dev/null 2>&1; }
where_plate() {  # -> "holder|pad|lost x_mm y_mm": platelook, then classify against holder A (data/plate_ref.json)
    local out; out=$(A platelook)
    if ! echo "$out" | grep -q "PLATELOOK:"; then  # one retry (light, glare); a failed look leaves the controller halted
        A rest >/dev/null; A clear >/dev/null; out=$(A platelook)
    fi
    echo "$out" | grep -q "PLATELOOK:" || { echo "lost 0 0"; return; }
    python3 - <<'EOF'
import json, math
r = json.load(open("touch/data/plate_ref.json"))["T"]; p = json.load(open("touch/data/plate_pose.json"))["T"]
x, y = p[0][3], p[1][3]
d = math.hypot(x - r[0][3], y - r[1][3])
print(("holder" if d < 0.015 else "pad" if 0.236 <= x <= 0.532 and 0.292 <= y <= 0.45 else "elsewhere"), round(x * 1000), round(y * 1000))
EOF
}
can_watch() {  # during a rollout: CAN frames stop for 3 s while it runs -> reset the link (frees a stalled adapter)
    # stops once the rollout has parked / disconnected the arm (the idle bus after that is not a stall: 2026-11-04)
    local last=-1 same=0 n moved=0 rl
    while sg docker -c "docker ps -q -f ancestor=b601-policy -f status=running" | grep -q .; do
        rl=$(ls -t logs/rollout_*.log 2>/dev/null | grep -v _server | head -1)
        if [[ -n $rl && $rl -nt "$1" ]] && grep -qE "parked at the start pose|NOT parked|Follower disconnected|Rollout finished" "$rl"; then
            return 0
        fi
        n=$(cat /sys/class/net/canb601/statistics/tx_packets 2>/dev/null || echo 0)
        if [[ $n == "$last" ]]; then same=$((same + 1)); else (( last >= 0 )) && moved=1; same=0; last=$n; fi
        # only once frames have flowed (start-up: camera and connect take ~20 s with no traffic)
        if (( moved && same >= 3 )) && pgrep -f "b601_rollout.py run" >/dev/null; then
            say "  CAN STALL during the rollout (no frames for ${same} s): resetting the link"
            sg docker -c "docker run --rm --network host --privileged -v /usr/sbin/ip:/usr/local/bin/ip:ro \
                -v /lib/aarch64-linux-gnu/libbpf.so.1:/lib/aarch64-linux-gnu/libbpf.so.1:ro \
                -v /lib/aarch64-linux-gnu/libmnl.so.0:/lib/aarch64-linux-gnu/libmnl.so.0:ro \
                -v /lib/aarch64-linux-gnu/libelf.so.1:/lib/aarch64-linux-gnu/libelf.so.1:ro b601-policy \
                sh -c 'ip link set canb601 down; ip link set canb601 up type can bitrate 1000000'" >/dev/null 2>&1
            same=0
        fi
        sleep 1
    done
}
declare -A tries ok
tries[holder]=0; tries[table]=0; ok[holder]=0; ok[table]=0
# a restart of the same label keeps its earlier results (touch/trials.jsonl)
for t in holder table; do
    tries[$t]=$(grep -c "\"label\":\"$LABEL\",\"task\":\"$t\"" "$OUT" 2>/dev/null) || true
    ok[$t]=$(grep "\"label\":\"$LABEL\",\"task\":\"$t\"" "$OUT" 2>/dev/null | grep -c '"result":"success"') || true
done
say "trials start: $LABEL, $N per task, $SECS s each"
agent_on || exit 1
while (( tries[holder] < N || tries[table] < N )); do
    read -r loc x y < <(where_plate)
    if [[ $loc == lost || $loc == elsewhere ]]; then
        say "STOP: plate not found on the pad or in holder A ($loc $x $y): needs a person"; break
    fi
    # the task with fewer tries goes next; set the plate up for it with the scripted controller if needed
    if (( tries[table] >= N )); then task=holder
    elif (( tries[holder] >= N )); then task=table
    elif (( tries[holder] <= tries[table] )); then task=holder
    else task=table; fi
    if [[ $task == holder && $loc == holder ]]; then out=$(A kpick rand); fi
    # a pad start outside the demos' pad goals (x 322-451, y 350-396 mm) gets a fresh random place first: a set-down
    # at (427, 302) was pushed off the pad with open jaws (v6sq2 trial 11, 2026-11-07)
    if [[ $task == holder && $loc == pad ]] && (( ${x%.*} < 325 || ${x%.*} > 450 || ${y%.*} < 348 || ${y%.*} > 398 )); then
        say "  plate at ($x, $y) is outside the demos' pad starts: re-placing it"; out=$(A kpick rand); fi
    if [[ $task == table && $loc == pad ]]; then out=$(A kpick home); fi
    if [[ -n ${out:-} ]]; then
        if echo "$out" | grep -q "let it cool"; then say "  shoulder warm: cooling 5 min"; A rest >/dev/null; sleep 300; A clear >/dev/null; continue; fi
        # a holder-A wall catch falls back to a pad place, which also logs KPICK SUCCEEDED (v6 trial 2, 2026-11-06)
        if ! echo "$out" | grep -q "KPICK SUCCEEDED" || echo "$out" | grep -q "HOLDER FALLBACK"; then
            say "  setup pick for '$task' failed: $(echo "$out" | grep -E "HALT|FROZEN" | tail -1 | cut -c1-120)"
            setup_fails=$((${setup_fails:-0} + 1))
            if (( setup_fails >= 4 )); then say "STOP: $setup_fails setup picks failed in a row"; break; fi
            A rest >/dev/null; A clear >/dev/null; unset out; continue
        fi
        unset out
        read -r loc x y < <(where_plate)
        if [[ $task == holder && $loc != pad ]] || [[ $task == table && $loc != holder ]]; then
            say "  setup for '$task' left the plate at $loc ($x, $y): retrying"
            setup_fails=$((${setup_fails:-0} + 1))
            if (( setup_fails >= 4 )); then say "STOP: $setup_fails setup picks failed in a row"; break; fi
            A rest >/dev/null; A clear >/dev/null; continue
        fi
        setup_fails=0
    fi
    A rest >/dev/null; A record off >/dev/null
    start="$loc $x $y"
    agent_off
    tries[$task]=$((tries[$task] + 1)); t0=$(date +%s)
    say "trial $((tries[holder] + tries[table])): task '$task' (#${tries[$task]}), plate starts $start"
    mark=$(mktemp); can_watch "$mark" & cw=$!
    rlog=$(sg docker -c "./jetson.sh rollout $task $SECS" 2>&1 | grep -oE "saving this run to \S+" | head -1)
    kill $cw 2>/dev/null; rm -f "$mark"
    agent_on || break
    read -r loc x y < <(where_plate)
    held=""
    # the rollout's park set the plate down itself (the run ended holding it): not the policy's doing -> failure
    grep -q "run ended HOLDING" "${rlog#saving this run to }" 2>/dev/null && held=" (run ended holding the plate; the park set it down)"
    if [[ -z $held ]] && { [[ $task == holder && $loc == holder ]] || [[ $task == table && $loc == pad ]]; }; then
        res=success; ok[$task]=$((ok[$task] + 1))
    else
        res=failure
    fi
    say "  -> $res$held: plate ends $loc ($x, $y) mm, $(( $(date +%s) - t0 )) s; ${rlog#saving this run to }"
    say "  score: holder ${ok[holder]}/${tries[holder]}, table ${ok[table]}/${tries[table]}"
    printf '{"label":"%s","task":"%s","start":"%s","end":"%s %s %s","result":"%s","time":"%s","log":"%s"}\n' \
        "$LABEL" "$task" "$start" "$loc" "$x" "$y" "$res" "$(date -u '+%F %T')" "${rlog#saving this run to }" >> "$OUT"
done
say "trials done: $LABEL holder ${ok[holder]}/${tries[holder]}, table ${ok[table]}/${tries[table]}"
A rest >/dev/null; agent_off
