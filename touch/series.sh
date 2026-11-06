#!/usr/bin/env bash
# Run N `kpick rand` cycles through the agent; wait out shoulder cooling, stop at any other halt.
# Usage: bash touch/series.sh N
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
log="$here/log.txt"
send() { echo "$1" > "$here/ctl"; }
wait_done() {  # wait_done START_LINE CMD -> prints the new log lines once "< done CMD" appears
    until tail -n +"$1" "$log" | grep -q "< done $2"; do sleep 3; done
    tail -n +"$1" "$log"
}
for i in $(seq "${1:-10}"); do
    n=$(($(wc -l < "$log") + 1))
    send "kpick rand"
    out=$(wait_done "$n" kpick)
    if echo "$out" | grep -q "KPICK SUCCEEDED"; then
        echo "$(date +%T) cycle $i: $(echo "$out" | grep -E 'kpick: goal' | cut -c10-) -> $(echo "$out" | grep 'KPICK SUCCEEDED' | cut -c10-)"
        continue
    fi
    if echo "$out" | grep -q "let it cool"; then
        echo "$(date +%T) cycle $i: shoulder warm, cooling 5 min"
        n=$(($(wc -l < "$log") + 1)); send rest; wait_done "$n" rest >/dev/null
        sleep 300
        continue
    fi
    echo "$(date +%T) cycle $i STOPPED:"; echo "$out" | grep -E "FROZEN|halt|Abort|kpick: goal" | cut -c1-250
    exit 1
done
echo "$(date +%T) series done"
