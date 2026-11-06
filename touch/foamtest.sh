#!/usr/bin/env bash
# Gripper contact-sensing test: open the jaws, give you 20 s to hold foam between the pads,
# then close in 1-degree steps until contact is sensed (force capped at 0.8 Nm).
# Usage: bash ~/robot-setup/touch/foamtest.sh
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
log="$here/log.txt"

send() {  # send one command and wait for it to finish
    local n; n=$(wc -l < "$log")
    echo "$1" > "$here/ctl"
    for _ in $(seq 1 120); do
        tail -n +$((n + 1)) "$log" | grep -q "< done" && break
        sleep 1
    done
    tail -n +$((n + 1)) "$log" | grep -v "^.*> \|< done"
}

echo "Opening the jaws..."
send "goto gripper=-324" > /dev/null
for s in $(seq 20 -1 1); do
    printf "\rHold the foam between the dark pads (fingers clear): closing in %2d s " "$s"
    sleep 1
done
echo
echo "Closing with contact sensing..."
send "gripclose -3"
