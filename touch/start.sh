#!/usr/bin/env bash
# Start the AprilTag touch agent in the background (follower stays limp until commanded).
# Usage: bash ~/robot-setup/touch/start.sh        Stop: echo quit > ~/robot-setup/touch/ctl
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if pgrep -f "$here/agent.py" >/dev/null; then
    echo "Touch agent is already running."
    exit 0
fi
setsid nohup "$here/../env/bin/python" "$here/agent.py" </dev/null >/dev/null 2>&1 &
sleep 8
tail -n 3 "$here/log.txt"
