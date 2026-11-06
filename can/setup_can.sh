#!/usr/bin/env bash
# One-time setup (run with sudo): USB CAN adapter (candleLight / gs_usb, 1d50:606f) for the B601 RobStride arm.
# JetPack 7's kernel ships without gs_usb, so this installs a module built from the v6.8.12 kernel source
# (gs_usb.c next to this script) for kernel 6.8.12-1021-tegra.
# After this, the adapter appears as "canb601", up at 1 Mbit/s, whenever it is plugged in (systemd-networkd).
# ("can0" on this Jetson is the Orin's built-in CAN controller, not the arm.)
set -euo pipefail
cd "$(dirname "$0")"
[ "$(id -u)" = 0 ] || { echo "run with sudo" >&2; exit 1; }
KVER=$(uname -r)
[ "$KVER" = "6.8.12-1021-tegra" ] || { echo "gs_usb.ko was built for 6.8.12-1021-tegra, this is $KVER: rebuild it (see README)" >&2; exit 1; }

install -D -m 644 gs_usb.ko "/lib/modules/$KVER/updates/gs_usb.ko"
depmod -a "$KVER"
echo gs_usb > /etc/modules-load.d/gs_usb.conf

# Name + queue length (.link, applied by udev) and bitrate + bring-up on every plug-in (.network, networkd).
rm -f /etc/udev/rules.d/80-b601-can.rules    # earlier version of this script (its RUN step failed)
mkdir -p /etc/systemd/network
cat > /etc/systemd/network/80-b601-can.link <<'LINK'
# B601 arm USB CAN adapter (candleLight / gs_usb)
[Match]
Driver=gs_usb

[Link]
Name=canb601
TransmitQueueLength=1000
LINK
cat > /etc/systemd/network/80-b601-can.network <<'NET'
[Match]
Name=canb601

[Link]
RequiredForOnline=no

[CAN]
BitRate=1M
# No RestartSec: gs_usb rejects automatic bus-off restart ("Device doesn't support restart from Bus Off")
# and that fails the whole configuration. After a bus-off: sudo networkctl reconfigure canb601 (or replug).
NET
udevadm control --reload
systemctl enable --now systemd-networkd >/dev/null 2>&1 || true
networkctl reload

modprobe gs_usb
# Re-apply to an adapter that is already plugged in.
udevadm trigger --action=add --subsystem-match=net
udevadm settle
for _ in $(seq 10); do
    [ $(( $(cat /sys/class/net/canb601/flags 2>/dev/null || echo 0) & 1 )) = 1 ] && break
    networkctl reconfigure canb601 2>/dev/null || true
    sleep 1
done
ip -details link show canb601 2>/dev/null | grep -E 'canb601|bitrate|state' || { echo "canb601 not found: is the CAN adapter plugged in? unplug/replug it and rerun" >&2; exit 1; }
if [ $(( $(cat /sys/class/net/canb601/flags) & 1 )) = 1 ]; then
    echo "OK: canb601 is UP at 1 Mbit/s"
else
    echo "canb601 is still DOWN; networkctl status canb601 says:" >&2
    networkctl status canb601 --no-pager 2>&1 | tail -15 >&2
    echo "direct attempt (shows the kernel's error):" >&2
    ip link set canb601 type can bitrate 1000000 && ip link set canb601 up
    exit 1
fi
