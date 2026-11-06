"""List connected OAK devices and their sensors; optionally save one test frame per sensor.

Usage: python -m lerobot_camera_oak.find [--snapshot DIR] [--width 640 --height 480 --fps 30]
"""

import argparse
from pathlib import Path

import cv2
import depthai as dai

from .config_oak import OAKCameraConfig
from .oak import OAKCamera


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, help="Directory for one test frame per sensor.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--encoding", default="raw", choices=["raw", "mjpeg"])
    args = parser.parse_args()

    infos = dai.Device.getAllAvailableDevices()
    if not infos:
        print("No OAK devices found. Check cables and the udev rule (see robot-setup/root-setup.sh).")
        return

    devices = []
    for info in infos:
        with dai.Device(info) as device:
            sockets = [s.name for s in device.getConnectedCameras()]
            devices.append((info.getDeviceId(), sockets))
            print(
                f"{info.getDeviceId()}  product={device.getProductName()!r}  usb={device.getUsbSpeed().name}  "
                f"sensors={device.getCameraSensorNames()}"
            )

    if args.snapshot is None:
        return
    args.snapshot.mkdir(parents=True, exist_ok=True)
    for device_id, sockets in devices:
        for socket in sockets:
            config = OAKCameraConfig(
                device_id=device_id,
                socket=socket,
                width=args.width,
                height=args.height,
                fps=args.fps,
                encoding=args.encoding,
            )
            with OAKCamera(config) as camera:
                frame = camera.read()
                path = args.snapshot / f"{device_id}_{socket}.png"
                cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                print(f"Saved {path} ({camera.usb_speed})")


if __name__ == "__main__":
    main()
