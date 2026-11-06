"""Test the two OAK cameras: USB link speed, sensors, achieved frame rate, one snapshot each.

Run with: sudo ./jetson.sh cameras   (inside the b601-policy image; nothing else may be using the cameras)
"""

import sys
import time
from pathlib import Path

import cv2
import depthai as dai

CAMERAS = [  # (label, MXID, socket, width, height) as in b601_transfer/reference/robot.env
    ("wrist (OAK-D-SR)", "19443010A13F095A00", "CAM_B", 640, 400),
    ("overview (OAK-1-AF)", "19443010A155785A00", "CAM_A", 640, 480),
]
FPS = 15          # what the rollout requests for the wrist camera
TEST_S = 3.0
OUT = Path("/opt/b601/logs")
SPEED_NOTE = {"SUPER": "USB 3 (5 Gbit/s)", "SUPER_PLUS": "USB 3 (10 Gbit/s)", "HIGH": "USB 2 (480 Mbit/s)",
              "FULL": "USB 1 (12 Mbit/s) - bad cable/port", "LOW": "USB 1 (1.5 Mbit/s) - bad cable/port"}


def test(label: str, mxid: str, socket_name: str, w: int, h: int) -> bool:
    print(f"\n== {label}  {mxid}")
    device = None
    try:
        device = dai.Device(dai.DeviceInfo(mxid))
    except RuntimeError as e:
        print(f"   FAIL: cannot open ({e})")
        return False
    try:
        speed = device.getUsbSpeed().name
        sockets = [s.name for s in device.getConnectedCameras()]
        print(f"   model {device.getDeviceName()}, USB link {speed} = {SPEED_NOTE.get(speed, speed)}")
        print(f"   sensors: {', '.join(sockets)}")
        if socket_name not in sockets:
            print(f"   FAIL: socket {socket_name} not present")
            return False
        pipeline = dai.Pipeline(device)
        cam = pipeline.create(dai.node.Camera).build(getattr(dai.CameraBoardSocket, socket_name))
        q = cam.requestOutput((w, h), type=dai.ImgFrame.Type.BGR888i, fps=FPS).createOutputQueue(4, False)
        pipeline.start()
        frame, n, t0 = None, 0, None
        deadline = time.monotonic() + TEST_S + 5.0  # + time for the sensor to start
        while time.monotonic() < deadline:
            msg = q.tryGet()
            if msg is None:
                time.sleep(0.002)
                continue
            frame = msg.getCvFrame()
            if t0 is None:
                t0 = time.monotonic()  # count from the first frame
            n += 1
            if time.monotonic() - t0 >= TEST_S:
                break
        pipeline.stop()
        if frame is None:
            print("   FAIL: no frames")
            return False
        fps = (n - 1) / max(time.monotonic() - t0, 1e-3)
        mb_s = fps * w * h * 3 / 1e6
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / f"camera_{label.split()[0]}_{socket_name}.jpg"
        cv2.imwrite(str(path), frame)
        ok = fps >= 0.9 * FPS
        print(f"   {socket_name} {w}x{h} raw: {fps:.1f} fps of {FPS} requested ({mb_s:.1f} MB/s)"
              f" -> {'OK' if ok else 'LOW FRAME RATE'}")
        print(f"   snapshot: logs/{path.name}")
        return ok
    except Exception as e:
        print(f"   FAIL: {e}")
        return False
    finally:
        device.close()


def main() -> int:
    found = {d.getDeviceId(): d.state.name for d in dai.Device.getAllAvailableDevices()}
    print("OAK devices visible: " + (", ".join(f"{k} ({v})" for k, v in found.items()) or "none"))
    results = [test(*c) for c in CAMERAS]
    print("\nAll cameras OK" if all(results) else "\nSome cameras FAILED (see above)")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
