"""Record the overview OAK-1 at ~2 fps for a while, as independent evidence of a touch.

Usage: overview_grab.py SECONDS  -> touch/data/overview/<HHMMSS_ms>.jpg
"""

import sys
import time
from pathlib import Path

import cv2
import depthai as dai

OVERVIEW_ID = "19443010A155785A00"
OUT = Path(__file__).resolve().parent / "data" / "overview"


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 120
    OUT.mkdir(parents=True, exist_ok=True)
    with dai.Device(dai.DeviceInfo(OVERVIEW_ID)) as dev:
        pipeline = dai.Pipeline(dev)
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        q = cam.requestOutput((640, 480), type=dai.ImgFrame.Type.BGR888i, fps=5).createOutputQueue(2, False)
        pipeline.start()
        end, last = time.time() + seconds, 0.0
        while time.time() < end:
            frame = q.get().getCvFrame()
            if time.time() - last >= 0.5:
                last = time.time()
                name = time.strftime("%H%M%S") + f"_{int(last * 1000) % 1000:03d}.jpg"
                cv2.imwrite(str(OUT / name), frame)
        pipeline.stop()


if __name__ == "__main__":
    main()
