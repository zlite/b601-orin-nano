"""One-shot perception: wrist color + aligned stereo depth, intrinsics, AprilTag pose.

Writes touch/data/{wrist.png, wrist_depth.npy, intrinsics.json} and prints the tag pose.
The cameras must be free (stop the B601 Studio preview first).
"""

import json
import sys
import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

WRIST_ID = "19443010A13F095A00"
OVERVIEW_ID = "19443010A155785A00"
W, H = 640, 400
TAG_SIZE_M = 0.060
OUT = Path(__file__).resolve().parent / "data"


def open_device(device_id: str, attempts: int = 20) -> dai.Device:
    """OAKs re-enumerate for a few seconds after another process closes them."""
    for i in range(attempts):
        try:
            return dai.Device(dai.DeviceInfo(device_id))
        except RuntimeError:
            if i == attempts - 1:
                raise
            time.sleep(1.0)


def tag_pose(gray, K, dist, size=TAG_SIZE_M):
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
                                  cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None:
        return None
    c = corners[0].reshape(4, 2)
    h = size / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, c, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return {"id": int(ids[0][0]), "corners": c.tolist(), "rvec": rvec.ravel().tolist(), "tvec": tvec.ravel().tolist()}


def main():
    OUT.mkdir(exist_ok=True)
    intr = {}
    with open_device(OVERVIEW_ID) as dev:
        cal = dev.readCalibration()
        intr["overview"] = {"K": cal.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 640, 480),
                            "dist": cal.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A), "size": [640, 480]}
    time.sleep(1.0)

    dev = open_device(WRIST_ID)
    cal = dev.readCalibration()
    intr["wrist"] = {"K": cal.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, W, H),
                     "dist": cal.getDistortionCoefficients(dai.CameraBoardSocket.CAM_B), "size": [W, H],
                     "baseline_cm": cal.getBaselineDistance()}
    pipeline = dai.Pipeline(dev)
    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    color_q = left.requestOutput((W, H), type=dai.ImgFrame.Type.BGR888i, fps=10).createOutputQueue(4, False)
    stereo = pipeline.create(dai.node.StereoDepth).build(
        left.requestOutput((W, H), fps=10), right.requestOutput((W, H), fps=10))
    stereo.setExtendedDisparity(True)
    stereo.setLeftRightCheck(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_B)
    depth_q = stereo.depth.createOutputQueue(4, False)
    pipeline.start()
    try:
        for _ in range(15):  # let exposure settle
            color = color_q.get().getCvFrame()
            depth = depth_q.get().getFrame()
    finally:
        pipeline.stop()
        dev.close()

    cv2.imwrite(str(OUT / "wrist.png"), color)
    np.save(OUT / "wrist_depth.npy", depth)
    (OUT / "intrinsics.json").write_text(json.dumps(intr, indent=2))
    K = np.array(intr["wrist"]["K"]); dist = np.array(intr["wrist"]["dist"])
    print("wrist K:", np.round(K, 1).tolist(), "baseline cm:", intr["wrist"]["baseline_cm"])
    print("depth shape", depth.shape, "valid %", round(100 * (depth > 0).mean(), 1))
    pose = tag_pose(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), K, dist)
    print("tag:", json.dumps(pose))
    if pose:
        cx, cy = np.mean(pose["corners"], axis=0).astype(int)
        patch = depth[max(0, cy - 6):cy + 6, max(0, cx - 6):cx + 6]
        valid = patch[patch > 0]
        print(f"tag centre px ({cx},{cy}); PnP z = {pose['tvec'][2] * 1000:.0f} mm; "
              f"stereo depth = {np.median(valid) if valid.size else float('nan'):.0f} mm ({valid.size} px)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
