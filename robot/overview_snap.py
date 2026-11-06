"""Save one overview-camera (OAK-1-AF) picture: python3 overview_snap.py OUT.jpg"""
import sys, time
import cv2, depthai as dai
with dai.Device(dai.DeviceInfo("19443010A155785A00")) as dev:
    p = dai.Pipeline(dev)
    q = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A).requestOutput((640, 480), type=dai.ImgFrame.Type.BGR888i, fps=5).createOutputQueue(2, False)
    p.start(); t = time.monotonic(); f = None
    while time.monotonic() - t < 4:  # let exposure settle, keep the last frame
        m = q.tryGet()
        if m is not None: f = m.getCvFrame()
        time.sleep(0.05)
    p.stop()
cv2.imwrite(sys.argv[1], f); print("saved", sys.argv[1])
