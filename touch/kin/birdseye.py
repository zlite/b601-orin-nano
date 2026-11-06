"""Reproject a wrist-camera image onto a horizontal plane in base_link (a top-down map, 1 px = res_m).

Used to find the plate on the table without a tag: with the camera pose known from the calibrated
kinematics, the plate top (a known height) becomes an undistorted, metric, top-down rectangle.
"""
import cv2
import numpy as np


def plane_map(frame, K, dist, T_cam, z, x_range, y_range, res_m=0.001):
    """Top-down image of the plane z (base frame) over x_range x y_range (metres).
    Output row r, col c <-> base point (x_range[0] + c*res, y_range[1] - r*res): x right, y up.
    Pixels the camera does not see (or behind it) are black. Returns (image, to_base) where
    to_base(c, r) maps output pixels to base (x, y)."""
    xs = np.arange(x_range[0], x_range[1], res_m)
    ys = np.arange(y_range[1], y_range[0], -res_m)
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X, Y, np.full_like(X, z)], -1).reshape(-1, 3)
    Tinv = np.linalg.inv(T_cam)
    Pc = P @ Tinv[:3, :3].T + Tinv[:3, 3]
    front = Pc[:, 2] > 0.02
    uv, _ = cv2.projectPoints(Pc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist)
    uv = uv.reshape(-1, 2)
    uv[~front] = -1
    mapx = uv[:, 0].reshape(X.shape).astype(np.float32)
    mapy = uv[:, 1].reshape(X.shape).astype(np.float32)
    img = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def to_base(c, r):
        return x_range[0] + c * res_m, y_range[1] - r * res_m
    return img, to_base
