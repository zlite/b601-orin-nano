"""Find an empty black plate holder (open frame) in a top-down plane map (birdseye.plane_map at the
table height): the wood seen through the frame is a plate-sized bright hole in a dark ring."""
import math

import cv2
import numpy as np

INNER_M = (0.125, 0.085)  # inner opening as seen from above (measured 2026-10-19)
SIZE_TOL_M = 0.012
DARK_FRAC = 0.6


def find(plane_img, res_m, prior, window_m=0.10):
    """prior: rough centre (col, row) px. -> dict(c=(col, row) px, angle_deg (long side vs map x, CCW,
    in [-90, 90)), size_m=(long, short), fill) or None."""
    gray = cv2.cvtColor(plane_img, cv2.COLOR_BGR2GRAY) if plane_img.ndim == 3 else plane_img
    w = int(window_m / res_m)
    x0, y0 = max(0, int(prior[0]) - w), max(0, int(prior[1]) - w)
    win = gray[y0:int(prior[1]) + w, x0:int(prior[0]) + w]
    valid = win > 0
    if valid.mean() < 0.8:
        return None  # the holder runs off the image
    t = DARK_FRAC * np.median(win[valid])  # mostly wood; Otsu splits the darker wood grain off instead
    dark = ((win < t) & valid).astype(np.uint8) * 255
    k = max(3, int(0.004 / res_m) | 1)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    cs, hier = cv2.findContours(dark, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    best = None
    for i, c in enumerate(cs):
        if hier[0][i][3] < 0:
            continue  # holes in the dark region only
        (cx, cy), (a_px, b_px), ang = cv2.minAreaRect(c)
        if a_px < b_px:
            a_px, b_px, ang = b_px, a_px, ang + 90
        size = (a_px * res_m, b_px * res_m)
        if abs(size[0] - INNER_M[0]) > SIZE_TOL_M or abs(size[1] - INNER_M[1]) > SIZE_TOL_M:
            continue
        d = math.hypot(cx + x0 - prior[0], cy + y0 - prior[1])
        if best is None or d < best[0]:
            fill = cv2.contourArea(c) / (a_px * b_px)
            best = (d, {"c": (cx + x0, cy + y0), "angle_deg": -((ang + 90) % 180 - 90),  # rows grow downwards
                        "size_m": size, "fill": fill})
    return None if best is None or best[1]["fill"] < 0.85 else best[1]
