"""Find a 96-well plate in a top-down plane map (birdseye.plane_map at the plate-top height).

Matches a rendered 12 x 8 well lattice (9 mm pitch) over rotations. Seen from above each well is a
dark disc in a bright lattice (a zero-sum dark-centre / bright-surround kernel);
a ring of "no well" positions just outside the lattice scores negatively, so a
lattice shifted by one pitch (11/12 overlap) loses clearly to the true position.
"""
import math

import cv2
import numpy as np

PITCH_M, COLS, ROWS = 0.009, 12, 8
HOLE_M, CELL_M = 0.0028, 0.0040  # dark disc radius, bright surround radius


def render(angle_deg, res_m, outside_w=0.7):
    """Template image (float32) of the lattice rotated by angle_deg (CCW, about its centre)."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    half = 0.5 * math.hypot((COLS + 1) * PITCH_M, (ROWS + 1) * PITCH_M) + CELL_M
    n = int(math.ceil(2 * half / res_m)) | 1
    c0 = n // 2
    t = np.zeros((n, n), np.float32)
    hole, cell = HOLE_M / res_m, CELL_M / res_m
    k_out = hole ** 2 / (cell ** 2 - hole ** 2)  # zero sum per well
    for i in range(-1, COLS + 1):
        for j in range(-1, ROWS + 1):
            inside = 0 <= i < COLS and 0 <= j < ROWS
            w = 1.0 if inside else -outside_w
            x, y = (i - (COLS - 1) / 2) * PITCH_M, (j - (ROWS - 1) / 2) * PITCH_M
            u, v = (ca * x - sa * y) / res_m, -(sa * x + ca * y) / res_m  # image rows grow downwards
            p = (int(round((c0 + u) * 8)), int(round((c0 + v) * 8)))
            cv2.circle(t, p, int(round(cell * 8)), w * k_out, -1, cv2.LINE_AA, shift=3)
            cv2.circle(t, p, int(round(hole * 8)), -w, -1, cv2.LINE_AA, shift=3)
    return t


def highpass(gray, res_m, valid):
    g = gray.astype(np.float32)
    s = 0.004 / res_m
    hp = g - cv2.GaussianBlur(g, (0, 0), s)
    hp /= cv2.GaussianBlur(np.abs(hp), (0, 0), 3 * s) + 2.0  # local contrast normalisation
    hp[~valid] = 0
    return hp


def detect(plane_img, res_m, angles=None, prior=None, prior_r_m=0.005):
    """-> dict(c=(col, row) px of the lattice centre, angle_deg (lattice x axis vs map x, CCW, in
    [-90, 90)), score, second=best score at a different place/angle). None if nothing fits.
    prior=(col, row): only consider centres within prior_r_m of it (a known rough position rules out
    the lattice shifted by whole pitches, which can score nearly as well)."""
    gray = cv2.cvtColor(plane_img, cv2.COLOR_BGR2GRAY) if plane_img.ndim == 3 else plane_img
    valid = cv2.erode((gray > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    hp = highpass(gray, res_m, valid)
    best = []
    for a in (np.arange(-90, 90, 2.0) if angles is None else angles):
        t = render(a, res_m)
        if t.shape[0] > hp.shape[0] or t.shape[1] > hp.shape[1]:
            pad = max(t.shape[0] - hp.shape[0], t.shape[1] - hp.shape[1]) // 2 + 1
            hp = cv2.copyMakeBorder(hp, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
            offset = pad
        else:
            offset = 0
        r = cv2.matchTemplate(hp, t, cv2.TM_CCORR) / float(np.abs(t).sum())
        if prior is not None:  # response index = centre - half template + padding
            pr = int(round(prior_r_m / res_m))
            x0 = int(round(prior[0])) - t.shape[1] // 2 + offset
            y0 = int(round(prior[1])) - t.shape[0] // 2 + offset
            keep = np.zeros_like(r, bool)
            keep[max(0, y0 - pr):max(0, y0 + pr + 1), max(0, x0 - pr):max(0, x0 + pr + 1)] = True
            r = np.where(keep, r, 0.0)
        mn, mx, lmn, lmx = cv2.minMaxLoc(r)
        for v, loc in ((mx, lmx), (-mn, lmn)):  # wells look dark from afar, light up close: either polarity
            best.append((v, a, loc[0] + t.shape[1] // 2 - offset, loc[1] + t.shape[0] // 2 - offset))
        if offset:
            hp = hp[offset:-offset, offset:-offset]
    best.sort(reverse=True)
    if not best or best[0][0] <= 0:
        return None
    s, a, cx, cy = best[0]
    # second-best clearly different hypothesis (another place, or a 90-deg rotation)
    far = [b for b in best[1:] if (prior is None and math.hypot(b[2] - cx, b[3] - cy) * res_m > PITCH_M / 2)
           or abs((b[1] - a + 90) % 180 - 90) > 10]
    return {"c": (cx, cy), "angle_deg": a, "score": s, "second": far[0][0] if far else 0.0}


def refine(plane_img, res_m, coarse, span=5.0, step=0.25, prior=None, prior_r_m=0.005, keep=0.97):
    """Fine fit around a coarse one. The score is nearly flat over a few degrees (with a spurious spike
    where the template is pixel-aligned), so the angle is the centre of the smoothed score's plateau
    (>= keep x max) rather than its argmax; the centre is the fit at that angle."""
    angles = np.arange(coarse["angle_deg"] - span, coarse["angle_deg"] + span + 1e-6, step)
    fits = [detect(plane_img, res_m, [a], prior=prior, prior_r_m=prior_r_m) for a in angles]
    if any(f is None for f in fits):
        return detect(plane_img, res_m, angles, prior=prior, prior_r_m=prior_r_m)
    s = np.array([f["score"] for f in fits])
    sm = np.convolve(np.pad(s, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    i0 = int(np.argmax(sm))
    lo, hi = i0, i0
    while lo > 0 and sm[lo - 1] >= keep * sm[i0]:
        lo -= 1
    while hi < len(sm) - 1 and sm[hi + 1] >= keep * sm[i0]:
        hi += 1
    a = float(angles[lo:hi + 1].mean())
    best = detect(plane_img, res_m, [a], prior=prior, prior_r_m=prior_r_m)
    best["second"] = 0.0 if prior is not None else coarse["second"]  # with a prior, the window rules out rivals
    best["score"] = max(best["score"], float(s.max()))
    return best
