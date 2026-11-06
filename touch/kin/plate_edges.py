"""The plate's outline in the wrist view over the grasp (jaws fully open): the clear plate is grey (low
saturation) against the orange wood, so its silhouette between the pads is clean, and straight lines fitted to
its long sides and its near short edge locate it without the well lattice (whose repeats fool image matching).
Image coordinates: x right, y down; a side is censored (dropped) where it runs into a pad."""
import cv2
import numpy as np

ROI_X = (226, 476)   # between the fully open pads' inner edges
ROI_Y = (0, 220)
S_MAX, V_MIN = 85, 90
V_GLARE = 200       # unsaturated and this bright: plate under glare
TEX_SD = 22         # local grey-level std (9 px): plate ~35 (10 % < 27), wood ~18, glare ~27
MIN_ROWS = 25        # boundary points needed for a side line


def silhouette(img):
    """Grey (unsaturated) and finely textured (the wells): pale glare on the wood is unsaturated but smooth."""
    h = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mu = cv2.blur(g, (9, 9))
    sd = np.sqrt(np.maximum(cv2.blur(g * g, (9, 9)) - mu * mu, 0))
    tex = cv2.blur((sd > TEX_SD).astype(np.float32), (15, 15)) > 0.6
    # glare washes the wells out (no texture) but leaves the plate grey and bright; glare on the wood stays orange
    m = ((h[..., 1] < S_MAX) & (h[..., 2] > V_MIN) & (tex | (h[..., 2] > V_GLARE))).astype(np.uint8)
    roi = np.zeros_like(m)
    roi[ROI_Y[0]:ROI_Y[1], ROI_X[0]:ROI_X[1]] = 1
    m &= roi
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (lab == big).astype(np.uint8)
    cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m = np.zeros_like(m)
    cv2.drawContours(m, cnt, -1, 1, cv2.FILLED)  # fill the holes (dark well bottoms, reflections)
    return m


def _robust_line(a, b):
    """b = k * a + c through (a, b) points, refit without outliers (> 2.5 px). -> (k, c, n) or None."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < MIN_ROWS:
        return None
    keep = np.ones(len(a), bool)
    for _ in range(3):
        k, c = np.polyfit(a[keep], b[keep], 1)
        keep = np.abs(b - (k * a + c)) < 2.5
        if keep.sum() < MIN_ROWS:
            return None
    return float(k), float(c), int(keep.sum())


def outline(img):
    """-> dict with the left / right side lines x = k*y + c and the near edge y = k*x + c (each None if not
    seen), or None if no plate silhouette."""
    m = silhouette(img)
    if m is None:
        return None
    rows = np.where(m.any(1))[0]
    bottom = rows.max()
    L, R = ([], []), ([], [])
    for y in rows:
        if y > bottom - 12:  # stay clear of the rounded corners
            continue
        xs = np.where(m[y])[0]
        if xs[0] > ROI_X[0] + 2:
            L[0].append(y); L[1].append(xs[0])
        if xs[-1] < ROI_X[1] - 3:
            R[0].append(y); R[1].append(xs[-1])
    cols = np.where(m.any(0))[0]
    B = ([], [])
    for x in cols:
        if x < cols.min() + 15 or x > cols.max() - 15:
            continue
        ys = np.where(m[:, x])[0]
        if ys[-1] < ROI_Y[1] - 3:
            B[0].append(x); B[1].append(ys[-1])
    return {"left": _robust_line(*L), "right": _robust_line(*R), "near": _robust_line(*B), "mask": m}


def offset(ref, cur, y0=80.0):
    """Plate in `cur` vs `ref`: (dx sideways px, dy along px, dang deg) at row y0, or None if not measurable.
    dx > 0: the plate is further right in cur; dy > 0: further down; dang: the sides' slope change."""
    a, b = outline(ref), outline(cur)
    if a is None or b is None or a["near"] is None or b["near"] is None:
        return None
    xs = []
    for side in ("left", "right"):
        if a[side] and b[side]:
            xs.append((b[side][0] * y0 + b[side][1]) - (a[side][0] * y0 + a[side][1]))
    if not xs:
        return None
    dx = float(np.mean(xs))
    angs = [np.degrees(np.arctan(b[s][0]) - np.arctan(a[s][0])) for s in ("left", "right") if a[s] and b[s]]
    xc = (ROI_X[0] + ROI_X[1]) / 2
    dy = (b["near"][0] * (xc + dx) + b["near"][1]) - (a["near"][0] * xc + a["near"][1])
    return {"dx": dx, "dy": float(dy), "dang": float(np.mean(angs)), "sides": len(xs),
            "spread": float(np.ptp(xs)) if len(xs) > 1 else 0.0}


def mask_shift(ref, cur, search=60, angles=np.arange(-3.0, 3.01, 0.5)):
    """Slide (and turn about its centroid) the reference silhouette over the current one; the best overlap
    (IoU) gives the plate's image shift. -> {"dx", "dy", "dang", "iou", "mask_ref", "mask_cur"} or None."""
    a, b = silhouette(ref), silhouette(cur)
    if a is None or b is None:
        return None
    ys, xs = np.nonzero(a)
    cx, cy = xs.mean(), ys.mean()
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    bp = np.pad(b.astype(np.float32), search)
    best = None
    for ang in angles:
        R = cv2.getRotationMatrix2D((float(cx), float(cy)), float(ang), 1.0)
        ar = cv2.warpAffine(a, R, (a.shape[1], a.shape[0]), flags=cv2.INTER_NEAREST)
        t = ar[y0:y1, x0:x1].astype(np.float32)
        # overlap = correlation of the 0/1 masks; IoU = overlap / (|A| + |B in window| - overlap)
        ov = cv2.matchTemplate(bp[y0:y1 + 2 * search, x0:x1 + 2 * search], t, cv2.TM_CCORR)
        bs = cv2.matchTemplate(bp[y0:y1 + 2 * search, x0:x1 + 2 * search], np.ones_like(t), cv2.TM_CCORR)
        iou = ov / np.maximum(t.sum() + bs - ov, 1.0)
        j = np.unravel_index(int(np.argmax(iou)), iou.shape)
        if best is None or iou[j] > best[0]:
            best = (float(iou[j]), j[1] - search, j[0] - search, float(ang))
    iou, dx, dy, ang = best
    return {"dx": float(dx), "dy": float(dy), "dang": ang, "iou": iou, "mask_ref": a, "mask_cur": b}


def _robust_line(y, x):
    """x = p(y) fitted with iterative outlier rejection (residuals > 2.5 robust sigma)."""
    keep = np.ones(len(y), bool)
    for _ in range(4):
        p = np.polyfit(y[keep], x[keep], 1)
        r = x - np.polyval(p, y)
        s = 1.4826 * np.median(np.abs(r[keep])) + 0.5
        keep = np.abs(r) < 2.5 * s
    return p


def side_edges(img):
    """Lines (x = p(y)) along the left and right edges of the plate silhouette's widest band, or None."""
    m = silhouette(img)
    if m is None:
        return None
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    if n < 2:
        return None
    m = lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    w = m.sum(1)
    rows = np.nonzero(w > 0.6 * w.max())[0]
    if len(rows) < 20:
        return None
    left = np.array([np.nonzero(m[r])[0][0] for r in rows], float)
    right = np.array([np.nonzero(m[r])[0][-1] for r in rows], float)
    return _robust_line(rows, left), _robust_line(rows, right)


def edge_turn(ref, cur):
    """Image turn (deg, CCW on screen positive, as mask_shift's dang) of the plate's long side edges, current
    vs reference: (left, right) or None. IoU over the ragged silhouette is flat in angle (0.914-0.928 over
    +-3 deg, 2026-10-21); a line along each straight side is not."""
    a, b = side_edges(ref), side_edges(cur)
    if a is None or b is None:
        return None
    return tuple(float(np.degrees(np.arctan(pb[0]) - np.arctan(pa[0]))) for pa, pb in zip(a, b))
