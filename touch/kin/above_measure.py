"""Where the plate sits relative to the jaws, from the wrist view over the grasp vs a reference view (same
relative pose, taken over the plate in holder A): an affine image match on the plate between the pads,
coarse (blurred until the well lattice is gone, so it cannot slip a well) then fine. The match's scale gives
the change in distance to the plate, its shift where the plate centre moved in the image."""
import cv2
import numpy as np

MASK = (slice(0, 175), slice(240, 455))  # rows, cols: the plate between the pads (FLAT_OPEN_DEG), 30 mm over it


def _gray(img, sigma):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.GaussianBlur(g, (0, 0), sigma)


def match(ref, frame):
    """-> (W 2x3 affine, ref -> frame pixels, cc of the fine level) or (None, 0.0)."""
    mask = np.zeros(ref.shape[:2], np.uint8)
    mask[MASK] = 255
    W = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-6)
    cc = 0.0
    for sigma in (8.0, 4.0, 1.5):
        try:
            cc, W = cv2.findTransformECC(_gray(ref, sigma), _gray(frame, sigma), W, cv2.MOTION_AFFINE, crit, mask, 5)
        except cv2.error:
            return None, 0.0
    return W, float(cc)


def plate_offset(ref, frame, K, dist, plate_in_cam):
    """-> dict: the plate centre's offset in camera coordinates (m) from where the reference has it, plus
    the image numbers behind it; None if the images do not match."""
    W, cc = match(ref, frame)
    if W is None:
        return None
    P = np.asarray(plate_in_cam, float)
    uv0 = cv2.projectPoints(P.reshape(1, 1, 3), np.zeros(3), np.zeros(3), K, dist)[0].reshape(2)
    uv1 = W[:, :2] @ uv0 + W[:, 2]
    s = float(np.sqrt(abs(np.linalg.det(W[:, :2]))))
    ang = float(np.degrees(np.arctan2(W[1, 0] - W[0, 1], W[0, 0] + W[1, 1])))
    depth = P[2] / s
    ray = cv2.undistortPoints(uv1.reshape(1, 1, 2).astype(np.float64), K, dist).reshape(2)
    P1 = np.array([ray[0] * depth, ray[1] * depth, depth])
    return {"d_cam": P1 - P, "scale": s, "ang": ang, "duv": uv1 - uv0, "cc": cc}
