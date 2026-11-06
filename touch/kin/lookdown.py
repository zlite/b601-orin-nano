"""Camera poses that view a table point like VIEW_POSE does (same tilt, yaw about vertical free),
with the point at a chosen pixel and distance; used to find the plate without a tag. (The wrist
cannot pitch far enough to look straight down; the top-down remap removes the perspective anyway.)"""
import numpy as np

from model import PUBLIC

LOOK_D_M = 0.33           # distance along the ray to the plate centre
LOOK_PX = (325.0, 260.0)  # where the plate centre should appear (the closed fingers cover the top centre)
DOWN = np.diag([1.0, -1.0, -1.0])


def view_tool(K, C_view, depth=LOOK_D_M, px=LOOK_PX):
    """Frame on the camera ray through px at `depth`, oriented as a z-down frame was at C_view."""
    T = np.eye(4)
    T[:3, :3] = C_view[:3, :3].T @ DOWN
    T[:3, 3] = [(px[0] - K[0][2]) / K[0][0] * depth, (px[1] - K[1][2]) / K[1][1] * depth, depth]
    return T


def look_q(kin, K, p_base, q_seed, q_view, limits=None, margin_deg=3.0, depth=LOOK_D_M):
    """Joints viewing p_base at LOOK_PX from `depth`, with the camera tilt of pose q_view.
    Returns (q, pos_err_mm, rot_err_deg)."""
    G = np.eye(4)
    G[:3, :3] = DOWN
    G[:3, 3] = p_base
    lim = None if limits is None else {j: (limits[j][0] + margin_deg, limits[j][1] - margin_deg) for j in PUBLIC}
    return kin.ik(G, q_seed, tool=view_tool(K, kin.cam(q_view), depth), rot_w=(1.0, 1.0, 0.0), limits=lim)
