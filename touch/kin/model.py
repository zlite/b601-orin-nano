"""Calibrated B601-RS kinematics (URDF + kin_calib.json): forward kinematics, camera pose, IK.

All poses are 4x4 in base_link (metres); joint angles are driver ("public") degrees.
"""
import json
import math
from pathlib import Path

import numpy as np

import fk as fkmod

HERE = Path(__file__).parent
PUBLIC = fkmod.PUBLIC


def log_rot(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    if th > math.pi - 1e-6:  # 180 deg: axis from the symmetric part
        w = np.sqrt(np.maximum(np.diag(R) + 1, 0) / 2)
        return w * th
    return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th)) * th


class Kin:
    def __init__(self, calib=HERE / "kin_calib.json"):
        K = json.loads(Path(calib).read_text())
        self.signs, self.offs = K["signs"], K["offsets_deg"]
        self.X = np.array(K["X_cam_in_end"])  # camera in gripper_end
        self.tags = {int(t): np.array(v) for t, v in K["tags_in_base"].items()}

    def end(self, q: dict) -> np.ndarray:
        return fkmod.fk({j: -(self.signs[k] * q[j] + self.offs[k]) for k, j in enumerate(PUBLIC)})

    def cam(self, q: dict) -> np.ndarray:
        return self.end(q) @ self.X

    def point(self, q: dict, p_cam) -> np.ndarray:
        """A point fixed in the camera frame (e.g. the fingertip midpoint), in base."""
        C = self.cam(q)
        return C[:3, :3] @ np.asarray(p_cam) + C[:3, 3]

    def ik(self, T_goal, q0: dict, tool=np.eye(4), rot_w=(1.0, 1.0, 1.0), limits=None, iters=200,
           tol_mm=0.2, tol_deg=0.1):
        """Joints placing `cam(q) @ tool` at T_goal (damped least squares from q0).
        rot_w weights the rotation error about the goal's x, y, z axes (0 frees that axis).
        Returns (q, pos_err_mm, rot_err_deg)."""
        q = np.array([q0[j] for j in PUBLIC], float)
        W = np.diag([1, 1, 1, *[0.05 * w for w in rot_w]])  # 0.05 m per rad

        def err(qv):
            T = self.cam(dict(zip(PUBLIC, qv))) @ tool
            dp = T_goal[:3, 3] - T[:3, 3]
            dr_goal = log_rot(T_goal[:3, :3].T @ T[:3, :3])  # in goal axes
            return np.concatenate([dp, -T_goal[:3, :3] @ (np.array(rot_w) * dr_goal)]), dp, dr_goal

        lam = 1e-8  # J is in m per degree (~5e-3), so J^T J ~ 1e-5
        for _ in range(iters):
            e, dp, dr = err(q)
            if np.linalg.norm(dp) * 1000 < tol_mm and math.degrees(np.linalg.norm(np.array(rot_w) * dr)) < tol_deg:
                break
            J = np.empty((6, 6))
            for i in range(6):
                dq = np.zeros(6); dq[i] = 1e-3
                J[:, i] = (e - err(q + dq)[0]) / 1e-3
            Jw, ew = W @ J, W @ e
            step = np.linalg.solve(Jw.T @ Jw + lam * np.eye(6), Jw.T @ ew)
            step *= min(1.0, 5.0 / max(1e-9, np.abs(step).max()))  # <= 5 deg per iteration
            q = q + step
            if limits:
                q = np.array([min(limits[j][1], max(limits[j][0], v)) for j, v in zip(PUBLIC, q)])
        e, dp, dr = err(q)
        return (dict(zip(PUBLIC, q.tolist())), float(np.linalg.norm(dp) * 1000),
                float(math.degrees(np.linalg.norm(np.array(rot_w) * dr))))


def rot_vec(w):
    """Rotation matrix of the rotation vector w (rad)."""
    th = float(np.linalg.norm(w))
    return np.eye(3) if th < 1e-12 else fkmod.axis_angle(np.asarray(w) / th, th)
