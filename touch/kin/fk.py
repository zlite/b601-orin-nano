"""Forward kinematics of the reBot Arm B601-RS from Seeed's URDF (ReBot_Arm_RS.urdf).

Driver ("public") joint angles are the negated motor angles; the URDF uses motor angles with the
same zero (the folded rest pose), so q_urdf = -radians(q_public).
"""
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

URDF = Path(__file__).with_name("ReBot_Arm_RS.urdf")
PUBLIC = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll")


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_angle(axis, a):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def load_chain(end="gripper_end"):
    root = ET.parse(URDF).getroot()
    by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
    chain, link = [], end
    while link in by_child:
        j = by_child[link]
        o = j.find("origin")
        xyz = [float(v) for v in (o.get("xyz") or "0 0 0").split()]
        rpy = [float(v) for v in (o.get("rpy") or "0 0 0").split()]
        ax = j.find("axis")
        chain.append((j.get("name"), j.get("type"), np.array(xyz), rpy_matrix(*rpy),
                      np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else None))
        link = j.find("parent").get("link")
    return chain[::-1]


CHAIN = load_chain()


def fk(q_public: dict, end_offset=None) -> np.ndarray:
    """4x4 pose of gripper_end (or gripper_end @ end_offset) in base_link, from driver angles (deg)."""
    q = [-math.radians(q_public[j]) for j in PUBLIC]
    T, k = np.eye(4), 0
    for name, typ, xyz, R0, axis in CHAIN:
        A = np.eye(4)
        A[:3, :3], A[:3, 3] = R0, xyz
        T = T @ A
        if typ == "revolute":
            M = np.eye(4)
            M[:3, :3] = axis_angle(axis, q[k])
            T = T @ M
            k += 1
    if end_offset is not None:
        T = T @ end_offset
    return T


if __name__ == "__main__":
    import json
    L = json.loads((Path(__file__).parents[1] / "data" / "pick_learned.json").read_text())
    poses = {"rest": dict.fromkeys(PUBLIC, 0.0),
             "view": {"shoulder_pan": 41.0, "shoulder_lift": -38.0, "elbow_flex": -38.5, "wrist_flex": 51.0,
                      "wrist_yaw": 0.0, "wrist_roll": 3.0},
             "vertical grasp": L["grasp_cmd"], "flat grasp (measured)": L["flat_grasp_q"],
             "flat lift (measured)": L["flat_lift_q"]}
    for n, q in poses.items():
        T = fk(q)
        z_axis = T[:3, 2]
        print(f"{n:24s} gripper_end at {np.round(T[:3, 3] * 1000, 1).tolist()} mm, tool z axis {np.round(z_axis, 2).tolist()}")
