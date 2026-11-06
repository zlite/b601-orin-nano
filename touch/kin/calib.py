"""Fit the B601-RS kinematics to wrist-camera tag observations.

Model: FK(q) @ X @ C_i = Y_tag for every sample i, where FK is the URDF chain (fk.py) with per-joint
signs and zero offsets, X the camera pose in gripper_end (hand-eye), C_i the measured tag pose in the
camera and Y_tag the tag pose in base_link (one per tag). Samples: data/kin_samples.json (kincollect).

Usage: calib.py            -> prints the fit and writes kin/kin_calib.json
"""
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fk as fkmod  # noqa: E402

HERE = Path(__file__).parent
SAMPLES = HERE.parent / "data" / "kin_samples.json"
OUT = HERE / "kin_calib.json"
ROT_W = 0.05  # m per rad in the residual (1 deg ~ 0.9 mm)
OUTLIER_DEG = 10.0


def rodrigues(v):
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    return fkmod.axis_angle(v / th, th)


def log_rot(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th


def homog(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def inv(T):
    R, t = T[:3, :3], T[:3, 3]
    return homog(R.T, -R.T @ t)


def fk_signed(q, signs, offsets):
    """FK with driver angles mapped as q_urdf = sign * radians(q_public) + offset (fk.fk uses sign -1)."""
    qq = {j: -(signs[k] * q[j] + math.degrees(offsets[k])) for k, j in enumerate(fkmod.PUBLIC)}
    return fkmod.fk(qq)


def hand_eye(E, C):
    """Park-Martin AX = XB from pairs of samples: A = E_i^-1 E_j, B = C_i C_j^-1."""
    alphas, betas, As, Bs = [], [], [], []
    n = len(E)
    for i in range(n):
        for j in range(i + 1, n):
            A = inv(E[i]) @ E[j]
            B = C[i] @ inv(C[j])
            a, b = log_rot(A[:3, :3]), log_rot(B[:3, :3])
            if np.linalg.norm(a) < math.radians(3) or np.linalg.norm(b) < math.radians(3):
                continue
            alphas.append(a); betas.append(b); As.append(A); Bs.append(B)
    if len(alphas) < 3:
        return None
    H = sum(np.outer(b, a) for a, b in zip(alphas, betas))
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    Rx = Vt.T @ D @ U.T
    M = np.vstack([A[:3, :3] - np.eye(3) for A in As])
    v = np.concatenate([Rx @ B[:3, 3] - A[:3, 3] for A, B in zip(As, Bs)])
    tx = np.linalg.lstsq(M, v, rcond=None)[0]
    return homog(Rx, tx)


def residuals(params, data, signs, tags, fit_offsets):
    X = homog(rodrigues(params[0:3]), params[3:6])
    k = 6
    Y = {}
    for t in tags:
        Y[t] = homog(rodrigues(params[k:k + 3]), params[k + 3:k + 6])
        k += 6
    offsets = params[k:k + 6] if fit_offsets else np.zeros(6)
    res = []
    for d in data:
        pred = inv(X) @ inv(fk_signed(d["q"], signs, offsets)) @ Y[d["tag"]]  # tag in camera
        res.extend(pred[:3, 3] - d["C"][:3, 3])
        res.extend(ROT_W * log_rot(d["C"][:3, :3].T @ pred[:3, :3]))
    return np.array(res)


def lm(f, x0, iters=60):
    x, lam = x0.copy(), 1e-3
    r = f(x)
    cost = r @ r
    for _ in range(iters):
        J = np.empty((len(r), len(x)))
        for i in range(len(x)):
            dx = np.zeros(len(x)); dx[i] = 1e-6
            J[:, i] = (f(x + dx) - r) / 1e-6
        g, Hm = J.T @ r, J.T @ J
        while True:
            step = -np.linalg.solve(Hm + lam * np.diag(np.diag(Hm) + 1e-12), g)
            r2 = f(x + step)
            if r2 @ r2 < cost:
                x, r, cost, lam = x + step, r2, r2 @ r2, lam * 0.3
                break
            lam *= 10
            if lam > 1e8:
                return x, r
        if np.linalg.norm(step) < 1e-9:
            break
    return x, r


def fit(data, signs, fit_offsets=False, x0=None):
    tags = sorted({d["tag"] for d in data})
    offsets = np.zeros(6)
    E = [fk_signed(d["q"], signs, offsets) for d in data]
    if x0 is None:
        X = hand_eye(E, [d["C"] for d in data])
        if X is None:
            return None
        x0 = list(log_rot(X[:3, :3])) + list(X[:3, 3])
        for t in tags:
            Ys = [E[i] @ X @ d["C"] for i, d in enumerate(data) if d["tag"] == t]
            x0 += list(log_rot(Ys[0][:3, :3])) + list(np.mean([Yi[:3, 3] for Yi in Ys], axis=0))
        if fit_offsets:
            x0 += [0.0] * 6
        x0 = np.array(x0)
    x, r = lm(lambda p: residuals(p, data, signs, tags, fit_offsets), x0)
    rr = r.reshape(-1, 6)
    return {"x": x, "pos_rms_mm": float(np.sqrt((rr[:, :3] ** 2).sum(1).mean()) * 1000),
            "rot_rms_deg": float(np.degrees(np.sqrt((rr[:, 3:] ** 2).sum(1).mean()) / ROT_W)), "tags": tags}


def main():
    raw = json.loads(SAMPLES.read_text())
    data = [{"tag": s["tag"], "q": s["q"], "C": homog(np.array(s["R"]), np.array(s["p"]))} for s in raw]
    print(f"{len(data)} samples, tags {sorted({d['tag'] for d in data})}")
    results = []
    for signs in itertools.product((-1.0, 1.0), repeat=6):
        res = fit(data, np.array(signs))
        if res:
            results.append((res["pos_rms_mm"] + res["rot_rms_deg"], signs, res))
    results.sort(key=lambda t: t[0])
    # Drop samples with a flipped tag pose (planar-tag ambiguity): rotation residual > OUTLIER_DEG.
    _, signs0, res0 = results[0]
    rr = residuals(res0["x"], data, np.array(signs0), res0["tags"], False).reshape(-1, 6)
    rot_err = np.degrees(np.linalg.norm(rr[:, 3:], axis=1) / ROT_W)
    keep = rot_err < OUTLIER_DEG
    if not keep.all():
        print(f"dropping {int((~keep).sum())} outlier samples: {np.nonzero(~keep)[0].tolist()}")
        data = [d for d, k in zip(data, keep) if k]
        results = [(r["pos_rms_mm"] + r["rot_rms_deg"], sg, r) for sg in [signs0]
                   for r in [fit(data, np.array(sg))]]
    for score, signs, res in results[:5]:
        print(f"signs {signs}: pos rms {res['pos_rms_mm']:.1f} mm, rot rms {res['rot_rms_deg']:.2f} deg")
    _, signs, res = results[0]
    tags = res["tags"]
    x0 = np.concatenate([res["x"], np.zeros(6)])
    res2 = fit(data, np.array(signs), fit_offsets=True, x0=x0)
    k = 6 + 6 * len(tags)
    offs = np.degrees(res2["x"][k:k + 6])
    print(f"with zero offsets: pos rms {res2['pos_rms_mm']:.1f} mm, rot rms {res2['rot_rms_deg']:.2f} deg; "
          f"offsets {np.round(offs, 2).tolist()} deg")
    X = homog(rodrigues(res2["x"][0:3]), res2["x"][3:6])
    print("camera in gripper_end: t", np.round(X[:3, 3] * 1000, 1).tolist(), "mm, R\n", np.round(X[:3, :3], 3))
    Ys = {}
    for i, t in enumerate(tags):
        Yt = homog(rodrigues(res2["x"][6 + 6 * i:9 + 6 * i]), res2["x"][9 + 6 * i:12 + 6 * i])
        Ys[t] = Yt.tolist()
        print(f"tag {t} in base: {np.round(Yt[:3, 3] * 1000, 1).tolist()} mm, normal {np.round(Yt[:3, 2], 3).tolist()}")
    OUT.write_text(json.dumps({"signs": list(signs), "offsets_deg": offs.tolist(), "X_cam_in_end": X.tolist(),
                               "tags_in_base": Ys, "pos_rms_mm": res2["pos_rms_mm"],
                               "rot_rms_deg": res2["rot_rms_deg"], "n": len(data)}, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
