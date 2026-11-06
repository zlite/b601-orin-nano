"""Export the plate-policy dashboard's data files: episodes, HF training jobs and their loss curves.

Usage: env/bin/python touch/dash_export.py OUT_DIR  -> OUT_DIR/{episodes,jobs,loss}.json
"""
import json
import math
import re
import sys
from pathlib import Path

from huggingface_hub import HfApi

HERE = Path(__file__).resolve().parent
DATASETS = {"plate_shuttle": "holder A <-> holder B shuttle", "plate_random": "random table / holder A"}
STEP_RE = re.compile(r"step:(\d+(?:\.\d+)?)([KM]?) .*?loss:([0-9.]+)")


def episodes():
    rows = []
    for name, kind in DATASETS.items():
        f = HERE / "data" / "lerobot" / name / "episode_outcomes.json"
        if not f.exists():
            continue
        for e in json.loads(f.read_text()):
            err = e.get("place_err_mm")
            rows.append({
                "dataset": name,
                "episode": e["episode"],
                "time": e["time"].replace(" ", "T"),
                "day": e["time"][:10],
                "success": bool(e["success"]),
                "kind": "holder" if "holder" in e.get("task", "") else "table",
                "frames": e.get("frames", 0),
                "place_err_mm": round(math.hypot(*err), 1) if err else None,
                "place_err_deg": round(abs(e["place_err_deg"]), 2) if "place_err_deg" in e else None,
            })
    return rows


def steps(n, unit):
    return int(float(n) * {"": 1, "K": 1_000, "M": 1_000_000}[unit])


def jobs(api):
    out, losses = [], []
    for j in api.list_jobs():
        env = j.environment or {}
        if "POLICY_REPO" not in env:
            continue
        log = [line for line in api.fetch_job_logs(job_id=j.id) if "step:" in line] if j.status.stage != "SCHEDULING" else []
        curve = []
        for line in log:
            m = STEP_RE.search(line)
            if m:
                curve.append({"job": j.id[:8], "step": steps(m[1], m[2]), "loss": float(m[3])})
        losses += curve
        n_eps = len(json.loads(env.get("EPISODES", "[]")))
        out.append({
            "job": j.id[:8],
            "policy": env["POLICY_REPO"].split("/")[-1],
            "dataset": env.get("DATASET", "").split("/")[-1],
            "base": env.get("BASE_POLICY", "lerobot/smolvla_base").split("/")[-1],
            "episodes": n_eps,
            "steps_target": int(env.get("STEPS", "20000")),
            "steps_done": curve[-1]["step"] if curve else 0,
            "final_loss": curve[-1]["loss"] if curve else None,
            "status": j.status.stage.lower(),
            "gpu": j.flavor,
            "created": j.created_at.isoformat(timespec="seconds") if j.created_at else None,
            "url": j.url,
        })
    out.sort(key=lambda r: r["created"] or "")
    return out, losses


def main():
    dst = Path(sys.argv[1])
    dst.mkdir(parents=True, exist_ok=True)
    job_rows, loss_rows = jobs(HfApi())
    for name, rows in (("episodes", episodes()), ("jobs", job_rows), ("loss", loss_rows)):
        (dst / f"{name}.json").write_text(json.dumps(rows))
        print(name, len(rows))


if __name__ == "__main__":
    main()
