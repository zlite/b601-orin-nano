"""Training progress for a SmolVLA fine-tune on HF Jobs: one row per logged step (loss, grad norm, lr, speed,
ETA, cost so far), then a summary when the job ends.

Run inside the b601-policy image with the HF cache mounted (./jetson.sh train-progress does this):
  python3 /w/training/progress.py [JOB_ID] [--watch SECONDS]
JOB_ID defaults to the last job training/launch.py started (training/last_job.json).
--watch keeps polling until the job ends; without it, prints what is logged so far and exits.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

PRICE_PER_H = {"l40sx1": 1.80, "a10g-small": 1.00, "a10g-large": 1.50, "a100-large": 4.00, "t4-small": 0.40}
LAST = Path(__file__).with_name("last_job.json")
STEP = re.compile(r"step:(\S+) smpl:\S+ ep:\S+ epch:(\S+) loss:(\S+) grdn:(\S+) lr:(\S+).*?step_s:(\S+)")
BAR = re.compile(r"(\d+)/(\d+) \[([\d:]+)<([\d:?]+)")
DONE = ("COMPLETED", "ERROR", "CANCELED", "CANCELLED", "DELETED")


def hms(sec):
    sec = int(max(sec, 0))
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job_id", nargs="?")
    ap.add_argument("--watch", type=float, default=0, help="poll every N s until the job ends")
    ap.add_argument("--log-freq", type=int, default=200, help="lerobot-train --log_freq (its default 200)")
    a = ap.parse_args()
    meta = json.loads(LAST.read_text()) if LAST.exists() else {}
    job_id = a.job_id or meta.get("job_id")
    if not job_id:
        sys.exit("no job id given and no training/last_job.json")
    api = HfApi()
    seen, rows, total = set(), [], None
    header = False
    while True:
        job = api.inspect_job(job_id=job_id)
        stage = str(job.status.stage).split(".")[-1]
        flavor = str(getattr(job, "flavor", "") or meta.get("flavor", "")).split(".")[-1]
        price = PRICE_PER_H.get(flavor)
        started = job.created_at.timestamp() if job.created_at else time.time()
        lines = list(api.fetch_job_logs(job_id=job_id)) if stage != "SCHEDULING" else []
        text = "\n".join(lines)
        bars = BAR.findall(text)
        if bars:
            total = int(bars[-1][1])
        # lerobot rounds the step above 1000 ("step:1K" for 1000..1499), so number the rows by their order
        for i, m in enumerate(STEP.finditer(text)):
            step = (i + 1) * a.log_freq
            if step in seen:
                continue
            seen.add(step)
            epoch, loss, grdn, lr, step_s = m.group(2), float(m.group(3)), float(m.group(4)), m.group(5), float(m.group(6))
            if not header:
                print(f"job {job_id} ({flavor}{f', ${price:.2f}/h' if price else ''}): {job.url if hasattr(job, 'url') else ''}")
                print(f"{'step':>11} {'done':>5} {'epoch':>6} {'loss':>7} {'grad':>6} {'lr':>8} {'step/s':>6} {'ETA':>8}")
                header = True
            left = (total - step) * step_s if total else 0
            pct = f"{100 * step / total:4.0f}%" if total else "   ?"
            print(f"{step:>5}/{total or '?':<5} {pct} {epoch:>6} {loss:7.4f} {grdn:6.3f} {lr:>8} {1 / step_s:6.2f} "
                  f"{hms(left) if total else '?':>8}", flush=True)
            rows.append((step, loss))
        elapsed = time.time() - started
        # billed from the first training line, not from creation (a job can wait for hardware in SCHEDULING)
        if stage == "SCHEDULING" or not price or not bars:
            cost = ""
        else:
            run_s = sum(int(x) * 60 ** i for i, x in enumerate(reversed(bars[-1][2].split(":"))))
            cost = f", ~${price * run_s / 3600:.2f} of GPU time so far" if stage not in DONE else ""
        if stage in DONE or not a.watch:
            if rows:
                losses = [l for _, l in rows]
                print(f"summary: {stage}; loss {losses[0]:.4f} at step {rows[0][0]} -> {losses[-1]:.4f} at step "
                      f"{rows[-1][0]} (min {min(losses):.4f}); last bar {bars[-1][0] if bars else '?'}/{total or '?'}"
                      f" in {bars[-1][2] if bars else '?'}{cost}")
            else:
                print(f"{time.strftime('%H:%M:%S')} job {job_id}: {stage}, no training steps logged yet{cost}")
            if stage in DONE:
                tail = [l for l in lines[-40:] if "pushed to" in l or "Error" in l or "Traceback" in l]
                for l in tail[-3:]:
                    print("  " + l.strip()[-200:])
            break
        time.sleep(a.watch)


if __name__ == "__main__":
    main()
