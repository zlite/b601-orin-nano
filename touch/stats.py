"""Plain-text summary of data collection and training, for a terminal (./jetson.sh stats refreshes it).

Reads only files: the recorded episodes (data/lerobot/plate_jetson/episode_outcomes.json), the controller log
(touch/log.txt), its status (touch/status.json) and the training progress cache (training/progress_cache.txt, written
by ./jetson.sh stats while a training job runs). Standard library only; uses no camera, arm or network.
  python3 touch/stats.py [--append FILE]
"""
import argparse
import datetime as dt
import json
import re
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUTCOMES = HERE / "data" / "lerobot" / "plate_jetson" / "episode_outcomes.json"
LOG = HERE / "log.txt"
STATUS = HERE / "status.json"
TRAIN = ROOT / "training" / "progress_cache.txt"
GOAL = HERE / "collect_goal.json"  # {"successes_today": 40}: the collection target shown as a progress bar
UTC = dt.timezone.utc


def local(t_utc: str) -> dt.datetime:  # outcome times are UTC ("2026-11-04 19:34:45")
    return dt.datetime.strptime(t_utc, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC).astimezone()


def tail_lines(path: Path, n: int = 4000) -> list[str]:
    if not path.exists():
        return []
    with path.open("rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 400_000))
        return f.read().decode("utf-8", "replace").splitlines()[-n:]


def halt_before(lines, t: dt.datetime, within_s: int = 300) -> str:
    """The controller's last HALT/FROZEN reason in the few minutes before an episode ended (log times are UTC)."""
    end = t.astimezone(UTC)
    best = ""
    for ln in lines:
        m = re.match(r"(\d\d):(\d\d):(\d\d) (HALT|FROZEN): (.*)", ln)
        if not m:
            continue
        lt = end.replace(hour=int(m[1]), minute=int(m[2]), second=int(m[3]))
        if 0 <= (end - lt).total_seconds() <= within_s:
            best = m[5]
    best = re.sub(r"^halted: ", "", best)
    return re.sub(r" \(jaws left open.*$|; not (going down|closing)$", "", best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--append", help="also append this summary to a file")
    a = ap.parse_args()
    now = dt.datetime.now().astimezone()
    out = [f"B601 plate collection — {now:%a %d %b %H:%M:%S}"]

    eps = json.loads(OUTCOMES.read_text()) if OUTCOMES.exists() else []
    lines = tail_lines(LOG)
    today = [e for e in eps if local(e["time"]).date() == now.date()]
    real = [e for e in today if e.get("frames", 0) > 0]  # 0-frame entries are picks skipped for motor heat
    ok = [e for e in real if e["success"]]
    out.append("")
    out.append(f"Today      {len(ok)} successful picks, {len(real) - len(ok)} failed"
               f"{f', {len(today) - len(real)} skipped (motor heat)' if len(today) > len(real) else ''}")
    tot_ok = sum(1 for e in eps if e["success"])
    frames = sum(e.get("frames", 0) for e in eps if e["success"])
    out.append(f"Dataset    plate_jetson: {len(eps)} episodes, {tot_ok} successful "
               f"({frames:,} frames, {frames / 10 / 60:.0f} min at 10 fps)")
    if real:
        span = (local(real[-1]["time"]) - local(real[0]["time"])).total_seconds() / 3600
        if span > 0.2:
            out.append(f"Rate       {len(ok) / span:.1f} successes per hour today")
    G = json.loads(GOAL.read_text()) if GOAL.exists() else {}
    if "holder" in G:  # pad <-> holder cycle (touch/cycle.sh): successes since its start, per direction
        since = G.get("since", "1970-01-01 00:00:00")
        cyc = [e for e in eps if e["time"] >= since and e["success"]]
        for key, task, label in (("holder", "put the 96-well plate in the holder", "pad -> holder"),
                                 ("pad", "pick up the 96-well plate and put it down somewhere else on the table",
                                  "holder -> pad")):
            n, goal = sum(1 for e in cyc if e["task"] == task), int(G[key])
            k = round(30 * min(n, goal) / goal)
            out.append(f"Goal       [{'#' * k}{'.' * (30 - k)}] {n}/{goal} {label}" + (" DONE" if n >= goal else ""))
        tries = [e for e in eps if e["time"] >= since and e.get("frames", 0) > 0]
        if tries:
            out.append(f"Cycle      {len(cyc)}/{len(tries)} picks succeeded since {local(since):%H:%M} "
                       f"({100 * len(cyc) / len(tries):.0f}%)")
    elif G:
        goal = int(G.get("successes_today", 0))
        if goal > 0:
            done = min(len(ok), goal)
            bar = "#" * round(30 * done / goal) + "." * (30 - round(30 * done / goal))
            left = goal - len(ok)
            eta = ""
            if left > 0 and real:
                # the pace of the last 2 h (motor cool-downs included), else the day's
                recent = [e for e in ok if (now - local(e["time"])).total_seconds() < 7200]
                hrs = min(2.0, (now - local(real[0]["time"])).total_seconds() / 3600)
                rate = len(recent) / hrs if hrs > 0.2 else 0
                if rate > 0:
                    fin = now + dt.timedelta(hours=left / rate)
                    eta = f", {left} to go: ~{left / rate:.1f} h at {rate:.1f}/h (about {fin:%H:%M})"
            out.append(f"Goal       [{bar}] {len(ok)}/{goal} successes today ({100 * len(ok) / goal:.0f}%)"
                       + (" DONE" if left <= 0 else eta))
    errs = [(e["place_err_mm"][0] ** 2 + e["place_err_mm"][1] ** 2) ** 0.5 for e in ok
            if e.get("place_err_mm") and not (e.get("read_vs_arm_mm") and
                                                (e["read_vs_arm_mm"][0] ** 2 + e["read_vs_arm_mm"][1] ** 2) ** 0.5 > 15)]
    if errs:
        errs.sort()
        out.append(f"Placement  median {errs[len(errs) // 2]:.1f} mm from the goal, worst {errs[-1]:.1f} mm "
                   f"(arm's own pose; picks re-anchored on the camera excluded)")

    # controller state
    out.append("")
    if STATUS.exists():
        s = json.loads(STATUS.read_text())
        age = time.time() - s.get("time", 0)
        state = "FROZEN" if s.get("frozen") else s.get("stage", "?")
        out.append(f"Controller {state}: {s.get('note', '')[:90]}  ({age / 60:.0f} min ago)")
        silent = time.time() - LOG.stat().st_mtime if LOG.exists() else 0
        if s.get("can") == "dead" or (silent > 120 and (s.get("torque") or s.get("stage") not in ("idle", None))):
            out.append(f"!!! WARNING: controller silent for {silent / 60:.0f} min with the motors possibly holding"
                       f" (or CAN link dead). Support the arm and switch its power off if the motors are warm. !!!")
        stalls = [ln for ln in lines if "CAN STALL" in ln or "CAN DEAD" in ln]
        if stalls:
            out.append(f"CAN        {len(stalls)} stall(s) in the log; last: {stalls[-1][:100]}")
    warm = [ln for ln in lines if re.match(r"\d\d:\d\d:\d\d (warning: motors warm|FROZEN: shoulder at)", ln)]
    if warm:
        out.append(f"Motor heat last: {warm[-1][:90]} (UTC)")
    re_a = [ln for ln in lines if "re-anchored the set-down pose" in ln]
    if re_a:
        out.append(f"Drift      {len(re_a)} reset(s) in the log; last {re_a[-1][:8]} UTC: "
                   f"{re.search(r'\((.*)\)', re_a[-1])[1]}")

    # recent picks
    out.append("")
    out.append("Recent picks (local time)")
    for e in today[-10:][::-1] if today else eps[-10:][::-1]:
        t = local(e["time"])
        if e.get("frames", 0) == 0:
            res = "skipped: " + (halt_before(lines, t) or "motor heat")
        elif e["success"]:
            pe = e.get("place_err_mm")
            rv = e.get("read_vs_arm_mm")
            res = "ok" + (f", {(pe[0] ** 2 + pe[1] ** 2) ** 0.5:4.1f} mm from goal" if pe else "")
            r = (rv[0] ** 2 + rv[1] ** 2) ** 0.5 if rv else 0.0
            if r > 15:  # re-anchored: the goal error then equals the read-vs-arm gap
                p = (pe[0] ** 2 + pe[1] ** 2) ** 0.5 if pe else 0.0
                res += f"  (drift reset {r:.0f} mm)" if abs(p - r) < 3 else f"  (camera read {r:.0f} mm off)"
        else:
            res = "FAILED: " + (halt_before(lines, t) or "?")
        out.append(f"  ep {e['episode']:>3}  {t:%H:%M}  {e.get('frames', 0):>5} fr  {res[:95]}")

    # training
    if TRAIN.exists():
        tl = [ln for ln in TRAIN.read_text().splitlines() if ln.strip()]
        age = (time.time() - TRAIN.stat().st_mtime) / 60
        out.append("")
        out.append(f"Training   (checked {age:.0f} min ago)")
        head = [ln for ln in tl if ln.startswith("job ") or ln.lstrip().startswith("step ")]
        rows = [ln for ln in tl if re.match(r"\s*\d+/\d+", ln)]
        rest = [ln for ln in tl if ln.startswith(("summary", "  ")) and ln not in rows and ln not in head]
        for ln in head + rows[-6:] + rest[-3:] + [ln for ln in tl if "no training steps" in ln][-1:]:
            out.append("  " + ln[:140])

    text = "\n".join(out)
    print(text)
    if a.append:
        with open(a.append, "a") as f:
            f.write(text + "\n\n")


if __name__ == "__main__":
    main()
