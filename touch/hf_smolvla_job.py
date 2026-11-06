# /// script
# requires-python = ">=3.12"
# dependencies = ["lerobot[smolvla,training] @ git+https://github.com/huggingface/lerobot@ae2b57f12223c8c5184fcd814eb63cd9ce7de203"]
# ///
"""HF Jobs entry point: fine-tune SmolVLA on the plate_shuttle dataset and push the policy to the Hub.

Settings come from the environment: DATASET, EPISODES (JSON list), POLICY_REPO, STEPS, BATCH, and optional
BASE_POLICY (default lerobot/smolvla_base; e.g. a previous fine-tune to continue from) and JOB_NAME.
Optional CKPT_DIR (a bucket mounted read-write, e.g. -v hf://buckets/chr1sa/jobs-artifacts:/ckpt) gets a copy
of every finished checkpoint under CKPT_DIR/RUN_NAME; relaunching with the same RUN_NAME resumes from the newest.
"""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

JOB = os.environ.get("JOB_NAME", "smolvla_plate_shuttle")
OUT = Path("/tmp/outputs") / JOB
ckpt_root = os.environ.get("CKPT_DIR")
store = Path(ckpt_root) / os.environ.get("RUN_NAME", "smolvla_plate_shuttle") if ckpt_root else None


def saved_steps(d):
    return sorted(p for p in d.iterdir() if p.name.isdigit() and (p / "pretrained_model" / "train_config.json").exists()) if d.is_dir() else []


def mirror(stop):
    # copy each new local checkpoint to the bucket once lerobot has finished writing it
    done = {p.name for p in saved_steps(store)}
    while True:
        last = stop.is_set()
        for p in saved_steps(OUT / "checkpoints"):
            if p.name in done or (not last and time.time() - max(f.stat().st_mtime for f in p.rglob("*")) < 60):
                continue
            tmp = store / f".{p.name}.part"
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.copytree(p, tmp)
            tmp.rename(store / p.name)
            done.add(p.name)
            print(f"[ckpt] copied step {p.name} to {store}", flush=True)
        if last:
            return
        stop.wait(30)


episodes = json.loads(os.environ["EPISODES"])
cmd = [
    "lerobot-train",
    f"--policy.path={os.environ.get('BASE_POLICY', 'lerobot/smolvla_base')}",
    f"--policy.repo_id={os.environ['POLICY_REPO']}",
    "--policy.push_to_hub=true",
    "--policy.private=true",
    "--policy.device=cuda",
    # one wrist camera; SmolVLA base skips its absent camera2/3
    '--rename_map={"observation.images.wrist": "observation.images.camera1"}',
    f"--dataset.repo_id={os.environ['DATASET']}",
    f"--dataset.episodes={json.dumps(episodes)}",
    "--dataset.video_backend=pyav",
    f"--steps={os.environ.get('STEPS', '20000')}",
    f"--batch_size={os.environ.get('BATCH', '64')}",
    "--save_freq=5000",
    "--num_workers=8",
    f"--output_dir={OUT}",
    f"--job_name={JOB}",
    "--wandb.enable=false",
]

if store:
    store.mkdir(parents=True, exist_ok=True)
    prev = saved_steps(store)
    if prev:
        # restore the newest checkpoint locally (with the `last` link lerobot expects) and resume from it
        ck = OUT / "checkpoints"
        ck.mkdir(parents=True, exist_ok=True)
        shutil.copytree(prev[-1], ck / prev[-1].name)
        (ck / "last").symlink_to(prev[-1].name)
        cfg = ck / prev[-1].name / "pretrained_model" / "train_config.json"
        cmd = ["lerobot-train", f"--config_path={cfg}", "--resume=true", f"--output_dir={OUT}",
               f"--steps={os.environ.get('STEPS', '20000')}"]
        print(f"[ckpt] resuming from step {prev[-1].name}", flush=True)

print(" ".join(cmd), flush=True)
stop = threading.Event()
watcher = threading.Thread(target=mirror, args=(stop,)) if store else None
if watcher:
    watcher.start()
rc = subprocess.call(cmd)
if watcher:
    stop.set()
    watcher.join()
sys.exit(rc)
