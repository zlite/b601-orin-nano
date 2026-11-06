"""Upload the Jetson-recorded episodes and start a SmolVLA fine-tune on HF Jobs (v3's episodes + the new successes).

Run inside the b601-policy image with the HF cache mounted:
  docker run --rm -v ~/.cache/huggingface:/root/.cache/huggingface -v $PWD:/w b601-policy python3 /w/training/launch.py
Prints the job id and URL. Settings below.
"""
import json
from pathlib import Path

from huggingface_hub import HfApi, get_token

USER = "chr1sa"
LOCAL = Path("/w/touch/data/lerobot/plate_jetson")
NEW_REPO = f"{USER}/plate_jetson"
BASE_POLICY = f"{USER}/smolvla_plate_random_v3"
import os
POLICY_REPO = os.environ.get("POLICY_REPO", f"{USER}/smolvla_plate_jetson_v5")  # v4: pad -> pad only; v5 adds pad <-> holder A
STEPS, BATCH, FLAVOR = int(os.environ.get("STEPS", 10000)), 64, os.environ.get("FLAVOR", "l40sx1")  # v5: ~2.5x the Jetson episodes of v4   # ~0.8 h at v3's 3 steps/s on an L40S (~$1.8/h)

api = HfApi()
outcomes = json.loads((LOCAL / "episode_outcomes.json").read_text())
ok = [o["episode"] for o in outcomes if o["success"]]
print(f"{len(outcomes)} Jetson episodes, {len(ok)} successful: {ok}")
if not ok:
    raise SystemExit("no successful episodes to train on")

api.create_repo(NEW_REPO, repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(folder_path=str(LOCAL), repo_id=NEW_REPO, repo_type="dataset",
                  ignore_patterns=["images/**"], commit_message=f"{len(outcomes)} episodes recorded on the Jetson")
print(f"uploaded {LOCAL} -> {NEW_REPO}")

v3_eps = json.loads(Path("/w/training/v3_episodes.json").read_text())
sources = [{"repo": f"{USER}/plate_random", "episodes": v3_eps}, {"repo": NEW_REPO, "episodes": ok}]
job = api.run_uv_job(
    "/w/training/hf_smolvla_merge_job.py",
    flavor=FLAVOR, timeout="3h",
    env={"SOURCES": json.dumps(sources), "BASE_POLICY": BASE_POLICY, "POLICY_REPO": POLICY_REPO,
         "STEPS": str(STEPS), "BATCH": str(BATCH), "JOB_NAME": POLICY_REPO.split("/")[1]},
    secrets={"HF_TOKEN": get_token()},
)
print(f"job {job.id} ({FLAVOR}, {STEPS} steps): {job.url}")
Path("/w/training/last_job.json").write_text(json.dumps({"job_id": job.id, "flavor": FLAVOR, "steps": STEPS,
                                                          "policy_repo": POLICY_REPO, "url": job.url}))
print("follow it with: ./jetson.sh train-progress   (loss, lr, speed, ETA and cost every 200 steps)")
