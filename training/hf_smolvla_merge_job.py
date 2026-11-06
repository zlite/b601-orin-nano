# /// script
# requires-python = ">=3.12"
# dependencies = ["lerobot[smolvla,training] @ git+https://github.com/huggingface/lerobot@ae2b57f12223c8c5184fcd814eb63cd9ce7de203"]
# ///
"""HF Jobs entry point: merge several LeRobot datasets (selected episodes of each), fine-tune SmolVLA on the
result and push the policy to the Hub. Same LeRobot commit and training flags as touch/hf_smolvla_job.py.

Environment:
  SOURCES      JSON list of {"repo": "user/name", "episodes": [..] or null (= all)}
  POLICY_REPO  where the policy is pushed (private)
  BASE_POLICY  policy to continue from (default lerobot/smolvla_base)
  STEPS, BATCH, JOB_NAME
"""
import json
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from lerobot.datasets.aggregate import aggregate_datasets

sources = json.loads(os.environ["SOURCES"])
job = os.environ.get("JOB_NAME", "smolvla_merged")
work = Path("/tmp/work")
roots, repo_ids, episodes, offset = [], [], [], 0
for s in sources:
    root = work / "src" / s["repo"].replace("/", "__")
    snapshot_download(s["repo"], repo_type="dataset", local_dir=root)
    total = json.loads((root / "meta" / "info.json").read_text())["total_episodes"]
    keep = list(range(total)) if s.get("episodes") is None else s["episodes"]
    episodes += [offset + e for e in keep]
    print(f"[merge] {s['repo']}: {total} episodes, keeping {len(keep)} (merged indices {offset}..{offset + total - 1})",
          flush=True)
    roots.append(root)
    repo_ids.append(s["repo"])
    offset += total

merged = work / "merged"
aggregate_datasets(repo_ids=repo_ids, aggr_repo_id="local/merged", roots=roots, aggr_root=merged,
                   concatenate_videos=False)
print(f"[merge] {offset} episodes in {merged}; training on {len(episodes)}", flush=True)

out = Path("/tmp/outputs") / job
cmd = [
    "lerobot-train",
    f"--policy.path={os.environ.get('BASE_POLICY', 'lerobot/smolvla_base')}",
    f"--policy.repo_id={os.environ['POLICY_REPO']}",
    "--policy.push_to_hub=true",
    "--policy.private=true",
    "--policy.device=cuda",
    '--rename_map={"observation.images.wrist": "observation.images.camera1"}',
    "--dataset.repo_id=local/merged",
    f"--dataset.root={merged}",
    f"--dataset.episodes={json.dumps(episodes)}",
    "--dataset.video_backend=pyav",
    f"--steps={os.environ.get('STEPS', '8000')}",
    f"--batch_size={os.environ.get('BATCH', '64')}",
    "--save_freq=4000",
    "--num_workers=8",
    f"--output_dir={out}",
    f"--job_name={job}",
    "--wandb.enable=false",
]
print(" ".join(cmd), flush=True)
sys.exit(subprocess.call(cmd))
