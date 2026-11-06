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

HELD_BAND = (-275.0, -225.0)  # gripper.pos with the plate in the jaws (open -294..-324, closed -21)
GRIP_MAX_CLOSE_DEG = -240.0   # agent.py: never command further closed than this on the plate


def squeeze_carry(root: Path, deg: float):
    """Relabel the carry: where jaws and command are both at the plate's width, command `deg` past the measured
    jaws (capped at GRIP_MAX_CLOSE_DEG, never looser than recorded). The demos carried with the command only ~3 deg
    past the jaws, so the policy learned an absolute ~-250 and its grasps did not squeeze when the plate met the
    jaws wider (v6 trials 1 and 7, 2026-11-06). agent.py squeezes 8 deg past contact (GRIP_SQUEEZE_DEG)."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    lo, hi = HELD_BAND
    changed = total = 0
    grips = []
    for f in sorted((root / "data").glob("*/*.parquet")):
        t = pq.read_table(f)
        a = np.array(t.column("action").combine_chunks().flatten(), dtype=np.float32).reshape(-1, 7)
        s = np.array(t.column("observation.state").combine_chunks().flatten(), dtype=np.float32).reshape(-1, 7)
        m = (s[:, 6] >= lo) & (s[:, 6] <= hi) & (a[:, 6] >= lo) & (a[:, 6] <= hi)
        new = np.maximum(a[:, 6], np.minimum(s[:, 6] + deg, GRIP_MAX_CLOSE_DEG))
        changed += int((m & (new != a[:, 6])).sum())
        total += len(a)
        a[m, 6] = new[m]
        grips.append(a[:, 6].copy())
        col = pa.FixedSizeListArray.from_arrays(pa.array(a.reshape(-1), pa.float32()), 7)
        i = t.schema.get_field_index("action")
        pq.write_table(t.set_column(i, t.schema.field(i), col), f)
    g = np.concatenate(grips)
    sp = root / "meta" / "stats.json"
    st = json.loads(sp.read_text())
    act = st["action"]
    for k, v in (("min", g.min()), ("max", g.max()), ("mean", g.mean()), ("std", g.std())):
        act[k][6] = float(v)
    for k in act:
        if k.startswith("q") and k[1:].isdigit():
            act[k][6] = float(np.quantile(g, int(k[1:]) / 100))
    sp.write_text(json.dumps(st, indent=4))
    print(f"[squeeze] {root.name}: {changed}/{total} frames' gripper action moved to jaws+{deg:g} "
          f"(cap {GRIP_MAX_CLOSE_DEG:g})", flush=True)


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
    if os.environ.get("GRIP_SQUEEZE_DEG"):
        squeeze_carry(root, float(os.environ["GRIP_SQUEEZE_DEG"]))
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
