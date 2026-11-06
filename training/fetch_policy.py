"""Download a fine-tuned policy for the server: python3 fetch_policy.py REPO DIR (inside b601-policy, HF cache mounted).
Applies the v3 fix: config.json observation.state shape 6 -> 7 (the training data has 7 joints)."""
import json, sys
from pathlib import Path
from huggingface_hub import snapshot_download
repo, out = sys.argv[1], Path(sys.argv[2])
snapshot_download(repo, local_dir=out)
cfg = out / "config.json"
c = json.loads(cfg.read_text())
s = c["input_features"]["observation.state"]["shape"]
if s != [7]:
    c["input_features"]["observation.state"]["shape"] = [7]
    cfg.write_text(json.dumps(c, indent=4))
    print(f"config.json: observation.state shape {s} -> [7]")
print(f"{repo} -> {out}")
