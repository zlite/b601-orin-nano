"""Time the plate policy on this GPU: one action chunk (50 actions = 5 s of motion at 10 Hz) per call.

    python3 bench.py [n_calls]

Runs inside the container (jetson.sh bench). Uses dummy inputs shaped like the robot's: 7 joint values,
one 200x320 wrist image, the training task text. Prints load time, per-call times and GPU memory,
for full precision and for bfloat16 autocast.
"""
import sys
import time

import torch

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
TASK = "pick up the 96-well plate and put it down somewhere else on the table\n"
assert torch.cuda.is_available(), "CUDA not available: run the container with --runtime nvidia"
print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__, "| CUDA", torch.version.cuda, flush=True)

t = time.time()
p = SmolVLAPolicy.from_pretrained("/opt/b601/checkpoint").to("cuda").eval()
print(f"model load {time.time() - t:.1f} s, {sum(x.numel() for x in p.parameters()) / 1e6:.0f}M params", flush=True)
tok = p.model.vlm_with_expert.processor.tokenizer([TASK], padding="max_length",
                                                 max_length=p.config.tokenizer_max_length, return_tensors="pt")
batch = {"observation.state": torch.rand(1, 7, device="cuda"),
         "observation.images.camera1": torch.rand(1, 3, 200, 320, device="cuda"),
         "observation.language.tokens": tok["input_ids"].cuda(),
         "observation.language.attention_mask": tok["attention_mask"].bool().cuda()}

for label, ctx in (("float32", torch.autocast("cuda", enabled=False)),
                   ("bfloat16 autocast", torch.autocast("cuda", dtype=torch.bfloat16))):
    times = []
    with torch.inference_mode(), ctx:
        for i in range(N + 2):
            torch.cuda.synchronize()
            t = time.perf_counter()
            a = p.predict_action_chunk(batch)
            torch.cuda.synchronize()
            if i >= 2:  # first calls include CUDA warmup
                times.append(time.perf_counter() - t)
    times.sort()
    print(f"{label}: chunk {tuple(a.shape)}; median {times[len(times) // 2] * 1000:.0f} ms, "
          f"max {times[-1] * 1000:.0f} ms over {N} calls", flush=True)
print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
print("Budget: each chunk is 5 s of motion; async inference needs a call well under that (target < 1.5 s).")
