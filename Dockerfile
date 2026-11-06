# LeRobot policy server for the B601 plate policy on a Jetson Orin (JetPack 7 / L4T R39, driver 595, CUDA <= 13.2).
# Base: NVIDIA NGC PyTorch 26.05 (arm64): PyTorch 2.12, CUDA 13.2.1, Ubuntu 24.04 (Python 3.12, which this LeRobot needs).
# Do not move to 26.07+ unless the Jetson driver is upgraded: those need CUDA 13.3+.
FROM nvcr.io/nvidia/pytorch:26.05-py3

# PIP_CONSTRAINT is cleared so NGC's own pin file cannot block the versions below; torch/torchvision stay pinned
# through /opt/constraints.txt.
ENV PIP_INDEX_URL=https://pypi.org/simple \
    PIP_EXTRA_INDEX_URL= \
    PIP_CONSTRAINT= \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1

# libsndfile: the Python soundfile package is present and transformers imports it, which fails without the C library.
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*

# Keep the image's CUDA PyTorch: pin torch/torchvision to what is installed so pip never swaps in a CPU build.
RUN python3 -c "import torch, torchvision; print(f'torch=={torch.__version__}'); print(f'torchvision=={torchvision.__version__}')" \
        > /opt/constraints.txt && cat /opt/constraints.txt

# LeRobot at the robot PC's exact version (commit ae2b57f, copied from robot-setup/lerobot), without its
# dependency resolution (it would pull torchcodec and other extras the server does not use). The server's
# needs are installed explicitly below.
COPY lerobot /opt/lerobot-src
RUN pip install --no-deps /opt/lerobot-src && \
    pip install -c /opt/constraints.txt \
        "numpy>=2.0.0,<2.3.0" "opencv-python-headless>=4.9.0,<4.14.0" "Pillow>=10.0.0,<13.0.0" "einops>=0.8.0,<0.9.0" \
        "draccus>=0.11.6,<0.12.0" "huggingface-hub>=1.6.0,<2.0.0" "filelock>=3.12.0,<4.0.0" "fsspec>=2023.5.0" \
        "httpx>=0.27.0,<1.0.0" "requests>=2.32.0,<3.0.0" "gymnasium>=1.1.1,<2.0.0" "safetensors>=0.4.3,<1.0.0" \
        "packaging>=24.2,<26.0" "termcolor>=2.4.0,<4.0.0" "tqdm>=4.66.0,<5.0.0" \
        "transformers>=5.4.0,<5.6.0" "num2words>=0.5.14,<0.6.0" "psutil" \
        "datasets>=4.8.0,<5.0.0" "pandas>=2.0.0,<3.0.0" "pyarrow>=21.0.0,<30.0.0" "jsonlines>=4.0.0,<5.0.0" "av>=15.0.0,<16.0.0" \
        "eclipse-zenoh==1.10.2" "msgpack>=1.1.0,<2.0.0" "pyyaml" && \
    # accelerate is the only one here that requires torch, and pip will not accept NGC's pre-release torch
    # (2.12.0a0+...nv26.05) as satisfying torch>=2.0.0. Its other deps are in the list above.
    pip install --no-deps "accelerate>=1.14.0,<2.0.0" && \
    python3 -c "import torch; print('torch', torch.__version__, 'built with CUDA', torch.version.cuda); assert torch.version.cuda, 'torch was replaced by a CPU build'" && \
    python3 -c "import lerobot.remote_inference.server, lerobot.policies.smolvla.modeling_smolvla; print('lerobot server imports OK')"

# Arm side (same image, second container; see `jetson.sh check` / `jetson.sh rollout`):
# motorbridge drives the RobStride motors over SocketCAN, depthai + the OAK plugin read the wrist camera.
COPY robot/lerobot_camera_oak /opt/b601/robot/lerobot_camera_oak
RUN pip install -c /opt/constraints.txt "motorbridge>=0.5,<0.6" "depthai>=3.0,<4" && \
    pip install --no-deps -e /opt/b601/robot/lerobot_camera_oak && \
    python3 -c "import motorbridge, depthai, lerobot_camera_oak, lerobot.robots.rebot_b601_follower; print('arm imports OK, depthai', depthai.__version__)"
COPY robot/b601_rollout.py /opt/b601/robot/

COPY server.yaml bench.py /opt/b601/
WORKDIR /opt/b601
CMD ["lerobot-policy-server", "--config_path=/opt/b601/server.yaml"]
