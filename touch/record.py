"""Record the agent's own pick/place runs as a LeRobot dataset (for policy training).

Frames are sampled from the agent's command stream, not a separate thread: every motion command
refreshes the motors' cached states, so sampling there adds no CAN traffic. Each frame holds the
measured joints (observation.state), the commanded joints (action) and the camera images (stored
at 320x200; the policies resize to 224-512 px anyway).

Frames stream straight into the dataset (an episode in RAM would be ~1 GB). Every episode is kept;
episode_outcomes.json says which succeeded, so training can select episodes.
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "gripper")
ROOT = Path(__file__).resolve().parent / "data" / "lerobot"
IMG_W, IMG_H = 320, 200


def features(cams) -> dict:
    names = [f"{j}.pos" for j in JOINTS]
    f = {"observation.state": {"dtype": "float32", "shape": (len(JOINTS),), "names": names},
         "action": {"dtype": "float32", "shape": (len(JOINTS),), "names": names}}
    for cam in cams:
        f[f"observation.images.{cam}"] = {"dtype": "video", "shape": (IMG_H, IMG_W, 3),
                                          "names": ["height", "width", "channels"]}
    return f


class EpisodeRecorder:
    def __init__(self, name: str, cams=("wrist",), fps: int = 10, user: str = "local"):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # heavy import, only when recording

        self.fps, self.cams = fps, tuple(cams)
        self.root = ROOT / name
        self.repo_id = f"{user}/{name}"
        if (self.root / "meta" / "info.json").exists():
            self.ds = LeRobotDataset.resume(self.repo_id, root=self.root, image_writer_threads=2)
        else:
            self.ds = LeRobotDataset.create(self.repo_id, fps, features(self.cams), root=self.root,
                                            robot_type="rebot_b601_rs", use_videos=True, image_writer_threads=2)
        self.outcomes_path = self.root / "episode_outcomes.json"
        self.outcomes = json.loads(self.outcomes_path.read_text()) if self.outcomes_path.exists() else []
        self.task, self.n, self.last, self.info = None, 0, 0.0, {}

    @property
    def active(self) -> bool:
        return self.task is not None

    def start(self, task: str, **info):
        self.task, self.n, self.last, self.info = task, 0, 0.0, info

    def sample(self, state: dict, action: dict, images: dict):
        """Call often (each motion command); keeps one frame per 1/fps s. images: BGR frames per camera."""
        if self.task is None:
            return
        now = time.monotonic()
        if now - self.last < 1.0 / self.fps - 1e-3:
            return
        frame = {"observation.state": np.array([state[j] for j in JOINTS], np.float32),
                 "action": np.array([action.get(j, state[j]) for j in JOINTS], np.float32), "task": self.task}
        for cam in self.cams:
            img = images.get(cam)
            if img is None:
                return  # drop the whole frame rather than record a gap
            small = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
            frame[f"observation.images.{cam}"] = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        self.ds.add_frame(frame)
        self.last = now
        self.n += 1

    def stop(self, success: bool, **info) -> int:
        """Close the episode; every episode is saved, with its outcome in episode_outcomes.json."""
        if self.task is None:
            return 0
        n = self.n
        if n >= 5:
            self.ds.save_episode()
            # Finalize every episode: the parquet footers are only written there, and the 2026-10-20 reboot
            # (agent killed before close()) left the whole dataset unreadable. Each episode gets its own files.
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            self.ds.finalize()
            self.ds = LeRobotDataset.resume(self.repo_id, root=self.root, image_writer_threads=2)
            self.outcomes.append({"episode": len(self.outcomes), "task": self.task, "success": bool(success),
                                  "frames": n, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **self.info, **info})
            self.outcomes_path.write_text(json.dumps(self.outcomes, indent=1))
        else:
            self.ds.clear_episode_buffer()
        self.task = None
        return n

    def close(self):
        self.ds.finalize()
