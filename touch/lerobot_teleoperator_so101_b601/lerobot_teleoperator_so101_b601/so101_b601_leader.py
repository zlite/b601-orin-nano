"""SO-101 leader arm producing reBot B601 follower actions."""

import json
import logging
from pathlib import Path

from lerobot.teleoperators.so_leader.so_leader import SOLeader

from .config_so101_b601_leader import SO101B601LeaderConfig

logger = logging.getLogger(__name__)

# SO-101 joints that have a B601 counterpart with the same name.
MAPPED_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")

# B601 action order.
B601_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "gripper")


def default_mapping_path(calibration_dir: Path, leader_id: str | None) -> Path:
    return calibration_dir / f"{leader_id}_b601_mapping.json"


def load_mapping(path: Path) -> dict[str, tuple[float, float]]:
    """Return {joint: (scale, offset)} from a mapping file."""
    data = json.loads(path.read_text())
    joints = data["joints"]
    missing = [j for j in MAPPED_JOINTS if j not in joints]
    if missing:
        raise ValueError(f"Mapping file {path} is missing joints: {missing}")
    return {j: (float(joints[j]["scale"]), float(joints[j]["offset"])) for j in MAPPED_JOINTS}


class SO101B601Leader(SOLeader):
    """SO-101 leader whose ``get_action`` returns B601 joint targets in degrees.

    ``name`` is inherited from ``SOLeader`` so the SO-101 motor calibration is shared with
    ``lerobot-calibrate --teleop.type=so101_leader``.
    """

    config_class = SO101B601LeaderConfig

    def __init__(self, config: SO101B601LeaderConfig):
        super().__init__(config)
        self.config = config
        self.mapping_path = config.mapping_file or default_mapping_path(self.calibration_dir, self.id)
        self.mapping: dict[str, tuple[float, float]] | None = None

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{joint}.pos": float for joint in B601_JOINTS}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    def connect(self, calibrate: bool = True) -> None:
        if not self.mapping_path.is_file():
            raise FileNotFoundError(
                f"No SO-101 -> B601 joint mapping at {self.mapping_path}. Run "
                f"`python -m lerobot_teleoperator_so101_b601.map_joints` (see robot-setup/robot.sh map)."
            )
        self.mapping = load_mapping(self.mapping_path)
        logger.info(f"Loaded SO-101 -> B601 mapping from {self.mapping_path}")
        super().connect(calibrate=calibrate)

    def get_action(self) -> dict[str, float]:
        if self.mapping is None:
            raise RuntimeError(f"{self} has no joint mapping loaded; call connect() first.")
        leader = super().get_action()
        action = {}
        for joint in B601_JOINTS:
            if joint == "wrist_yaw":
                action["wrist_yaw.pos"] = self.config.wrist_yaw_deg
            else:
                scale, offset = self.mapping[joint]
                action[f"{joint}.pos"] = scale * leader[f"{joint}.pos"] + offset
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # Force feedback would need the inverse mapping; the SO-101 leader is passive here.
        pass
