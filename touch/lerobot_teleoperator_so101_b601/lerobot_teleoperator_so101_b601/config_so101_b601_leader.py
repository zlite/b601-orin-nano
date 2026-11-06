from dataclasses import dataclass
from pathlib import Path

from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig


@TeleoperatorConfig.register_subclass("so101_b601_leader")
@dataclass
class SO101B601LeaderConfig(SOLeaderTeleopConfig):
    """SO-101 leader whose actions are expressed in reBot B601 follower joint degrees.

    Each shared joint is mapped linearly, ``follower = scale * leader + offset``, using the
    file written by ``python -m lerobot_teleoperator_so101_b601.map_joints``. The SO-101
    has no wrist yaw joint, so the B601 ``wrist_yaw`` is held at ``wrist_yaw_deg``.

    The underlying SO-101 calibration is shared with the plain ``so101_leader`` type: it is
    read from ``teleoperators/so_leader/<id>.json``.
    """

    # Joint mapping JSON. Defaults to teleoperators/so_leader/<id>_b601_mapping.json.
    mapping_file: Path | None = None

    # Fixed B601 wrist_yaw target in degrees (the SO-101 has no equivalent joint).
    wrist_yaw_deg: float = 0.0

    def __post_init__(self) -> None:
        # Mapping files are fitted in degrees; normalized ranges would invalidate them.
        self.use_degrees = True
