from dataclasses import dataclass

from lerobot.cameras.configs import CameraConfig, ColorMode, Cv2Rotation

__all__ = ["OAKCameraConfig", "ColorMode", "Cv2Rotation"]


@CameraConfig.register_subclass("oak")
@dataclass
class OAKCameraConfig(CameraConfig):
    """Configuration for a Luxonis OAK camera driven through DepthAI v3.

    Example (CLI):
        --robot.cameras='{wrist: {type: oak, device_id: "1944301011B4F01300", socket: CAM_B,
                                  width: 640, height: 400, fps: 30}}'

    Attributes:
        device_id: DepthAI device ID (MXID) or name. ``None`` opens the first free device,
            which is only safe when a single OAK is connected. List IDs with
            ``python -m lerobot_camera_oak.find``.
        socket: Board socket of the sensor to stream. OAK-1 and most OAK-D models have the
            color sensor on ``CAM_A``. The OAK-D SR has no CAM_A; its two color OV9782
            sensors are ``CAM_B`` (left) and ``CAM_C`` (right).
        encoding: ``"raw"`` streams uncompressed BGR frames; ``"mjpeg"`` encodes on the
            device and decodes on the host, which cuts USB bandwidth roughly 10x. Use
            ``mjpeg`` when several cameras share a USB 2 link.
        mjpeg_quality: JPEG quality (1-100) used when ``encoding="mjpeg"``.
        color_mode: Output color order. LeRobot expects RGB.
        rotation: Image rotation applied on the host.
        warmup_s: Seconds spent reading frames in ``connect`` so auto-exposure settles.

    When ``fps``, ``width`` and ``height`` are all omitted they default to 640x480 at 30 fps.
    """

    device_id: str | None = None
    socket: str = "CAM_A"
    encoding: str = "raw"
    mjpeg_quality: int = 90
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: float = 1.0
    # Optional sensor output size. When set (both), frames are captured at this size and resized on the
    # host to width x height with cv2.INTER_AREA, e.g. capture 640x400, deliver 320x200 (how the B601
    # plate dataset was made). Unset: captured at width x height directly.
    capture_width: int | None = None
    capture_height: int | None = None

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        self.rotation = Cv2Rotation(self.rotation)
        self.socket = self.socket.upper()
        if self.socket not in {"CAM_A", "CAM_B", "CAM_C", "CAM_D"}:
            raise ValueError(f"`socket` must be CAM_A, CAM_B, CAM_C or CAM_D, got '{self.socket}'.")
        if self.encoding not in {"raw", "mjpeg"}:
            raise ValueError(f"`encoding` must be 'raw' or 'mjpeg', got '{self.encoding}'.")
        values = (self.fps, self.width, self.height)
        if all(v is None for v in values):
            self.fps, self.width, self.height = 30, 640, 480
        elif any(v is None for v in values):
            raise ValueError("For `fps`, `width` and `height`, either all must be set or none.")
        if (self.capture_width is None) != (self.capture_height is None):
            raise ValueError("Set both `capture_width` and `capture_height`, or neither.")
