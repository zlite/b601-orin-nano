"""LeRobot camera backed by a Luxonis OAK device (DepthAI v3)."""

import datetime
import logging
import os
import time
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

import cv2
import depthai as dai
import numpy as np
from numpy.typing import NDArray

from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import ColorMode
from lerobot.cameras.utils import get_cv2_rotation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config_oak import OAKCameraConfig

logger = logging.getLogger(__name__)

_QUEUE_TIMEOUT = datetime.timedelta(milliseconds=500)

# When set, each camera also writes its latest frame as <dir>/<device_id>_<socket>.jpg a few
# times per second, so another process (e.g. the robot-setup web app) can show a live view
# while LeRobot owns the device.
PREVIEW_DIR_ENV = "LEROBOT_OAK_PREVIEW_DIR"
_PREVIEW_PERIOD_S = 0.2


class OAKCamera(Camera):
    """Streams one color sensor of an OAK device on a background thread.

    Frames are delivered as ``(height, width, 3)`` uint8 arrays in the configured color
    order, with rotation applied, matching LeRobot's OpenCV camera.
    """

    def __init__(self, config: OAKCameraConfig):
        super().__init__(config)
        self.config = config
        self.device_id = config.device_id
        self.rotation = get_cv2_rotation(config.rotation)

        # Sensor output size before host rotation.
        if config.rotation in (90, -90):
            self.capture_size = (config.height, config.width)
        else:
            self.capture_size = (config.width, config.height)

        self.device: dai.Device | None = None
        self.pipeline: dai.Pipeline | None = None
        self.queue: dai.MessageQueue | None = None
        self.usb_speed: str | None = None

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None
        self.new_frame_event: Event = Event()

        preview_dir = os.environ.get(PREVIEW_DIR_ENV)
        self.preview_path = (
            Path(preview_dir) / f"{config.device_id or 'first'}_{config.socket}.jpg" if preview_dir else None
        )
        self._last_preview_time = 0.0

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.device_id or 'first'}:{self.config.socket})"

    @property
    def is_connected(self) -> bool:
        return self.pipeline is not None and self.pipeline.isRunning()

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """List OAK devices visible to DepthAI. Devices already in use are not reported."""
        found = []
        for info in dai.Device.getAllAvailableDevices():
            found.append(
                {
                    "name": info.name,
                    "type": "oak",
                    "id": info.getDeviceId(),
                    "state": info.state.name,
                    "protocol": info.protocol.name,
                }
            )
        return found

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        info = dai.DeviceInfo(self.device_id) if self.device_id else None
        try:
            self.device = dai.Device(info) if info is not None else dai.Device()
        except RuntimeError as e:
            raise ConnectionError(
                f"Failed to open {self}. Check the device ID, udev rules and that no other "
                f"process is using it. Available: {self.find_cameras()}"
            ) from e

        self.usb_speed = self.device.getUsbSpeed().name
        socket = getattr(dai.CameraBoardSocket, self.config.socket)
        connected = self.device.getConnectedCameras()
        if socket not in connected:
            names = [s.name for s in connected]
            self.device.close()
            self.device = None
            raise ValueError(f"{self}: socket {self.config.socket} not present; this device has {names}.")

        self.pipeline = dai.Pipeline(self.device)
        cam = self.pipeline.create(dai.node.Camera).build(socket)
        if self.config.encoding == "mjpeg":
            out = cam.requestOutput(self.capture_size, type=dai.ImgFrame.Type.NV12, fps=self.fps)
            encoder = self.pipeline.create(dai.node.VideoEncoder).build(
                out,
                frameRate=float(self.fps),
                profile=dai.VideoEncoderProperties.Profile.MJPEG,
                quality=self.config.mjpeg_quality,
            )
            out = encoder.out
        else:
            out = cam.requestOutput(self.capture_size, type=dai.ImgFrame.Type.BGR888i, fps=self.fps)
        self.queue = out.createOutputQueue(maxSize=2, blocking=False)
        self.pipeline.start()

        if self.config.encoding == "raw" and self.usb_speed not in ("SUPER", "SUPER_PLUS"):
            logger.warning(
                f"{self} is linked at USB {self.usb_speed}. Use a USB 3 port/cable or encoding='mjpeg'."
            )

        self._start_read_thread()

        if warmup and self.config.warmup_s > 0:
            deadline = time.time() + self.config.warmup_s
            while time.time() < deadline:
                try:
                    self.async_read(timeout_ms=2000)
                except TimeoutError:
                    pass
            if self.latest_frame is None:
                self.disconnect()
                raise ConnectionError(f"{self} produced no frames during warmup.")

        logger.info(f"{self} connected (USB {self.usb_speed}, {self.config.encoding}).")

    def _decode(self, msg: Any) -> NDArray[Any]:
        if self.config.encoding == "mjpeg":
            image = cv2.imdecode(np.asarray(msg.getData(), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"{self} failed to decode MJPEG frame.")
        else:
            image = msg.getCvFrame()

        if self.config.color_mode == ColorMode.RGB:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.rotation is not None:
            image = cv2.rotate(image, self.rotation)

        h, w = image.shape[:2]
        if (h, w) != (self.height, self.width):
            raise RuntimeError(f"{self} frame is {w}x{h}, expected {self.width}x{self.height}.")
        return image

    def _read_from_hardware(self) -> NDArray[Any] | None:
        if self.queue is None:
            raise RuntimeError(f"{self} has no output queue.")
        msg = self.queue.get(_QUEUE_TIMEOUT)
        if msg is None:
            return None
        image = self._decode(msg)
        if self.preview_path is not None and time.monotonic() - self._last_preview_time > _PREVIEW_PERIOD_S:
            self._last_preview_time = time.monotonic()
            self._write_preview(msg, image)
        return image

    def _write_preview(self, msg: Any, image: NDArray[Any]) -> None:
        if self.config.encoding == "mjpeg" and self.rotation is None:
            data = bytes(msg.getData())
        else:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if self.config.color_mode == ColorMode.RGB else image
            data = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
        try:
            self.preview_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.preview_path.with_suffix(".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, self.preview_path)
        except OSError as e:
            logger.debug(f"{self} could not write preview: {e}")

    @check_if_not_connected
    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any]:
        """Block until the next frame arrives and return it."""
        if color_mode is not None and color_mode != self.config.color_mode:
            raise ValueError(f"{self} is configured for {self.config.color_mode.value}; got {color_mode}.")
        return self.async_read(timeout_ms=2000)

    def _read_loop(self) -> None:
        if self.stop_event is None:
            raise RuntimeError(f"{self}: stop_event is not initialized.")

        failures = 0
        while not self.stop_event.is_set():
            try:
                frame = self._read_from_hardware()
                if frame is None:
                    continue
                with self.frame_lock:
                    self.latest_frame = frame
                    self.latest_timestamp = time.perf_counter()
                self.new_frame_event.set()
                failures = 0
            except Exception as e:
                if self.stop_event.is_set():
                    break
                failures += 1
                logger.warning(f"Error reading frame in background thread for {self}: {e}")
                if failures >= 10:
                    logger.error(f"{self} read thread stopping after repeated errors.")
                    break

    def _start_read_thread(self) -> None:
        self._stop_read_thread()
        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, name=f"{self}_read_loop", daemon=True)
        self.thread.start()

    def _stop_read_thread(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None
        self.stop_event = None
        with self.frame_lock:
            self.latest_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """Return the latest unconsumed frame, waiting up to ``timeout_ms`` for a new one."""
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(f"Timed out waiting for frame from camera {self} after {timeout_ms} ms.")

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")
        return frame

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        """Return the most recent frame without waiting."""
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        with self.frame_lock:
            frame = self.latest_frame
            timestamp = self.latest_timestamp

        if frame is None or timestamp is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")

        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(f"{self} latest frame is too old: {age_ms:.1f} ms (max allowed: {max_age_ms} ms).")
        return frame

    def disconnect(self) -> None:
        self._stop_read_thread()
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                logger.exception(f"Error stopping pipeline for {self}.")
        if self.device is not None and not self.device.isClosed():
            self.device.close()
        self.pipeline = None
        self.queue = None
        self.device = None
        logger.info(f"{self} disconnected.")
