"""Luxonis OAK camera plugin for LeRobot (camera type: ``oak``)."""

from .config_oak import OAKCameraConfig
from .oak import OAKCamera

__all__ = ["OAKCamera", "OAKCameraConfig"]
