"""
OpenCV-backed webcam wrapper.

On Windows we try multiple backends in order (DirectShow first, then Media
Foundation, then "any"). Some drivers reject one and accept the other; at
least one almost always works if a camera is physically available.
"""
from __future__ import annotations

import sys
import threading

import cv2
import numpy as np

from src.interface.logger import ILogger


class CameraNotAvailableError(RuntimeError):
    pass


def _candidate_backends() -> list[tuple[str, int]]:
    if sys.platform == "win32":
        return [
            ("CAP_DSHOW", cv2.CAP_DSHOW),
            ("CAP_MSMF", cv2.CAP_MSMF),
            ("CAP_ANY", cv2.CAP_ANY),
        ]
    return [("CAP_ANY", cv2.CAP_ANY)]


class OpenCVCamera:
    def __init__(
        self,
        logger: ILogger,
        device_index: int = 0,
        width: int = 1280,
        height: int = 720,
        target_fps: int = 30,
        flip_horizontal: bool = True,
    ) -> None:
        self._logger = logger
        self._device_index = device_index
        self._width = width
        self._height = height
        self._target_fps = target_fps
        self._flip_horizontal = flip_horizontal

        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._cap is not None and self._cap.isOpened()

    def open(self) -> None:
        with self._lock:
            if self._cap is not None and self._cap.isOpened():
                return

            attempted: list[str] = []
            for name, backend in _candidate_backends():
                attempted.append(name)
                cap = cv2.VideoCapture(self._device_index, backend)
                if not cap.isOpened():
                    cap.release()
                    self._logger.debug(
                        "camera.backend_rejected",
                        device_index=self._device_index,
                        backend=name,
                    )
                    continue

                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
                cap.set(cv2.CAP_PROP_FPS, self._target_fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                self._cap = cap
                self._logger.info(
                    "camera.opened",
                    device_index=self._device_index,
                    backend=name,
                    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    fps=cap.get(cv2.CAP_PROP_FPS),
                )
                return

            self._logger.error(
                "camera.open_failed",
                device_index=self._device_index,
                tried=attempted,
            )
            raise CameraNotAvailableError(
                f"Could not open camera {self._device_index}. "
                "Run `python scripts/probe_camera.py` to find a device that works."
            )

    def read(self) -> np.ndarray | None:
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                return None
            ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        if self._flip_horizontal:
            frame = cv2.flip(frame, 1)
        return frame

    def close(self) -> None:
        with self._lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                finally:
                    self._cap = None
                    self._logger.info("camera.closed")
