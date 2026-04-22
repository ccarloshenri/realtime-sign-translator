"""
MediaPipe Hands adapter (Tasks API).

Uses `mediapipe.tasks.python.vision.HandLandmarker` — the legacy
`mediapipe.solutions.hands` API was removed from the Python 3.12 wheels,
so this new backend is required on modern Python.

The asset file (`hand_landmarker.task`) is downloaded separately by
`scripts/download_models.py` into `artifacts/`.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from src.interface.logger import ILogger
from src.models.confidence import Confidence
from src.models.hand_landmarks import HandLandmarks
from src.models.handedness import Handedness


class MediaPipeHandLandmarkExtractor:
    def __init__(
        self,
        logger: ILogger,
        model_path: str | Path,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,  # kept for backward config compat; unused
    ) -> None:
        self._logger = logger
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe model asset not found at {model_path}. "
                "Run `python scripts/download_models.py` first."
            )

        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(options)
        self._lock = threading.Lock()
        self._closed = False
        self._start_ns = time.monotonic_ns()
        logger.info("mediapipe.loaded", model=str(model_path))

    def _timestamp_ms(self) -> int:
        # VIDEO mode requires strictly-increasing millisecond timestamps.
        return (time.monotonic_ns() - self._start_ns) // 1_000_000

    def extract(self, frame_bgr: np.ndarray) -> tuple[HandLandmarks, ...]:
        if self._closed:
            return ()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        with self._lock:
            if self._closed:
                return ()
            try:
                result = self._detector.detect_for_video(
                    mp_image, self._timestamp_ms()
                )
            except Exception:
                self._logger.exception("mediapipe.detect_failed")
                return ()

        landmarks_list = getattr(result, "hand_landmarks", None) or []
        handedness_list = getattr(result, "handedness", None) or []
        if not landmarks_list:
            return ()

        out: list[HandLandmarks] = []
        for i, hand_landmarks in enumerate(landmarks_list):
            points = np.asarray(
                [(lm.x, lm.y, lm.z) for lm in hand_landmarks],
                dtype=np.float32,
            )

            label = "unknown"
            confidence = 0.0
            if handedness_list and i < len(handedness_list):
                category = handedness_list[i][0]
                label = category.category_name
                confidence = float(category.score)

            out.append(
                HandLandmarks(
                    points=points,
                    handedness=Handedness.from_mediapipe(label),
                    detection_confidence=Confidence(
                        max(0.0, min(1.0, confidence))
                    ),
                )
            )
        return tuple(out)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self._detector.close()
            except Exception:
                self._logger.exception("mediapipe.close_failed")
            self._closed = True
