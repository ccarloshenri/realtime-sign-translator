"""
MediaPipe Hands adapter.

Wraps `mediapipe.solutions.hands.Hands` behind the `IHandLandmarkExtractor`
contract. Converts MediaPipe's proto-style output into our immutable
`HandLandmarks` value objects.
"""
from __future__ import annotations

import threading

import cv2
import numpy as np

from src.interface.logger import ILogger
from src.models.confidence import Confidence
from src.models.hand_landmarks import HandLandmarks
from src.models.handedness import Handedness

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "mediapipe is required. Install it with `pip install mediapipe`."
    ) from exc


class MediaPipeHandLandmarkExtractor:
    def __init__(
        self,
        logger: ILogger,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
    ) -> None:
        self._logger = logger
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._lock = threading.Lock()
        self._closed = False

    def extract(self, frame_bgr: np.ndarray) -> tuple[HandLandmarks, ...]:
        if self._closed:
            return ()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        with self._lock:
            if self._closed:
                return ()
            results = self._hands.process(frame_rgb)

        landmarks_list = getattr(results, "multi_hand_landmarks", None)
        handedness_list = getattr(results, "multi_handedness", None)
        if not landmarks_list:
            return ()

        out: list[HandLandmarks] = []
        for i, hand_landmarks in enumerate(landmarks_list):
            points = np.asarray(
                [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark],
                dtype=np.float32,
            )

            label = "unknown"
            confidence = 0.0
            if handedness_list and i < len(handedness_list):
                classification = handedness_list[i].classification[0]
                label = classification.label
                confidence = float(classification.score)

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
                self._hands.close()
            except Exception:
                self._logger.exception("mediapipe.close_failed")
            self._closed = True
