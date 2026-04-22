from __future__ import annotations

from typing import Protocol

import numpy as np

from src.models.hand_landmarks import HandLandmarks


class IHandLandmarkExtractor(Protocol):
    """Extracts hand landmarks from a single BGR frame."""

    def extract(self, frame_bgr: np.ndarray) -> tuple[HandLandmarks, ...]: ...

    def close(self) -> None: ...
