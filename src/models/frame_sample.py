"""A single captured frame plus any hand landmarks extracted from it."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from src.models.hand_landmarks import HandLandmarks


@dataclass(slots=True)
class FrameSample:
    image_bgr: np.ndarray                     # HxWx3 uint8 (OpenCV order)
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    hands: tuple[HandLandmarks, ...] = ()

    @property
    def has_hand(self) -> bool:
        return len(self.hands) > 0

    @property
    def primary_hand(self) -> HandLandmarks | None:
        if not self.hands:
            return None
        return max(self.hands, key=lambda h: float(h.detection_confidence))
