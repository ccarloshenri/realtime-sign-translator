"""
Hand landmarks — one hand = 21 points × (x, y, z).

MediaPipe emits normalized coordinates in [0, 1] for x/y and relative depth in
z. We keep them raw here; normalization for the classifier is the job of the
LandmarkNormalizer service.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.models.confidence import Confidence
from src.models.handedness import Handedness

NUM_LANDMARKS = 21
NUM_COORDS = 3  # x, y, z


@dataclass(frozen=True, slots=True)
class HandLandmarks:
    """Immutable snapshot of one hand detected in a single frame."""

    points: np.ndarray           # shape: (NUM_LANDMARKS, NUM_COORDS), dtype=float32
    handedness: Handedness
    detection_confidence: Confidence

    def __post_init__(self) -> None:
        if self.points.shape != (NUM_LANDMARKS, NUM_COORDS):
            raise ValueError(
                f"points shape must be ({NUM_LANDMARKS}, {NUM_COORDS}), "
                f"got {self.points.shape}"
            )
        if self.points.dtype != np.float32:
            object.__setattr__(self, "points", self.points.astype(np.float32))

    @classmethod
    def from_xyz_iterable(
        cls,
        xyz: Iterable[tuple[float, float, float]],
        handedness: Handedness,
        detection_confidence: Confidence,
    ) -> "HandLandmarks":
        arr = np.asarray(list(xyz), dtype=np.float32)
        return cls(arr, handedness, detection_confidence)

    def flatten(self) -> np.ndarray:
        """Flatten to a 1-D feature vector of length NUM_LANDMARKS * NUM_COORDS."""
        return self.points.reshape(-1)
