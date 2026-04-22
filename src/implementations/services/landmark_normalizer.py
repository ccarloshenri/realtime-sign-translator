"""
Landmark normalization.

MediaPipe returns coordinates in image/normalized space, which depend on where
the hand is in the frame and how close it is to the camera. For a sign
classifier we want features that are translation- and scale-invariant:

    1. Translate so the wrist (landmark 0) sits at the origin.
    2. Scale so the largest distance from the wrist equals 1.

We also optionally encode both hands into a single feature vector so the
classifier sees the full scene at once. When a hand is missing, its slot is
filled with zeros.
"""
from __future__ import annotations

import numpy as np

from src.models.hand_landmarks import HandLandmarks, NUM_COORDS, NUM_LANDMARKS
from src.models.handedness import Handedness

_WRIST_INDEX = 0
_SINGLE_HAND_FEATURES = NUM_LANDMARKS * NUM_COORDS            # 63
_TWO_HAND_FEATURES = 2 * _SINGLE_HAND_FEATURES                # 126


class LandmarkNormalizer:
    def __init__(self, include_both_hands: bool = True) -> None:
        self._include_both_hands = include_both_hands

    @property
    def feature_size(self) -> int:
        return _TWO_HAND_FEATURES if self._include_both_hands else _SINGLE_HAND_FEATURES

    def normalize(self, hands: tuple[HandLandmarks, ...]) -> np.ndarray:
        if not hands:
            return np.zeros(self.feature_size, dtype=np.float32)

        if not self._include_both_hands:
            primary = max(hands, key=lambda h: float(h.detection_confidence))
            return self._normalize_single(primary.points)

        left_slot = np.zeros(_SINGLE_HAND_FEATURES, dtype=np.float32)
        right_slot = np.zeros(_SINGLE_HAND_FEATURES, dtype=np.float32)

        unassigned: list[HandLandmarks] = []
        for hand in hands:
            normalized = self._normalize_single(hand.points)
            if hand.handedness is Handedness.LEFT:
                left_slot = normalized
            elif hand.handedness is Handedness.RIGHT:
                right_slot = normalized
            else:
                unassigned.append(hand)

        for hand in unassigned:
            normalized = self._normalize_single(hand.points)
            if not np.any(left_slot):
                left_slot = normalized
            elif not np.any(right_slot):
                right_slot = normalized

        return np.concatenate([left_slot, right_slot], axis=0)

    @staticmethod
    def _normalize_single(points: np.ndarray) -> np.ndarray:
        centered = points - points[_WRIST_INDEX]
        max_norm = float(np.max(np.linalg.norm(centered, axis=1)))
        if max_norm > 1e-8:
            centered = centered / max_norm
        return centered.reshape(-1).astype(np.float32)
