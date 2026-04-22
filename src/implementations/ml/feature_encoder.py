"""
Feature encoder used by the training pipeline.

Wraps LandmarkNormalizer so the offline training code produces features in the
exact same format as the live pipeline.
"""
from __future__ import annotations

import numpy as np

from src.implementations.services.landmark_normalizer import LandmarkNormalizer
from src.models.hand_landmarks import HandLandmarks


class FeatureEncoder:
    def __init__(self, include_both_hands: bool = True) -> None:
        self._normalizer = LandmarkNormalizer(include_both_hands=include_both_hands)

    @property
    def feature_size(self) -> int:
        return self._normalizer.feature_size

    def encode(self, hands: tuple[HandLandmarks, ...]) -> np.ndarray:
        return self._normalizer.normalize(hands)
