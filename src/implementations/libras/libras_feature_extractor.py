"""
Libras feature extractor.

Takes the per-frame normalized landmark features the pipeline already
produces (shape `(T, F)`) and enriches them with a per-frame velocity
channel to produce `(T, 2F)`. Motion matters for Libras: many signs
only differ from each other by *how* the hand moves between
configurations (e.g. "obrigado" vs. "amor" share handshapes but differ
in trajectory). Static position channels alone would erase that
information.

The same extractor is used by:
  - the runtime classifier (`LibrasSequenceClassifier`)
  - the offline training script (`training/libras/train_libras_model.py`)

so the tensor shapes seen in production always match what the model
was trained on. If you change the encoding here, retrain the model.
"""
from __future__ import annotations

import numpy as np

from src.implementations.services.landmark_normalizer import LandmarkNormalizer


class LibrasFeatureExtractor:
    def __init__(self, include_both_hands: bool = True) -> None:
        self._normalizer = LandmarkNormalizer(include_both_hands=include_both_hands)
        self._position_size = self._normalizer.feature_size

    @property
    def position_size(self) -> int:
        """Size of the per-frame position feature vector (one hand = 63, two = 126)."""
        return self._position_size

    @property
    def feature_size(self) -> int:
        """Full per-frame feature size after velocity enrichment (position + velocity)."""
        return self._position_size * 2

    def enrich(self, sequence: np.ndarray) -> np.ndarray:
        """
        Enrich a `(T, F)` sequence of normalized landmark vectors into a
        `(T, 2F)` tensor where the second half of each frame is the
        per-frame velocity (difference from the previous frame). The
        first frame's velocity is filled with zeros.
        """
        if sequence.ndim != 2:
            raise ValueError(f"Expected 2-D sequence, got shape {sequence.shape}")
        if sequence.shape[1] != self._position_size:
            raise ValueError(
                f"Sequence feature size {sequence.shape[1]} does not match "
                f"expected {self._position_size}"
            )

        velocity = np.zeros_like(sequence, dtype=np.float32)
        if sequence.shape[0] > 1:
            velocity[1:] = np.diff(sequence, axis=0).astype(np.float32)

        return np.concatenate([sequence.astype(np.float32), velocity], axis=1)

    # -- Single-frame helpers (used by training-time feature pipelines) --

    def normalize_frame(self, hands) -> np.ndarray:
        """Convenience pass-through to the underlying landmark normalizer."""
        return self._normalizer.normalize(hands)
