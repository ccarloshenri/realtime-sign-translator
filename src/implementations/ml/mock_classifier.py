"""
Mock temporal classifier.

Used when no trained model is available yet. It is NOT a sign-language
recognizer — it cycles through the configured vocabulary on a slow timer
whenever a hand is present, purely so the UI can be demoed end-to-end
without a trained model. Replace it with a real backend
(`classifier.backend: keras`) once you have one.
"""
from __future__ import annotations

import time

import numpy as np

from src.interface.sequence_classifier import ClassifierOutput
from src.models.confidence import Confidence

_CYCLE_SECONDS = 3.0  # How long each mock label stays on-screen


class MockSequenceClassifier:
    def __init__(
        self,
        labels: tuple[str, ...],
        sequence_length: int,
        feature_size: int,
    ) -> None:
        if not labels:
            raise ValueError("labels must not be empty")
        self._labels = labels
        self._sequence_length = sequence_length
        self._feature_size = feature_size
        self._start = time.monotonic()

    @property
    def labels(self) -> tuple[str, ...]:
        return self._labels

    @property
    def sequence_length(self) -> int:
        return self._sequence_length

    def predict(self, sequence: np.ndarray) -> ClassifierOutput:
        if sequence.shape != (self._sequence_length, self._feature_size):
            raise ValueError(
                f"Expected sequence shape ({self._sequence_length}, "
                f"{self._feature_size}), got {sequence.shape}"
            )

        nonzero_ratio = float(np.mean(np.any(sequence != 0.0, axis=1)))
        if nonzero_ratio < 0.5:
            uniform = np.full(len(self._labels), 1.0 / len(self._labels), dtype=np.float32)
            return ClassifierOutput(
                label=self._labels[0],
                confidence=Confidence(float(uniform[0])),
                probabilities=uniform,
            )

        # Cycle through the vocabulary on a wall clock so the demo stays
        # visually alive regardless of what the user is signing. A real
        # backend would replace this entire function.
        elapsed = time.monotonic() - self._start
        motion_energy = float(np.mean(np.abs(np.diff(sequence, axis=0))))
        bucket = int(elapsed // _CYCLE_SECONDS) % len(self._labels)

        logits = np.full(len(self._labels), 0.1, dtype=np.float32)
        logits[bucket] = 2.8 + min(motion_energy * 2.0, 2.0)
        probs = _softmax(logits)

        top = int(np.argmax(probs))
        return ClassifierOutput(
            label=self._labels[top],
            confidence=Confidence(float(probs[top])),
            probabilities=probs,
        )


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp = np.exp(shifted)
    return exp / np.sum(exp)
