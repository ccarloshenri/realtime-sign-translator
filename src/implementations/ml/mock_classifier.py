"""
Mock temporal classifier.

Used when no trained model is available yet. It's deliberately NOT random —
it produces deterministic, plausible-looking predictions by hashing simple
statistics of the input sequence (mean hand position, motion energy). That
gives us a working end-to-end pipeline we can demo, while making it easy to
swap in a real model by flipping a config flag.

TODO(replace): swap the `classifier.backend` config to `keras` once a real
model ships, and remove this from the production config.
"""
from __future__ import annotations

import math

import numpy as np

from src.interface.sequence_classifier import ClassifierOutput
from src.models.confidence import Confidence


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

        mean_pos = float(np.mean(sequence[:, :3]))
        motion_energy = float(np.mean(np.abs(np.diff(sequence, axis=0))))

        bucket = int(
            abs(math.floor(mean_pos * 97.0) + math.floor(motion_energy * 131.0))
            % len(self._labels)
        )

        logits = np.full(len(self._labels), 0.1, dtype=np.float32)
        logits[bucket] = 1.0 + min(motion_energy * 4.0, 2.5)
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
