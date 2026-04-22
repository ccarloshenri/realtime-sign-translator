"""
Temporal stabilization for classifier output.

Raw per-sequence predictions flicker: the top label can flip every few frames,
and confidence bounces around the threshold. The smoother combines three
techniques:

    1. Moving-average of probabilities over the last N predictions.
    2. Confidence gate — below `min_confidence`, no caption is emitted.
    3. Dwell counter — the same label must be the top-1 for `min_dwell`
       consecutive predictions before it is promoted to "stable".
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np

from src.interface.sequence_classifier import ClassifierOutput
from src.models.confidence import Confidence


@dataclass(frozen=True, slots=True)
class SmoothedPrediction:
    label: str
    confidence: Confidence
    should_publish: bool
    changed: bool


class PredictionSmoother:
    def __init__(
        self,
        labels: tuple[str, ...],
        min_confidence: float,
        smoothing_window: int,
        min_dwell_frames: int,
        publish_unchanged: bool = False,
    ) -> None:
        if not labels:
            raise ValueError("labels must not be empty")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if smoothing_window < 1:
            raise ValueError("smoothing_window must be >= 1")
        if min_dwell_frames < 1:
            raise ValueError("min_dwell_frames must be >= 1")

        self._labels = labels
        self._min_confidence = min_confidence
        self._min_dwell = min_dwell_frames
        self._publish_unchanged = publish_unchanged

        self._prob_history: Deque[np.ndarray] = deque(maxlen=smoothing_window)
        self._pending_label: str | None = None
        self._pending_count = 0
        self._stable_label: str | None = None

    def reset(self) -> None:
        self._prob_history.clear()
        self._pending_label = None
        self._pending_count = 0
        self._stable_label = None

    def observe(self, output: ClassifierOutput) -> SmoothedPrediction:
        label, confidence = self._smooth(output)

        if confidence < self._min_confidence:
            return SmoothedPrediction(
                label=label,
                confidence=Confidence(confidence),
                should_publish=False,
                changed=False,
            )

        if label == self._pending_label:
            self._pending_count += 1
        else:
            self._pending_label = label
            self._pending_count = 1

        promoted = self._pending_count >= self._min_dwell
        changed = promoted and label != self._stable_label

        if promoted:
            self._stable_label = label

        should_publish = changed or (promoted and self._publish_unchanged)
        return SmoothedPrediction(
            label=label,
            confidence=Confidence(confidence),
            should_publish=should_publish,
            changed=changed,
        )

    def _smooth(self, output: ClassifierOutput) -> tuple[str, float]:
        if output.probabilities is None:
            return output.label, float(output.confidence)

        probs = np.asarray(output.probabilities, dtype=np.float32)
        if probs.shape != (len(self._labels),):
            raise ValueError(
                f"probabilities shape {probs.shape} does not match "
                f"labels count {len(self._labels)}"
            )
        self._prob_history.append(probs)
        averaged = np.mean(np.stack(self._prob_history, axis=0), axis=0)
        idx = int(np.argmax(averaged))
        return self._labels[idx], float(averaged[idx])
