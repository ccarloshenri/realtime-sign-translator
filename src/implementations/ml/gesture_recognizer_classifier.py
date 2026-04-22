"""
ISequenceClassifier backed by Google's pretrained MediaPipe
GestureRecognizer.

Covers 7 static hand gestures out of the box (plus a "None" sentinel
when no gesture is recognized):

    👍 Thumb_Up       👎 Thumb_Down     ✌️ Victory
    ✊ Closed_Fist    🖐 Open_Palm      ☝️ Pointing_Up
    ❤️ ILoveYou  (ASL "I love you")

This is NOT a Libras translator — it is a plug-and-play real recognizer
useful as a bridge between the mock backend and a custom-trained model.

The classifier does not look at the 30-frame temporal sequence it
receives — the underlying MediaPipe model is itself a per-frame static
recognizer. Instead, it reads the latest classification cached by the
`MediaPipeGestureExtractor` it is composed with.
"""
from __future__ import annotations

import numpy as np

from src.implementations.vision.mediapipe_gesture_extractor import (
    MediaPipeGestureExtractor,
)
from src.interface.sequence_classifier import ClassifierOutput
from src.models.confidence import Confidence


# (MediaPipe category name, display label)
GESTURE_LABELS: tuple[tuple[str, str], ...] = (
    ("Closed_Fist", "punho"),
    ("Open_Palm", "mão aberta"),
    ("Pointing_Up", "apontando"),
    ("Thumb_Down", "polegar p/ baixo"),
    ("Thumb_Up", "polegar p/ cima"),
    ("Victory", "vitória"),
    ("ILoveYou", "eu te amo"),
)


class GestureRecognizerClassifier:
    def __init__(
        self,
        extractor: MediaPipeGestureExtractor,
        sequence_length: int,
    ) -> None:
        self._extractor = extractor
        self._sequence_length = sequence_length
        self._mp_to_display: dict[str, str] = dict(GESTURE_LABELS)
        self._display_labels: tuple[str, ...] = tuple(
            display for _, display in GESTURE_LABELS
        )
        self._uniform = np.full(
            len(self._display_labels),
            1.0 / len(self._display_labels),
            dtype=np.float32,
        )

    @property
    def labels(self) -> tuple[str, ...]:
        return self._display_labels

    @property
    def sequence_length(self) -> int:
        return self._sequence_length

    def predict(self, sequence: np.ndarray) -> ClassifierOutput:
        # We ignore the sequence — the underlying model is per-frame.
        mp_label, score = self._extractor.latest_gesture

        if mp_label == "None" or mp_label not in self._mp_to_display:
            # Emit a uniform distribution so the smoother's confidence gate
            # filters this out and the last stable caption sticks.
            return ClassifierOutput(
                label=self._display_labels[0],
                confidence=Confidence(float(self._uniform[0])),
                probabilities=self._uniform.copy(),
            )

        display = self._mp_to_display[mp_label]
        idx = self._display_labels.index(display)

        # Build a peaked distribution around the top-1 so the smoother can
        # still compute a moving-average confidence across frames.
        probs = np.full(
            len(self._display_labels),
            (1.0 - score) / max(1, len(self._display_labels) - 1),
            dtype=np.float32,
        )
        probs[idx] = score
        probs = probs / probs.sum()

        return ClassifierOutput(
            label=display,
            confidence=Confidence(max(0.0, min(1.0, score))),
            probabilities=probs,
        )
