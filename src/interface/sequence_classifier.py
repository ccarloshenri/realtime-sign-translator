"""
Temporal classifier contract.

The input is a normalized sequence of shape (T, F) where T = sequence_length
and F = per-frame feature size.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.models.confidence import Confidence


@dataclass(frozen=True, slots=True)
class ClassifierOutput:
    label: str
    confidence: Confidence
    probabilities: np.ndarray | None = None


class ISequenceClassifier(Protocol):
    """Stateless classifier over a fixed-length temporal window."""

    @property
    def labels(self) -> tuple[str, ...]: ...

    @property
    def sequence_length(self) -> int: ...

    def predict(self, sequence: np.ndarray) -> ClassifierOutput: ...
