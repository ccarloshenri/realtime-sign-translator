"""
Thread-safe fixed-length temporal buffer of landmark feature vectors.

The buffer is the bridge between the per-frame extractor and the temporal
classifier: it keeps the last N feature vectors and, once full, exposes them
as an (N, F) NumPy array ready for inference.
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Deque

import numpy as np


class SequenceBuffer:
    def __init__(self, sequence_length: int, feature_size: int) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if feature_size <= 0:
            raise ValueError("feature_size must be positive")
        self._sequence_length = sequence_length
        self._feature_size = feature_size
        self._frames: Deque[np.ndarray] = deque(maxlen=sequence_length)
        self._lock = threading.Lock()

    @property
    def sequence_length(self) -> int:
        return self._sequence_length

    @property
    def feature_size(self) -> int:
        return self._feature_size

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)

    def is_ready(self) -> bool:
        with self._lock:
            return len(self._frames) == self._sequence_length

    def append(self, features: np.ndarray) -> None:
        if features.shape != (self._feature_size,):
            raise ValueError(
                f"Expected feature vector shape ({self._feature_size},), "
                f"got {features.shape}"
            )
        with self._lock:
            self._frames.append(features.astype(np.float32, copy=False))

    def append_zero(self) -> None:
        with self._lock:
            self._frames.append(np.zeros(self._feature_size, dtype=np.float32))

    def snapshot(self) -> np.ndarray | None:
        with self._lock:
            if len(self._frames) < self._sequence_length:
                return None
            return np.stack(self._frames, axis=0)

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()
