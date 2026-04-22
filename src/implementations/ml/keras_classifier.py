"""
Keras-backed temporal classifier.

Loads a saved Keras model that takes input of shape (1, T, F) and outputs a
softmax over `labels`. The labels vocabulary is read from a JSON file, so the
training pipeline and the runtime agree on label order.

TensorFlow is imported lazily so that users running the MVP with the mock
backend never need TF installed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.interface.logger import ILogger
from src.interface.sequence_classifier import ClassifierOutput
from src.models.confidence import Confidence


class KerasSequenceClassifier:
    def __init__(
        self,
        model_path: str | Path,
        labels_path: str | Path,
        logger: ILogger,
    ) -> None:
        try:
            import tensorflow as tf  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is required for the Keras backend. "
                "Install with `pip install tensorflow`."
            ) from exc

        from tensorflow.keras.models import load_model

        self._logger = logger
        self._model_path = Path(model_path)
        self._labels_path = Path(labels_path)

        if not self._model_path.exists():
            raise FileNotFoundError(f"Keras model not found at {self._model_path}")
        if not self._labels_path.exists():
            raise FileNotFoundError(f"Labels file not found at {self._labels_path}")

        self._labels: tuple[str, ...] = tuple(
            json.loads(self._labels_path.read_text(encoding="utf-8"))
        )
        self._model = load_model(str(self._model_path))

        input_shape = self._model.input_shape
        if len(input_shape) != 3 or input_shape[1] is None:
            raise ValueError(
                f"Unexpected model input_shape {input_shape}; "
                "expected (batch, sequence_length, feature_size)"
            )
        self._sequence_length = int(input_shape[1])
        self._feature_size = int(input_shape[2]) if input_shape[2] is not None else -1

        logger.info(
            "classifier.keras_loaded",
            labels=len(self._labels),
            sequence_length=self._sequence_length,
            feature_size=self._feature_size,
        )

    @property
    def labels(self) -> tuple[str, ...]:
        return self._labels

    @property
    def sequence_length(self) -> int:
        return self._sequence_length

    def predict(self, sequence: np.ndarray) -> ClassifierOutput:
        if sequence.ndim != 2 or sequence.shape[0] != self._sequence_length:
            raise ValueError(
                f"Expected sequence of shape ({self._sequence_length}, F), "
                f"got {sequence.shape}"
            )
        batched = sequence[np.newaxis, ...].astype(np.float32)
        probs = np.asarray(self._model.predict(batched, verbose=0)[0], dtype=np.float32)
        if probs.shape != (len(self._labels),):
            raise ValueError(
                f"Model output shape {probs.shape} does not match "
                f"labels count {len(self._labels)}"
            )
        idx = int(np.argmax(probs))
        return ClassifierOutput(
            label=self._labels[idx],
            confidence=Confidence(float(probs[idx])),
            probabilities=probs,
        )
