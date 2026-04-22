"""
Libras sequence classifier.

Loads a Keras model trained specifically on Libras signs
(`artifacts/libras_lstm.keras` by default) together with its label
vocabulary (`artifacts/libras_labels.json`). Internally feeds every
incoming sequence through `LibrasFeatureExtractor` so the tensor the
model sees at inference time has the same (T, 2F) shape it was trained
on — no hidden schema drift between training and production.

If the model file does not exist yet, construction raises
`FileNotFoundError` with an actionable hint. The bootstrap layer catches
this and falls back to the mock backend so the app still boots.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.implementations.libras.libras_feature_extractor import (
    LibrasFeatureExtractor,
)
from src.interface.logger import ILogger
from src.interface.sequence_classifier import ClassifierOutput
from src.models.confidence import Confidence


LIBRAS_MODEL_MISSING_HINT = (
    "Train the Libras model first: "
    "`python -m training.data_collection.collect_samples --label <sign>`, "
    "then `python -m training.preprocessing.build_dataset`, "
    "then `python -m training.libras.train_libras_model`."
)


class LibrasSequenceClassifier:
    def __init__(
        self,
        model_path: str | Path,
        labels_path: str | Path,
        logger: ILogger,
        feature_extractor: LibrasFeatureExtractor | None = None,
    ) -> None:
        try:
            import tensorflow as tf  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is required for the Libras backend. "
                "Install with `pip install tensorflow`."
            ) from exc

        from tensorflow.keras.models import load_model

        self._logger = logger
        self._feature_extractor = feature_extractor or LibrasFeatureExtractor()

        self._model_path = Path(model_path)
        self._labels_path = Path(labels_path)

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Libras model not found at {self._model_path}. "
                f"{LIBRAS_MODEL_MISSING_HINT}"
            )
        if not self._labels_path.exists():
            raise FileNotFoundError(
                f"Libras labels file not found at {self._labels_path}. "
                f"{LIBRAS_MODEL_MISSING_HINT}"
            )

        self._labels: tuple[str, ...] = tuple(
            json.loads(self._labels_path.read_text(encoding="utf-8"))
        )
        if not self._labels:
            raise ValueError(
                f"Libras labels file {self._labels_path} is empty"
            )

        self._model = load_model(str(self._model_path))
        input_shape = self._model.input_shape
        if len(input_shape) != 3 or input_shape[1] is None:
            raise ValueError(
                f"Unexpected Libras model input_shape {input_shape}; "
                "expected (batch, sequence_length, feature_size)"
            )
        self._sequence_length = int(input_shape[1])
        expected_feature_size = int(input_shape[2]) if input_shape[2] is not None else -1
        if expected_feature_size not in (-1, self._feature_extractor.feature_size):
            raise ValueError(
                f"Libras model expects feature size {expected_feature_size}, "
                f"LibrasFeatureExtractor produces {self._feature_extractor.feature_size}. "
                "Retrain the model with the current extractor."
            )

        logger.info(
            "libras.model_loaded",
            labels=len(self._labels),
            sequence_length=self._sequence_length,
            feature_size=self._feature_extractor.feature_size,
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

        enriched = self._feature_extractor.enrich(sequence)
        batched = enriched[np.newaxis, ...].astype(np.float32)
        probs = np.asarray(
            self._model.predict(batched, verbose=0)[0], dtype=np.float32
        )
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
