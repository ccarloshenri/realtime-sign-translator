"""
Libras-specific trainer.

Differences from the generic `training.model_training.train_model`:

  1. Validates that every label in the dataset is part of the declared
     Libras seed vocabulary (`LIBRAS_BASE_VOCABULARY`). Unknown labels
     abort the run with a clear error instead of silently training on
     typos.
  2. Applies `LibrasFeatureExtractor` to every training sample so the
     model is trained on velocity-enriched (T, 2F) tensors — the same
     tensors the runtime classifier feeds it. No training/inference
     schema drift.
  3. Writes the artifacts under Libras-specific filenames
     (`artifacts/libras_lstm.keras`, `artifacts/libras_labels.json`),
     which is what the `libras` runtime backend reads.

Usage:

    python -m training.libras.train_libras_model

The dataset is expected to be the same `.npz` file produced by
`training.preprocessing.build_dataset` (run that first).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.implementations.config.yaml_configuration import (  # noqa: E402
    YamlConfigurationProvider,
)
from src.implementations.libras.libras_feature_extractor import (  # noqa: E402
    LibrasFeatureExtractor,
)
from src.implementations.logging.structured_logger import build_logger  # noqa: E402
from src.models.libras_vocabulary import LIBRAS_BASE_VOCABULARY  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="training/datasets/signflow.npz")
    parser.add_argument("--output", default="artifacts/libras_lstm.keras")
    parser.add_argument("--labels-out", default="artifacts/libras_labels.json")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--allow-unknown-labels",
        action="store_true",
        help="Skip the vocabulary check and train whatever labels are "
        "present in the dataset. Use only if you are deliberately "
        "extending beyond LIBRAS_BASE_VOCABULARY.",
    )
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def _enrich_dataset(
    X: np.ndarray, extractor: LibrasFeatureExtractor
) -> np.ndarray:
    """Apply feature enrichment to every (T, F) sample in the dataset."""
    enriched = np.empty(
        (X.shape[0], X.shape[1], extractor.feature_size), dtype=np.float32
    )
    for i in range(X.shape[0]):
        enriched[i] = extractor.enrich(X[i])
    return enriched


def main() -> int:
    args = _parse_args()
    config = YamlConfigurationProvider(args.config).get()
    logger = build_logger("signflow.libras.train", config.logging)

    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow.keras import callbacks, layers, models, optimizers
    except ImportError:
        logger.error("train.tensorflow_missing", hint="pip install tensorflow")
        return 1

    data = np.load(args.dataset, allow_pickle=True)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    labels = [str(label) for label in data["labels"]]

    # Vocabulary validation -------------------------------------------------
    unknown = sorted(set(labels) - set(LIBRAS_BASE_VOCABULARY))
    if unknown and not args.allow_unknown_labels:
        logger.error(
            "libras.unknown_labels",
            unknown=unknown,
            known=list(LIBRAS_BASE_VOCABULARY),
            hint="Pass --allow-unknown-labels to extend the vocabulary.",
        )
        return 1
    if unknown and args.allow_unknown_labels:
        logger.warning("libras.extending_vocabulary", new_labels=unknown)

    # Feature enrichment ----------------------------------------------------
    extractor = LibrasFeatureExtractor(include_both_hands=True)
    seq_len = X_train.shape[1]
    raw_feature_size = X_train.shape[2]
    if raw_feature_size != extractor.position_size:
        logger.error(
            "libras.feature_size_mismatch",
            dataset=raw_feature_size,
            extractor=extractor.position_size,
            hint="Rebuild the dataset so feature_size matches LandmarkNormalizer.",
        )
        return 1

    X_train_enriched = _enrich_dataset(X_train, extractor)
    X_val_enriched = _enrich_dataset(X_val, extractor)

    logger.info(
        "libras.train.dataset_ready",
        train=len(X_train_enriched),
        val=len(X_val_enriched),
        classes=len(labels),
        seq_len=seq_len,
        feature_size=extractor.feature_size,
    )

    model = models.Sequential(
        [
            layers.Input(shape=(seq_len, extractor.feature_size)),
            layers.Masking(mask_value=0.0),
            layers.Bidirectional(layers.LSTM(96, return_sequences=True)),
            layers.Dropout(0.3),
            layers.Bidirectional(layers.LSTM(48)),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dense(len(labels), activation="softmax"),
        ]
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(print_fn=lambda s: logger.info("libras.model.layer", line=s))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.labels_out).parent.mkdir(parents=True, exist_ok=True)

    history = model.fit(
        X_train_enriched,
        y_train,
        validation_data=(X_val_enriched, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=8,
                restore_best_weights=True,
            ),
            callbacks.ModelCheckpoint(
                args.output,
                monitor="val_accuracy",
                save_best_only=True,
            ),
        ],
        verbose=2,
    )

    # Persist the label vocabulary in the order the model's output axis uses.
    Path(args.labels_out).write_text(
        json.dumps(labels, ensure_ascii=False), encoding="utf-8"
    )

    best_val_acc = float(max(history.history.get("val_accuracy", [0.0])))
    logger.info(
        "libras.train.done",
        best_val_accuracy=round(best_val_acc, 4),
        model=args.output,
        labels=args.labels_out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
