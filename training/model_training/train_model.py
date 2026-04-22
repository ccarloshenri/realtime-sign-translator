"""
Baseline temporal classifier training script.

Architecture: mask-aware Bi-LSTM → Dense → Softmax. Small on purpose — the
goal is a reproducible baseline that runs on a CPU, not a SOTA model.

Usage:
    python -m training.model_training.train_model
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.implementations.config.yaml_configuration import (  # noqa: E402
    YamlConfigurationProvider,
)
from src.implementations.logging.structured_logger import build_logger  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="training/datasets/signflow.npz")
    parser.add_argument("--output", default="artifacts/signflow_lstm.keras")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = YamlConfigurationProvider(args.config).get()
    logger = build_logger("signflow.train", config.logging)

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
    labels = list(data["labels"])

    num_classes = len(labels)
    seq_len, feature_size = X_train.shape[1], X_train.shape[2]

    logger.info(
        "train.dataset_loaded",
        train=len(X_train),
        val=len(X_val),
        classes=num_classes,
        seq_len=seq_len,
        feature_size=feature_size,
    )

    model = models.Sequential(
        [
            layers.Input(shape=(seq_len, feature_size)),
            layers.Masking(mask_value=0.0),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.Dropout(0.3),
            layers.Bidirectional(layers.LSTM(32)),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(print_fn=lambda s: logger.info("model.layer", line=s))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=6,
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

    best_val_acc = float(max(history.history.get("val_accuracy", [0.0])))
    logger.info(
        "train.done",
        best_val_accuracy=round(best_val_acc, 4),
        output=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
