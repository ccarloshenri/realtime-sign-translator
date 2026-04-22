"""
Evaluate the trained Keras model on the held-out test split.

Prints top-1 accuracy, per-class accuracy, and a confusion matrix.
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
    parser.add_argument("--model", default="artifacts/signflow_lstm.keras")
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = YamlConfigurationProvider(args.config).get()
    logger = build_logger("signflow.eval", config.logging)

    try:
        from tensorflow.keras.models import load_model
    except ImportError:
        logger.error("eval.tensorflow_missing")
        return 1

    data = np.load(args.dataset, allow_pickle=True)
    X_test = data["X_test"]
    y_test = data["y_test"]
    labels = list(data["labels"])

    model = load_model(args.model)
    preds = np.argmax(model.predict(X_test, verbose=0), axis=1)

    top1 = float(np.mean(preds == y_test))
    logger.info("eval.top1", accuracy=round(top1, 4), samples=len(y_test))

    print(f"\nTop-1 accuracy: {top1 * 100:.2f}%  ({len(y_test)} samples)\n")

    num_classes = len(labels)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(y_test, preds):
        confusion[true, pred] += 1

    header = " " * 14 + " ".join(f"{lbl[:8]:>8}" for lbl in labels)
    print(header)
    for i, label in enumerate(labels):
        row = " ".join(f"{confusion[i, j]:>8d}" for j in range(num_classes))
        per_class = confusion[i, i] / max(1, confusion[i].sum())
        print(f"{label[:12]:>12} | {row}  ({per_class * 100:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
