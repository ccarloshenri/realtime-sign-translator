"""
Turn a folder of collected `.npz` samples into a single dataset ready for
training.

Layout expected on disk:

    samples/
        ajuda/abc.npz
        ajuda/def.npz
        obrigado/...
        ...

Output:

    training/datasets/signflow.npz
        X_train, y_train, X_val, y_val, X_test, y_test, labels
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
from src.implementations.logging.structured_logger import build_logger  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a train/val/test dataset.")
    parser.add_argument("--samples", default="samples")
    parser.add_argument("--output", default="training/datasets/signflow.npz")
    parser.add_argument("--labels-out", default="artifacts/labels.json")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = YamlConfigurationProvider(args.config).get()
    logger = build_logger("signflow.dataset", config.logging)

    samples_root = Path(args.samples)
    if not samples_root.exists():
        logger.error("dataset.samples_missing", path=str(samples_root))
        return 1

    label_dirs = sorted(p for p in samples_root.iterdir() if p.is_dir())
    if not label_dirs:
        logger.error("dataset.no_labels_found", path=str(samples_root))
        return 1

    labels = [d.name for d in label_dirs]
    X: list[np.ndarray] = []
    y: list[int] = []

    expected_seq_len = config.pipeline.sequence_length

    for idx, label_dir in enumerate(label_dirs):
        for npz_path in sorted(label_dir.glob("*.npz")):
            data = np.load(npz_path, allow_pickle=True)
            features = data["features"]
            if features.shape[0] != expected_seq_len:
                logger.warning(
                    "dataset.sequence_length_mismatch",
                    path=str(npz_path),
                    got=features.shape[0],
                    expected=expected_seq_len,
                )
                continue
            X.append(features.astype(np.float32))
            y.append(idx)

    if not X:
        logger.error("dataset.empty")
        return 1

    X_arr = np.stack(X, axis=0)
    y_arr = np.asarray(y, dtype=np.int64)

    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(X_arr))

    n_total = len(indices)
    n_test = int(n_total * args.test_ratio)
    n_val = int(n_total * args.val_ratio)
    n_train = n_total - n_val - n_test

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.labels_out).parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        args.output,
        X_train=X_arr[train_idx],
        y_train=y_arr[train_idx],
        X_val=X_arr[val_idx],
        y_val=y_arr[val_idx],
        X_test=X_arr[test_idx],
        y_test=y_arr[test_idx],
        labels=np.array(labels, dtype=object),
    )
    Path(args.labels_out).write_text(
        json.dumps(labels, ensure_ascii=False), encoding="utf-8"
    )

    logger.info(
        "dataset.built",
        path=args.output,
        labels=labels,
        train=n_train,
        val=n_val,
        test=n_test,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
