"""
Interactive sample collector.

Usage:
    python -m training.data_collection.collect_samples --label ajuda --samples 40

Runtime controls (OpenCV preview window):
    SPACE  start recording a single sample
    ESC    quit
    R      discard current in-progress sample
    BACK   undo the last saved sample

Each sample is stored as `samples/<label>/<uuid>.npz` with the normalized
landmark sequence plus label metadata — the same feature layout the live
pipeline uses.
"""
from __future__ import annotations

import argparse
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.implementations.config.yaml_configuration import (  # noqa: E402
    YamlConfigurationProvider,
)
from src.implementations.logging.structured_logger import build_logger  # noqa: E402
from src.implementations.ml.feature_encoder import FeatureEncoder  # noqa: E402
from src.implementations.vision.mediapipe_extractor import (  # noqa: E402
    MediaPipeHandLandmarkExtractor,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect labeled landmark sequences.")
    parser.add_argument("--label", required=True, help="Sign label for this batch")
    parser.add_argument(
        "--samples", type=int, default=20, help="Target number of samples"
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to config.yaml"
    )
    parser.add_argument(
        "--output", default="samples", help="Root directory for saved samples"
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = YamlConfigurationProvider(args.config).get()
    logger = build_logger("signflow.collector", config.logging)

    output_dir = Path(args.output) / args.label
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = MediaPipeHandLandmarkExtractor(
        logger=logger,
        max_num_hands=config.vision.max_num_hands,
        min_detection_confidence=config.vision.min_detection_confidence,
        min_tracking_confidence=config.vision.min_tracking_confidence,
        model_complexity=config.vision.model_complexity,
    )
    encoder = FeatureEncoder(include_both_hands=True)

    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(config.camera.device_index, backend)
    if not cap.isOpened():
        logger.error("collector.camera_failed", device=config.camera.device_index)
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.height)
    cap.set(cv2.CAP_PROP_FPS, config.camera.target_fps)

    sequence_length = config.pipeline.sequence_length
    collected: list[Path] = list(output_dir.glob("*.npz"))
    recording_buffer: list[np.ndarray] | None = None

    logger.info(
        "collector.started",
        label=args.label,
        target=args.samples,
        existing=len(collected),
        sequence_length=sequence_length,
    )
    print(
        "SPACE: start sample  |  ESC: quit  |  R: discard current  |  BACKSPACE: undo last"
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            if config.camera.flip_horizontal:
                frame = cv2.flip(frame, 1)

            hands = extractor.extract(frame)
            features = encoder.encode(hands)

            hud = frame.copy()
            _draw_hud(
                hud,
                label=args.label,
                collected=len(collected),
                target=args.samples,
                recording_progress=(
                    None if recording_buffer is None else len(recording_buffer) / sequence_length
                ),
                hand_detected=bool(hands),
            )
            cv2.imshow("SignFlow — collector (SPACE to record)", hud)

            if recording_buffer is not None:
                recording_buffer.append(features)
                if len(recording_buffer) >= sequence_length:
                    saved_path = _save_sample(
                        output_dir=output_dir,
                        label=args.label,
                        buffer=recording_buffer,
                    )
                    collected.append(saved_path)
                    logger.info(
                        "collector.sample_saved",
                        path=str(saved_path),
                        total=len(collected),
                    )
                    recording_buffer = None
                    if len(collected) >= args.samples:
                        logger.info("collector.target_reached")
                        break

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == 32 and recording_buffer is None:
                recording_buffer = []
                logger.info("collector.recording_started")
            elif key in (ord("r"), ord("R")):
                recording_buffer = None
                logger.info("collector.recording_discarded")
            elif key == 8 and collected:
                last = collected.pop()
                try:
                    last.unlink()
                    logger.info("collector.undo", path=str(last))
                except OSError:
                    logger.exception("collector.undo_failed")

            time.sleep(0.001)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()

    return 0


def _save_sample(
    output_dir: Path,
    label: str,
    buffer: list[np.ndarray],
) -> Path:
    features = np.stack(buffer, axis=0).astype(np.float32)
    path = output_dir / f"{uuid.uuid4().hex}.npz"
    np.savez_compressed(
        path,
        features=features,
        meta=np.array(
            {
                "label": label,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sequence_length": features.shape[0],
            },
            dtype=object,
        ),
    )
    return path


def _draw_hud(
    frame: np.ndarray,
    label: str,
    collected: int,
    target: int,
    recording_progress: float | None,
    hand_detected: bool,
) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 70), (15, 17, 21), -1)
    cv2.putText(
        frame, f"label: {label}", (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (238, 241, 247), 2,
    )
    cv2.putText(
        frame, f"saved: {collected}/{target}", (20, 55),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (138, 147, 166), 1,
    )

    status = "REC" if recording_progress is not None else ("HAND" if hand_detected else "IDLE")
    color = (243, 117, 115) if recording_progress is not None else (
        (50, 213, 131) if hand_detected else (138, 147, 166)
    )
    cv2.putText(
        frame, status, (w - 120, 45),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2,
    )

    if recording_progress is not None:
        bar_w = int(max(0.0, min(1.0, recording_progress)) * (w - 40))
        cv2.rectangle(frame, (20, h - 30), (w - 20, h - 10), (40, 50, 70), -1)
        cv2.rectangle(frame, (20, h - 30), (20 + bar_w, h - 10), (78, 161, 255), -1)


if __name__ == "__main__":
    raise SystemExit(main())
