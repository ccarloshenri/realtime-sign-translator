"""
Main realtime pipeline.

Runs in its own background thread so the UI stays responsive. Every iteration:

    1. Grab a BGR frame from the camera.
    2. Extract hand landmarks.
    3. Normalize them into a feature vector.
    4. Push to the temporal SequenceBuffer.
    5. If the buffer is full, run the classifier.
    6. Smooth/stabilize the prediction.
    7. Notify listeners (UI, WebSocket) when a stable caption is ready.

This use case depends only on the interfaces in `src.interface`, never on
concrete adapters. That's what keeps the classifier swappable and the
WebSocket layer optional.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, replace
from typing import Callable

from src.implementations.services.landmark_normalizer import LandmarkNormalizer
from src.implementations.services.prediction_smoother import (
    PredictionSmoother,
    SmoothedPrediction,
)
from src.implementations.services.sequence_buffer import SequenceBuffer
from src.interface.camera_provider import ICameraProvider
from src.interface.hand_landmark_extractor import IHandLandmarkExtractor
from src.interface.logger import ILogger
from src.interface.sequence_classifier import ISequenceClassifier
from src.models.frame_sample import FrameSample
from src.models.sign_prediction import SignPrediction


FrameListener = Callable[[FrameSample], None]
PredictionListener = Callable[[SignPrediction], None]
StateListener = Callable[["PipelineState"], None]


@dataclass(slots=True)
class PipelineState:
    running: bool = False
    hand_detected: bool = False
    buffer_fill: float = 0.0
    last_label: str | None = None
    last_confidence: float = 0.0
    fps: float = 0.0


@dataclass(slots=True)
class PipelineCallbacks:
    on_frame: list[FrameListener] = field(default_factory=list)
    on_prediction: list[PredictionListener] = field(default_factory=list)
    on_state: list[StateListener] = field(default_factory=list)


class RunTranslationPipeline:
    def __init__(
        self,
        camera: ICameraProvider,
        extractor: IHandLandmarkExtractor,
        normalizer: LandmarkNormalizer,
        buffer: SequenceBuffer,
        classifier: ISequenceClassifier,
        smoother: PredictionSmoother,
        logger: ILogger,
        target_fps: int = 30,
        source: str = "camera",
    ) -> None:
        self._camera = camera
        self._extractor = extractor
        self._normalizer = normalizer
        self._buffer = buffer
        self._classifier = classifier
        self._smoother = smoother
        self._logger = logger
        self._target_period = 1.0 / max(target_fps, 1)
        self._source = source

        self._callbacks = PipelineCallbacks()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._state = PipelineState()
        self._state_lock = threading.Lock()

    @property
    def callbacks(self) -> PipelineCallbacks:
        return self._callbacks

    def snapshot_state(self) -> PipelineState:
        with self._state_lock:
            return replace(self._state)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._logger.warning("pipeline.start called but already running")
            return
        self._stop_event.clear()
        self._buffer.clear()
        self._smoother.reset()
        self._camera.open()
        self._thread = threading.Thread(
            target=self._run, name="signflow-pipeline", daemon=True
        )
        self._update_state(running=True)
        self._thread.start()
        self._logger.info("pipeline.started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._camera.close()
        self._extractor.close()
        self._update_state(running=False, hand_detected=False, buffer_fill=0.0)
        self._logger.info("pipeline.stopped")

    def _run(self) -> None:
        frame_count = 0
        fps_window_start = time.monotonic()

        while not self._stop_event.is_set():
            iteration_start = time.monotonic()

            frame = self._camera.read()
            if frame is None:
                self._sleep_budget(iteration_start)
                continue

            try:
                hands = self._extractor.extract(frame)
            except Exception:
                self._logger.exception("extractor.failed")
                hands = ()

            sample = FrameSample(image_bgr=frame, hands=hands)
            features = self._normalizer.normalize(hands)
            if hands:
                self._buffer.append(features)
            else:
                self._buffer.append_zero()

            self._notify_frame(sample)

            prediction: SmoothedPrediction | None = None
            snapshot = self._buffer.snapshot()
            if snapshot is not None:
                try:
                    output = self._classifier.predict(snapshot)
                    prediction = self._smoother.observe(output)
                except Exception:
                    self._logger.exception("classifier.failed")

            if prediction is not None and prediction.should_publish:
                sign = SignPrediction(
                    label=prediction.label,
                    confidence=prediction.confidence,
                    sequence_size=self._buffer.sequence_length,
                    source=self._source,
                )
                self._notify_prediction(sign)
                self._logger.info(
                    "prediction.published",
                    label=sign.label,
                    confidence=round(float(sign.confidence), 3),
                )

            frame_count += 1
            elapsed = time.monotonic() - fps_window_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                self._update_state(
                    hand_detected=bool(hands),
                    buffer_fill=len(self._buffer) / self._buffer.sequence_length,
                    last_label=prediction.label if prediction else None,
                    last_confidence=float(prediction.confidence) if prediction else 0.0,
                    fps=fps,
                )
                frame_count = 0
                fps_window_start = time.monotonic()
            else:
                self._update_state(
                    hand_detected=bool(hands),
                    buffer_fill=len(self._buffer) / self._buffer.sequence_length,
                )

            self._sleep_budget(iteration_start)

    def _sleep_budget(self, iteration_start: float) -> None:
        spent = time.monotonic() - iteration_start
        remaining = self._target_period - spent
        if remaining > 0:
            time.sleep(remaining)

    def _update_state(self, **kwargs) -> None:
        with self._state_lock:
            for k, v in kwargs.items():
                setattr(self._state, k, v)
            snapshot = replace(self._state)
        for listener in list(self._callbacks.on_state):
            try:
                listener(snapshot)
            except Exception:
                self._logger.exception("state_listener.failed")

    def _notify_frame(self, sample: FrameSample) -> None:
        for listener in list(self._callbacks.on_frame):
            try:
                listener(sample)
            except Exception:
                self._logger.exception("frame_listener.failed")

    def _notify_prediction(self, prediction: SignPrediction) -> None:
        for listener in list(self._callbacks.on_prediction):
            try:
                listener(prediction)
            except Exception:
                self._logger.exception("prediction_listener.failed")
