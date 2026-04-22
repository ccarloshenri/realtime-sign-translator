"""
MVVM-style ViewModel for the desktop UI.

Subscribes to the pipeline callbacks and keeps a thread-safe snapshot of the
data the UI cares about: latest frame, latest caption, hand detection flag,
buffer fill ratio, FPS. The UI polls `take_state()` from its Tk loop.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np

from src.implementations.pipeline.run_translation_pipeline import (
    PipelineState,
    RunTranslationPipeline,
)
from src.models.frame_sample import FrameSample
from src.models.language import Language
from src.models.sign_prediction import SignPrediction


@dataclass(slots=True)
class UIState:
    latest_frame: np.ndarray | None = None
    caption: str = ""
    confidence: float = 0.0
    hand_detected: bool = False
    buffer_fill: float = 0.0
    fps: float = 0.0
    running: bool = False
    landmarks_per_hand: tuple[np.ndarray, ...] = field(default_factory=tuple)


class TranslationViewModel:
    def __init__(self, pipeline: RunTranslationPipeline, language: Language) -> None:
        self._pipeline = pipeline
        self._language = language
        self._lock = threading.Lock()
        self._state = UIState(caption=language.waiting_text)

        pipeline.callbacks.on_frame.append(self._on_frame)
        pipeline.callbacks.on_prediction.append(self._on_prediction)
        pipeline.callbacks.on_state.append(self._on_state)

    def start(self) -> None:
        self._pipeline.start()

    def stop(self) -> None:
        self._pipeline.stop()

    def take_state(self) -> UIState:
        with self._lock:
            return UIState(
                latest_frame=self._state.latest_frame,
                caption=self._state.caption,
                confidence=self._state.confidence,
                hand_detected=self._state.hand_detected,
                buffer_fill=self._state.buffer_fill,
                fps=self._state.fps,
                running=self._state.running,
                landmarks_per_hand=self._state.landmarks_per_hand,
            )

    def _on_frame(self, sample: FrameSample) -> None:
        landmarks = tuple(hand.points for hand in sample.hands)
        with self._lock:
            self._state.latest_frame = sample.image_bgr
            self._state.landmarks_per_hand = landmarks
            if not sample.has_hand and not self._state.caption:
                self._state.caption = self._language.no_hand_text

    def _on_prediction(self, prediction: SignPrediction) -> None:
        with self._lock:
            self._state.caption = prediction.label
            self._state.confidence = float(prediction.confidence)

    def _on_state(self, state: PipelineState) -> None:
        with self._lock:
            self._state.hand_detected = state.hand_detected
            self._state.buffer_fill = state.buffer_fill
            self._state.fps = state.fps
            self._state.running = state.running
            if not state.hand_detected and self._state.confidence < 0.1:
                self._state.caption = self._language.no_hand_text
