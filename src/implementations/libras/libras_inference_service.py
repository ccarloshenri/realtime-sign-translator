"""
Libras inference service.

High-level facade around the generic `RunTranslationPipeline`, exposing
a Libras-oriented API to callers (UI, CLI, API). It is intentionally
thin: its job is not to duplicate the pipeline, but to give downstream
code a stable, Libras-named entry point and a place for future
Libras-specific concerns (language filtering, sentence compounding,
post-translation cleanup) to land without touching the generic
pipeline.
"""
from __future__ import annotations

from typing import Callable

from src.implementations.pipeline.run_translation_pipeline import (
    RunTranslationPipeline,
)
from src.models.sign_prediction import SignPrediction


CaptionListener = Callable[[SignPrediction], None]


class LibrasInferenceService:
    def __init__(self, pipeline: RunTranslationPipeline) -> None:
        self._pipeline = pipeline

    def start(self) -> None:
        self._pipeline.start()

    def stop(self) -> None:
        self._pipeline.stop()

    def on_caption(self, listener: CaptionListener) -> None:
        """Subscribe a listener to every stabilized Libras caption."""
        self._pipeline.callbacks.on_prediction.append(listener)

    @property
    def is_running(self) -> bool:
        return self._pipeline.snapshot_state().running
