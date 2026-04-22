"""
Libras caption service.

Wraps the generic `ICaptionPublisher` so every caption produced while
the Libras backend is active is re-tagged with `source="libras"` before
being forwarded. External consumers (WebSocket clients, OBS overlays,
log pipelines) can then tell Libras captions apart from other sources
like ASR or static gesture recognizers — important once the project
grows to support multi-modal input.
"""
from __future__ import annotations

from dataclasses import replace

from src.interface.caption_publisher import ICaptionPublisher
from src.models.sign_prediction import SignPrediction

LIBRAS_SOURCE = "libras"


class LibrasCaptionService:
    def __init__(self, downstream: ICaptionPublisher) -> None:
        self._downstream = downstream

    def publish(self, prediction: SignPrediction) -> None:
        tagged = (
            prediction
            if prediction.source == LIBRAS_SOURCE
            else replace(prediction, source=LIBRAS_SOURCE)
        )
        self._downstream.publish(tagged)
