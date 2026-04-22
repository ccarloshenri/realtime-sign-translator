from __future__ import annotations

from typing import Protocol

from src.models.sign_prediction import SignPrediction


class ICaptionPublisher(Protocol):
    """Pushes stabilized captions to some downstream channel (UI, WS, logs)."""

    def publish(self, prediction: SignPrediction) -> None: ...
