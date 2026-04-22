"""
Thread-safe shared state for the API layer.

The realtime pipeline runs in its own thread and the API runs in an asyncio
event loop, so everything shared between the two lives here behind a lock.
"""
from __future__ import annotations

import threading

from src.models.sign_prediction import SignPrediction


class ApiState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: SignPrediction | None = None
        self._pipeline_running = False

    def set_latest(self, prediction: SignPrediction) -> None:
        with self._lock:
            self._latest = prediction

    def get_latest(self) -> SignPrediction | None:
        with self._lock:
            return self._latest

    def set_pipeline_running(self, running: bool) -> None:
        with self._lock:
            self._pipeline_running = running

    def is_pipeline_running(self) -> bool:
        with self._lock:
            return self._pipeline_running
