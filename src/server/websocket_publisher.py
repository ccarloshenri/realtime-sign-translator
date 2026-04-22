"""
Implements ICaptionPublisher by fanning predictions out over the WebSocket
broadcaster and updating the shared API state.
"""
from __future__ import annotations

from src.models.sign_prediction import SignPrediction
from src.server.state import ApiState
from src.server.websocket import WebSocketBroadcaster


class WebSocketCaptionPublisher:
    def __init__(self, state: ApiState, broadcaster: WebSocketBroadcaster) -> None:
        self._state = state
        self._broadcaster = broadcaster

    def publish(self, prediction: SignPrediction) -> None:
        self._state.set_latest(prediction)
        self._broadcaster.publish_from_any_thread(prediction.to_payload())
