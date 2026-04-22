"""Local API + WebSocket — exposes captions to external integrations."""

from .server import ApiServer, build_api
from .state import ApiState
from .websocket import WebSocketBroadcaster
from .websocket_publisher import WebSocketCaptionPublisher

__all__ = [
    "ApiServer",
    "ApiState",
    "WebSocketBroadcaster",
    "WebSocketCaptionPublisher",
    "build_api",
]
