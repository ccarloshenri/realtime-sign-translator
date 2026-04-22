"""
Headless entry point.

Starts the API + pipeline with no UI. Useful for servers or for integrating
SignFlow as a background service that exposes captions over WebSocket.
"""
from __future__ import annotations

import signal
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bootstrap import bootstrap  # noqa: E402


def main() -> int:
    services = bootstrap()
    services.api_server.start()
    services.view_model.start()

    stop_event = threading.Event()

    def _handle_signal(*_: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)

    services.logger.info("headless.ready", api_port=services.config.api.port)
    try:
        stop_event.wait()
    finally:
        services.view_model.stop()
        services.api_server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
