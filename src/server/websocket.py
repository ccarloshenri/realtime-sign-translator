"""
Async WebSocket broadcaster.

Receives predictions from the (non-async) pipeline thread via
`publish_from_any_thread` and fans them out to every connected client over
the server's event loop.
"""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import WebSocket

from src.interface.logger import ILogger


class WebSocketBroadcaster:
    def __init__(self, logger: ILogger) -> None:
        self._logger = logger
        self._clients: set[WebSocket] = set()
        self._clients_lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    @property
    def client_count(self) -> int:
        return len(self._clients)

    async def register(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._clients_lock:
            self._clients.add(ws)
        self._logger.info("ws.connected", clients=len(self._clients))

    async def unregister(self, ws: WebSocket) -> None:
        async with self._clients_lock:
            self._clients.discard(ws)
        self._logger.info("ws.disconnected", clients=len(self._clients))

    async def broadcast(self, payload: dict[str, Any]) -> None:
        if not self._clients:
            return
        dead: list[WebSocket] = []
        for ws in list(self._clients):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._clients_lock:
                for ws in dead:
                    self._clients.discard(ws)
            self._logger.warning("ws.pruned_dead_clients", count=len(dead))

    def publish_from_any_thread(self, payload: dict[str, Any]) -> None:
        if self._loop is None or self._loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(self.broadcast(payload), self._loop)
