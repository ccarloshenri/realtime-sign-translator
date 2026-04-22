"""
Threaded Uvicorn host for the FastAPI app.

The pipeline and UI run on the main thread; the API event loop runs here.
`start()` spins up a daemon thread; `stop()` asks uvicorn to exit cleanly.
"""
from __future__ import annotations

import asyncio
import threading

import uvicorn
from fastapi import FastAPI

from src.implementations.config.yaml_configuration import AppConfig
from src.interface.logger import ILogger
from src.server.routes import register_routes
from src.server.state import ApiState
from src.server.websocket import WebSocketBroadcaster


class ApiServer:
    def __init__(
        self,
        config: AppConfig,
        state: ApiState,
        broadcaster: WebSocketBroadcaster,
        logger: ILogger,
    ) -> None:
        self._config = config
        self._state = state
        self._broadcaster = broadcaster
        self._logger = logger

        self._app = FastAPI(title="SignFlow Realtime API", version="0.1.0")
        register_routes(self._app, self._state, self._config, self._broadcaster)
        self._app.router.add_event_handler("startup", self._on_startup)

        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def app(self) -> FastAPI:
        return self._app

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        api_cfg = self._config.api
        uv_config = uvicorn.Config(
            self._app,
            host=api_cfg.host,
            port=api_cfg.port,
            log_level="warning",
            loop="asyncio",
            lifespan="on",
        )
        self._server = uvicorn.Server(uv_config)

        self._thread = threading.Thread(
            target=self._serve, name="signflow-api", daemon=True
        )
        self._thread.start()
        self._logger.info(
            "api.started", host=api_cfg.host, port=api_cfg.port
        )

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._logger.info("api.stopped")

    def _serve(self) -> None:
        assert self._server is not None
        try:
            self._server.run()
        except Exception:
            self._logger.exception("api.crashed")

    async def _on_startup(self) -> None:
        self._broadcaster.bind_loop(asyncio.get_running_loop())


def build_api(
    config: AppConfig,
    state: ApiState,
    broadcaster: WebSocketBroadcaster,
    logger: ILogger,
) -> ApiServer:
    return ApiServer(config, state, broadcaster, logger)
