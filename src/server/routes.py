"""FastAPI routes: status, config, latest prediction, WebSocket."""
from __future__ import annotations

from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect

from src.implementations.config.yaml_configuration import AppConfig
from src.server.state import ApiState
from src.server.websocket import WebSocketBroadcaster


def register_routes(
    app: FastAPI,
    state: ApiState,
    config: AppConfig,
    broadcaster: WebSocketBroadcaster,
) -> None:
    router = APIRouter()

    @router.get("/status")
    def get_status() -> dict:
        latest = state.get_latest()
        return {
            "service": "signflow-realtime",
            "version": "0.1.0",
            "pipeline_running": state.is_pipeline_running(),
            "ws_clients": broadcaster.client_count,
            "latest_prediction": latest.to_payload() if latest else None,
        }

    @router.get("/config")
    def get_config() -> dict:
        return config.model_dump(mode="json")

    @router.get("/predictions/latest")
    def get_latest_prediction() -> dict:
        latest = state.get_latest()
        if latest is None:
            return {"prediction": None}
        return {"prediction": latest.to_payload()}

    app.include_router(router)

    if config.api.enable_websocket:

        @app.websocket("/ws/captions")
        async def captions_ws(websocket: WebSocket) -> None:
            await broadcaster.register(websocket)
            try:
                latest = state.get_latest()
                if latest is not None:
                    await websocket.send_json(latest.to_payload())
                while True:
                    await websocket.receive_text()
            except WebSocketDisconnect:
                pass
            finally:
                await broadcaster.unregister(websocket)
