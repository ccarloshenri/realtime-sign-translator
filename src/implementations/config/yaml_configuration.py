"""
YAML-backed configuration loader.

All top-level sections are validated by Pydantic v2 models, which gives us
type-checked access throughout the rest of the codebase.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from src.models.language import Language


class CameraConfig(BaseModel):
    device_index: int = 0
    width: int = 1280
    height: int = 720
    target_fps: int = 30
    flip_horizontal: bool = True


class VisionConfig(BaseModel):
    max_num_hands: int = Field(default=2, ge=1, le=4)
    min_detection_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    min_tracking_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    model_complexity: Literal[0, 1] = 1
    model_path: str = "artifacts/hand_landmarker.task"


class PipelineConfig(BaseModel):
    sequence_length: int = Field(default=30, ge=1)
    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    smoothing_window: int = Field(default=5, ge=1)
    min_dwell_frames: int = Field(default=4, ge=1)
    publish_unchanged: bool = False


class ClassifierConfig(BaseModel):
    backend: Literal["mock", "keras", "gesture_recognizer", "libras"] = "mock"
    model_path: str = "artifacts/signflow_lstm.keras"
    labels_path: str = "artifacts/labels.json"
    gesture_model_path: str = "artifacts/gesture_recognizer.task"
    mock_vocabulary: list[str] = Field(
        default_factory=lambda: ["olá", "obrigado", "ajuda"]
    )

    @field_validator("mock_vocabulary")
    @classmethod
    def _vocab_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("mock_vocabulary must contain at least one label")
        return v


class LibrasConfig(BaseModel):
    model_path: str = "artifacts/libras_lstm.keras"
    labels_path: str = "artifacts/libras_labels.json"


class UIConfig(BaseModel):
    language: Language = Language.PT_BR
    theme: Literal["dark", "light", "system"] = "dark"
    window_title: str = "SignFlow Realtime"
    preview_width: int = 720
    preview_height: int = 405


class ApiConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1, le=65535)
    enable_websocket: bool = True


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    file: str | None = "logs/signflow.log"
    console: bool = True


class AppConfig(BaseModel):
    camera: CameraConfig = Field(default_factory=CameraConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    libras: LibrasConfig = Field(default_factory=LibrasConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


class YamlConfigurationProvider:
    """Loads (and optionally reloads) an AppConfig from a YAML file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._cached: AppConfig | None = None

    def get(self) -> AppConfig:
        if self._cached is None:
            self._cached = self._load()
        return self._cached

    def reload(self) -> AppConfig:
        self._cached = self._load()
        return self._cached

    def _load(self) -> AppConfig:
        if not self._path.exists():
            return AppConfig()
        with self._path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return AppConfig.model_validate(raw)
