from __future__ import annotations

from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from src.implementations.config.yaml_configuration import AppConfig


class IConfigurationProvider(Protocol):
    """Access to validated, typed configuration."""

    def get(self) -> "AppConfig": ...

    def reload(self) -> "AppConfig": ...
