"""
Settings panel.

Exposes a few read-only-ish summaries of the active configuration plus the
start/stop toggle. Configuration lives in `config.yaml`; the UI only
*reflects* it. Full editing belongs elsewhere (a settings dialog wired to
the config provider's `reload()`).
"""
from __future__ import annotations

from typing import Callable

import customtkinter as ctk

from src.implementations.config.yaml_configuration import AppConfig
from src.ui.theme import FONT_HEADING, FONT_MONO, FONT_SMALL, PALETTE


class SettingsPanel(ctk.CTkFrame):
    def __init__(
        self,
        master,
        config: AppConfig,
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
    ) -> None:
        super().__init__(master, fg_color=PALETTE.bg_card, corner_radius=12)
        self._on_start = on_start
        self._on_stop = on_stop

        ctk.CTkLabel(
            self,
            text="Configuração ativa",
            font=FONT_HEADING,
            text_color=PALETTE.text_primary,
        ).pack(anchor="w", padx=20, pady=(16, 8))

        self._info_label = ctk.CTkLabel(
            self,
            text=self._render_config(config),
            font=FONT_MONO,
            text_color=PALETTE.text_muted,
            justify="left",
        )
        self._info_label.pack(anchor="w", padx=20, pady=(0, 12))

        self._api_label = ctk.CTkLabel(
            self,
            text=f"API local: http://{config.api.host}:{config.api.port}",
            font=FONT_SMALL,
            text_color=PALETTE.accent,
        )
        self._api_label.pack(anchor="w", padx=20, pady=(0, 12))

        button_row = ctk.CTkFrame(self, fg_color="transparent")
        button_row.pack(fill="x", padx=20, pady=(0, 16))

        self._toggle_btn = ctk.CTkButton(
            button_row,
            text="Iniciar captura",
            command=self._handle_toggle,
            fg_color=PALETTE.accent,
            hover_color=PALETTE.accent_dim,
            height=40,
        )
        self._toggle_btn.pack(fill="x")

        self._running = False

    def set_running(self, running: bool) -> None:
        self._running = running
        if running:
            self._toggle_btn.configure(
                text="Parar captura",
                fg_color=PALETTE.danger,
                hover_color=PALETTE.danger,
            )
        else:
            self._toggle_btn.configure(
                text="Iniciar captura",
                fg_color=PALETTE.accent,
                hover_color=PALETTE.accent_dim,
            )

    def _handle_toggle(self) -> None:
        if self._running:
            self._on_stop()
        else:
            self._on_start()

    @staticmethod
    def _render_config(config: AppConfig) -> str:
        return (
            f"câmera          #{config.camera.device_index} "
            f"{config.camera.width}x{config.camera.height}@{config.camera.target_fps}fps\n"
            f"seq. temporal   {config.pipeline.sequence_length} frames\n"
            f"limiar conf.    {config.pipeline.min_confidence:.2f}\n"
            f"smoothing       {config.pipeline.smoothing_window} preds / "
            f"{config.pipeline.min_dwell_frames} dwell\n"
            f"classificador   {config.classifier.backend}\n"
            f"idioma          {config.ui.language.value}"
        )
