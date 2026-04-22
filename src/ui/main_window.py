"""
Desktop shell.

Polls the TranslationViewModel at ~30 Hz and pushes its current state into
the child panels. The UI is intentionally stateless — it just mirrors what
the ViewModel tells it to show.
"""
from __future__ import annotations

import customtkinter as ctk

from src.implementations.config.yaml_configuration import AppConfig
from src.ui.components import (
    CaptionPanel,
    ConfidencePanel,
    PreviewPanel,
    SettingsPanel,
)
from src.ui.theme import FONT_SMALL, FONT_TITLE, PALETTE
from src.ui.viewmodels.translation_viewmodel import TranslationViewModel


class MainWindow(ctk.CTk):
    _REFRESH_INTERVAL_MS = 33  # ~30 fps UI refresh

    def __init__(self, view_model: TranslationViewModel, config: AppConfig) -> None:
        ctk.set_appearance_mode(config.ui.theme)
        ctk.set_default_color_theme("dark-blue")

        super().__init__()
        self._view_model = view_model
        self._config = config

        self.title(config.ui.window_title)
        self.geometry("1280x760")
        self.minsize(1100, 680)
        self.configure(fg_color=PALETTE.bg_primary)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_layout()
        self._schedule_refresh()

    def _build_layout(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent", height=64)
        header.pack(fill="x", padx=24, pady=(20, 8))

        ctk.CTkLabel(
            header,
            text="SignFlow Realtime",
            font=FONT_TITLE,
            text_color=PALETTE.text_primary,
        ).pack(side="left")

        ctk.CTkLabel(
            header,
            text=f"realtime sign translator · backend: {self._config.classifier.backend}",
            font=FONT_SMALL,
            text_color=PALETTE.text_muted,
        ).pack(side="left", padx=16, pady=4)

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=24, pady=(8, 16))

        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=2)
        body.grid_rowconfigure(0, weight=0)
        body.grid_rowconfigure(1, weight=1)

        preview_size = (
            self._config.ui.preview_width,
            self._config.ui.preview_height,
        )

        self._preview = PreviewPanel(body, preview_size=preview_size)
        self._preview.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 12))

        self._caption = CaptionPanel(
            body,
            placeholder_text=self._config.ui.language.waiting_text,
        )
        self._caption.grid(row=0, column=1, sticky="nsew", pady=(0, 12))

        right = ctk.CTkFrame(body, fg_color="transparent")
        right.grid(row=1, column=1, sticky="nsew")
        right.grid_rowconfigure(0, weight=0)
        right.grid_rowconfigure(1, weight=0)
        right.grid_columnconfigure(0, weight=1)

        self._confidence = ConfidencePanel(right)
        self._confidence.grid(row=0, column=0, sticky="ew", pady=(0, 12))

        self._settings = SettingsPanel(
            right,
            config=self._config,
            on_start=self._view_model.start,
            on_stop=self._view_model.stop,
        )
        self._settings.grid(row=1, column=0, sticky="ew")

    def _schedule_refresh(self) -> None:
        self.after(self._REFRESH_INTERVAL_MS, self._refresh)

    def _refresh(self) -> None:
        try:
            state = self._view_model.take_state()
            self._preview.update_view(
                frame_bgr=state.latest_frame,
                landmarks_per_hand=state.landmarks_per_hand,
                hand_detected=state.hand_detected,
                buffer_fill=state.buffer_fill,
                fps=state.fps,
            )
            self._caption.update_view(
                caption=state.caption,
                running=state.running,
                hand_detected=state.hand_detected,
            )
            self._confidence.update_view(
                confidence=state.confidence,
                buffer_fill=state.buffer_fill,
                fps=state.fps,
            )
            self._settings.set_running(state.running)
        finally:
            self._schedule_refresh()

    def _on_close(self) -> None:
        try:
            self._view_model.stop()
        finally:
            self.destroy()
