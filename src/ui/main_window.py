"""
Minimal desktop shell.

Layout (no nested panels, no metric dashboards):

    ┌────────────────────────────────────────────────────────┐
    │ SignFlow                                    [ Parar ]  │
    │                                                        │
    │                                                        │
    │                    OLÁ                                 │
    │                                                        │
    │              92% · mão detectada                       │
    │                                                        │
    │                                         ┌───────────┐  │
    │                                         │  preview  │  │
    │                                         └───────────┘  │
    └────────────────────────────────────────────────────────┘

The caption dominates the screen. Everything else (status, preview) is
intentionally discreet.
"""
from __future__ import annotations

import customtkinter as ctk

from src.implementations.config.yaml_configuration import AppConfig
from src.ui.components import PreviewPanel
from src.ui.theme import PALETTE
from src.ui.viewmodels.translation_viewmodel import TranslationViewModel

_FONT_TITLE = ("Segoe UI Semibold", 20)
_FONT_CAPTION = ("Segoe UI", 88, "bold")
_FONT_STATUS = ("Segoe UI", 14)
_FONT_BUTTON = ("Segoe UI Semibold", 13)


class MainWindow(ctk.CTk):
    _REFRESH_INTERVAL_MS = 33  # ~30 fps UI refresh

    def __init__(self, view_model: TranslationViewModel, config: AppConfig) -> None:
        ctk.set_appearance_mode(config.ui.theme)
        ctk.set_default_color_theme("dark-blue")

        super().__init__()
        self._view_model = view_model
        self._config = config
        self._running = False

        self.title(config.ui.window_title)
        self.geometry("1200x720")
        self.minsize(960, 600)
        self.configure(fg_color=PALETTE.bg_primary)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_layout()
        self._schedule_refresh()

    def _build_layout(self) -> None:
        # --- Header -----------------------------------------------------
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=28, pady=(20, 0))

        ctk.CTkLabel(
            header,
            text="SignFlow",
            font=_FONT_TITLE,
            text_color=PALETTE.text_primary,
        ).pack(side="left")

        self._toggle_btn = ctk.CTkButton(
            header,
            text="Parar",
            width=110,
            height=36,
            font=_FONT_BUTTON,
            command=self._handle_toggle,
            fg_color=PALETTE.danger,
            hover_color=PALETTE.danger,
        )
        self._toggle_btn.pack(side="right")

        # --- Center body (caption + status, absolutely positioned) ------
        body = ctk.CTkFrame(self, fg_color=PALETTE.bg_primary)
        body.pack(fill="both", expand=True, padx=28, pady=20)

        self._caption_label = ctk.CTkLabel(
            body,
            text=self._config.ui.language.waiting_text,
            font=_FONT_CAPTION,
            text_color=PALETTE.text_primary,
            wraplength=900,
            justify="center",
        )
        self._caption_label.place(relx=0.5, rely=0.45, anchor="center")

        self._status_label = ctk.CTkLabel(
            body,
            text="",
            font=_FONT_STATUS,
            text_color=PALETTE.text_muted,
        )
        self._status_label.place(relx=0.5, rely=0.60, anchor="center")

        # --- Small preview (bottom-right corner) ------------------------
        self._preview = PreviewPanel(body, preview_size=(320, 180))
        self._preview.place(relx=1.0, rely=1.0, anchor="se")

    # ---- refresh ------------------------------------------------------

    def _schedule_refresh(self) -> None:
        self.after(self._REFRESH_INTERVAL_MS, self._refresh)

    def _refresh(self) -> None:
        try:
            state = self._view_model.take_state()

            self._caption_label.configure(text=state.caption or "…")
            self._status_label.configure(
                text=self._format_status(state),
                text_color=self._status_color(state),
            )

            self._preview.update_view(
                frame_bgr=state.latest_frame,
                landmarks_per_hand=state.landmarks_per_hand,
            )

            self._set_running(state.running)
        finally:
            self._schedule_refresh()

    # ---- helpers ------------------------------------------------------

    @staticmethod
    def _format_status(state) -> str:
        if not state.running:
            return "captura parada"
        if not state.hand_detected:
            return "aproxime a mão da câmera"
        if state.confidence <= 0.0:
            return f"analisando sequência… · buffer {int(state.buffer_fill * 100)}%"
        return f"{int(state.confidence * 100)}% · mão detectada"

    @staticmethod
    def _status_color(state) -> str:
        if not state.running:
            return PALETTE.danger
        if not state.hand_detected:
            return PALETTE.warning
        return PALETTE.accent

    def _set_running(self, running: bool) -> None:
        if running == self._running:
            return
        self._running = running
        if running:
            self._toggle_btn.configure(
                text="Parar",
                fg_color=PALETTE.danger,
                hover_color=PALETTE.danger,
            )
        else:
            self._toggle_btn.configure(
                text="Iniciar",
                fg_color=PALETTE.accent,
                hover_color=PALETTE.accent_dim,
            )

    def _handle_toggle(self) -> None:
        if self._running:
            self._view_model.stop()
        else:
            self._view_model.start()

    def _on_close(self) -> None:
        try:
            self._view_model.stop()
        finally:
            self.destroy()
