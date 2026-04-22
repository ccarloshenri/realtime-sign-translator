"""Confidence meter + buffer fill + FPS indicator."""
from __future__ import annotations

import customtkinter as ctk

from src.ui.theme import FONT_HEADING, FONT_SMALL, PALETTE


class ConfidencePanel(ctk.CTkFrame):
    def __init__(self, master) -> None:
        super().__init__(master, fg_color=PALETTE.bg_card, corner_radius=12)

        ctk.CTkLabel(
            self,
            text="Métricas",
            font=FONT_HEADING,
            text_color=PALETTE.text_primary,
        ).pack(anchor="w", padx=20, pady=(16, 8))

        self._confidence_row, self._confidence_value, self._confidence_bar = (
            self._build_metric_row("Confiança", PALETTE.success)
        )
        self._confidence_row.pack(fill="x", padx=20, pady=4)

        self._buffer_row, self._buffer_value, self._buffer_bar = (
            self._build_metric_row("Buffer temporal", PALETTE.accent)
        )
        self._buffer_row.pack(fill="x", padx=20, pady=4)

        self._fps_row, self._fps_value, _ = (
            self._build_metric_row("FPS", PALETTE.warning, show_bar=False)
        )
        self._fps_row.pack(fill="x", padx=20, pady=(4, 16))

    def _build_metric_row(
        self,
        label_text: str,
        color: str,
        show_bar: bool = True,
    ) -> tuple[ctk.CTkFrame, ctk.CTkLabel, ctk.CTkProgressBar | None]:
        row = ctk.CTkFrame(self, fg_color="transparent")

        header = ctk.CTkFrame(row, fg_color="transparent")
        header.pack(fill="x")
        ctk.CTkLabel(
            header,
            text=label_text,
            font=FONT_SMALL,
            text_color=PALETTE.text_muted,
        ).pack(side="left")
        value_label = ctk.CTkLabel(
            header,
            text="—",
            font=FONT_SMALL,
            text_color=PALETTE.text_primary,
        )
        value_label.pack(side="right")

        bar: ctk.CTkProgressBar | None = None
        if show_bar:
            bar = ctk.CTkProgressBar(
                row,
                height=8,
                progress_color=color,
                fg_color=PALETTE.bg_secondary,
            )
            bar.set(0.0)
            bar.pack(fill="x", pady=(6, 0))

        return row, value_label, bar

    def update_view(self, confidence: float, buffer_fill: float, fps: float) -> None:
        self._confidence_value.configure(text=f"{confidence * 100:.1f}%")
        if self._confidence_bar is not None:
            self._confidence_bar.set(max(0.0, min(1.0, confidence)))

        self._buffer_value.configure(text=f"{buffer_fill * 100:.0f}%")
        if self._buffer_bar is not None:
            self._buffer_bar.set(max(0.0, min(1.0, buffer_fill)))

        self._fps_value.configure(text=f"{fps:.1f}")
