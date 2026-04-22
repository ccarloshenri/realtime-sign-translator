"""Central caption area — the main thing the user looks at."""
from __future__ import annotations

import customtkinter as ctk

from src.ui.theme import (
    FONT_CAPTION,
    FONT_HEADING,
    FONT_SMALL,
    PALETTE,
)


class CaptionPanel(ctk.CTkFrame):
    def __init__(self, master, placeholder_text: str) -> None:
        super().__init__(master, fg_color=PALETTE.bg_card, corner_radius=12)

        ctk.CTkLabel(
            self,
            text="Legenda em tempo real",
            font=FONT_HEADING,
            text_color=PALETTE.text_muted,
        ).pack(anchor="w", padx=20, pady=(16, 4))

        self._caption_label = ctk.CTkLabel(
            self,
            text=placeholder_text,
            font=FONT_CAPTION,
            text_color=PALETTE.text_primary,
            wraplength=640,
            justify="center",
        )
        self._caption_label.pack(expand=True, fill="both", padx=20, pady=8)

        self._hint_label = ctk.CTkLabel(
            self,
            text="aguardando sequência temporal…",
            font=FONT_SMALL,
            text_color=PALETTE.text_muted,
        )
        self._hint_label.pack(anchor="w", padx=20, pady=(0, 16))

    def update_view(self, caption: str, running: bool, hand_detected: bool) -> None:
        self._caption_label.configure(text=caption or "…")

        if not running:
            hint = "captura parada"
            color = PALETTE.danger
        elif not hand_detected:
            hint = "aproxime a mão da câmera"
            color = PALETTE.warning
        else:
            hint = "analisando sequência…"
            color = PALETTE.accent

        self._hint_label.configure(text=hint, text_color=color)
