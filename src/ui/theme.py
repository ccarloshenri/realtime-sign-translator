"""
Shared color palette and sizing constants for the desktop UI.

Keeping this in one place makes it trivial to reskin the app and avoids
magic-string colors scattered across the components.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Palette:
    bg_primary: str = "#0f1115"
    bg_secondary: str = "#161a22"
    bg_card: str = "#1c2230"
    accent: str = "#4ea1ff"
    accent_dim: str = "#2b6bc4"
    success: str = "#32d583"
    danger: str = "#f27573"
    warning: str = "#f8c555"
    text_primary: str = "#eef1f7"
    text_muted: str = "#8a93a6"
    border: str = "#232a3a"


PALETTE = Palette()

FONT_TITLE = ("Segoe UI Semibold", 22)
FONT_CAPTION = ("Segoe UI", 48, "bold")
FONT_HEADING = ("Segoe UI Semibold", 14)
FONT_BODY = ("Segoe UI", 12)
FONT_SMALL = ("Segoe UI", 11)
FONT_MONO = ("Consolas", 11)
