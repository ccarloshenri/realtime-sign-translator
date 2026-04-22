"""
Minimal webcam preview.

Shows the current frame with MediaPipe hand landmarks drawn on top — no
pills, no FPS, no bar charts. The parent window owns status indicators.
"""
from __future__ import annotations

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image

from src.ui.theme import PALETTE

_HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


class PreviewPanel(ctk.CTkFrame):
    def __init__(self, master, preview_size: tuple[int, int]) -> None:
        super().__init__(
            master,
            fg_color=PALETTE.bg_card,
            corner_radius=12,
            border_width=1,
            border_color=PALETTE.border,
        )
        self._preview_size = preview_size

        self._image_label = ctk.CTkLabel(
            self,
            text="",
            width=preview_size[0],
            height=preview_size[1],
            fg_color=PALETTE.bg_primary,
            corner_radius=8,
        )
        self._image_label.pack(padx=6, pady=6)

        self._placeholder = self._build_placeholder()
        self._image_label.configure(image=self._placeholder)
        self._current_image: ctk.CTkImage | None = self._placeholder

    def update_view(
        self,
        frame_bgr: np.ndarray | None,
        landmarks_per_hand: tuple[np.ndarray, ...],
    ) -> None:
        if frame_bgr is None:
            self._image_label.configure(image=self._placeholder)
            self._current_image = self._placeholder
            return

        rendered = self._render(frame_bgr, landmarks_per_hand)
        self._image_label.configure(image=rendered)
        self._current_image = rendered  # Keep the reference alive.

    def _render(
        self,
        frame_bgr: np.ndarray,
        landmarks_per_hand: tuple[np.ndarray, ...],
    ) -> ctk.CTkImage:
        h, w = frame_bgr.shape[:2]
        if landmarks_per_hand:
            self._draw_landmarks(frame_bgr, landmarks_per_hand, w, h)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb).resize(
            self._preview_size, Image.BILINEAR
        )
        return ctk.CTkImage(
            light_image=pil_image,
            dark_image=pil_image,
            size=self._preview_size,
        )

    @staticmethod
    def _draw_landmarks(
        frame: np.ndarray,
        landmarks_per_hand: tuple[np.ndarray, ...],
        w: int,
        h: int,
    ) -> None:
        for pts in landmarks_per_hand:
            px = (pts[:, :2] * np.array([w, h], dtype=np.float32)).astype(int)
            for a, b in _HAND_CONNECTIONS:
                if a < len(px) and b < len(px):
                    cv2.line(frame, tuple(px[a]), tuple(px[b]), (78, 161, 255), 2)
            for (x, y) in px:
                cv2.circle(frame, (int(x), int(y)), 3, (255, 255, 255), -1)

    def _build_placeholder(self) -> ctk.CTkImage:
        placeholder = Image.new("RGB", self._preview_size, (15, 17, 21))
        return ctk.CTkImage(
            light_image=placeholder,
            dark_image=placeholder,
            size=self._preview_size,
        )
