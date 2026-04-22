"""
Live webcam preview.

Overlays MediaPipe hand landmarks on the frame so the user can see that the
extractor is actually tracking their hands. Shows a "HAND DETECTED" /
"NO HAND" pill, an FPS counter, and the temporal buffer fill bar.
"""
from __future__ import annotations

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image

from src.ui.theme import FONT_HEADING, FONT_SMALL, PALETTE

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
        super().__init__(master, fg_color=PALETTE.bg_card, corner_radius=12)
        self._preview_size = preview_size

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=16, pady=(12, 8))

        ctk.CTkLabel(
            header,
            text="Preview",
            font=FONT_HEADING,
            text_color=PALETTE.text_primary,
        ).pack(side="left")

        self._status_pill = ctk.CTkLabel(
            header,
            text="● aguardando",
            font=FONT_SMALL,
            fg_color=PALETTE.bg_secondary,
            text_color=PALETTE.text_muted,
            corner_radius=999,
            padx=12,
            pady=4,
        )
        self._status_pill.pack(side="right")

        self._fps_label = ctk.CTkLabel(
            header,
            text="0 FPS",
            font=FONT_SMALL,
            text_color=PALETTE.text_muted,
        )
        self._fps_label.pack(side="right", padx=(0, 12))

        self._image_label = ctk.CTkLabel(
            self,
            text="",
            width=preview_size[0],
            height=preview_size[1],
            fg_color=PALETTE.bg_primary,
            corner_radius=8,
        )
        self._image_label.pack(padx=16, pady=(0, 8))

        self._buffer_bar = ctk.CTkProgressBar(
            self,
            height=6,
            progress_color=PALETTE.accent,
            fg_color=PALETTE.bg_secondary,
        )
        self._buffer_bar.set(0.0)
        self._buffer_bar.pack(fill="x", padx=16, pady=(0, 12))

        self._placeholder = self._build_placeholder()
        self._image_label.configure(image=self._placeholder)
        self._current_image: ctk.CTkImage | None = self._placeholder

    def update_view(
        self,
        frame_bgr: np.ndarray | None,
        landmarks_per_hand: tuple[np.ndarray, ...],
        hand_detected: bool,
        buffer_fill: float,
        fps: float,
    ) -> None:
        self._buffer_bar.set(max(0.0, min(1.0, buffer_fill)))
        self._fps_label.configure(text=f"{fps:.0f} FPS")

        if hand_detected:
            self._status_pill.configure(
                text="● mão detectada",
                text_color=PALETTE.success,
            )
        else:
            self._status_pill.configure(
                text="○ sem mão",
                text_color=PALETTE.text_muted,
            )

        if frame_bgr is None:
            self._image_label.configure(image=self._placeholder)
            self._current_image = self._placeholder
            return

        rendered = self._render(frame_bgr, landmarks_per_hand)
        self._image_label.configure(image=rendered)
        self._current_image = rendered

    def _render(
        self,
        frame_bgr: np.ndarray,
        landmarks_per_hand: tuple[np.ndarray, ...],
    ) -> ctk.CTkImage:
        h, w = frame_bgr.shape[:2]
        if landmarks_per_hand:
            self._draw_landmarks(frame_bgr, landmarks_per_hand, w, h)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        pil_image = pil_image.resize(self._preview_size, Image.BILINEAR)
        return ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=self._preview_size)

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
