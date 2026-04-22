"""
Initial Libras vocabulary.

These 10 signs are the seed vocabulary used by the MVP — enough to
exercise the full collect/train/infer loop end-to-end. Real Libras has
more than 10.000 signs; everything here is designed to grow (the live
classifier reads the actual vocabulary from `artifacts/libras_labels.json`
after training, so this enum is only the starting point).

Keeping it as an Enum — instead of free-floating strings scattered across
collectors, trainers and the runtime — gives us a single typed
declaration every layer can import without risk of typos.
"""
from __future__ import annotations

from enum import Enum


class LibrasSign(str, Enum):
    OLA = "olá"
    TCHAU = "tchau"
    SIM = "sim"
    NAO = "não"
    OBRIGADO = "obrigado"
    POR_FAVOR = "por favor"
    AJUDA = "ajuda"
    EU = "eu"
    VOCE = "você"
    AMOR = "amor"

    @classmethod
    def from_label(cls, label: str) -> "LibrasSign | None":
        normalized = (label or "").strip().casefold()
        for sign in cls:
            if sign.value.casefold() == normalized:
                return sign
        return None


LIBRAS_BASE_VOCABULARY: tuple[str, ...] = tuple(sign.value for sign in LibrasSign)
