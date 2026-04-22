from enum import Enum


class Handedness(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"

    @classmethod
    def from_mediapipe(cls, label: str) -> "Handedness":
        normalized = (label or "").strip().lower()
        if normalized == "left":
            return cls.LEFT
        if normalized == "right":
            return cls.RIGHT
        return cls.UNKNOWN
