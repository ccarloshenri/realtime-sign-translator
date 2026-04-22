"""Stable, serializable output of the temporal classifier."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.models.confidence import Confidence


@dataclass(frozen=True, slots=True)
class SignPrediction:
    label: str
    confidence: Confidence
    sequence_size: int
    source: str = "camera"
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_payload(self) -> dict:
        return {
            "text": self.label,
            "confidence": round(float(self.confidence), 4),
            "timestamp": self.timestamp.astimezone(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "source": self.source,
            "sequence_size": self.sequence_size,
        }
