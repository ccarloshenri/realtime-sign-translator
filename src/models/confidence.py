"""Bounded [0, 1] probability."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Confidence:
    value: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.value!r}")

    def __float__(self) -> float:
        return self.value

    def __lt__(self, other: "Confidence | float") -> bool:
        return self.value < float(other)

    def __ge__(self, other: "Confidence | float") -> bool:
        return self.value >= float(other)

    def as_percent(self) -> float:
        return self.value * 100.0
