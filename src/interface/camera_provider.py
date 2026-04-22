from __future__ import annotations

from typing import Protocol

import numpy as np


class ICameraProvider(Protocol):
    """Produces BGR frames from a capture device."""

    def open(self) -> None: ...

    def read(self) -> np.ndarray | None: ...

    def close(self) -> None: ...

    @property
    def is_open(self) -> bool: ...
