"""
Downloads the MediaPipe HandLandmarker model asset required by the new
Tasks API (the legacy `mediapipe.solutions` backend is deprecated on
Python 3.12+).

Usage:
    python scripts/download_models.py
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
_DEST = Path("artifacts") / "hand_landmarker.task"


def main() -> int:
    _DEST.parent.mkdir(parents=True, exist_ok=True)
    if _DEST.exists() and _DEST.stat().st_size > 0:
        print(f"[ok] already present: {_DEST} ({_DEST.stat().st_size / 1e6:.1f} MB)")
        return 0

    print(f"[..] downloading {_URL}")

    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        pct = min(100.0, downloaded * 100 / max(total_size, 1))
        sys.stdout.write(
            f"\r    {pct:5.1f}%  {downloaded / 1e6:5.1f} MB / {total_size / 1e6:.1f} MB"
        )
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(_URL, _DEST, reporthook=_progress)
    except Exception as exc:
        print(f"\n[error] download failed: {exc}")
        if _DEST.exists():
            _DEST.unlink()
        return 1

    print(f"\n[ok] saved to {_DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
