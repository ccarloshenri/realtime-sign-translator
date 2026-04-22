"""
Downloads the MediaPipe model assets used by the Tasks API backends.

Usage:
    python scripts/download_models.py

Files downloaded into `artifacts/`:
    - hand_landmarker.task        (required by every backend for UI landmarks)
    - gesture_recognizer.task     (required only if classifier.backend == "gesture_recognizer")
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

_ARTIFACTS = Path("artifacts")

_ASSETS = [
    (
        _ARTIFACTS / "hand_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task",
    ),
    (
        _ARTIFACTS / "gesture_recognizer.task",
        "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
        "gesture_recognizer/float16/1/gesture_recognizer.task",
    ),
]


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    pct = min(100.0, downloaded * 100 / max(total_size, 1))
    sys.stdout.write(
        f"\r    {pct:5.1f}%  {downloaded / 1e6:5.1f} MB / {total_size / 1e6:.1f} MB"
    )
    sys.stdout.flush()


def _download_one(dest: Path, url: str) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[ok] already present: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return True

    print(f"[..] downloading {url}")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
    except Exception as exc:
        print(f"\n[error] download failed: {exc}")
        if dest.exists():
            dest.unlink()
        return False
    print(f"\n[ok] saved to {dest}")
    return True


def main() -> int:
    failures = 0
    for dest, url in _ASSETS:
        if not _download_one(dest, url):
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
