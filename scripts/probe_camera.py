"""
Probe available camera devices + backends.

Run this when `python -m src.main` fails with
`Could not open camera <N>`. It tries every combination of device index
and OpenCV backend and prints which ones deliver a usable frame.

Usage:
    python scripts/probe_camera.py
"""
from __future__ import annotations

import cv2

_BACKENDS = [
    ("CAP_DSHOW", cv2.CAP_DSHOW),
    ("CAP_MSMF", cv2.CAP_MSMF),
    ("CAP_ANY", cv2.CAP_ANY),
]


def main() -> int:
    print("Probing cameras... (this takes a few seconds)\n")
    working: list[tuple[int, str]] = []

    for index in range(6):
        for name, backend in _BACKENDS:
            cap = cv2.VideoCapture(index, backend)
            opened = cap.isOpened()
            got_frame = False
            shape = None
            if opened:
                ok, frame = cap.read()
                if ok and frame is not None:
                    got_frame = True
                    shape = frame.shape
            cap.release()

            status = "OK" if got_frame else ("OPEN_NO_FRAME" if opened else "FAIL")
            shape_str = f" shape={shape}" if shape is not None else ""
            print(f"  index={index}  backend={name:<10}  {status}{shape_str}")
            if got_frame:
                working.append((index, name))

    print()
    if not working:
        print("No camera returned a frame. Check that:")
        print("  1. No other app (Teams, Zoom, browser) is holding the camera.")
        print("  2. Windows Settings > Privacy > Camera allows desktop apps.")
        print("  3. The camera is actually attached (open Windows Camera app).")
        return 1

    print("Working combinations:")
    for index, name in working:
        print(f"  device_index={index} backend={name}")
    print()
    print("Set the winning device_index in config.yaml under `camera:`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
