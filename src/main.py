"""
Desktop entry point.

Starts the API server, opens the main window, and runs the Tk main loop.
Camera capture begins automatically; the user can stop/start it from the UI.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _ensure_project_on_path() -> None:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_ensure_project_on_path()

from src.bootstrap import bootstrap  # noqa: E402
from src.ui.main_window import MainWindow  # noqa: E402


def main() -> int:
    services = bootstrap()
    services.api_server.start()
    try:
        services.view_model.start()
        window = MainWindow(
            view_model=services.view_model, config=services.config
        )
        window.mainloop()
    finally:
        services.view_model.stop()
        services.api_server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
