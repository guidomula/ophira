"""
main.py — OPHIRA entry point (Optical Thin Film Reflectance Analyzer)
Creates AppState, FitEngine and MainWindow, then starts the Qt loop.

Compatibility: PyQt6 (preferred) with automatic fallback to PyQt5.
"""
import sys
import os

# PyQt6 / PyQt5 compatibility
try:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
    PYQT_VERSION = 6
except ImportError:
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt
    PYQT_VERSION = 5

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(__file__))

import physics as _ph
_ph._reload_material_cache()   # clear the CSV cache at every startup

from app_state import AppState
from fit_engine import FitEngine
from windows.main_window import MainWindow


def main():
    # In Spyder a Qt app may already exist — reuse it if present
    app = QApplication.instance()
    if app is None:
        # Enable DPI scaling BEFORE creating the app (required by PyQt5)
        if PYQT_VERSION == 5:
            QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
            QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        app = QApplication(sys.argv)

    app.setApplicationName("PSi Reflectivity fitting App")
    app.setOrganizationName("Laboratorio PoroSiLab - UniCA")

    # State shared across all windows
    state = AppState()

    # Fit engine — no UI dependency
    engine = FitEngine(state)

    # Main window
    window = MainWindow(state, engine)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
