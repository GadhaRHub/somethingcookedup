"""Entry point: ``python -m csvplot [files...]``."""

import sys

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from .main_window import APP_NAME, MainWindow, app_icon
from .theme import DEFAULT_THEME, apply_theme


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    apply_theme(app, QSettings("csvplot", "csvplot").value("theme", DEFAULT_THEME))
    app.setWindowIcon(app_icon())
    window = MainWindow()
    window.show()
    window.setFocus()  # don't start with the cursor in the session-name box
    window.open_files(sys.argv[1:])
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
