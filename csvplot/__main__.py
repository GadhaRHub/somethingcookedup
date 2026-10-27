"""Entry point: ``python -m csvplot [files...]``."""

import sys

from PySide6.QtWidgets import QApplication

from .main_window import APP_NAME, MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    window = MainWindow()
    window.show()
    window.open_files(sys.argv[1:])
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
