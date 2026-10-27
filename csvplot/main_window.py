"""Main application window: menus, file opening and the tab of plots."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QTabWidget,
)

from . import __version__
from .loaders import file_dialog_filter, load_file
from .plot_tab import PlotTab

APP_NAME = "CSV Plotter"

MOUSE_HELP = """
<h3>Mouse &amp; keyboard</h3>
<table cellpadding="3">
<tr><td><b>Mouse wheel</b></td><td>Zoom around the cursor (Shift: x only, Ctrl: y only)</td></tr>
<tr><td><b>Pan tool</b> (✥)</td><td>Left-drag to move the plot, right-drag to stretch axes</td></tr>
<tr><td><b>Zoom tool</b> (🔍)</td><td>Drag a rectangle to zoom into it</td></tr>
<tr><td><b>Home / ← / →</b></td><td>Reset view, previous / next view</td></tr>
<tr><td><b>Mark point</b></td><td>Click to mark a point (snaps to data if enabled)</td></tr>
<tr><td><b>Mark region</b></td><td>Drag horizontally to shade an x-range</td></tr>
<tr><td><b>Right-click a mark</b></td><td>Remove it</td></tr>
<tr><td><b>Legend / mark labels</b></td><td>Drag them to move them</td></tr>
</table>
<p>Cursor coordinates are shown under the plot.</p>
"""


class DetachedWindow(QMainWindow):
    """Hosts a PlotTab in its own window; closing it docks the plot back."""

    def __init__(self, tab: PlotTab, main: "MainWindow"):
        super().__init__(main, Qt.Window)
        self.tab = tab
        self.main = main
        self.setWindowTitle(f"{tab.title} — {APP_NAME}")
        self.setCentralWidget(tab)
        self.resize(1200, 750)

    def closeEvent(self, event) -> None:
        tab = self.takeCentralWidget()
        if tab is not None:
            self.main.attach_tab(tab)
        self.main.detached.remove(self)
        super().closeEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1300, 820)
        self.setAcceptDrops(True)
        self.settings = QSettings("csvplot", "csvplot")
        self.detached: list[DetachedWindow] = []

        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.setMovable(True)
        self.tabs.setDocumentMode(True)
        self.tabs.tabCloseRequested.connect(self.close_tab)
        self.tabs.currentChanged.connect(self._update_actions)

        empty = QLabel(
            "<h2>No file open</h2>"
            "<p>Use <b>File → Open…</b> (Ctrl+O) or drop files here.</p>"
        )
        empty.setAlignment(Qt.AlignCenter)
        empty.setStyleSheet("color: #777;")

        self.stack = QStackedWidget()
        self.stack.addWidget(empty)
        self.stack.addWidget(self.tabs)
        self.setCentralWidget(self.stack)

        self._build_actions()
        self.statusBar().showMessage("Ready")
        self._update_actions()

    def _build_actions(self) -> None:
        def action(text, slot, shortcut=None, tip=None):
            act = QAction(text, self)
            act.triggered.connect(slot)
            if shortcut:
                act.setShortcut(QKeySequence(shortcut))
            if tip:
                act.setStatusTip(tip)
            return act

        self.act_open = action("&Open…", self.open_dialog, QKeySequence.Open,
                               "Open one or more files, each in a new tab")
        self.act_add = action("&Add to current plot…", self.add_dialog, "Ctrl+Shift+O",
                              "Overlay another file on the current plot")
        self.act_save = action("&Save figure…", self.save_figure, QKeySequence.Save,
                               "Export the current plot as PNG, SVG, PDF…")
        self.act_close = action("&Close tab", lambda: self.close_tab(self.tabs.currentIndex()),
                                QKeySequence.Close)
        self.act_quit = action("&Quit", self.close, QKeySequence.Quit)
        self.act_detach = action("&Detach plot to window", self.detach_current, "Ctrl+D",
                                 "Move the current plot into its own movable window")
        self.act_controls = action("Show/hide &controls panel", self.toggle_controls, "Ctrl+B")
        self.act_help = action("&Mouse controls", self.show_help, QKeySequence.HelpContents)
        self.act_about = action("&About", self.show_about)

        file_menu = self.menuBar().addMenu("&File")
        for act in (self.act_open, self.act_add, None, self.act_save, None,
                    self.act_close, None, self.act_quit):
            file_menu.addSeparator() if act is None else file_menu.addAction(act)
        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.act_detach)
        view_menu.addAction(self.act_controls)
        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction(self.act_help)
        help_menu.addAction(self.act_about)

        toolbar = self.addToolBar("Main")
        toolbar.setObjectName("main_toolbar")
        toolbar.setMovable(False)
        for act in (self.act_open, self.act_add, self.act_save, self.act_detach):
            toolbar.addAction(act)

    def _update_actions(self, *_):
        has_tab = self.tabs.count() > 0
        for act in (self.act_add, self.act_save, self.act_close, self.act_detach, self.act_controls):
            act.setEnabled(has_tab)
        self.stack.setCurrentIndex(1 if has_tab else 0)

    def current_tab(self) -> PlotTab | None:
        return self.tabs.currentWidget()

    # --------------------------------------------------------------- files
    def _ask_for_files(self, title: str) -> list[str]:
        start = self.settings.value("last_dir", str(Path.home()))
        paths, _ = QFileDialog.getOpenFileNames(self, title, start, file_dialog_filter())
        if paths:
            self.settings.setValue("last_dir", str(Path(paths[0]).parent))
        return paths

    def open_dialog(self) -> None:
        self.open_files(self._ask_for_files("Open data files"))

    def add_dialog(self) -> None:
        tab = self.current_tab()
        if tab is None:
            return
        for path in self._ask_for_files("Add files to the current plot"):
            dataset = self._load(path)
            if dataset is not None:
                tab.add_dataset(dataset)
        self.tabs.setTabText(self.tabs.currentIndex(), tab.title)

    def _load(self, path):
        try:
            return load_file(path)
        except Exception as exc:  # noqa: BLE001 - show any parse error to the user
            QMessageBox.warning(self, "Could not open file", f"{path}\n\n{exc}")
            return None

    def open_files(self, paths) -> None:
        for path in paths:
            dataset = self._load(path)
            if dataset is None:
                continue
            tab = PlotTab(dataset)
            index = self.tabs.addTab(tab, tab.title)
            self.tabs.setTabToolTip(index, str(dataset.path))
            self.tabs.setCurrentIndex(index)
            rows, cols = dataset.frame.shape
            self.statusBar().showMessage(f"Opened {dataset.path}  ({rows} rows, {cols} columns)", 8000)
        self._update_actions()

    def close_tab(self, index: int) -> None:
        if index < 0:
            return
        tab = self.tabs.widget(index)
        self.tabs.removeTab(index)
        tab.deleteLater()
        self._update_actions()

    def save_figure(self) -> None:
        if (tab := self.current_tab()) is not None:
            tab.save_figure()

    # ------------------------------------------------------------ windows
    def detach_current(self) -> None:
        tab = self.current_tab()
        if tab is None:
            return
        self.tabs.removeTab(self.tabs.currentIndex())
        window = DetachedWindow(tab, self)
        self.detached.append(window)
        tab.show()
        window.show()
        self._update_actions()

    def attach_tab(self, tab: PlotTab) -> None:
        index = self.tabs.addTab(tab, tab.title)
        self.tabs.setTabToolTip(index, "\n".join(str(d.path) for d in tab.datasets))
        self.tabs.setCurrentIndex(index)
        self._update_actions()

    def toggle_controls(self) -> None:
        if (tab := self.current_tab()) is not None:
            tab.controls.setVisible(not tab.controls.isVisible())

    def show_help(self) -> None:
        QMessageBox.information(self, "Mouse controls", MOUSE_HELP)

    def show_about(self) -> None:
        QMessageBox.about(self, f"About {APP_NAME}",
                          f"<b>{APP_NAME}</b> {__version__}<br>Interactive viewer for CSV data files.")

    # ------------------------------------------------------- drag & drop
    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        self.open_files(paths)

    def closeEvent(self, event) -> None:
        for window in list(self.detached):
            window.takeCentralWidget()
            window.close()
        super().closeEvent(event)
