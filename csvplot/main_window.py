"""Main application window: menus, file opening and the tab of plots."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QActionGroup, QColor, QIcon, QKeySequence, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from . import __version__, theme
from .loaders import file_dialog_filter, load_file
from .plot_tab import PlotTab
from .session import SESSION_FILTER, SESSION_SUFFIX, load_session, save_session

APP_NAME = "CSV Plotter"
RESOURCES = Path(__file__).resolve().parent / "resources"


def icon_path() -> Path:
    """The icon drawn in the current theme's colours (see tools/make_icon.py)."""
    path = RESOURCES / f"icon-{theme.current.name.lower()}.svg"
    return path if path.exists() else RESOURCES / "icon.svg"


def app_icon() -> QIcon:
    return QIcon(str(icon_path()))

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
        self.setCentralWidget(tab)
        self.resize(1200, 750)
        self.update_title()

    def update_title(self) -> None:
        self.setWindowTitle(f"{self.tab.display_name} — {self.main.windowTitle()}")

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
        self.setWindowIcon(app_icon())
        self.resize(1300, 820)
        self.setAcceptDrops(True)
        self.settings = QSettings("csvplot", "csvplot")
        self.detached: list[DetachedWindow] = []
        self.session_path: Path | None = None

        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.setMovable(True)
        self.tabs.setDocumentMode(True)
        self.tabs.tabCloseRequested.connect(self.close_tab)
        self.tabs.tabBarDoubleClicked.connect(self.rename_tab)
        self.tabs.tabBar().setToolTip("Double-click to rename")
        self.tabs.currentChanged.connect(self._update_actions)

        self.empty_label = empty = QLabel()
        empty.setObjectName("empty_state")
        empty.setAlignment(Qt.AlignCenter)
        empty_page = QWidget()
        ep = QVBoxLayout(empty_page)
        ep.addWidget(empty, 0, Qt.AlignCenter)

        self.stack = QStackedWidget()
        self.stack.addWidget(empty_page)
        self.stack.addWidget(self.tabs)
        self.setCentralWidget(self.stack)

        self._build_actions()
        self._refresh_empty_label()
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
        self.act_save = action("&Export figure…", self.save_figure, QKeySequence.Save,
                               "Save the current plot as high-resolution PNG, or vector SVG / PDF / EPS")
        self.act_close = action("&Close tab", lambda: self.close_tab(self.tabs.currentIndex()),
                                QKeySequence.Close)
        self.act_quit = action("&Quit", self.close, QKeySequence.Quit)
        self.act_open_session = action("Open s&ession…", self.open_session_dialog, "Ctrl+Shift+E",
                                       "Open a saved session with all its plots and settings")
        self.act_save_session = action("Save sessio&n", self.save_session, "Ctrl+Alt+S",
                                       "Save all plots, settings and marks to a session file")
        self.act_save_session_as = action("Save session &as…", lambda: self.save_session(ask=True),
                                          "Ctrl+Shift+S")
        self.act_session = action("Name &session…", self.edit_session_name, "F2",
                                  "Name this session; the name appears in the window title")
        self.act_rename_tab = action("&Rename plot…", lambda: self.rename_tab(self.tabs.currentIndex()),
                                     "Ctrl+R", "Rename the current plot tab")
        self.act_detach = action("&Detach plot to window", self.detach_current, "Ctrl+D",
                                 "Move the current plot into its own movable window")
        self.act_controls = action("Show/hide &controls panel", self.toggle_controls, "Ctrl+B")
        self.act_data = action("Show/hide &data table", self.toggle_data, "Ctrl+T",
                               "Show the raw values next to the plot (below or beside it)")
        self.act_data_auto = QAction("Open data table with new &files", self, checkable=True)
        self.act_data_auto.setStatusTip("Also open the data table automatically when a file is opened")
        self.act_data_auto.setChecked(self.settings.value("data_table_auto", False, type=bool))
        self.act_data_auto.toggled.connect(lambda on: self.settings.setValue("data_table_auto", on))
        self.act_fft = action("&Windowed FFT / periods…", self.open_fft, "Ctrl+Shift+F",
                              "Spectral analysis of the current plot: window, overlap, periods")
        self.act_help = action("&Mouse controls", self.show_help, QKeySequence.HelpContents)
        self.act_about = action("&About", self.show_about)

        file_menu = self.menuBar().addMenu("&File")
        for act in (self.act_open, self.act_add, None, self.act_open_session, self.act_save_session,
                    self.act_save_session_as, None, self.act_session, self.act_rename_tab, None,
                    self.act_save, None, self.act_close, None, self.act_quit):
            file_menu.addSeparator() if act is None else file_menu.addAction(act)
        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.act_detach)
        view_menu.addAction(self.act_controls)
        view_menu.addSeparator()
        view_menu.addAction(self.act_data)
        view_menu.addAction(self.act_data_auto)
        view_menu.addSeparator()
        theme_menu = view_menu.addMenu("&Theme")
        self.theme_group = QActionGroup(self)
        for name, t in theme.THEMES.items():
            act = QAction(name, self, checkable=True)
            act.setChecked(t is theme.current)
            act.setStatusTip(f"Use the {name} colour theme")
            act.triggered.connect(lambda _=False, n=name: self.set_theme(n))
            self.theme_group.addAction(act)
            theme_menu.addAction(act)
        analysis_menu = self.menuBar().addMenu("&Analysis")
        analysis_menu.addAction(self.act_fft)
        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction(self.act_help)
        help_menu.addAction(self.act_about)

        toolbar = self.addToolBar("Main")
        toolbar.setObjectName("main_toolbar")
        toolbar.setMovable(False)
        for act in (self.act_open, self.act_add, self.act_save, self.act_detach):
            toolbar.addAction(act)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)
        caption = QLabel("Session")
        caption.setObjectName("session_caption")
        toolbar.addWidget(caption)
        self.session_edit = QLineEdit()
        self.session_edit.setObjectName("session_name")
        self.session_edit.setPlaceholderText("Untitled session")
        self.session_edit.setToolTip("Name this session (F2). It is shown in the window title.")
        self.session_edit.setFixedWidth(240)
        self.session_edit.setClearButtonEnabled(True)
        self.session_edit.textChanged.connect(self._update_title)
        self.session_edit.returnPressed.connect(lambda: self.tabs.setFocus())
        toolbar.addWidget(self.session_edit)
        self._style_session_edit()

    def _style_session_edit(self) -> None:
        # Placeholder colour can't be set from a stylesheet.
        palette = self.session_edit.palette()
        color = QColor(theme.current.bar_text)
        color.setAlphaF(0.55)
        palette.setColor(QPalette.PlaceholderText, color)
        self.session_edit.setPalette(palette)

    def _update_actions(self, *_):
        has_tab = self.tabs.count() > 0
        for act in (self.act_add, self.act_save, self.act_close, self.act_detach, self.act_controls,
                    self.act_rename_tab, self.act_data, self.act_fft, self.act_save_session,
                    self.act_save_session_as):
            act.setEnabled(has_tab)
        self.stack.setCurrentIndex(1 if has_tab else 0)

    # ------------------------------------------------------------- sessions
    def current_theme_name(self) -> str:
        return theme.current.name

    def _all_plot_count(self) -> int:
        return self.tabs.count() + len(self.detached)

    def save_session(self, ask: bool = False) -> bool:
        if self._all_plot_count() == 0:
            return False
        path = self.session_path
        if ask or path is None:
            folder = self.settings.value("last_dir", str(Path.home()))
            name = (self.session_name or "session").replace("/", "-") + SESSION_SUFFIX
            chosen, _ = QFileDialog.getSaveFileName(self, "Save session", str(Path(folder) / name),
                                                    SESSION_FILTER)
            if not chosen:
                return False
            if not chosen.lower().endswith(SESSION_SUFFIX):
                chosen += SESSION_SUFFIX
            path = Path(chosen)
        try:
            save_session(self, path)
        except Exception as exc:  # noqa: BLE001 - show any write error
            QMessageBox.warning(self, "Could not save session", f"{path}\n\n{exc}")
            return False
        self.session_path = Path(path)
        self.statusBar().showMessage(f"Session saved to {path}", 8000)
        return True

    def open_session_dialog(self) -> None:
        folder = self.settings.value("last_dir", str(Path.home()))
        path, _ = QFileDialog.getOpenFileName(self, "Open session", folder, SESSION_FILTER)
        if path:
            self.open_session(path)

    def open_session(self, path) -> None:
        if self._all_plot_count():
            choice = QMessageBox.question(
                self, "Open session",
                "Close the plots that are open now?\n\n"
                "Yes: replace them with the session.  No: add the session's plots.",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Yes)
            if choice == QMessageBox.Cancel:
                return
            if choice == QMessageBox.Yes:
                self.close_all()
        try:
            notes = load_session(self, path)
        except Exception as exc:  # noqa: BLE001 - corrupt or foreign file
            QMessageBox.warning(self, "Could not open session", f"{path}\n\n{exc}")
            return
        self.session_path = Path(path)
        self.settings.setValue("last_dir", str(Path(path).parent))
        self._update_actions()
        if notes:
            QMessageBox.information(self, "Session opened", "\n".join(notes))
        else:
            self.statusBar().showMessage(f"Opened session {path}", 8000)

    def close_all(self) -> None:
        for window in list(self.detached):
            tab = window.takeCentralWidget()
            window.close()
            if tab is not None:
                tab.deleteLater()
        while self.tabs.count():
            self.close_tab(0)

    def current_tab(self) -> PlotTab | None:
        return self.tabs.currentWidget()

    # --------------------------------------------------------------- files
    def _ask_for_files(self, title: str) -> list[str]:
        start = self.settings.value("last_dir", str(Path.home()))
        paths, _ = QFileDialog.getOpenFileNames(
            self, title, start, f"{file_dialog_filter()};;{SESSION_FILTER}")
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
        self.tabs.setTabText(self.tabs.currentIndex(), tab.display_name)

    def _load(self, path):
        try:
            return load_file(path)
        except Exception as exc:  # noqa: BLE001 - show any parse error to the user
            QMessageBox.warning(self, "Could not open file", f"{path}\n\n{exc}")
            return None

    def open_files(self, paths) -> None:
        for path in paths:
            if str(path).lower().endswith(SESSION_SUFFIX):
                self.open_session(path)
                continue
            dataset = self._load(path)
            if dataset is None:
                continue
            tab = self._new_tab(dataset)
            self.tabs.setTabToolTip(self.tabs.indexOf(tab), str(dataset.path))
            rows, cols = dataset.frame.shape
            self.statusBar().showMessage(f"Opened {dataset.path}  ({rows} rows, {cols} columns)", 8000)
        self._update_actions()

    def _new_tab(self, dataset) -> PlotTab:
        tab = PlotTab(dataset)
        tab.statusMessage.connect(lambda msg: self.statusBar().showMessage(msg, 8000))
        tab.openDatasetRequested.connect(self.open_dataset)
        if self.act_data_auto.isChecked():
            tab.show_data_preview(focus=False)
        index = self.tabs.addTab(tab, tab.display_name)
        self.tabs.setCurrentIndex(index)
        self._update_actions()
        return tab

    def open_dataset(self, dataset) -> None:
        """Open data made inside the app (e.g. a spectrum) as a new plot."""
        self.stack.setCurrentIndex(1)
        self._new_tab(dataset)
        self.statusBar().showMessage(f"Opened {dataset.name}", 8000)

    def open_fft(self) -> None:
        if (tab := self.current_tab()) is not None:
            tab.open_fft()

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
        index = self.tabs.addTab(tab, tab.display_name)
        self.tabs.setTabToolTip(index, "\n".join(str(d.path) for d in tab.datasets))
        self.tabs.setCurrentIndex(index)
        self._update_actions()

    def toggle_data(self) -> None:
        if (tab := self.current_tab()) is not None:
            tab.toggle_data_preview()

    def toggle_controls(self) -> None:
        if (tab := self.current_tab()) is not None:
            tab.controls.setVisible(not tab.controls.isVisible())

    def set_theme(self, name: str) -> None:
        old = theme.current
        new = theme.apply_theme(QApplication.instance(), name)
        QApplication.instance().setWindowIcon(app_icon())
        self.setWindowIcon(app_icon())
        self.settings.setValue("theme", name)
        self._refresh_empty_label()
        self._style_session_edit()
        tabs = [self.tabs.widget(i) for i in range(self.tabs.count())]
        for tab in tabs + [w.tab for w in self.detached]:
            tab.change_theme(old, new)

    def _refresh_empty_label(self) -> None:
        self.empty_label.setText(
            f"<img src='{icon_path().as_posix()}' width='96' height='96'>"
            f"<h2 style='color:{theme.current.accent}'>No file open</h2>"
            "<p>Use <b>Open</b> (Ctrl+O) or drop files here.</p>"
        )

    # ------------------------------------------------------------- naming
    @property
    def session_name(self) -> str:
        return self.session_edit.text().strip()

    def set_session_name(self, name: str) -> None:
        self.session_edit.setText(name)

    def edit_session_name(self) -> None:
        self.session_edit.setFocus()
        self.session_edit.selectAll()

    def _update_title(self, *_) -> None:
        name = self.session_name
        self.setWindowTitle(f"{name} — {APP_NAME}" if name else APP_NAME)
        for window in self.detached:
            window.update_title()

    def rename_tab(self, index: int) -> None:
        tab = self.tabs.widget(index)
        if tab is None:
            return
        name, ok = QInputDialog.getText(self, "Rename plot", "Plot name (empty for the file name):",
                                        QLineEdit.Normal, tab.custom_name or tab.title)
        if ok:
            tab.custom_name = "" if name.strip() == tab.title else name.strip()
            self.tabs.setTabText(index, tab.display_name)

    def show_help(self) -> None:
        QMessageBox.information(self, "Mouse controls", MOUSE_HELP)

    def show_about(self) -> None:
        box = QMessageBox(self)
        box.setWindowTitle(f"About {APP_NAME}")
        box.setIconPixmap(app_icon().pixmap(96, 96))
        box.setText(f"<h3>{APP_NAME} {__version__}</h3>Interactive viewer for CSV data files.")
        box.exec()

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
