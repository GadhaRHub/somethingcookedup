"""Common base for analysis windows (spectral analysis, background fit, …).

It provides the layout (settings panel, figure, results table), the choice of
lines and range, live updates when the plot's view changes, figure export with
the export dialog, table copy/save, and saving the settings in sessions.
Subclasses add their own settings groups and implement ``run`` and ``draw``.
"""

from __future__ import annotations

import io
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .export import export_figure
from .plot_tab import PlotCanvas

ALL_SHOWN = "All lines shown on the plot"
CURRENT_VIEW = "Current view (x range on screen)"
FULL_DATA = "Full data"


class AnalysisWindow(QDialog):
    """Non-modal analysis window for one plot tab."""

    window_title = "Analysis"
    table_title = ""
    export_suffix = "analysis"

    def __init__(self, tab, parent=None):
        super().__init__(parent or tab, Qt.Window)
        self.tab = tab
        self.results: list = []
        self.errors: list[str] = []
        self._export_theme: theme.Theme | None = None
        self._export_font: float | None = None
        self._export_grid: bool | None = None
        self.setWindowTitle(f"{self.window_title} — {tab.display_name}")
        self.resize(1300, 840)

        self._timer = QTimer(self, singleShot=True, interval=300)
        self._timer.timeout.connect(self.run)

        controls = QWidget()
        controls.setObjectName("controls_panel")
        cl = QVBoxLayout(controls)
        cl.setSpacing(10)
        cl.addWidget(self._build_data_group())
        for group in self.build_groups():
            cl.addWidget(group)
        cl.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(controls)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(390)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.figure = Figure(figsize=(8, 6), layout="constrained")
        self.canvas = PlotCanvas(self.figure, self._rc)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        save_action = self.toolbar._actions.get("save_figure")
        if save_action is not None:
            save_action.triggered.disconnect()
            save_action.triggered.connect(self.export)
            save_action.setToolTip("Export these plots (PNG, SVG, PDF, …)")
        self.info = QLabel()
        self.info.setObjectName("hint")
        self.info.setWordWrap(True)
        plot_card = QFrame()
        plot_card.setObjectName("card")
        pl = QVBoxLayout(plot_card)
        pl.setContentsMargins(12, 10, 12, 12)
        pl.addWidget(self.toolbar)
        pl.addWidget(self.canvas, 1)
        pl.addWidget(self.info)

        self.table = QTableWidget(0, 0)
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table_card = QFrame()
        table_card.setObjectName("card")
        tl = QVBoxLayout(table_card)
        tl.setContentsMargins(12, 10, 12, 12)
        tl.addWidget(QLabel(self.table_title))
        tl.addWidget(self.table)

        right = QSplitter(Qt.Vertical)
        right.addWidget(plot_card)
        right.addWidget(table_card)
        right.setSizes([620, 200])

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(scroll)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([400, 900])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(splitter)
        for combo in controls.findChildren(QComboBox):
            # Long entries shouldn't widen (and clip) the settings panel.
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(12)

        self.refresh_choices()
        self.on_ready()
        tab.viewChanged.connect(self._on_view_changed)
        self.run()

    # ------------------------------------------------------- for subclasses
    def build_groups(self) -> list[QGroupBox]:
        """Settings groups shown below 'Data to analyse'."""
        return []

    def settings_widgets(self) -> list[tuple[str, QWidget]]:
        """(name, widget) pairs saved in sessions, in the order to restore them."""
        return []

    def on_ready(self) -> None:
        """Called once the UI exists, before the first run."""

    def run(self) -> None:
        raise NotImplementedError

    def draw(self) -> None:
        raise NotImplementedError

    def headers(self) -> list[str]:
        return []

    def rows(self) -> list[list]:
        return []

    def format_cell(self, column: int, value) -> str:
        return value if isinstance(value, str) else f"{value:.6g}"

    def add_buttons(self, form: QFormLayout, buttons) -> None:
        """Add (text, slot, primary) buttons to a settings form."""
        for text, slot, primary in buttons:
            btn = QPushButton(text)
            if not primary:
                btn.setProperty("secondary", True)
            btn.clicked.connect(slot)
            form.addRow(btn)

    # ------------------------------------------------------------ data choice
    def _build_data_group(self) -> QGroupBox:
        box = QGroupBox("Data to analyse")
        form = QFormLayout(box)
        self.series_combo = QComboBox()
        self.series_combo.setToolTip("Analyse one line, or every line currently shown on the plot.")
        self.range_combo = QComboBox()
        self.range_combo.setToolTip("Current view follows the plot as you zoom and pan.\n"
                                    "Marked regions are listed too.")
        for combo in (self.series_combo, self.range_combo):
            combo.currentTextChanged.connect(self._schedule)
        refresh = QPushButton("Refresh lines && regions")
        refresh.setProperty("secondary", True)
        refresh.clicked.connect(self.refresh_choices)
        form.addRow("Lines", self.series_combo)
        form.addRow("Range", self.range_combo)
        form.addRow(refresh)
        return box

    def refresh_choices(self) -> None:
        current_series, current_range = self.series_combo.currentData(), self.range_combo.currentText()
        self.series_combo.blockSignals(True)
        self.series_combo.clear()
        self.series_combo.addItem(ALL_SHOWN)
        for s in self.tab._plottable():
            self.series_combo.addItem(self.tab._series_label(s), s)
            if s is current_series:
                # Keep the same line even if its name changed (e.g. a file
                # prefix appears once a second dataset is added).
                self.series_combo.setCurrentIndex(self.series_combo.count() - 1)
        self.series_combo.blockSignals(False)

        self.range_combo.blockSignals(True)
        self.range_combo.clear()
        self.range_combo.addItems([CURRENT_VIEW, FULL_DATA])
        for mark in self.tab.region_marks:
            lo, hi = sorted((mark.xmin, mark.xmax))
            self.range_combo.addItem(f"Region {lo:.6g} → {hi:.6g}", (lo, hi))
        self.range_combo.setCurrentText(current_range or CURRENT_VIEW)
        self.range_combo.blockSignals(False)
        self._schedule()

    def _on_view_changed(self) -> None:
        if self.isVisible() and self.range_combo.currentText() == CURRENT_VIEW:
            self._schedule()

    def _schedule(self, *_) -> None:
        self._timer.start()

    def selected_series(self) -> list:
        data = self.series_combo.currentData()
        if data is not None:
            return [data]
        return [s for s in self.tab._plottable() if s.visible]

    def x_range(self) -> tuple[float, float] | None:
        text = self.range_combo.currentText()
        if text == FULL_DATA:
            return None
        if text == CURRENT_VIEW:
            return tuple(self.tab.ax.get_xlim())
        return self.range_combo.currentData()

    def series_data(self, s, x_range):
        """(x, y, range in data coordinates, colour) for a series. The plot shows
        x + offset, so the range is shifted back to the data's own x values."""
        x, _ = self.tab._x_values(s.dataset)
        y = s.dataset.frame[s.column].to_numpy(dtype=float)
        rng = None if x_range is None else (x_range[0] - s.x_offset, x_range[1] - s.x_offset)
        colors = {id(series): c for series, c in self.tab._series_colors()}
        return x, y, rng, colors.get(id(s), s.color)

    @staticmethod
    def number(edit: QLineEdit) -> float | None:
        try:
            return float(edit.text())
        except ValueError:
            return None

    # ---------------------------------------------------------------- session
    def _state_widgets(self) -> list[tuple[str, QWidget]]:
        return [("lines", self.series_combo), ("range", self.range_combo), *self.settings_widgets()]

    def get_state(self) -> dict:
        state = {"open": self.isVisible()}
        for name, widget in self._state_widgets():
            if isinstance(widget, QComboBox):
                state[name] = widget.currentText()
            elif isinstance(widget, QLineEdit):
                state[name] = widget.text()
            elif isinstance(widget, QCheckBox):
                state[name] = widget.isChecked()
            else:
                state[name] = widget.value()
        return state

    def apply_state(self, state: dict) -> None:
        self.refresh_choices()   # the plot's lines and regions may have changed
        for name, widget in self._state_widgets():
            if name not in state:
                continue
            value = state[name]
            if isinstance(widget, QComboBox):
                if widget.findText(value) >= 0:   # e.g. a region that no longer exists
                    widget.setCurrentText(value)
            elif isinstance(widget, QLineEdit):
                widget.setText(value)
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            else:
                widget.setValue(value)
        self._timer.stop()
        self.run()

    # ----------------------------------------------------------------- export
    def draw_theme(self) -> theme.Theme:
        return self._export_theme or theme.current

    def grid_on(self) -> bool:
        return self._export_grid if self._export_grid is not None else True

    def _rc(self) -> dict:
        rc = theme.mpl_rc(self.draw_theme())
        rc["font.size"] = self._export_font or 10
        return rc

    @contextmanager
    def export_style(self, draw_theme: theme.Theme, font_size: float | None,
                     grid: bool | None = None):
        """Temporarily redraw for export (see export.save_figure)."""
        self._export_theme, self._export_font, self._export_grid = draw_theme, font_size, grid
        try:
            self.draw()
            yield
        finally:
            self._export_theme = self._export_font = self._export_grid = None
            self.draw()

    def export(self) -> None:
        name = f"{self.tab.display_name}_{self.export_suffix}"
        if (path := export_figure(self.figure, name, self, restyle=self.export_style)):
            self.info.setText(f"Saved {path}")

    # ------------------------------------------------------------------ table
    def fill_table(self) -> None:
        rows, headers = self.rows(), self.headers()
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                item = QTableWidgetItem(self.format_cell(j, value))
                if not isinstance(value, str):
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(i, j, item)

    def table_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows(), columns=self.headers())

    def copy_table(self) -> None:
        buffer = io.StringIO()
        self.table_frame().to_csv(buffer, sep="\t", index=False)
        QApplication.clipboard().setText(buffer.getvalue())
        self.info.setText("Table copied to the clipboard.")

    def ask_save(self, title: str, name: str) -> str | None:
        folder = QSettings("csvplot", "csvplot").value("export/dir", str(Path.home()))
        path, _ = QFileDialog.getSaveFileName(self, title, str(Path(folder) / name), "CSV (*.csv)")
        if path and not path.lower().endswith(".csv"):
            path += ".csv"
        return path or None

    def save_table(self) -> None:
        if (path := self.ask_save("Save table", f"{self.tab.display_name}_{self.export_suffix}.csv")):
            self.table_frame().to_csv(path, index=False)
