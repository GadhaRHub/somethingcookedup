"""One plot: an interactive matplotlib canvas plus its control panel."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

import matplotlib
import numpy as np
from matplotlib import cbook
from matplotlib.backend_bases import _Mode
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.colors import ListedColormap, to_hex
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, LogLocator, MultipleLocator, NullLocator
from matplotlib.widgets import SpanSelector
from PySide6.QtCore import QByteArray, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QDoubleValidator, QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .data_view import DataPreview
from .export import export_figure
from .loaders import Dataset
from .symbols import SymbolButton, latex_available, latex_escape, safe_text

ROW_INDEX = "(row index)"

COLORMAPS = [
    *theme.THEMES, "viridis", "plasma", "inferno", "magma", "cividis", "turbo",
    "coolwarm", "Spectral", "RdYlBu", "rainbow", "jet",
    "tab10", "tab20", "Set1", "Dark2", "Paired",
]

LINE_STYLES = {"Solid": "-", "Dashed": "--", "Dotted": ":", "Dash-dot": "-.", "None": "None"}
MARKERS = {"None": "", "Circle": "o", "Square": "s", "Triangle": "^", "Diamond": "D",
           "Cross": "x", "Plus": "+", "Point": "."}

LEGEND_LOCATIONS = [
    "best", "upper right", "upper left", "lower left", "lower right",
    "center left", "center right", "lower center", "upper center", "center",
]

PLOT_BACKGROUNDS = ["Grid", "Plain"]
MINOR_TICKS = ["Off", "Auto", "2", "4", "5", "10"]  # subdivisions between major ticks
TICK_DIRECTIONS = {"Outside": "out", "Inside": "in", "Both": "inout"}
MAX_TICKS = 500

PICK_RADIUS_PX = 12
EDGE_GRAB_PX = 6


@dataclass
class Series:
    dataset: Dataset
    column: str
    color: str
    visible: bool = True
    label: str = ""          # custom legend name; empty means automatic
    width: float = 1.5
    linestyle: str = "-"
    marker: str = ""
    markersize: float = 4.0
    alpha: float = 1.0
    x_offset: float = 0.0   # added to this series only, e.g. to stack curves
    y_offset: float = 0.0


@dataclass
class PointMark:
    x: float
    y: float
    color: str
    offset: tuple[float, float] = (10.0, 10.0)
    artists: list = field(default_factory=list)

    @property
    def text(self) -> str:
        return f"({self.x:.6g}, {self.y:.6g})"


@dataclass
class TextMark:
    """Free text or an equation placed on the plot (supports $...$ math)."""

    x: float
    y: float
    content: str
    color: str
    artists: list = field(default_factory=list)

    @property
    def text(self) -> str:
        return self.content


@dataclass
class RegionMark:
    xmin: float
    xmax: float
    color: str
    artists: list = field(default_factory=list)

    @property
    def text(self) -> str:
        return f"{self.xmin:.6g} → {self.xmax:.6g}"


class PlotCanvas(FigureCanvasQTAgg):
    """Canvas that draws (and saves) with its own rcParams, e.g. theme colours
    and the optional full-LaTeX mode. ``on_error`` may handle a failed draw
    (return True) instead of raising, e.g. for LaTeX errors."""

    def __init__(self, figure, rc_params, on_error=None):
        super().__init__(figure)
        self._rc_params = rc_params
        self._on_error = on_error

    def draw(self):
        with matplotlib.rc_context(self._rc_params()):
            try:
                super().draw()
            except Exception as exc:  # noqa: BLE001 - e.g. LaTeX errors surface here
                if self._on_error is None or not self._on_error(exc):
                    raise

    def print_figure(self, *args, **kwargs):
        with matplotlib.rc_context(self._rc_params()):
            return super().print_figure(*args, **kwargs)


class ColorButton(QPushButton):
    """A small swatch that opens a colour picker when clicked."""

    colorChanged = Signal(str)

    def __init__(self, color: str, parent=None):
        super().__init__(parent)
        self.setFixedSize(36, 22)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Choose colour")
        self.clicked.connect(self._pick)
        self.set_color(color)

    @property
    def color(self) -> str:
        return self._color

    def set_color(self, color: str) -> None:
        self._color = color
        self.setStyleSheet(
            f"QPushButton {{ background: {color}; border: 2px solid {theme.current.card}; border-radius: 8px;"
            " padding: 0; }"
            f"QPushButton:hover {{ border-color: {theme.current.border}; }}"
            f"QPushButton:disabled {{ border: 2px dashed {theme.current.border}; }}"
        )

    def _pick(self) -> None:
        chosen = QColorDialog.getColor(QColor(self._color), self, "Choose colour")
        if chosen.isValid():
            self.set_color(chosen.name())
            self.colorChanged.emit(chosen.name())


def _parse_float(text: str, default: float = 0.0) -> float:
    try:
        return float(text)
    except ValueError:
        return default


def _float_edit(placeholder: str) -> QLineEdit:
    edit = QLineEdit()
    edit.setPlaceholderText(placeholder)
    edit.setValidator(QDoubleValidator())
    return edit


class PlotTab(QWidget):
    """Interactive plot of one or more datasets."""

    statusMessage = Signal(str)
    viewChanged = Signal()                 # x/y limits changed (zoom, pan, ranges)
    openDatasetRequested = Signal(object)  # e.g. a spectrum to open as a new plot

    def __init__(self, dataset: Dataset, parent=None):
        super().__init__(parent)
        self.datasets: list[Dataset] = []
        self.custom_name = ""
        self.data_preview: DataPreview | None = None
        self._export_theme: theme.Theme | None = None
        self._export_font: float | None = None
        self._export_grid: bool | None = None
        self._selected_marks: set[int] = set()   # ids of marks selected in the list
        self._drag: dict | None = None           # region being moved/resized
        self._line_series: dict = {}
        self.text_marks: list[TextMark] = []
        self._legend = None
        self._legend_pos: tuple[float, float] | None = None   # set by dragging the legend
        self._legend_drawn_loc = None
        self._math_errors: list[str] = []
        self._loading = False   # restoring a session: skip intermediate redraws
        self._fft_state: dict | None = None   # analysis settings waiting for the window
        self.series: list[Series] = []
        self.point_marks: list[PointMark] = []
        self.region_marks: list[RegionMark] = []
        self._data_lines: list = []
        self._span: SpanSelector | None = None
        self._syncing = False
        self._last_scales = ("linear", "linear")

        self._build_ui()
        self.add_dataset(dataset)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        theme.apply_matplotlib_style()
        self.figure = Figure(figsize=(8, 5), layout="tight")
        self.canvas = PlotCanvas(self.figure, self.rc_params, self._on_draw_error)
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.ax = self.figure.add_subplot()
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        # The toolbar's save button opens our high-quality export dialog.
        save_action = self.toolbar._actions.get("save_figure")
        if save_action is not None:
            save_action.triggered.disconnect()
            save_action.triggered.connect(self.save_figure)
            save_action.setToolTip("Export figure (PNG, SVG, PDF, …)")

        # Second toolbar row: marking tools and the data table, so they stay
        # visible even when the plot is narrow (e.g. table beside the plot).
        self.tools_bar = QToolBar()
        self.tools_bar.setObjectName("plot_tools")
        self.act_mark_point = QAction("Mark point", self, checkable=True)
        self.act_mark_point.setToolTip(
            "Click on the plot to mark a point (snaps to the nearest data point).\n"
            "Right-click a mark to remove it."
        )
        self.act_mark_region = QAction("Mark region", self, checkable=True)
        self.act_mark_region.setToolTip("Drag horizontally on the plot to shade an x-range.")
        self.act_add_text = QAction("Add text / equation", self, checkable=True)
        self.act_add_text.setToolTip("Click on the plot to place text or an equation, e.g. $E = mc^2$.\n"
                                     "Drag it to move, double-click to edit, right-click to remove.")
        self.act_add_text.toggled.connect(lambda on: self._set_mark_mode("text", on))
        self.tools_bar.addAction(self.act_mark_point)
        self.tools_bar.addAction(self.act_mark_region)
        self.tools_bar.addAction(self.act_add_text)
        self.tools_bar.addSeparator()
        self.act_data = QAction("Data table", self, checkable=True)
        self.act_data.setToolTip("Show the raw values next to the plot (Ctrl+T)")
        self.act_data.toggled.connect(lambda on: self.show_data_preview() if on else self.close_data_preview())
        self.tools_bar.addAction(self.act_data)
        self.tools_bar.addSeparator()
        self.act_fft = QAction("Spectral analysis (FFT)", self)
        self.act_fft.setToolTip("Windowed FFT / Welch / spectrogram and period extraction")
        self.act_fft.triggered.connect(self.open_fft)
        self.tools_bar.addAction(self.act_fft)
        self.apply_toolbar_icons()
        self.act_mark_point.toggled.connect(lambda on: self._set_mark_mode("point", on))
        self.act_mark_region.toggled.connect(lambda on: self._set_mark_mode("region", on))
        # Turning on matplotlib's pan/zoom switches marking off, and vice versa.
        for name in ("pan", "zoom"):
            if name in self.toolbar._actions:
                self.toolbar._actions[name].triggered.connect(self._on_nav_mode)

        self.coord_label = QLabel("Move the mouse over the plot to see coordinates")
        self.coord_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.coord_label.setObjectName("coords")

        plot_area = QFrame()
        plot_area.setObjectName("card")
        pl = QVBoxLayout(plot_area)
        pl.setContentsMargins(12, 10, 12, 12)
        pl.setSpacing(6)
        pl.addWidget(self.toolbar)
        pl.addWidget(self.tools_bar)
        pl.addWidget(self.canvas, 1)
        pl.addWidget(self.coord_label)

        panel = QWidget()
        panel.setObjectName("controls_panel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(2, 0, 6, 0)
        panel_layout.setSpacing(10)
        panel_layout.addWidget(self._build_data_group())
        panel_layout.addWidget(self._build_series_style_group())
        panel_layout.addWidget(self._build_color_group())
        panel_layout.addWidget(self._build_label_group())
        panel_layout.addWidget(self._build_legend_group())
        panel_layout.addWidget(self._build_axes_group())
        panel_layout.addWidget(self._build_ticks_group())
        panel_layout.addWidget(self._build_marks_group())
        panel_layout.addStretch(1)

        self.controls = QScrollArea()
        self.controls.setWidget(panel)
        self.controls.setWidgetResizable(True)
        self.controls.setMinimumWidth(320)
        self.controls.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # The plot and the (optional) data table share a splitter, so both are
        # visible at once next to the controls panel.
        self.views = QSplitter(Qt.Vertical)
        self.views.setChildrenCollapsible(False)
        self.views.addWidget(plot_area)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.views)
        splitter.addWidget(self.controls)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([900, 340])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 8)
        layout.addWidget(splitter)

        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("pick_event", self._on_pick)
        self.canvas.mpl_connect("axes_leave_event", lambda e: self.coord_label.setText(""))

    def _build_data_group(self) -> QGroupBox:
        box = QGroupBox("Data")
        form = QFormLayout(box)

        self.x_combo = QComboBox()
        self.x_combo.currentTextChanged.connect(self._on_x_changed)
        form.addRow("X axis", self.x_combo)

        # Same value as "X label" under Title & labels; kept in sync.
        self.xname_edit = QLineEdit()
        self.xname_edit.setToolTip("Name shown on the X axis. Leave empty for the column name.\n"
                                   "Tip: double-click an axis label or the title on the plot to rename it.")
        self.xname_edit.editingFinished.connect(self.redraw)
        form.addRow("X name", self.xname_edit)

        self.series_table = QTableWidget(0, 3)
        self.series_table.setHorizontalHeaderLabels(["Y series", "Colour", "Width"])
        self.series_table.setToolTip("Tick to show, untick to hide. Select a row to edit its style below.")
        self.series_table.verticalHeader().hide()
        self.series_table.setShowGrid(False)
        self.series_table.setFocusPolicy(Qt.NoFocus)
        self.series_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.series_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.series_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.series_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.series_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.series_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.series_table.itemChanged.connect(self._on_series_toggled)
        self.series_table.currentCellChanged.connect(lambda row, *_: self._load_series_editor(row))
        form.addRow(self.series_table)

        hint = QLabel("Tip: click a legend entry to hide or show that line.")
        hint.setWordWrap(True)
        hint.setObjectName("hint")
        form.addRow(hint)
        return box

    def _build_series_style_group(self) -> QGroupBox:
        box = QGroupBox("Line style")
        form = QFormLayout(box)
        self.style_target = QLabel()
        self.style_target.setWordWrap(True)
        form.addRow(self.style_target)

        self.series_name = QLineEdit()
        self.series_name.setToolTip("Name shown in the legend. Leave empty for the column name.")
        self.series_name.editingFinished.connect(self._on_series_style_changed)
        form.addRow("Name", self.series_name)

        self.series_linestyle = QComboBox()
        self.series_linestyle.addItems(LINE_STYLES)
        form.addRow("Line", self.series_linestyle)

        self.series_marker = QComboBox()
        self.series_marker.addItems(MARKERS)
        form.addRow("Marker", self.series_marker)

        self.series_markersize = QDoubleSpinBox(minimum=0.5, maximum=30, singleStep=0.5)
        form.addRow("Marker size", self.series_markersize)

        self.series_alpha = QSpinBox(minimum=5, maximum=100, singleStep=5, suffix=" %")
        form.addRow("Opacity", self.series_alpha)

        self.series_x_offset, self.series_y_offset = _float_edit("0"), _float_edit("0")
        for edit, axis in ((self.series_x_offset, "X"), (self.series_y_offset, "Y")):
            edit.setToolTip(f"Added to this line's {axis} values only (the data is unchanged).\n"
                            "Useful for stacking curves.")
            edit.editingFinished.connect(self._on_series_style_changed)
        offsets = QHBoxLayout()
        offsets.addWidget(QLabel("X"))
        offsets.addWidget(self.series_x_offset)
        offsets.addWidget(QLabel("Y"))
        offsets.addWidget(self.series_y_offset)
        form.addRow("Offset", offsets)

        for widget in (self.series_linestyle, self.series_marker):
            widget.currentTextChanged.connect(lambda _: self._on_series_style_changed())
        for widget in (self.series_markersize, self.series_alpha):
            widget.valueChanged.connect(lambda _: self._on_series_style_changed())

        apply_all = QPushButton("Apply this style to all lines")
        apply_all.setProperty("secondary", True)
        apply_all.setToolTip("Copy width, line, marker and opacity to every series (not colour or name).")
        apply_all.clicked.connect(self._apply_style_to_all)
        form.addRow(apply_all)
        return box

    def _build_color_group(self) -> QGroupBox:
        box = QGroupBox("Colours")
        form = QFormLayout(box)
        self.use_cmap = QCheckBox("Colour series with a colormap")
        self.use_cmap.setToolTip("Useful when several series are plotted together.")
        self.use_cmap.toggled.connect(self._on_cmap_toggled)
        form.addRow(self.use_cmap)

        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(COLORMAPS)
        self.cmap_combo.setCurrentText(theme.current.name)
        self.cmap_combo.setEnabled(False)
        self.cmap_combo.currentTextChanged.connect(lambda _: self.redraw())
        self.cmap_reverse = QCheckBox("Reverse")
        self.cmap_reverse.setEnabled(False)
        self.cmap_reverse.toggled.connect(lambda _: self.redraw())
        row = QHBoxLayout()
        row.addWidget(self.cmap_combo, 1)
        row.addWidget(self.cmap_reverse)
        form.addRow("Colormap", row)
        return box

    def _build_label_group(self) -> QGroupBox:
        box = QGroupBox("Title && labels")
        form = QFormLayout(box)
        self.title_edit = QLineEdit()
        self.xlabel_edit = QLineEdit()
        self.ylabel_edit = QLineEdit()
        for edit in (self.title_edit, self.xlabel_edit, self.ylabel_edit):
            edit.editingFinished.connect(self.redraw)
        self.xname_edit.textChanged.connect(
            lambda text: self.xlabel_edit.text() != text and self.xlabel_edit.setText(text))
        self.xlabel_edit.textChanged.connect(
            lambda text: self.xname_edit.text() != text and self.xname_edit.setText(text))
        form.addRow("Title", self.title_edit)
        form.addRow("X label", self.xlabel_edit)
        form.addRow("Y label", self.ylabel_edit)

        self.font_size = QDoubleSpinBox(minimum=5, maximum=40, singleStep=1, value=11)
        self.font_size.valueChanged.connect(lambda _: self.redraw())
        form.addRow("Font size", self.font_size)

        self.symbols = SymbolButton([self.title_edit, self.xlabel_edit, self.ylabel_edit,
                                     self.xname_edit, self.series_name])
        form.addRow(self.symbols)
        math_hint = QLabel("Math: put LaTeX between $…$, e.g. $\\Delta R/R$, cm$^{-1}$, $\\tau_{1}$")
        math_hint.setObjectName("hint")
        math_hint.setWordWrap(True)
        math_hint.setTextFormat(Qt.PlainText)
        form.addRow(math_hint)

        available, why = latex_available()
        self.full_latex = QCheckBox("Full LaTeX (uses your LaTeX installation)")
        self.full_latex.setToolTip(
            "Render all text with real LaTeX: packages, \\text{…}, \\begin{aligned}…, and LaTeX fonts.\n"
            "Slower than the built-in math. Exports use it too." if available else why)
        self.full_latex.setEnabled(available)
        self.full_latex.toggled.connect(self._on_full_latex)
        form.addRow(self.full_latex)
        self.latex_preamble = QLineEdit()
        self.latex_preamble.setPlaceholderText(r"Extra preamble, e.g. \usepackage{siunitx}")
        self.latex_preamble.setToolTip("Added after \\usepackage{amsmath}.")
        self.latex_preamble.setEnabled(False)
        self.latex_preamble.editingFinished.connect(lambda: self.usetex() and self.redraw())
        form.addRow("Preamble", self.latex_preamble)
        if not available:
            missing = QLabel(why)
            missing.setObjectName("hint")
            missing.setWordWrap(True)
            form.addRow(missing)

        self.accent_axes = QCheckBox("Accent-coloured axes && ticks")
        self.accent_axes.setToolTip("Draw the axes, ticks, labels and title in the theme's accent colour.")
        self.accent_axes.toggled.connect(lambda _: self.redraw())
        form.addRow(self.accent_axes)

        self.plot_background = QComboBox()
        self.plot_background.addItems(PLOT_BACKGROUNDS)
        self.plot_background.setToolTip("Plain: no gridlines. Grid: gridlines at the major ticks\n"
                                        "(minor gridlines can be added under Ticks).")
        self.plot_background.currentTextChanged.connect(lambda _: self.redraw())
        form.addRow("Background", self.plot_background)
        return box

    def _build_legend_group(self) -> QGroupBox:
        box = QGroupBox("Legend")
        form = QFormLayout(box)
        self.legend_check = QCheckBox("Show legend")
        self.legend_check.setChecked(True)
        self.legend_check.toggled.connect(lambda _: self.redraw())
        form.addRow(self.legend_check)

        self.legend_loc = QComboBox()
        self.legend_loc.addItems(LEGEND_LOCATIONS)
        self.legend_loc.currentTextChanged.connect(lambda _: self._reset_legend_position())
        form.addRow("Position", self.legend_loc)

        self.legend_reset = QPushButton("Reset dragged position")
        self.legend_reset.setProperty("secondary", True)
        self.legend_reset.setToolTip("Drag the legend on the plot to move it; this puts it back.")
        self.legend_reset.setEnabled(False)
        self.legend_reset.clicked.connect(self._reset_legend_position)
        form.addRow(self.legend_reset)

        self.legend_line_length = QDoubleSpinBox(minimum=0.5, maximum=10, singleStep=0.5, value=2.0)
        self.legend_line_length.setToolTip("Length of the line sample in the legend (in font sizes).")
        self.legend_line_length.valueChanged.connect(lambda _: self.redraw())
        form.addRow("Line length", self.legend_line_length)

        self.legend_cols = QSpinBox(minimum=1, maximum=8, value=1)
        self.legend_cols.valueChanged.connect(lambda _: self.redraw())
        form.addRow("Columns", self.legend_cols)

        self.legend_box = QCheckBox("Box around legend")
        self.legend_box.setChecked(True)
        self.legend_box.toggled.connect(lambda _: self.redraw())
        form.addRow(self.legend_box)
        return box

    def _reset_legend_position(self) -> None:
        self._legend_pos = None
        self._legend = None  # forget the drag so redraw uses the chosen position
        self.legend_reset.setEnabled(False)
        self.redraw()

    def _build_axes_group(self) -> QGroupBox:
        box = QGroupBox("Axis ranges")
        grid = QGridLayout(box)
        self.xmin_edit, self.xmax_edit = _float_edit("x min"), _float_edit("x max")
        self.ymin_edit, self.ymax_edit = _float_edit("y min"), _float_edit("y max")
        grid.addWidget(QLabel("X"), 0, 0)
        grid.addWidget(self.xmin_edit, 0, 1)
        grid.addWidget(self.xmax_edit, 0, 2)
        grid.addWidget(QLabel("Y"), 1, 0)
        grid.addWidget(self.ymin_edit, 1, 1)
        grid.addWidget(self.ymax_edit, 1, 2)
        for edit in (self.xmin_edit, self.xmax_edit, self.ymin_edit, self.ymax_edit):
            edit.returnPressed.connect(self.apply_ranges)

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.apply_ranges)
        auto_btn = QPushButton("Auto-fit")
        auto_btn.setProperty("secondary", True)
        auto_btn.clicked.connect(self.autoscale)
        grid.addWidget(apply_btn, 2, 1)
        grid.addWidget(auto_btn, 2, 2)

        self.logx_check = QCheckBox("Log X")
        self.logy_check = QCheckBox("Log Y")
        self.logx_check.toggled.connect(lambda _: self.redraw(keep_view=False))
        self.logy_check.toggled.connect(lambda _: self.redraw(keep_view=False))
        grid.addWidget(self.logx_check, 3, 1)
        grid.addWidget(self.logy_check, 3, 2)
        return box

    def _build_ticks_group(self) -> QGroupBox:
        box = QGroupBox("Ticks")
        grid = QGridLayout(box)
        grid.addWidget(QLabel("Major step"), 0, 1)
        grid.addWidget(QLabel("Minor ticks"), 0, 2)
        self.major_step, self.minor_ticks = {}, {}
        for row, axis in enumerate(("x", "y"), start=1):
            step = _float_edit("auto")
            step.setToolTip("Distance between major ticks. Leave empty for automatic.\n"
                            "(Ignored on log axes.)")
            step.editingFinished.connect(self.redraw)
            minor = QComboBox()
            minor.addItems(MINOR_TICKS)
            minor.setToolTip("Number of minor intervals between major ticks.")
            minor.currentTextChanged.connect(lambda _: self.redraw())
            self.major_step[axis], self.minor_ticks[axis] = step, minor
            grid.addWidget(QLabel(axis.upper()), row, 0)
            grid.addWidget(step, row, 1)
            grid.addWidget(minor, row, 2)

        self.tick_direction = QComboBox()
        self.tick_direction.addItems(TICK_DIRECTIONS)
        self.tick_direction.currentTextChanged.connect(lambda _: self.redraw())
        grid.addWidget(QLabel("Direction"), 3, 0)
        grid.addWidget(self.tick_direction, 3, 1, 1, 2)

        self.mirror_ticks = QCheckBox("Ticks on all four sides (boxed)")
        self.mirror_ticks.toggled.connect(lambda _: self.redraw())
        grid.addWidget(self.mirror_ticks, 4, 0, 1, 3)
        self.minor_grid = QCheckBox("Minor gridlines")
        self.minor_grid.toggled.connect(lambda _: self.redraw())
        grid.addWidget(self.minor_grid, 5, 0, 1, 3)

        grid.addWidget(QLabel("Width (pt)"), 6, 1)
        grid.addWidget(QLabel("Length (pt)"), 6, 2)
        self.tick_width, self.tick_length = {}, {}
        for row, (which, width, length) in enumerate((("major", 0.8, 5.0), ("minor", 0.6, 2.5)), start=7):
            w = QDoubleSpinBox(minimum=0.1, maximum=5, singleStep=0.1, decimals=1, value=width)
            l = QDoubleSpinBox(minimum=0, maximum=20, singleStep=0.5, decimals=1, value=length)
            for spin in (w, l):
                spin.valueChanged.connect(lambda _: self.redraw())
            self.tick_width[which], self.tick_length[which] = w, l
            grid.addWidget(QLabel(which.capitalize()), row, 0)
            grid.addWidget(w, row, 1)
            grid.addWidget(l, row, 2)
        return box

    def _apply_ticks(self, ax, t: theme.Theme, fs: float) -> None:
        for name, axis, lim in (("x", ax.xaxis, ax.get_xlim()), ("y", ax.yaxis, ax.get_ylim())):
            is_log = (ax.get_xscale() if name == "x" else ax.get_yscale()) == "log"
            try:
                step = float(self.major_step[name].text())
            except ValueError:
                step = 0
            if step > 0 and not is_log:
                if abs(lim[1] - lim[0]) / step > MAX_TICKS:
                    self.statusMessage.emit(f"{name.upper()} major step {step:g} is too small for "
                                            "this range; using automatic ticks.")
                else:
                    axis.set_major_locator(MultipleLocator(step))
            minor = self.minor_ticks[name].currentText()
            if minor == "Off":
                axis.set_minor_locator(NullLocator())
            elif is_log:
                axis.set_minor_locator(LogLocator(subs="auto"))
            else:
                axis.set_minor_locator(AutoMinorLocator(None if minor == "Auto" else int(minor)))

        mirror = self.mirror_ticks.isChecked()
        ax.tick_params(which="both", direction=TICK_DIRECTIONS[self.tick_direction.currentText()],
                       top=mirror, right=mirror)
        for which in ("major", "minor"):
            ax.tick_params(which=which, width=self.tick_width[which].value(),
                           length=self.tick_length[which].value())
        for side in ("top", "right"):
            ax.spines[side].set_visible(mirror)
            ax.spines[side].set_color(ax.spines["left"].get_edgecolor())
        if self._grid_on() and self.minor_grid.isChecked():
            ax.grid(True, which="minor", color=t.border, alpha=0.35 if t.dark else 0.18, linewidth=0.6)

    def _build_marks_group(self) -> QGroupBox:
        box = QGroupBox("Marks")
        layout = QVBoxLayout(box)

        hint = QLabel(
            "Use <b>Mark point</b> / <b>Mark region</b> in the plot toolbar. "
            "Select a region here (or click it on the plot) to highlight it; drag it to move, "
            "or drag its edges to resize. Right-click a mark to remove it."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        row = QHBoxLayout()
        row.addWidget(QLabel("Colour"))
        self.mark_color = ColorButton(theme.current.accent)
        row.addWidget(self.mark_color)
        self.snap_check = QCheckBox("Snap to data")
        self.snap_check.setChecked(True)
        row.addWidget(self.snap_check)
        row.addStretch(1)
        layout.addLayout(row)

        self.marks_list = QListWidget()
        self.marks_list.setMinimumHeight(80)
        self.marks_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.marks_list.itemSelectionChanged.connect(self._on_mark_selection)
        layout.addWidget(self.marks_list)

        row = QHBoxLayout()
        remove_btn = QPushButton("Remove selected")
        remove_btn.setProperty("secondary", True)
        remove_btn.clicked.connect(self._remove_selected_marks)
        clear_btn = QPushButton("Clear all")
        clear_btn.setProperty("secondary", True)
        clear_btn.clicked.connect(self.clear_marks)
        row.addWidget(remove_btn)
        row.addWidget(clear_btn)
        layout.addLayout(row)
        return box

    # ------------------------------------------------------------ datasets
    def add_dataset(self, dataset: Dataset) -> None:
        self.datasets.append(dataset)
        x_default = self.x_column or (dataset.numeric_columns[0] if dataset.numeric_columns else ROW_INDEX)
        for col in dataset.numeric_columns:
            # Hand out palette colours to the plotted series first.
            n_used = sum(s.visible for s in self.series)
            visible = col != x_default
            colors = theme.current.series_colors
            index = n_used if visible else len(colors) - 1 - len(self.series)
            color = colors[index % len(colors)]
            self.series.append(Series(dataset, col, color, visible=visible))
        if not any(s.visible for s in self.series) and self.series:
            self.series[-1].visible = True
        self._refresh_x_combo(x_default)
        self._refresh_series_table()
        self.redraw(keep_view=False)
        if getattr(self, "data_preview", None) is not None:
            self.data_preview.refresh()

    @property
    def display_name(self) -> str:
        """Name shown on the tab: the user's name for the plot, else the file names."""
        return self.custom_name or self.title

    @property
    def title(self) -> str:
        return " + ".join(d.name for d in self.datasets)

    @property
    def x_column(self) -> str | None:
        return self.x_combo.currentText() or None

    def _refresh_x_combo(self, select: str) -> None:
        # Offer columns shared by every dataset; files lacking the chosen
        # column fall back to their own first column.
        common = list(self.datasets[0].numeric_columns)
        for ds in self.datasets[1:]:
            common = [c for c in common if c in ds.numeric_columns]
        options = [ROW_INDEX] + (common or self.datasets[0].numeric_columns)
        self.x_combo.blockSignals(True)
        self.x_combo.clear()
        self.x_combo.addItems(options)
        self.x_combo.setCurrentText(select if select in options else options[min(1, len(options) - 1)])
        self.x_combo.blockSignals(False)

    def _default_label(self, s: Series) -> str:
        return s.column if len(self.datasets) == 1 else f"{s.dataset.path.stem}: {s.column}"

    def _series_label(self, s: Series) -> str:
        return s.label or self._default_label(s)

    def _is_x(self, s: Series) -> bool:
        """True if the series is the column currently used as its X axis."""
        return self._x_values(s.dataset)[1] == s.column

    def _plottable(self) -> list[Series]:
        return [s for s in self.series if not self._is_x(s)]

    def _on_x_changed(self, _text: str) -> None:
        self._refresh_series_table()
        self.redraw(keep_view=False)

    def _refresh_series_table(self) -> None:
        table = self.series_table
        current = table.currentRow()
        table.blockSignals(True)
        table.setRowCount(len(self.series))
        for row, s in enumerate(self.series):
            is_x = self._is_x(s)
            item = QTableWidgetItem(self._series_label(s))
            item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsSelectable
                          | (Qt.NoItemFlags if is_x else Qt.ItemIsEnabled))
            item.setCheckState(Qt.Checked if s.visible else Qt.Unchecked)
            item.setToolTip(f"{s.dataset.path}\n{s.column}" + ("\n(used as the X axis)" if is_x else ""))
            table.setItem(row, 0, item)

            button = ColorButton(s.color)
            button.colorChanged.connect(lambda c, s=s: self._set_series_color(s, c))
            width = QDoubleSpinBox(minimum=0.1, maximum=20, singleStep=0.25, decimals=2, value=s.width)
            width.setToolTip("Line width")
            width.setFixedWidth(78)
            width.valueChanged.connect(lambda v, s=s: self._set_series_width(s, v))
            for col, widget in ((1, button), (2, width)):
                holder = QWidget()
                hl = QHBoxLayout(holder)
                hl.setContentsMargins(4, 1, 4, 1)
                hl.addWidget(widget)
                holder.setEnabled(not is_x)
                table.setCellWidget(row, col, holder)
        table.blockSignals(False)
        if not 0 <= current < len(self.series) or self._is_x(self.series[current]):
            current = next((i for i, s in enumerate(self.series) if not self._is_x(s)), -1)
        table.selectRow(current) if current >= 0 else None
        self._load_series_editor(current)
        table.resizeRowsToContents()
        rows_height = sum(table.rowHeight(r) for r in range(table.rowCount()))
        table.setFixedHeight(min(260, table.horizontalHeader().height() + rows_height + 12))

    def _color_buttons(self) -> list[ColorButton]:
        return [self.series_table.cellWidget(r, 1).findChild(ColorButton)
                for r in range(self.series_table.rowCount())]

    def _width_spins(self) -> list[QDoubleSpinBox]:
        return [self.series_table.cellWidget(r, 2).findChild(QDoubleSpinBox)
                for r in range(self.series_table.rowCount())]

    def _on_series_toggled(self, item: QTableWidgetItem) -> None:
        s = self.series[item.row()]
        visible = item.checkState() == Qt.Checked
        if visible == s.visible:
            return  # e.g. the row was only renamed; keep the current view
        s.visible = visible
        self.redraw(keep_view=False)

    def set_series_visible(self, s: Series, visible: bool) -> None:
        s.visible = visible
        item = self.series_table.item(self.series.index(s), 0)
        self.series_table.blockSignals(True)
        item.setCheckState(Qt.Checked if visible else Qt.Unchecked)
        self.series_table.blockSignals(False)
        self.redraw()

    def _set_series_color(self, s: Series, color: str) -> None:
        s.color = color
        self.redraw()

    def _set_series_width(self, s: Series, width: float) -> None:
        s.width = width
        self.redraw()

    def _selected_series(self) -> Series | None:
        row = self.series_table.currentRow()
        return self.series[row] if 0 <= row < len(self.series) else None

    def _load_series_editor(self, row: int) -> None:
        s = self.series[row] if 0 <= row < len(self.series) else None
        widgets = (self.series_name, self.series_linestyle, self.series_marker,
                   self.series_markersize, self.series_alpha,
                   self.series_x_offset, self.series_y_offset)
        for w in widgets:
            w.setEnabled(s is not None)
            w.blockSignals(True)
        if s is None:
            self.style_target.setText("Select a series in the table above.")
        else:
            self.style_target.setText(f"Editing <b>{self._series_label(s)}</b>")
            self.series_name.setText(s.label)
            self.series_name.setPlaceholderText(self._default_label(s))
            self.series_linestyle.setCurrentText(
                next(k for k, v in LINE_STYLES.items() if v == s.linestyle))
            self.series_marker.setCurrentText(next(k for k, v in MARKERS.items() if v == s.marker))
            self.series_markersize.setValue(s.markersize)
            self.series_alpha.setValue(round(s.alpha * 100))
            self.series_x_offset.setText(f"{s.x_offset:g}" if s.x_offset else "")
            self.series_y_offset.setText(f"{s.y_offset:g}" if s.y_offset else "")
        for w in widgets:
            w.blockSignals(False)

    def _on_series_style_changed(self) -> None:
        s = self._selected_series()
        if s is None:
            return
        s.label = self.series_name.text().strip()
        s.linestyle = LINE_STYLES[self.series_linestyle.currentText()]
        s.marker = MARKERS[self.series_marker.currentText()]
        if s.linestyle == "None" and not s.marker:
            # Neither line nor marker would draw nothing at all.
            s.marker = "o"
            self.series_marker.blockSignals(True)
            self.series_marker.setCurrentText("Circle")
            self.series_marker.blockSignals(False)
        s.markersize = self.series_markersize.value()
        s.alpha = self.series_alpha.value() / 100
        s.x_offset = _parse_float(self.series_x_offset.text())
        s.y_offset = _parse_float(self.series_y_offset.text())
        self.series_table.item(self.series.index(s), 0).setText(self._series_label(s))
        self.style_target.setText(f"Editing <b>{self._series_label(s)}</b>")
        self.redraw()

    def _apply_style_to_all(self) -> None:
        src = self._selected_series()
        if src is None:
            return
        for s, spin in zip(self.series, self._width_spins()):
            s.linestyle, s.marker = src.linestyle, src.marker
            s.markersize, s.alpha, s.width = src.markersize, src.alpha, src.width
            spin.blockSignals(True)
            spin.setValue(src.width)
            spin.blockSignals(False)
        self.redraw()

    def _on_cmap_toggled(self, on: bool) -> None:
        self.cmap_combo.setEnabled(on)
        self.cmap_reverse.setEnabled(on)
        self.redraw()

    # -------------------------------------------------------------- colours
    def _series_colors(self) -> list[tuple[Series, str]]:
        """Colour of every plottable series (hidden ones too, so hiding never recolours)."""
        visible = [s for s in self._plottable() if s.visible]
        if not self.use_cmap.isChecked() or not visible:
            return [(s, s.color) for s in self._plottable()]
        cmap = matplotlib.colormaps[self.cmap_combo.currentText()]
        if self.cmap_reverse.isChecked():
            cmap = cmap.reversed()
        n = len(visible)
        if isinstance(cmap, ListedColormap) and cmap.N <= 20:
            # Qualitative maps: take consecutive distinct colours.
            colors = [cmap(i % cmap.N) for i in range(n)]
        else:
            # Stop short of the top end, which is near-white in many maps.
            colors = [cmap(v) for v in (np.linspace(0, 0.85, n) if n > 1 else [0.4])]
        by_id = {id(s): to_hex(c) for s, c in zip(visible, colors)}
        return [(s, by_id.get(id(s), s.color)) for s in self._plottable()]

    def _x_values(self, ds: Dataset) -> tuple[np.ndarray, str]:
        col = self.x_column
        if col == ROW_INDEX or not ds.numeric_columns:
            return np.arange(len(ds.frame), dtype=float), "row index"
        if col not in ds.frame.columns:
            col = ds.numeric_columns[0]
        return ds.frame[col].to_numpy(dtype=float), col

    # -------------------------------------------------------------- drawing
    # ----------------------------------------------------------- full LaTeX
    def usetex(self) -> bool:
        return hasattr(self, "full_latex") and self.full_latex.isChecked()

    def rc_params(self) -> dict:
        rc = theme.mpl_rc(self._draw_theme())
        if self.usetex():
            rc.update({
                "text.usetex": True,
                "font.family": "serif",
                "text.latex.preamble": "\\usepackage{amsmath}\n" + self.latex_preamble.text(),
            })
        return rc

    def _on_draw_error(self, exc: Exception) -> bool:
        if not self.usetex():
            return False
        self.latex_failed(exc)
        return True

    def latex_failed(self, exc: Exception) -> None:
        """LaTeX could not compile something: switch back to built-in math."""
        self.full_latex.blockSignals(True)
        self.full_latex.setChecked(False)
        self.full_latex.blockSignals(False)
        self.latex_preamble.setEnabled(False)
        lines = [ln for ln in str(exc).splitlines() if ln.strip()]
        detail = "\n".join(lines[-12:])
        QTimer.singleShot(0, self.redraw)
        box = QMessageBox(QMessageBox.Warning, "LaTeX error",
                          "LaTeX could not render a label, so built-in math is used again.",
                          QMessageBox.Ok, self)
        box.setDetailedText(detail)
        box.setAttribute(Qt.WA_DeleteOnClose)
        box.open()  # non-blocking: we may be inside a paint

    def _on_full_latex(self, on: bool) -> None:
        self.latex_preamble.setEnabled(on)
        if on:
            self.statusMessage.emit("Rendering with LaTeX… (the first draw can take a few seconds)")
        self.redraw()

    def _safe(self, text: str) -> str:
        if self.usetex():
            return latex_escape(text)
        shown, ok = safe_text(text)
        if not ok:
            self._math_errors.append(repr(text))
        return shown

    def _draw_theme(self) -> theme.Theme:
        return self._export_theme or theme.current

    def _grid_on(self) -> bool:
        if self._export_grid is not None:
            return self._export_grid
        return self.plot_background.currentText() == "Grid"

    def _font(self) -> float:
        return self._export_font or self.font_size.value()

    def redraw(self, keep_view: bool = True) -> None:
        if self._loading:
            return
        with matplotlib.rc_context(self.rc_params()):
            self._redraw(keep_view)

    def _redraw(self, keep_view: bool) -> None:
        ax = self.ax
        t = self._draw_theme()
        scales = ("log" if self.logx_check.isChecked() else "linear",
                  "log" if self.logy_check.isChecked() else "linear")
        keep_view = keep_view and bool(self._data_lines) and scales == self._last_scales
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        for mark in self.point_marks:
            # Remember where the user dragged the label to.
            if len(mark.artists) == 2:
                mark.offset = tuple(mark.artists[1].xyann)
        for mark in self.text_marks:
            if mark.artists:
                mark.x, mark.y = map(float, mark.artists[0].xyann)
        if self._legend is not None:
            loc = getattr(self._legend, "_loc", None)
            # Dragging changes the legend's location to axes coordinates.
            if isinstance(loc, tuple) and loc != self._legend_drawn_loc:
                self._legend_pos = tuple(map(float, loc))
                self.legend_reset.setEnabled(True)
        self._math_errors = []

        ax.clear()
        self.figure.set_facecolor(t.card)
        ax.set_facecolor(t.card)
        ax.set_xscale(scales[0])
        ax.set_yscale(scales[1])
        self._last_scales = scales

        fs = self._font()

        self._data_lines = []
        self._line_series = {}
        lines = []
        colored = self._series_colors()
        use_cmap = self.use_cmap.isChecked()
        color_of = {id(s): c for s, c in colored}
        for button, s in zip(self._color_buttons(), self.series):
            button.setEnabled(not use_cmap)
            button.set_color(color_of.get(id(s), s.color) if use_cmap else s.color)
        for s, color in colored:
            x, _ = self._x_values(s.dataset)
            x = x + s.x_offset
            y = s.dataset.frame[s.column].to_numpy(dtype=float) + s.y_offset
            (line,) = ax.plot(
                x, y, color=color, label=self._safe(self._series_label(s)), linestyle=s.linestyle,
                linewidth=s.width, marker=s.marker, markersize=s.markersize, alpha=s.alpha,
            )
            line.set_visible(s.visible)
            self._line_series[line] = s
            lines.append((s, line))
            if s.visible:
                self._data_lines.append(line)

        visible = [s for s, _ in colored if s.visible]
        x_names = {self._x_values(s.dataset)[1] for s in visible} or {self.x_column or ""}
        auto_x = " / ".join(sorted(x_names))
        auto_y = visible[0].column if len({s.column for s in visible}) == 1 else ""
        auto_title = self.title
        self.title_edit.setPlaceholderText(f"(auto) {auto_title}")
        self.xlabel_edit.setPlaceholderText(f"(auto) {auto_x}")
        self.xname_edit.setPlaceholderText(f"(auto) {auto_x}")
        self.ylabel_edit.setPlaceholderText(f"(auto) {auto_y}" if auto_y else "(none)")
        ax.set_title(self._safe(self.title_edit.text() or auto_title), fontsize=fs + 2)
        ax.set_xlabel(self._safe(self.xlabel_edit.text() or auto_x), fontsize=fs)
        ax.set_ylabel(self._safe(self.ylabel_edit.text() or auto_y), fontsize=fs)
        ax.tick_params(labelsize=fs - 1)
        if self.accent_axes.isChecked():
            for spine in ax.spines.values():
                spine.set_color(t.accent)
            ax.tick_params(which="both", colors=t.accent)
            for text in (ax.title, ax.xaxis.label, ax.yaxis.label):
                text.set_color(t.accent)
        if self._grid_on():
            ax.grid(True, color=t.border, alpha=0.6 if t.dark else 0.3)
        self._legend_map = {}
        self._legend = None
        if self.legend_check.isChecked() and lines:
            # Hidden lines stay in the legend, faded, so a click brings them back.
            loc = self._legend_pos or self.legend_loc.currentText()
            legend = ax.legend([l for _, l in lines], [l.get_label() for _, l in lines],
                               loc=loc, fontsize=fs - 1, ncols=self.legend_cols.value(),
                               handlelength=self.legend_line_length.value(),
                               frameon=self.legend_box.isChecked())
            legend.set_draggable(True)
            self._legend, self._legend_drawn_loc = legend, getattr(legend, "_loc", None)
            for (s, _), handle, text in zip(lines, legend.legend_handles, legend.get_texts()):
                handle.set_visible(True)
                for artist in (handle, text):
                    artist.set_picker(True)
                    artist.set_alpha(None if s.visible else 0.3)
                    self._legend_map[artist] = s

        self._draw_marks()

        if keep_view:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        else:
            ax.relim(visible_only=True)
            ax.autoscale(True)
            ax.autoscale_view()
        self._apply_ticks(ax, t, fs)
        if self._math_errors:
            self.statusMessage.emit("Could not read the math in: " + ", ".join(self._math_errors)
                                    + " (shown as plain text)")

        # ax.clear() drops axes callbacks, so reconnect every time.
        ax.callbacks.connect("xlim_changed", lambda _: self._sync_range_fields())
        ax.callbacks.connect("ylim_changed", lambda _: self._sync_range_fields())
        self._make_span_selector()
        self._sync_range_fields()
        self.canvas.draw_idle()
        if not keep_view:
            self.toolbar.update()  # reset the Home/Back/Forward history
            self.toolbar.push_current()

    def _draw_marks(self) -> None:
        fs = self._font()
        for mark in self.text_marks:
            selected = id(mark) in self._selected_marks
            ann = self.ax.annotate(
                self._safe(mark.content), (mark.x, mark.y), xycoords="data", fontsize=fs + 1,
                color=mark.color, zorder=7, annotation_clip=False,
                bbox=dict(boxstyle="round,pad=0.3", fc="none", ec=mark.color, lw=1, ls="--")
                if selected else None)
            ann.draggable(True)
            mark.artists = [ann]
        for mark in self.region_marks:
            selected = id(mark) in self._selected_marks
            mark.artists = [self.ax.axvspan(
                mark.xmin, mark.xmax, facecolor=mark.color, alpha=0.38 if selected else 0.2,
                zorder=0.5 if selected else 0, edgecolor=mark.color if selected else "none",
                linewidth=2 if selected else 0, linestyle="--")]
        for mark in self.point_marks:
            selected = id(mark) in self._selected_marks
            (dot,) = self.ax.plot([mark.x], [mark.y], marker="o", markersize=13 if selected else 9,
                                  markerfacecolor="none", markeredgewidth=3 if selected else 2,
                                  color=mark.color, zorder=5, scalex=False, scaley=False)
            ann = self.ax.annotate(
                mark.text, (mark.x, mark.y), xytext=mark.offset, textcoords="offset points",
                fontsize=fs - 1, color=mark.color, zorder=6,
                bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.6", fc=self._draw_theme().card, ec=mark.color, alpha=0.92),
                arrowprops=dict(arrowstyle="-", color=mark.color, lw=1),
            )
            ann.draggable(True)
            mark.artists = [dot, ann]

    def _make_span_selector(self) -> None:
        if self._span is not None:
            self._span.disconnect_events()
        self._span = SpanSelector(
            self.ax, self._on_span, "horizontal", useblit=True, button=1,
            props=dict(alpha=0.2, facecolor=self.mark_color.color),
        )
        self._span.set_active(self.act_mark_region.isChecked())

    # --------------------------------------------------------------- ranges
    def _sync_range_fields(self) -> None:
        if self._syncing:
            return
        self.viewChanged.emit()
        (x0, x1), (y0, y1) = self.ax.get_xlim(), self.ax.get_ylim()
        for edit, value in ((self.xmin_edit, x0), (self.xmax_edit, x1),
                            (self.ymin_edit, y0), (self.ymax_edit, y1)):
            if not edit.hasFocus():
                edit.setText(f"{value:.6g}")

    def apply_ranges(self) -> None:
        def parse(edit: QLineEdit, current: float) -> float:
            try:
                return float(edit.text())
            except ValueError:
                return current

        (x0, x1), (y0, y1) = self.ax.get_xlim(), self.ax.get_ylim()
        x0, x1 = parse(self.xmin_edit, x0), parse(self.xmax_edit, x1)
        y0, y1 = parse(self.ymin_edit, y0), parse(self.ymax_edit, y1)
        if x0 == x1 or y0 == y1:
            QMessageBox.warning(self, "Invalid range", "Minimum and maximum must differ.")
            return
        if (self.logx_check.isChecked() and min(x0, x1) <= 0) or (
            self.logy_check.isChecked() and min(y0, y1) <= 0
        ):
            QMessageBox.warning(self, "Invalid range", "Log axes need positive limits.")
            return
        self._syncing = True
        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y0, y1)
        self._syncing = False
        self._sync_range_fields()
        self.toolbar.push_current()
        self.canvas.draw_idle()

    def autoscale(self) -> None:
        self.ax.relim(visible_only=True)
        self.ax.autoscale(True)
        self.ax.autoscale_view()
        self.toolbar.push_current()
        self.canvas.draw_idle()

    # ---------------------------------------------------------------- marks
    def _set_mark_mode(self, mode: str, on: bool) -> None:
        if on:
            for other_mode, action in self._mode_actions().items():
                if other_mode != mode:
                    action.setChecked(False)
            # Leave matplotlib's pan/zoom mode so clicks reach us.
            if self.toolbar.mode == _Mode.PAN:
                self.toolbar.pan()
            elif self.toolbar.mode == _Mode.ZOOM:
                self.toolbar.zoom()
        if mode == "region" and self._span is not None:
            self._span.set_props(facecolor=self.mark_color.color)
            self._span.set_active(on)
        self.canvas.setCursor(Qt.CrossCursor if self._mark_mode() else Qt.ArrowCursor)

    def _mode_actions(self) -> dict:
        return {"point": self.act_mark_point, "region": self.act_mark_region, "text": self.act_add_text}

    def _mark_mode(self) -> str | None:
        return next((m for m, a in self._mode_actions().items() if a.isChecked()), None)

    def _on_nav_mode(self) -> None:
        if self.toolbar.mode != _Mode.NONE:
            for action in self._mode_actions().values():
                action.setChecked(False)

    def _on_span(self, xmin: float, xmax: float) -> None:
        if xmin == xmax:
            return
        self.region_marks.append(RegionMark(xmin, xmax, self.mark_color.color))
        self._after_marks_changed()

    def _nearest_data_point(self, event) -> tuple[float, float, Line2D | None, float]:
        """Nearest plotted point to the mouse, in screen pixels."""
        best = (event.xdata, event.ydata, None, np.inf)
        for line in self._data_lines:
            xy = np.column_stack(line.get_data()).astype(float)
            finite = np.isfinite(xy).all(axis=1)
            if not finite.any():
                continue
            xy = xy[finite]
            screen = self.ax.transData.transform(xy)
            d = np.hypot(screen[:, 0] - event.x, screen[:, 1] - event.y)
            i = int(np.argmin(d))
            if d[i] < best[3]:
                best = (xy[i, 0], xy[i, 1], line, d[i])
        return best

    def _text_mark_at(self, event) -> TextMark | None:
        for mark in reversed(self.text_marks):
            if mark.artists and mark.artists[0].contains(event)[0]:
                return mark
        return None

    def _mark_at(self, event):
        """Return the mark under the mouse, if any."""
        if (text := self._text_mark_at(event)) is not None:
            return text
        for mark in reversed(self.point_marks):
            sx, sy = self.ax.transData.transform((mark.x, mark.y))
            if np.hypot(sx - event.x, sy - event.y) < PICK_RADIUS_PX or (
                len(mark.artists) == 2 and mark.artists[1].contains(event)[0]
            ):
                return mark
        for mark in reversed(self.region_marks):
            if mark.artists and mark.artists[0].contains(event)[0]:
                return mark
        return None

    def _marks_of(self, mark) -> list:
        if isinstance(mark, PointMark):
            return self.point_marks
        if isinstance(mark, TextMark):
            return self.text_marks
        return self.region_marks

    def _ask_text(self, title: str, current: str = "") -> str | None:
        while True:
            text, ok = QInputDialog.getText(
                self, title, "Text or equation. Put math between $…$, e.g.\n"
                "$E = mc^2$,  $\\Delta R/R \\propto e^{-t/\\tau}$,  $\\frac{a}{b}$,  cm$^{-1}$",
                QLineEdit.Normal, current)
            if not ok or not text.strip():
                return None
            if self.usetex() or safe_text(text)[1]:
                return text
            QMessageBox.warning(self, "Invalid math",
                                "That math could not be read; please check the $…$ part.")
            current = text

    def _add_text_at(self, event) -> None:
        content = self._ask_text("Add text / equation")
        if content:
            self.text_marks.append(TextMark(float(event.xdata), float(event.ydata), content,
                                            self.mark_color.color))
            self._after_marks_changed()

    def _edit_text_mark(self, mark: TextMark) -> None:
        content = self._ask_text("Edit text / equation", mark.content)
        if content:
            mark.content = content
            self._after_marks_changed()

    def _region_hit(self, event):
        """(region, "left" | "right" | "move") under the mouse, preferring selected regions."""
        if event.xdata is None:
            return None, None
        ordered = sorted(reversed(self.region_marks), key=lambda m: id(m) not in self._selected_marks)
        for mark in ordered:
            left = self.ax.transData.transform((mark.xmin, 0))[0]
            right = self.ax.transData.transform((mark.xmax, 0))[0]
            left, right = min(left, right), max(left, right)
            if abs(event.x - left) <= EDGE_GRAB_PX:
                return mark, "left"
            if abs(event.x - right) <= EDGE_GRAB_PX:
                return mark, "right"
            if left < event.x < right:
                return mark, "move"
        return None, None

    def _rename_label_at(self, event) -> bool:
        """Double-click on the title or an axis label: rename it."""
        targets = ((self.ax.title, self.title_edit, "Title"),
                   (self.ax.xaxis.label, self.xlabel_edit, "X axis name"),
                   (self.ax.yaxis.label, self.ylabel_edit, "Y axis name"))
        for text, edit, what in targets:
            if text.get_visible() and text.contains(event)[0]:
                name, ok = QInputDialog.getText(self, f"Rename {what.lower()}",
                                                f"{what} (empty for automatic):",
                                                QLineEdit.Normal, edit.text() or text.get_text())
                if ok:
                    edit.setText(name.strip())
                    self.redraw()
                return True
        return False

    def _on_press(self, event) -> None:
        if self.toolbar.mode != _Mode.NONE:
            return
        if event.dblclick and event.button == 1 and self._rename_label_at(event):
            return
        if event.inaxes is not self.ax:
            return
        if event.button == 1 and event.dblclick and (text := self._text_mark_at(event)) is not None:
            self._edit_text_mark(text)
            return
        if event.button == 1 and not event.dblclick and self._text_mark_at(event) is not None:
            return  # matplotlib's draggable text handles the drag
        if event.button == 1 and not event.dblclick and self._mark_mode() == "text":
            self._add_text_at(event)
            return
        if event.button == 1 and not event.dblclick:
            mark, kind = self._region_hit(event)
            # Without a marking tool any region can be grabbed; with one, only
            # the selected region (so new marks can still be made on top of others).
            if mark is not None and (self._mark_mode() is None or id(mark) in self._selected_marks):
                self.select_mark(mark)
                self._drag = dict(mark=mark, kind=kind, x0=event.xdata,
                                  xmin=mark.xmin, xmax=mark.xmax)
                if self._span is not None:
                    self._span.set_active(False)
                self.redraw()
                return
        if event.button == 3:
            mark = self._mark_at(event)
            if mark is not None:
                self._marks_of(mark).remove(mark)
                self._after_marks_changed()
            return
        if event.button != 1 or self._mark_mode() != "point" or event.dblclick:
            return
        # Clicking an existing label drags it rather than adding a new mark.
        if any(len(m.artists) == 2 and m.artists[1].contains(event)[0] for m in self.point_marks):
            return
        x, y = event.xdata, event.ydata
        if self.snap_check.isChecked():
            x, y, _, _ = self._nearest_data_point(event)
        self.point_marks.append(PointMark(float(x), float(y), self.mark_color.color))
        self._after_marks_changed()

    def _on_mark_selection(self) -> None:
        self._selected_marks = {id(i.data(Qt.UserRole)) for i in self.marks_list.selectedItems()}
        self.redraw()

    def select_mark(self, mark) -> None:
        self._selected_marks = {id(mark)}
        self.marks_list.blockSignals(True)
        for i in range(self.marks_list.count()):
            item = self.marks_list.item(i)
            item.setSelected(item.data(Qt.UserRole) is mark)
        self.marks_list.blockSignals(False)

    def _after_marks_changed(self) -> None:
        alive = {id(m) for m in [*self.point_marks, *self.region_marks, *self.text_marks]}
        self._selected_marks &= alive
        self.marks_list.blockSignals(True)
        self.marks_list.clear()
        for mark in self.point_marks:
            item = QListWidgetItem(f"Point  {mark.text}")
            item.setForeground(QColor(mark.color))
            item.setData(Qt.UserRole, mark)
            self.marks_list.addItem(item)
        for mark in self.region_marks:
            item = QListWidgetItem(f"Region  {mark.text}")
            item.setForeground(QColor(mark.color))
            item.setData(Qt.UserRole, mark)
            self.marks_list.addItem(item)
        for mark in self.text_marks:
            item = QListWidgetItem(f"Text  {mark.text}")
            item.setForeground(QColor(mark.color))
            item.setData(Qt.UserRole, mark)
            self.marks_list.addItem(item)
        for i in range(self.marks_list.count()):
            item = self.marks_list.item(i)
            item.setSelected(id(item.data(Qt.UserRole)) in self._selected_marks)
        self.marks_list.blockSignals(False)
        self.redraw()

    def _remove_selected_marks(self) -> None:
        selected = {id(i.data(Qt.UserRole)) for i in self.marks_list.selectedItems()}
        self.point_marks = [m for m in self.point_marks if id(m) not in selected]
        self.region_marks = [m for m in self.region_marks if id(m) not in selected]
        self.text_marks = [m for m in self.text_marks if id(m) not in selected]
        self._after_marks_changed()

    def clear_marks(self) -> None:
        self.point_marks.clear()
        self.region_marks.clear()
        self.text_marks.clear()
        self._after_marks_changed()

    # ---------------------------------------------------------------- mouse
    def _on_release(self, event) -> None:
        if self._drag is None:
            return
        mark = self._drag["mark"]
        mark.xmin, mark.xmax = sorted((mark.xmin, mark.xmax))
        self._drag = None
        if self._span is not None:
            self._span.set_active(self.act_mark_region.isChecked())
        self._after_marks_changed()

    def _drag_region(self, event) -> None:
        d, mark = self._drag, self._drag["mark"]
        dx = event.xdata - d["x0"]
        if d["kind"] == "move":
            mark.xmin, mark.xmax = d["xmin"] + dx, d["xmax"] + dx
        elif d["kind"] == "left":
            mark.xmin = d["xmin"] + dx
        else:
            mark.xmax = d["xmax"] + dx
        patch = mark.artists[0] if mark.artists else None
        if hasattr(patch, "set_x") and hasattr(patch, "set_width"):
            lo, hi = sorted((mark.xmin, mark.xmax))
            patch.set_x(lo)
            patch.set_width(hi - lo)
            self.canvas.draw_idle()
        else:
            self.redraw()
        self.coord_label.setText(f"Region  {mark.xmin:.6g} → {mark.xmax:.6g}")

    def _update_cursor(self, event) -> None:
        mark, kind = self._region_hit(event)
        grabbable = mark is not None and (self._mark_mode() is None or id(mark) in self._selected_marks)
        if grabbable:
            self.canvas.setCursor(Qt.SizeHorCursor if kind in ("left", "right") else Qt.OpenHandCursor)
        else:
            self.canvas.setCursor(Qt.CrossCursor if self._mark_mode() else Qt.ArrowCursor)

    def _on_motion(self, event) -> None:
        if self._drag is not None and event.xdata is not None:
            self._drag_region(event)
            return
        if event.inaxes is not self.ax or event.xdata is None:
            return
        if self.toolbar.mode == _Mode.NONE:
            self._update_cursor(event)
        text = f"x = {event.xdata:<12.6g}  y = {event.ydata:<12.6g}"
        if self._data_lines:
            x, y, line, dist = self._nearest_data_point(event)
            if line is not None and dist < 40:
                s = self._line_series.get(line)
                if s is not None and (s.x_offset or s.y_offset):
                    # Report the data value, not the shifted position.
                    x, y = x - s.x_offset, y - s.y_offset
                    text += f"   │  nearest {line.get_label()} (data, offset removed): ({x:.6g}, {y:.6g})"
                else:
                    text += f"   │  nearest {line.get_label()}: ({x:.6g}, {y:.6g})"
        self.coord_label.setText(text)

    def _on_pick(self, event) -> None:
        s = getattr(self, "_legend_map", {}).get(event.artist)
        if s is not None and event.mouseevent.button == 1:
            self.set_series_visible(s, not s.visible)

    def _on_scroll(self, event) -> None:
        """Mouse-wheel zoom around the cursor. Shift: x only, Ctrl: y only."""
        if event.inaxes is not self.ax:
            return
        factor = 1 / 1.2 if event.button == "up" else 1.2
        key = (event.key or "").lower()
        zoom_x = "control" not in key and "ctrl" not in key
        zoom_y = "shift" not in key

        def zoom(lim, center, scale):
            if scale == "log":
                lo, hi, c = np.log10(lim[0]), np.log10(lim[1]), np.log10(center)
                return 10 ** (c - (c - lo) * factor), 10 ** (c + (hi - c) * factor)
            return center - (center - lim[0]) * factor, center + (lim[1] - center) * factor

        if zoom_x:
            self.ax.set_xlim(zoom(self.ax.get_xlim(), event.xdata, self.ax.get_xscale()))
        if zoom_y:
            self.ax.set_ylim(zoom(self.ax.get_ylim(), event.ydata, self.ax.get_yscale()))
        self.toolbar.push_current()
        self.canvas.draw_idle()

    def change_theme(self, old: theme.Theme, new: theme.Theme) -> None:
        """Swap colours that came from the old theme's defaults for the new theme's."""

        def swap(color: str) -> str:
            mapping = dict(zip(old.series_colors, new.series_colors))
            mapping[old.accent] = new.accent
            return mapping.get(color, color)

        for s in self.series:
            s.color = swap(s.color)
        for mark in [*self.point_marks, *self.region_marks, *self.text_marks]:
            mark.color = swap(mark.color)
        self.mark_color.set_color(swap(self.mark_color.color))
        if self.cmap_combo.currentText() == old.name:
            self.cmap_combo.blockSignals(True)
            self.cmap_combo.setCurrentText(new.name)
            self.cmap_combo.blockSignals(False)
        self.apply_toolbar_icons()
        self._after_marks_changed()  # refresh the marks list and redraw

    @contextmanager
    def export_style(self, draw_theme: theme.Theme, font_size: float | None,
                     grid: bool | None = None):
        """Temporarily redraw with another theme, font size and grid setting (for export)."""
        self._export_theme, self._export_font, self._export_grid = draw_theme, font_size, grid
        try:
            self.redraw()
            yield
        finally:
            self._export_theme = self._export_font = self._export_grid = None
            self.redraw()

    def apply_toolbar_icons(self) -> None:
        """Recolour matplotlib's toolbar icons with the theme's text colour."""
        color = theme.current.text.encode()
        for _text, _tip, image, callback in self.toolbar.toolitems:
            action = self.toolbar._actions.get(callback)
            path = cbook._get_data_path("images", f"{image}.svg") if image else None
            if action is None or path is None or not path.exists():
                continue
            svg = path.read_bytes()
            for old in (b"black", b"#000000"):
                svg = svg.replace(b"fill:" + old, b"fill:" + color).replace(b"stroke:" + old, b"stroke:" + color)
            renderer = QSvgRenderer(QByteArray(svg))
            pixmap = QPixmap(64, 64)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            renderer.render(painter)
            painter.end()
            action.setIcon(QIcon(pixmap))

    # ----------------------------------------------------------- data table
    def show_data_preview(self, focus: bool = True) -> None:
        if self.data_preview is None:
            self.data_preview = DataPreview(self.datasets)
            self.data_preview.closeRequested.connect(self.close_data_preview)
            self.data_preview.layout_button.clicked.connect(self.toggle_data_layout)
            self.views.addWidget(self.data_preview)
            total = sum(self.views.sizes()) or 800
            self.views.setSizes([int(total * 0.6), int(total * 0.4)])
            self._update_layout_button()
        if focus:
            self.data_preview.table.setFocus()
        self.act_data.blockSignals(True)
        self.act_data.setChecked(True)
        self.act_data.blockSignals(False)

    def close_data_preview(self) -> None:
        if self.data_preview is not None:
            self.data_preview.setParent(None)
            self.data_preview.deleteLater()
            self.data_preview = None
        self.act_data.blockSignals(True)
        self.act_data.setChecked(False)
        self.act_data.blockSignals(False)

    def open_fft(self) -> None:
        from .fft_dialog import FFTDialog  # imported lazily: pulls in scipy

        if getattr(self, "fft_dialog", None) is None:
            self.fft_dialog = FFTDialog(self)
            if self._fft_state:   # settings restored from a session
                self.fft_dialog.apply_state(self._fft_state)
                self._fft_state = None
        else:
            self.fft_dialog.refresh_choices()
        self.fft_dialog.show()
        self.fft_dialog.raise_()
        self.fft_dialog.activateWindow()

    def toggle_data_layout(self) -> None:
        """Switch the data table between below the plot and beside it."""
        vertical = self.views.orientation() == Qt.Vertical
        self.views.setOrientation(Qt.Horizontal if vertical else Qt.Vertical)
        total = sum(self.views.sizes()) or 800
        self.views.setSizes([int(total * 0.6), int(total * 0.4)])
        self._update_layout_button()

    def _update_layout_button(self) -> None:
        if self.data_preview is not None:
            below = self.views.orientation() == Qt.Vertical
            self.data_preview.layout_button.setText("Place beside plot" if below else "Place below plot")

    def toggle_data_preview(self) -> None:
        if self.data_preview is None:
            self.show_data_preview()
        else:
            self.close_data_preview()

    # -------------------------------------------------------------- session
    def _state_widgets(self) -> dict:
        """Settings saved in a session, by name."""
        w = {
            "title": self.title_edit, "x_label": self.xlabel_edit, "y_label": self.ylabel_edit,
            "font_size": self.font_size, "accent_axes": self.accent_axes,
            "background": self.plot_background, "full_latex": self.full_latex,
            "latex_preamble": self.latex_preamble,
            "use_colormap": self.use_cmap, "colormap": self.cmap_combo,
            "colormap_reverse": self.cmap_reverse,
            "legend": self.legend_check, "legend_position": self.legend_loc,
            "legend_line_length": self.legend_line_length, "legend_columns": self.legend_cols,
            "legend_box": self.legend_box,
            "log_x": self.logx_check, "log_y": self.logy_check,
            "tick_direction": self.tick_direction, "mirror_ticks": self.mirror_ticks,
            "minor_grid": self.minor_grid, "snap_to_data": self.snap_check,
            "mark_color": self.mark_color,
        }
        for axis in ("x", "y"):
            w[f"major_step_{axis}"] = self.major_step[axis]
            w[f"minor_ticks_{axis}"] = self.minor_ticks[axis]
        for which in ("major", "minor"):
            w[f"tick_width_{which}"] = self.tick_width[which]
            w[f"tick_length_{which}"] = self.tick_length[which]
        return w

    def get_state(self) -> dict:
        settings = {}
        for name, widget in self._state_widgets().items():
            if isinstance(widget, ColorButton):
                settings[name] = widget.color
            elif isinstance(widget, QLineEdit):
                settings[name] = widget.text()
            elif isinstance(widget, QComboBox):
                settings[name] = widget.currentText()
            elif isinstance(widget, QCheckBox):
                settings[name] = widget.isChecked()
            else:
                settings[name] = widget.value()
        if self._legend is not None:   # pick up a drag that hasn't been redrawn yet
            loc = getattr(self._legend, "_loc", None)
            if isinstance(loc, tuple) and loc != self._legend_drawn_loc:
                self._legend_pos = tuple(map(float, loc))
        for mark in self.text_marks:
            if mark.artists:
                mark.x, mark.y = map(float, mark.artists[0].xyann)
        for mark in self.point_marks:
            if len(mark.artists) == 2:
                mark.offset = tuple(mark.artists[1].xyann)
        series_fields = ("color", "visible", "label", "width", "linestyle", "marker",
                         "markersize", "alpha", "x_offset", "y_offset")
        return {
            "name": self.custom_name,
            "x_column": self.x_column,
            "settings": settings,
            "series": [{"dataset": self.datasets.index(s.dataset), "column": s.column,
                        **{f: getattr(s, f) for f in series_fields}} for s in self.series],
            "view": {"x": list(map(float, self.ax.get_xlim())), "y": list(map(float, self.ax.get_ylim()))},
            "legend_drag": list(self._legend_pos) if self._legend_pos else None,
            "marks": {
                "points": [{"x": m.x, "y": m.y, "color": m.color, "offset": list(m.offset)}
                           for m in self.point_marks],
                "regions": [{"xmin": m.xmin, "xmax": m.xmax, "color": m.color} for m in self.region_marks],
                "texts": [{"x": m.x, "y": m.y, "text": m.content, "color": m.color} for m in self.text_marks],
            },
            "data_table": None if self.data_preview is None else (
                "below" if self.views.orientation() == Qt.Vertical else "beside"),
            "analysis": (self.fft_dialog.get_state() if getattr(self, "fft_dialog", None)
                         else self._fft_state),
        }

    def apply_state(self, state: dict) -> None:
        """Restore everything get_state() saved (datasets must already be added)."""
        self._loading = True
        try:
            self.custom_name = state.get("name", "")
            if state.get("x_column"):
                self.x_combo.setCurrentText(state["x_column"])
            widgets = self._state_widgets()
            for name, value in state.get("settings", {}).items():
                widget = widgets.get(name)
                if widget is None:
                    continue
                if isinstance(widget, ColorButton):
                    widget.set_color(value)
                elif isinstance(widget, QLineEdit):
                    widget.setText(value)
                elif isinstance(widget, QComboBox):
                    widget.setCurrentText(value)
                elif isinstance(widget, QCheckBox):
                    if widget is not self.full_latex or widget.isEnabled():
                        widget.setChecked(bool(value))
                else:
                    widget.setValue(value)
            by_key = {(self.datasets.index(s.dataset), s.column): s for s in self.series}
            for saved in state.get("series", []):
                s = by_key.get((saved["dataset"], saved["column"]))
                if s is None:
                    continue  # the file no longer has this column
                for field_name, value in saved.items():
                    if field_name not in ("dataset", "column"):
                        setattr(s, field_name, value)
            marks = state.get("marks", {})
            self.point_marks = [PointMark(m["x"], m["y"], m["color"], tuple(m["offset"]))
                                for m in marks.get("points", [])]
            self.region_marks = [RegionMark(m["xmin"], m["xmax"], m["color"])
                                 for m in marks.get("regions", [])]
            self.text_marks = [TextMark(m["x"], m["y"], m["text"], m["color"])
                               for m in marks.get("texts", [])]
            self._legend_pos = tuple(state["legend_drag"]) if state.get("legend_drag") else None
            self.legend_reset.setEnabled(self._legend_pos is not None)
            self.latex_preamble.setEnabled(self.full_latex.isChecked())
            self.cmap_combo.setEnabled(self.use_cmap.isChecked())
            self.cmap_reverse.setEnabled(self.use_cmap.isChecked())
        finally:
            self._loading = False
        self._refresh_series_table()
        self._after_marks_changed()   # rebuilds the marks list and redraws
        self.redraw(keep_view=False)
        view = state.get("view")
        if view:
            self.ax.set_xlim(*view["x"])
            self.ax.set_ylim(*view["y"])
            self.toolbar.push_current()
            self.canvas.draw_idle()
        if state.get("data_table"):
            self.show_data_preview(focus=False)
            if state["data_table"] == "beside":
                self.toggle_data_layout()
        if state.get("analysis"):
            # Kept until the analysis window is opened, so closed windows keep
            # their settings too.
            self._fft_state = dict(state["analysis"])
            if self._fft_state.get("open"):
                self.open_fft()

    def save_figure(self) -> str | None:
        name = self.custom_name or "_".join(d.path.stem for d in self.datasets)
        path = export_figure(self.figure, name, self, restyle=self.export_style)
        if path:
            self.statusMessage.emit(f"Saved {path}")
        return path
