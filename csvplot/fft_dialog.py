"""Spectral analysis window: windowed FFT / Welch / spectrogram and period extraction."""

from __future__ import annotations

import io
from contextlib import contextmanager
from pathlib import Path

import matplotlib
import numpy as np
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
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .export import export_figure
from .analysis import (
    DETRENDS,
    METHODS,
    WINDOW_PARAMS,
    WINDOWS,
    FFTSettings,
    SpectrumResult,
    frequency_unit,
    spectrum,
)
from .loaders import Dataset
from .plot_tab import PlotCanvas

ALL_SHOWN = "All lines shown on the plot"
CURRENT_VIEW = "Current view (x range on screen)"
FULL_DATA = "Full data"
ZERO_PAD = {"1× (none)": 1, "2×": 2, "4×": 4, "8×": 8, "16×": 16}


class FFTDialog(QDialog):
    """Non-modal analysis window for one plot tab."""

    def __init__(self, tab, parent=None):
        super().__init__(parent or tab, Qt.Window)
        self.tab = tab
        self.results: list[SpectrumResult] = []
        self._export_theme: theme.Theme | None = None
        self._export_font: float | None = None
        self._export_grid: bool | None = None
        self.setWindowTitle(f"Spectral analysis — {tab.display_name}")
        self.resize(1300, 840)

        self._timer = QTimer(self, singleShot=True, interval=300)
        self._timer.timeout.connect(self.run)

        controls = QWidget()
        controls.setObjectName("controls_panel")
        cl = QVBoxLayout(controls)
        cl.setSpacing(10)
        cl.addWidget(self._build_data_group())
        cl.addWidget(self._build_fft_group())
        cl.addWidget(self._build_peaks_group())
        cl.addWidget(self._build_output_group())
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

        self.table = QTableWidget(0, 5)
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table_card = QFrame()
        table_card.setObjectName("card")
        tl = QVBoxLayout(table_card)
        tl.setContentsMargins(12, 10, 12, 12)
        title = QLabel("<b>Extracted periods</b> (strongest peaks first)")
        tl.addWidget(title)
        tl.addWidget(self.table)

        right = QSplitter(Qt.Vertical)
        right.addWidget(plot_card)
        right.addWidget(table_card)
        right.setSizes([600, 220])

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
        self._on_method()
        self._on_window()
        tab.viewChanged.connect(self._on_view_changed)
        self.run()

    # ------------------------------------------------------------------ UI
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

    def _build_fft_group(self) -> QGroupBox:
        box = QGroupBox("Window && FFT")
        form = QFormLayout(box)
        self.method = QComboBox()
        self.method.addItems(METHODS)
        self.method.currentTextChanged.connect(self._on_method)
        form.addRow("Method", self.method)

        self.window = QComboBox()
        self.window.addItems(WINDOWS)
        self.window.currentTextChanged.connect(self._on_window)
        form.addRow("Window", self.window)
        self.window_param = QDoubleSpinBox(decimals=2)
        self.window_param_label = QLabel("α")
        self.window_param.valueChanged.connect(self._schedule)
        form.addRow(self.window_param_label, self.window_param)

        self.segment = QDoubleSpinBox(minimum=0, maximum=1e12, decimals=4)
        self.segment.setSpecialValueText("auto (¼ of range)")
        self.segment.setToolTip("Length of each sliding window, in x units.")
        self.segment.valueChanged.connect(self._schedule)
        form.addRow("Window length", self.segment)
        self.overlap = QSpinBox(minimum=0, maximum=95, singleStep=5, value=50, suffix=" %")
        self.overlap.valueChanged.connect(self._schedule)
        form.addRow("Overlap", self.overlap)

        self.detrend = QComboBox()
        self.detrend.addItems(DETRENDS)
        self.detrend.setCurrentText("Remove linear trend")
        self.detrend.currentTextChanged.connect(self._schedule)
        form.addRow("Detrend", self.detrend)
        self.zero_pad = QComboBox()
        self.zero_pad.addItems(ZERO_PAD)
        self.zero_pad.setCurrentText("4×")
        self.zero_pad.setToolTip("Zero padding interpolates the spectrum (smoother peaks);\n"
                                 "it does not improve the true resolution.")
        self.zero_pad.currentTextChanged.connect(self._schedule)
        form.addRow("Zero padding", self.zero_pad)
        return box

    def _build_peaks_group(self) -> QGroupBox:
        box = QGroupBox("Periods (peak search)")
        form = QFormLayout(box)
        self.n_peaks = QSpinBox(minimum=1, maximum=30, value=5)
        self.min_height = QDoubleSpinBox(minimum=0, maximum=100, value=5, suffix=" % of max")
        self.f_min, self.f_max = QLineEdit(), QLineEdit()
        for edit in (self.f_min, self.f_max):
            edit.setPlaceholderText("auto")
            edit.editingFinished.connect(self._schedule)
        self.n_peaks.valueChanged.connect(self._schedule)
        self.min_height.valueChanged.connect(self._schedule)
        form.addRow("Peaks per line", self.n_peaks)
        form.addRow("Minimum height", self.min_height)
        frange = QHBoxLayout()
        frange.addWidget(self.f_min)
        frange.addWidget(QLabel("to"))
        frange.addWidget(self.f_max)
        self.freq_range_label = QLabel("Frequency range")
        form.addRow(self.freq_range_label, frange)
        return box

    def _build_output_group(self) -> QGroupBox:
        box = QGroupBox("Display && output")
        form = QFormLayout(box)
        self.axis_mode = QComboBox()
        self.axis_mode.addItems(["Frequency", "Period"])
        self.axis_mode.currentTextChanged.connect(lambda _: self.draw())
        form.addRow("Spectrum x axis", self.axis_mode)
        self.log_amp = QCheckBox("Logarithmic amplitude")
        self.log_amp.toggled.connect(lambda _: self.draw())
        form.addRow(self.log_amp)
        self.show_data = QCheckBox("Show the analysed data")
        self.show_data.setChecked(True)
        self.show_data.toggled.connect(lambda _: self.draw())
        form.addRow(self.show_data)

        buttons = [("Export figure (PNG, SVG, PDF…)", self.export),
                   ("Copy table", self.copy_table), ("Save table (CSV)…", self.save_table),
                   ("Save spectrum (CSV)…", self.save_spectrum),
                   ("Open spectrum as a new plot", self.open_spectrum)]
        for i, (text, slot) in enumerate(buttons):
            btn = QPushButton(text)
            if 0 < i < 4:
                btn.setProperty("secondary", True)
            btn.clicked.connect(slot)
            form.addRow(btn)
        return box

    # -------------------------------------------------------------- choices
    def refresh_choices(self) -> None:
        current_series, current_range = self.series_combo.currentText(), self.range_combo.currentText()
        self.series_combo.blockSignals(True)
        self.series_combo.clear()
        self.series_combo.addItem(ALL_SHOWN)
        for s in self.tab._plottable():
            self.series_combo.addItem(self.tab._series_label(s), s)
        self.series_combo.setCurrentText(current_series or ALL_SHOWN)
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

    def _on_method(self, *_) -> None:
        sliding = METHODS[self.method.currentText()] != "single"
        self.segment.setEnabled(sliding)
        self.overlap.setEnabled(sliding)
        self._schedule()

    def _on_window(self, *_) -> None:
        name = WINDOWS[self.window.currentText()]
        param = WINDOW_PARAMS.get(name)
        self.window_param.setVisible(param is not None)
        self.window_param_label.setVisible(param is not None)
        if param:
            label, default, lo, hi, step = param
            self.window_param_label.setText(label)
            self.window_param.blockSignals(True)
            self.window_param.setRange(lo, hi)
            self.window_param.setSingleStep(step)
            self.window_param.setValue(default)
            self.window_param.blockSignals(False)
        self._schedule()

    def _on_view_changed(self) -> None:
        if self.isVisible() and self.range_combo.currentText() == CURRENT_VIEW:
            self._schedule()

    def _schedule(self, *_) -> None:
        self._timer.start()

    # ------------------------------------------------------------- analysis
    def settings(self) -> FFTSettings:
        def number(edit):
            try:
                return float(edit.text())
            except ValueError:
                return None

        name = WINDOWS[self.window.currentText()]
        return FFTSettings(
            method=METHODS[self.method.currentText()], window=name,
            window_param=self.window_param.value() if name in WINDOW_PARAMS else None,
            segment=self.segment.value() or None, overlap=self.overlap.value(),
            detrend=DETRENDS[self.detrend.currentText()],
            zero_pad=ZERO_PAD[self.zero_pad.currentText()],
            n_peaks=self.n_peaks.value(), min_height=self.min_height.value(),
            f_min=number(self.f_min), f_max=number(self.f_max))

    def _selected_series(self) -> list:
        data = self.series_combo.currentData()
        if data is not None:
            return [data]
        return [s for s in self.tab._plottable() if s.visible]

    def _x_range(self) -> tuple[float, float] | None:
        text = self.range_combo.currentText()
        if text == FULL_DATA:
            return None
        if text == CURRENT_VIEW:
            return tuple(self.tab.ax.get_xlim())
        return self.range_combo.currentData()

    def run(self) -> None:
        settings = self.settings()
        x_range = self._x_range()
        self.results, errors = [], []
        for s in self._selected_series():
            x, _ = self.tab._x_values(s.dataset)
            y = s.dataset.frame[s.column].to_numpy(dtype=float)
            # The plot shows x + offset; analyse the data that is on screen.
            rng = None if x_range is None else (x_range[0] - s.x_offset, x_range[1] - s.x_offset)
            if settings.method != "single" and settings.segment is None:
                span = (max(rng) - min(rng)) if rng else float(np.nanmax(x) - np.nanmin(x))
                settings.segment = span / 4
            try:
                result = spectrum(x, y, settings, self.tab._series_label(s), rng)
            except ValueError as exc:
                errors.append(f"{self.tab._series_label(s)}: {exc}")
                continue
            result.x = result.x + s.x_offset
            colors = {id(series): c for series, c in self.tab._series_colors()}
            result.color = colors.get(id(s), s.color)
            self.results.append(result)
        self.errors = errors
        self.draw()
        self.fill_table()

    # --------------------------------------------------------------- output
    def units(self) -> tuple[str, str]:
        x_name = self.tab.xlabel_edit.text() or self.tab.ax.get_xlabel()
        return frequency_unit(x_name)

    def draw(self) -> None:
        fig = self.figure
        fig.clear()
        funit, xunit = self.units()
        period_axis = self.axis_mode.currentText() == "Period"
        spectro = any(r.spectrogram is not None for r in self.results)
        rows = int(self.show_data.isChecked()) + 1 + int(spectro)
        axes = fig.subplots(rows, 1, squeeze=False)[:, 0]
        i = 0
        t = self._export_theme or theme.current
        grid = self._export_grid if self._export_grid is not None else True
        fig.set_facecolor(t.card)
        with matplotlib.rc_context(self._rc()):
            if self.show_data.isChecked():
                ax = axes[i]
                i += 1
                for r in self.results:
                    ax.plot(r.x, r.y, color=r.color, lw=1.2, label=r.label)
                ax.set_title("Analysed data", fontsize="medium", loc="left")
                ax.set_xlabel(self.tab.ax.get_xlabel())
                ax.grid(grid, alpha=0.3)
            ax = axes[i]
            i += 1
            for n, r in enumerate(self.results):
                f, a = r.frequency, r.amplitude
                if period_axis:
                    keep = f > 0
                    f, a = 1 / f[keep], a[keep]
                ax.plot(f, a, color=r.color, lw=1.4, label=r.label)
                for k, p in enumerate(r.peaks):
                    px = p.period if period_axis else p.frequency
                    ax.plot([px], [p.amplitude], "o", ms=6, mfc="none", mec=r.color, mew=1.5)
                    if k < (3 if len(self.results) == 1 else 2):
                        ax.annotate(f"T = {p.period:.4g} {xunit}", (px, p.amplitude),
                                    xytext=(5, 4 + 11 * n + 11 * k * (len(self.results) == 1)),
                                    textcoords="offset points", fontsize="small", color=r.color)
            if period_axis:
                ax.set_xscale("log")
                ax.set_xlabel(f"Period ({xunit})" if xunit else "Period")
            else:
                ax.set_xlabel(f"Frequency ({funit})")
                ax.set_xlim(*self._frequency_view())
            if self.log_amp.isChecked():
                ax.set_yscale("log")
            ax.set_ylabel("Amplitude")
            ax.set_title("Amplitude spectrum", fontsize="medium", loc="left")
            ax.grid(grid, alpha=0.3)
            if len(self.results) > 1:
                ax.legend(fontsize="small")
            if spectro:
                ax = axes[i]
                r = next(r for r in self.results if r.spectrogram is not None)
                img = r.spectrogram
                if self.log_amp.isChecked():
                    img = np.log10(np.maximum(img, img.max() * 1e-6))
                mesh = ax.pcolormesh(r.seg_centers, r.frequency, img, shading="auto",
                                     cmap=t.name)
                fig.colorbar(mesh, ax=ax, label="log₁₀ amplitude" if self.log_amp.isChecked()
                             else "Amplitude")
                ax.set_xlabel(f"Window centre ({xunit})" if xunit else "Window centre")
                ax.set_ylabel(f"Frequency ({funit})")
                ax.set_title(f"Spectrogram: {r.label}", fontsize="medium", loc="left")
                ax.set_ylim(*self._frequency_view())
        self.canvas.draw_idle()
        self._update_info(funit, xunit)

    def _frequency_view(self) -> tuple[float, float]:
        """Frequency axis limits: the chosen range, else up to a few times the
        highest peak found (the interesting part is usually at low frequency)."""
        settings = self.settings()
        nyquist = max((r.frequency[-1] for r in self.results if len(r.frequency)), default=1.0)
        peak_f = [p.frequency for r in self.results for p in r.peaks]
        hi = min(nyquist, 4 * max(peak_f)) if peak_f else nyquist
        lo = settings.f_min if settings.f_min is not None else 0.0
        hi = settings.f_max if settings.f_max is not None else hi
        return lo, max(hi, lo * 1.01 + 1e-12)

    # -------------------------------------------------------------- session
    def _state_widgets(self) -> list:
        # Order matters: choosing a window resets its parameter, so the
        # parameter comes after it.
        return [("lines", self.series_combo), ("range", self.range_combo),
                ("method", self.method), ("window", self.window),
                ("window_param", self.window_param), ("window_length", self.segment),
                ("overlap", self.overlap), ("detrend", self.detrend), ("zero_padding", self.zero_pad),
                ("peaks", self.n_peaks), ("min_height", self.min_height),
                ("f_min", self.f_min), ("f_max", self.f_max), ("x_axis", self.axis_mode),
                ("log_amplitude", self.log_amp), ("show_data", self.show_data)]

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

    # --------------------------------------------------------------- export
    def _rc(self) -> dict:
        rc = theme.mpl_rc(self._export_theme or theme.current)
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
        name = f"{self.tab.display_name}_spectrum"
        if (path := export_figure(self.figure, name, self, restyle=self.export_style)):
            self.info.setText(f"Saved {path}")

    def _update_info(self, funit: str, xunit: str) -> None:
        if not self.results:
            self.info.setText("; ".join(self.errors) or "No line selected.")
            return
        r = self.results[0]
        parts = [f"{len(r.y)} points, spacing {r.dx:.4g} {xunit}",
                 f"window: {r.segment_points} points",
                 f"resolution Δf ≈ {r.resolution:.3g} {funit}"]
        if r.n_segments > 1:
            parts.append(f"{r.n_segments} windows averaged")
        if any(res.resampled for res in self.results):
            parts.append("uneven spacing → resampled onto a uniform grid")
        if self.errors:
            parts.append("skipped: " + "; ".join(self.errors))
        self.info.setText(" · ".join(parts))

    def rows(self) -> list[list]:
        funit, xunit = self.units()
        out = []
        for r in self.results:
            for p in r.peaks:
                out.append([r.label, p.frequency, p.period, p.amplitude, 100 * p.relative])
        return out

    def headers(self) -> list[str]:
        funit, xunit = self.units()
        return ["Line", f"Frequency ({funit})", f"Period ({xunit})" if xunit else "Period",
                "Amplitude", "Relative (%)"]

    def fill_table(self) -> None:
        rows = self.rows()
        self.table.setHorizontalHeaderLabels(self.headers())
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                text = value if isinstance(value, str) else (f"{value:.1f}" if j == 4 else f"{value:.6g}")
                item = QTableWidgetItem(text)
                if j:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(i, j, item)

    def _table_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows(), columns=self.headers())

    def copy_table(self) -> None:
        buffer = io.StringIO()
        self._table_frame().to_csv(buffer, sep="\t", index=False)
        QApplication.clipboard().setText(buffer.getvalue())
        self.info.setText("Table copied to the clipboard.")

    def _ask_save(self, title: str, name: str) -> str | None:
        folder = QSettings("csvplot", "csvplot").value("export/dir", str(Path.home()))
        path, _ = QFileDialog.getSaveFileName(self, title, str(Path(folder) / name), "CSV (*.csv)")
        if path and not path.lower().endswith(".csv"):
            path += ".csv"
        return path or None

    def save_table(self) -> None:
        if (path := self._ask_save("Save periods table", f"{self.tab.display_name}_periods.csv")):
            self._table_frame().to_csv(path, index=False)

    def spectrum_frame(self) -> pd.DataFrame:
        funit, xunit = self.units()
        if not self.results:
            return pd.DataFrame()
        frame = pd.DataFrame({f"Frequency ({funit})": self.results[0].frequency})
        with np.errstate(divide="ignore"):
            frame[f"Period ({xunit})" if xunit else "Period"] = 1 / self.results[0].frequency
        for r in self.results:
            # Lines analysed over different lengths have different frequency grids.
            frame[f"|FFT| {r.label}"] = np.interp(self.results[0].frequency, r.frequency, r.amplitude)
        return frame

    def save_spectrum(self) -> None:
        if (path := self._ask_save("Save spectrum", f"{self.tab.display_name}_spectrum.csv")):
            self.spectrum_frame().to_csv(path, index=False)

    def open_spectrum(self) -> None:
        frame = self.spectrum_frame()
        if frame.empty:
            QMessageBox.information(self, "Nothing to open", "Run an analysis first.")
            return
        frame = frame.drop(columns=[c for c in frame.columns if c.startswith("Period")])
        name = f"{self.tab.display_name} spectrum"
        self.tab.openDatasetRequested.emit(Dataset(Path(f"{name}.csv"), frame))
