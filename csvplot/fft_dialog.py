"""Spectral analysis window: windowed FFT / Welch / spectrogram and period extraction."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QSpinBox,
)

from .analysis import (
    DETRENDS,
    METHODS,
    WINDOW_PARAMS,
    WINDOWS,
    FFTSettings,
    frequency_unit,
    spectrum,
)
from .analysis_window import ALL_SHOWN, CURRENT_VIEW, FULL_DATA, AnalysisWindow  # noqa: F401
from .loaders import Dataset

ZERO_PAD = {"1× (none)": 1, "2×": 2, "4×": 4, "8×": 8, "16×": 16}


class FFTDialog(AnalysisWindow):
    window_title = "Spectral analysis"
    table_title = "<b>Extracted periods</b> (strongest peaks first)"
    export_suffix = "spectrum"

    # ------------------------------------------------------------------ UI
    def build_groups(self):
        return [self._build_fft_group(), self._build_peaks_group(), self._build_output_group()]

    def on_ready(self) -> None:
        self._on_method()
        self._on_window()

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
        form.addRow("Frequency range", frange)
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
        self.add_buttons(form, [
            ("Export figure (PNG, SVG, PDF…)", self.export, True),
            ("Copy table", self.copy_table, False),
            ("Save table (CSV)…", self.save_table, False),
            ("Save spectrum (CSV)…", self.save_spectrum, False),
            ("Open spectrum as a new plot", self.open_spectrum, True),
        ])
        return box

    def settings_widgets(self):
        # Order matters: choosing a window resets its parameter, so the
        # parameter comes after it.
        return [("method", self.method), ("window", self.window),
                ("window_param", self.window_param), ("window_length", self.segment),
                ("overlap", self.overlap), ("detrend", self.detrend), ("zero_padding", self.zero_pad),
                ("peaks", self.n_peaks), ("min_height", self.min_height),
                ("f_min", self.f_min), ("f_max", self.f_max), ("x_axis", self.axis_mode),
                ("log_amplitude", self.log_amp), ("show_data", self.show_data)]

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

    # ------------------------------------------------------------- analysis
    def settings(self) -> FFTSettings:
        name = WINDOWS[self.window.currentText()]
        return FFTSettings(
            method=METHODS[self.method.currentText()], window=name,
            window_param=self.window_param.value() if name in WINDOW_PARAMS else None,
            segment=self.segment.value() or None, overlap=self.overlap.value(),
            detrend=DETRENDS[self.detrend.currentText()],
            zero_pad=ZERO_PAD[self.zero_pad.currentText()],
            n_peaks=self.n_peaks.value(), min_height=self.min_height.value(),
            f_min=self.number(self.f_min), f_max=self.number(self.f_max))

    def run(self) -> None:
        settings = self.settings()
        x_range = self.x_range()
        self.results, self.errors = [], []
        for s in self.selected_series():
            x, y, rng, color = self.series_data(s, x_range)
            if settings.method != "single" and settings.segment is None:
                span = (max(rng) - min(rng)) if rng else float(np.nanmax(x) - np.nanmin(x))
                settings.segment = span / 4
            try:
                result = spectrum(x, y, settings, self.tab._series_label(s), rng)
            except ValueError as exc:
                self.errors.append(f"{self.tab._series_label(s)}: {exc}")
                continue
            result.x = result.x + s.x_offset
            result.color = color
            self.results.append(result)
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
        t, grid = self.draw_theme(), self.grid_on()
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
        return [[r.label, p.frequency, p.period, p.amplitude, 100 * p.relative]
                for r in self.results for p in r.peaks]

    def headers(self) -> list[str]:
        funit, xunit = self.units()
        return ["Line", f"Frequency ({funit})", f"Period ({xunit})" if xunit else "Period",
                "Amplitude", "Relative (%)"]

    def format_cell(self, column: int, value) -> str:
        if column == 4:
            return f"{value:.1f}"
        return super().format_cell(column, value)

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
        if (path := self.ask_save("Save spectrum", f"{self.tab.display_name}_spectrum.csv")):
            self.spectrum_frame().to_csv(path, index=False)

    def open_spectrum(self) -> None:
        frame = self.spectrum_frame()
        if frame.empty:
            QMessageBox.information(self, "Nothing to open", "Run an analysis first.")
            return
        frame = frame.drop(columns=[c for c in frame.columns if c.startswith("Period")])
        name = f"{self.tab.display_name} spectrum"
        self.tab.openDatasetRequested.emit(Dataset(Path(f"{name}.csv"), frame), {})
