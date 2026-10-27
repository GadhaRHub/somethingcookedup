"""Background fit / echo extraction window.

Pipeline (as in the Extracting_echos notebook):
  1. remove the slow background (double exponential, or polynomials),
  2. remove the oscillations (e.g. Brillouin) with windowed sinusoid fits,
  3. what remains is the residual: the echoes.
"""

from __future__ import annotations

import re
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

from .analysis import frequency_unit
from .analysis_window import AnalysisWindow
from .fitting import CRITERIA, MAX_SPLINE_DEGREE, MODELS, RESIDUALS, FitSettings, fit_background
from .help_widgets import add_full_row, add_row
from .loaders import Dataset
from .oscillations import OscillationSettings
from .plot_tab import ROW_INDEX


def frequency_display(tab) -> tuple[str, float]:
    """(unit shown to the user, factor from cycles-per-x-unit to that unit).
    For x in ps this is GHz (×1000), as in the notebooks."""
    funit, xunit = frequency_unit(tab.xlabel_edit.text() or tab.ax.get_xlabel())
    if funit == "THz":
        return "GHz", 1e3
    if funit == "PHz":
        return "THz", 1e3
    return funit, 1.0


def parse_bands(text: str) -> list[tuple[float, float]]:
    """'25-50, 60-90' -> [(25, 50), (60, 90)]. Raises ValueError on bad input."""
    bands = []
    for part in filter(None, (p.strip() for p in re.split(r"[;,]", text))):
        m = re.fullmatch(r"([-+]?[\d.eE+-]+?)\s*(?:-|–|to)\s*([-+]?[\d.eE+-]+)", part)
        if not m:
            raise ValueError(f"cannot read the band '{part}' (use e.g. 25-50)")
        lo, hi = sorted((float(m.group(1)), float(m.group(2))))
        bands.append((lo, hi))
    return bands


INFO = {
    "model": "How the slow background is described.\n"
             "• Double exponential a·e^(bx) + c·e^(dx): the model of minus_exp_fun_single3 in\n"
             "  diffR_functions_mine.py, used by the Extracting_echos notebooks.\n"
             "• Spline / independent pieces / single polynomial: polynomials (see below).",
    "force_decay": "Keep both exponential rates b and d ≤ 0, i.e. a decaying background.\n"
                   "Untick only if the background really rises.",
    "pieces": "Polynomial models only: the range is split into this many pieces, each with its\n"
              "own polynomial. More pieces follow the background more closely, but can start\n"
              "to follow the oscillations and echoes too.",
    "degrees": "Polynomial models only: the degrees tried for each piece.",
    "criterion": "Polynomial models only. Every degree in the range is fitted, then either:\n"
                 "• the best is kept (BIC prefers simpler curves, AIC a little less), or\n"
                 "• all are combined in a weighted average (Akaike/BIC weights, goodness of fit\n"
                 "  1/residual², or equal weights). The table shows each degree's share.",
    "blend": "Independent pieces only: neighbouring pieces overlap by this fraction of a piece\n"
             "and are blended smoothly, so there are no steps at the joins.",
    "exclude": "Leave the marked regions out of the background fit (they are shaded), so the\n"
               "background cannot bend to absorb an echo you have marked.",
    "remove_osc": "After the background, remove oscillations (e.g. Brillouin) with sinusoids\n"
                  "fitted in sliding windows. Without this step the residual is dominated by\n"
                  "the oscillations and looks like the signal itself.",
    "osc_window": "Length of each sliding window. Longer windows separate close frequencies\n"
                  "better; shorter windows follow changes along x (frequency resolution ≈ 1/length).",
    "osc_overlap": "How much neighbouring windows overlap. The overlap is cross-faded, so the\n"
                   "fitted oscillation curve is continuous. Must be shorter than the window.",
    "osc_range": "Frequencies searched for oscillations. Empty = from one cycle per window up\n"
                 "to the Nyquist frequency.",
    "osc_snr": "A spectral peak counts as an oscillation only if it is this many times higher\n"
               "than the median spectral level in the window (a robust noise estimate, since a\n"
               "CSV holds one averaged trace rather than the individual repeats).",
    "osc_rel": "…and at least this fraction of the strongest peak in the window, so small side\n"
               "bumps are not fitted.",
    "osc_max": "At most this many frequencies are fitted in each window.",
    "osc_bands": "Optional: only remove oscillations inside these bands, like the 'families' in\n"
                 "the notebook, e.g. 25-50, 60-90, 120-160. Empty = remove every peak found.",
    "residual": "Sign of the result: data − model (default) or model − data.",
}


class FitDialog(AnalysisWindow):
    window_title = "Background fit / echo extraction"
    table_title = "<b>Fit summary</b>"
    export_suffix = "background_fit"

    # ------------------------------------------------------------------ UI
    def build_groups(self):
        return [self._build_model_group(), self._build_osc_group(), self._build_output_group()]

    def on_ready(self) -> None:
        self._on_model()
        self._on_osc()

    def _build_model_group(self) -> QGroupBox:
        box = QGroupBox("1. Background")
        form = QFormLayout(box)
        self.model = QComboBox()
        self.model.addItems(MODELS)
        self.model.currentTextChanged.connect(self._on_model)
        add_row(form, "Model", self.model, INFO["model"])

        self.force_decay = QCheckBox("Force decay (rates ≤ 0)")
        self.force_decay.setChecked(True)
        self.force_decay.toggled.connect(self._schedule)
        add_full_row(form, self.force_decay, INFO["force_decay"])

        self.pieces = QSpinBox(minimum=1, maximum=500, value=8)
        self.pieces.valueChanged.connect(self._schedule)
        self.piece_length = QLabel()
        self.piece_length.setObjectName("hint")
        row = QHBoxLayout()
        row.addWidget(self.pieces)
        row.addWidget(self.piece_length, 1)
        add_row(form, "Polynomials", row, INFO["pieces"])

        self.degree_min = QSpinBox(minimum=0, maximum=15, value=1)
        self.degree_max = QSpinBox(minimum=0, maximum=15, value=5)
        for spin in (self.degree_min, self.degree_max):
            spin.valueChanged.connect(self._schedule)
        degrees = QHBoxLayout()
        degrees.addWidget(self.degree_min)
        degrees.addWidget(QLabel("to"))
        degrees.addWidget(self.degree_max)
        degrees.addStretch(1)
        add_row(form, "Degree range", degrees, INFO["degrees"])
        self.degree_note = QLabel(f"Splines use degrees up to {MAX_SPLINE_DEGREE}.")
        self.degree_note.setObjectName("hint")
        form.addRow(self.degree_note)

        self.criterion = QComboBox()
        self.criterion.addItems(CRITERIA)
        self.criterion.currentTextChanged.connect(self._schedule)
        add_row(form, "Degrees", self.criterion, INFO["criterion"])

        self.blend = QSpinBox(minimum=0, maximum=100, singleStep=5, value=20, suffix=" % of a piece")
        self.blend.valueChanged.connect(self._schedule)
        add_row(form, "Blend joins", self.blend, INFO["blend"])

        self.exclude = QCheckBox("Leave marked regions out of the fit")
        self.exclude.toggled.connect(self._schedule)
        add_full_row(form, self.exclude, INFO["exclude"])
        self._model_widgets = {"poly": [self.pieces, self.degree_min, self.degree_max,
                                        self.criterion], "exp": [self.force_decay]}
        return box

    def _build_osc_group(self) -> QGroupBox:
        box = QGroupBox("2. Oscillations")
        form = QFormLayout(box)
        self.remove_osc = QCheckBox("Remove oscillations (e.g. Brillouin)")
        self.remove_osc.setChecked(True)
        self.remove_osc.toggled.connect(self._on_osc)
        add_full_row(form, self.remove_osc, INFO["remove_osc"])

        self.osc_window = QDoubleSpinBox(minimum=0.001, maximum=1e9, decimals=3, value=50)
        self.osc_overlap = QDoubleSpinBox(minimum=0, maximum=1e9, decimals=3, value=20)
        add_row(form, "Window length", self.osc_window, INFO["osc_window"])
        add_row(form, "Overlap", self.osc_overlap, INFO["osc_overlap"])

        self.osc_fmin, self.osc_fmax = QLineEdit(), QLineEdit()
        for edit in (self.osc_fmin, self.osc_fmax):
            edit.setPlaceholderText("auto")
        frange = QHBoxLayout()
        frange.addWidget(self.osc_fmin)
        frange.addWidget(QLabel("to"))
        frange.addWidget(self.osc_fmax)
        self.osc_unit = QLabel()
        frange.addWidget(self.osc_unit)
        add_row(form, "Frequencies", frange, INFO["osc_range"])

        self.osc_snr = QDoubleSpinBox(minimum=0, maximum=1000, decimals=1, value=5.0, suffix=" × median")
        self.osc_rel = QDoubleSpinBox(minimum=0, maximum=100, decimals=0, value=30, suffix=" % of strongest")
        self.osc_max = QSpinBox(minimum=1, maximum=20, value=4)
        add_row(form, "Peak height", self.osc_snr, INFO["osc_snr"])
        add_row(form, "Relative height", self.osc_rel, INFO["osc_rel"])
        add_row(form, "Max. per window", self.osc_max, INFO["osc_max"])
        self.osc_bands = QLineEdit()
        self.osc_bands.setPlaceholderText("all peaks, or e.g. 25-50, 60-90, 120-160")
        add_row(form, "Only bands", self.osc_bands, INFO["osc_bands"])

        for spin in (self.osc_window, self.osc_overlap, self.osc_snr, self.osc_rel, self.osc_max):
            spin.valueChanged.connect(self._schedule)
        for edit in (self.osc_fmin, self.osc_fmax, self.osc_bands):
            edit.editingFinished.connect(self._schedule)
        self._osc_widgets = [self.osc_window, self.osc_overlap, self.osc_fmin, self.osc_fmax,
                             self.osc_snr, self.osc_rel, self.osc_max, self.osc_bands]

        self.residual = QComboBox()
        self.residual.addItems(RESIDUALS)
        self.residual.currentTextChanged.connect(self._schedule)
        add_row(form, "Residual", self.residual, INFO["residual"])
        return box

    def _build_output_group(self) -> QGroupBox:
        box = QGroupBox("Display && output")
        form = QFormLayout(box)
        self.show_edges = QCheckBox("Show where polynomial pieces join")
        self.show_edges.setChecked(True)
        self.show_edges.toggled.connect(lambda _: self.draw())
        form.addRow(self.show_edges)
        self.add_buttons(form, [
            ("Export figure (PNG, SVG, PDF…)", self.export, True),
            ("Copy table", self.copy_table, False),
            ("Save curves (CSV)…", self.save_curves, False),
            ("Add background, oscillations && residual to this plot", self.add_to_plot, False),
            ("Open residual as a new plot", self.open_residual, False),
            ("Analyse residual (FFT)", self.analyse_residual, True),
        ])
        return box

    def settings_widgets(self):
        return [("model", self.model), ("force_decay", self.force_decay),
                ("polynomials", self.pieces), ("degree_min", self.degree_min),
                ("degree_max", self.degree_max), ("criterion", self.criterion),
                ("blend", self.blend), ("exclude_marked", self.exclude),
                ("remove_oscillations", self.remove_osc), ("osc_window", self.osc_window),
                ("osc_overlap", self.osc_overlap), ("osc_f_min", self.osc_fmin),
                ("osc_f_max", self.osc_fmax), ("osc_snr", self.osc_snr), ("osc_rel", self.osc_rel),
                ("osc_max", self.osc_max), ("osc_bands", self.osc_bands),
                ("residual", self.residual), ("show_joins", self.show_edges)]

    def _on_model(self, *_) -> None:
        model = MODELS[self.model.currentText()]
        for w in self._model_widgets["poly"]:
            w.setEnabled(model != "exp2" and (w is not self.pieces or model != "single"))
        self.force_decay.setEnabled(model == "exp2")
        self.blend.setEnabled(model == "pieces")
        self.degree_note.setVisible(model == "spline")
        self._schedule()

    def _on_osc(self, *_) -> None:
        for w in self._osc_widgets:
            w.setEnabled(self.remove_osc.isChecked())
        unit, _ = frequency_display(self.tab)
        self.osc_unit.setText(unit)
        self._schedule()

    # ------------------------------------------------------------- analysis
    def oscillation_settings(self) -> OscillationSettings | None:
        if not self.remove_osc.isChecked():
            return None
        _, factor = frequency_display(self.tab)
        try:
            bands = [(lo / factor, hi / factor) for lo, hi in parse_bands(self.osc_bands.text())]
            self.osc_bands.setStyleSheet("")
        except ValueError as exc:
            self.errors.append(str(exc))
            self.osc_bands.setStyleSheet("border: 1px solid #d33;")
            bands = []
        f_min, f_max = self.number(self.osc_fmin), self.number(self.osc_fmax)
        return OscillationSettings(
            window=self.osc_window.value(), overlap=self.osc_overlap.value(),
            f_min=None if f_min is None else f_min / factor,
            f_max=None if f_max is None else f_max / factor,
            snr_min=self.osc_snr.value(), rel_min=self.osc_rel.value() / 100,
            max_freqs=self.osc_max.value(), bands=bands)

    def settings(self) -> FitSettings:
        return FitSettings(
            model=MODELS[self.model.currentText()], pieces=self.pieces.value(),
            degree_min=self.degree_min.value(), degree_max=self.degree_max.value(),
            criterion=CRITERIA[self.criterion.currentText()], blend=self.blend.value(),
            residual=RESIDUALS[self.residual.currentText()],
            force_decay=self.force_decay.isChecked(), oscillations=self.oscillation_settings())

    def _excluded(self, x_range) -> list[tuple[float, float]]:
        if not self.exclude.isChecked():
            return []
        regions = [tuple(sorted((m.xmin, m.xmax))) for m in self.tab.region_marks]
        # The region chosen as the analysis range is the data, not an echo to skip.
        if x_range is not None:
            regions = [r for r in regions if not np.allclose(r, sorted(x_range))]
        return regions

    def process(self, s, x_range):
        """Run the pipeline on one series; returns a FitResult (x already offset)."""
        x, y, rng, color = self.series_data(s, x_range)
        result = self.process_arrays(x, y, rng, self.tab._series_label(s), s.x_offset, x_range)
        result.color = color
        return result

    def process_arrays(self, x, y, rng, label: str = "", x_offset: float = 0.0, x_range=None):
        """The same pipeline on plain arrays (used by the echo windows).
        ``rng`` is in data coordinates; marked regions are shifted by ``x_offset``.
        Results are cached, so tuning an echo method doesn't refit the background."""
        settings = self.settings()
        settings.exclude = [(lo - x_offset, hi - x_offset) for lo, hi in self._excluded(x_range)]
        x, y = np.asarray(x, float), np.asarray(y, float)
        key = (repr(settings), label, x_offset, None if rng is None else tuple(map(float, rng)),
               x.size, hash(x.tobytes()), hash(y.tobytes()))
        cache = self.__dict__.setdefault("_cache", {})
        if key not in cache:
            result = fit_background(x, y, settings, label, rng)
            result.x = result.x + x_offset
            result.edges = result.edges + x_offset
            if len(cache) > 16:
                cache.pop(next(iter(cache)))
            cache[key] = result
        return cache[key]

    def run(self) -> None:
        x_range = self.x_range()
        self.results, self.errors = [], []
        for s in self.selected_series():
            try:
                self.results.append(self.process(s, x_range))
            except ValueError as exc:
                self.errors.append(f"{self.tab._series_label(s)}: {exc}")
        model = MODELS[self.model.currentText()]
        if self.results and model in ("spline", "pieces"):
            width = self.results[0].edges[1] - self.results[0].edges[0]
            self.piece_length.setText(f"each ≈ {width:.4g} wide")
        else:
            self.piece_length.setText("")
        self.draw()
        self.fill_table()

    # --------------------------------------------------------------- output
    def draw(self) -> None:
        fig = self.figure
        fig.clear()
        with_osc = any(r.oscillations is not None for r in self.results)
        axes = fig.subplots(3 if with_osc else 2, 1, sharex=True)
        ax_fit, ax_res = axes[0], axes[-1]
        t, grid = self.draw_theme(), self.grid_on()
        fig.set_facecolor(t.card)
        with matplotlib.rc_context(self._rc()):
            for ax in axes:
                ax.set_facecolor(t.card)
                ax.grid(grid, alpha=0.3)
            for lo, hi in self._excluded(self.x_range()):
                for ax in axes:
                    ax.axvspan(lo, hi, color=t.muted, alpha=0.15, lw=0)
            single = len(self.results) == 1
            for r in self.results:
                ax_fit.plot(r.x, r.y, color=r.color, lw=1.0, alpha=0.75, label=r.label)
                ax_fit.plot(r.x, r.fit, color=t.text, lw=1.6, ls="--",
                            label=self._fit_label(r) if single else None)
                if with_osc:
                    axes[1].plot(r.x, r.background_removed, color=r.color, lw=0.9, alpha=0.7,
                                 label="background removed" if single else r.label)
                    if r.oscillations is not None:
                        axes[1].plot(r.x, r.oscillations, color=t.text, lw=1.3, ls="--",
                                     label="fitted oscillations" if single else None)
                ax_res.plot(r.x, r.residual, color=r.color, lw=1.0, label=r.label)
            model = MODELS[self.model.currentText()]
            if (self.show_edges.isChecked() and model in ("spline", "pieces") and self.results
                    and len(self.results[0].edges) > 2):
                for edge in self.results[0].edges[1:-1]:
                    ax_fit.axvline(edge, color=t.muted, lw=0.8, ls=":")
            ax_res.axhline(0, color=t.muted, lw=0.8)
            ax_fit.set_title("1. Data and background", fontsize="medium", loc="left")
            if with_osc:
                axes[1].set_title("2. Background removed, with the fitted oscillations",
                                  fontsize="medium", loc="left")
                axes[1].axhline(0, color=t.muted, lw=0.8)
                axes[1].legend(fontsize="small", loc="upper right")
            what = "data − background − oscillations" if with_osc else "data − background"
            if self.residual.currentText() != "Data − fit":
                what = f"−({what})"
            ax_res.set_title(f"{3 if with_osc else 2}. Residual = {what}: the echoes",
                             fontsize="medium", loc="left")
            ax_res.set_xlabel(self.tab.ax.get_xlabel())
            if self.results:
                ax_fit.legend(fontsize="small", loc="upper right")
            if len(self.results) > 1:
                ax_res.legend(fontsize="small")
        self.canvas.draw_idle()
        self._update_info()

    def _fit_label(self, r) -> str:
        if r.exp_params is not None:
            a, b, c, d = r.exp_params
            return f"background {a:.3g}·e^({b:.3g}x) + {c:.3g}·e^({d:.3g}x)"
        if MODELS[self.model.currentText()] == "exp2":
            return "background (exponential fit failed: mean)"
        pieces = len(r.edges) - 1
        if r.weights:
            lo, hi = sorted((self.degree_min.value(), self.degree_max.value()))
            return f"background ({pieces} × weighted average of degrees {lo}–{hi})"
        return f"background ({pieces} × degree {r.degree_text})"

    def _update_info(self) -> None:
        if not self.results:
            self.info.setText("; ".join(self.errors) or "No line selected.")
            return
        parts = [f"{len(self.results[0].x)} points"]
        excluded = self._excluded(self.x_range())
        if excluded:
            parts.append(f"{len(excluded)} marked region(s) left out of the fit (shaded)")
        r = self.results[0]
        if r.oscillations is not None:
            unit, factor = frequency_display(self.tab)
            freqs = sorted({round(f * factor, 1) for _, fs in r.osc_windows for f in fs})
            parts.append(f"oscillations removed in {len(r.osc_windows)} windows"
                         + (f" (frequencies {', '.join(map(str, freqs[:8]))}"
                            f"{'…' if len(freqs) > 8 else ''} {unit})" if freqs else " (none found)"))
        if self.errors:
            parts.append("note: " + "; ".join(self.errors))
        self.info.setText(" · ".join(parts))

    def headers(self) -> list[str]:
        return ["Line", "Background", "Degree(s)", "RMS after background", "RMS of residual",
                "R² of background", "Points fitted"]

    def rows(self) -> list[list]:
        model = self.model.currentText().split(" (")[0].split(" a·")[0]
        return [[r.label, model, r.degree_text, float(np.std(r.background_removed)),
                 float(np.std(r.residual)), r.r2, int(r.used.sum())] for r in self.results]

    def format_cell(self, column: int, value) -> str:
        if column == 6:
            return str(value)
        if column == 5:
            return f"{value:.6f}"
        return super().format_cell(column, value)

    def curves_frame(self, include_data: bool = True, include_fit: bool = True) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame()
        x_col = self.tab.x_column if self.tab.x_column not in (None, ROW_INDEX) else "x"
        base = self.results[0].x
        frame = pd.DataFrame({x_col: base})
        for r in self.results:
            same = len(r.x) == len(base) and np.allclose(r.x, base)
            # Lines on different x grids are interpolated onto the first one.
            put = (lambda v, r=r: v) if same else (lambda v, r=r: np.interp(base, r.x, v))
            if include_data:
                frame[f"{r.label} data"] = put(r.y)
            if include_fit:
                frame[f"{r.label} background"] = put(r.fit)
                if r.oscillations is not None:
                    frame[f"{r.label} oscillations"] = put(r.oscillations)
            frame[f"{r.label} residual"] = put(r.residual)
        return frame

    def save_curves(self) -> None:
        if (path := self.ask_save("Save curves", f"{self.tab.display_name}_fit.csv")):
            self.curves_frame().to_csv(path, index=False)

    def _need_results(self) -> bool:
        if not self.results:
            QMessageBox.information(self, "Nothing to use", "Run a fit first.")
            return False
        return True

    def add_to_plot(self) -> None:
        if not self._need_results():
            return
        frame = self.curves_frame(include_data=False)
        self.tab.add_dataset(Dataset(Path(f"{self.tab.display_name} fit.csv"), frame))
        self.refresh_choices()
        self.info.setText("Background, oscillations and residual added to the plot.")

    def _residual_dataset(self) -> Dataset:
        frame = self.curves_frame(include_data=False, include_fit=False)
        return Dataset(Path(f"{self.tab.display_name} residual.csv"), frame)

    def open_residual(self) -> None:
        if self._need_results():
            self.tab.openDatasetRequested.emit(self._residual_dataset(), {})

    def analyse_residual(self) -> None:
        if self._need_results():
            self.tab.openDatasetRequested.emit(self._residual_dataset(), {"fft": True})
