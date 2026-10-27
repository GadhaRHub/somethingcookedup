"""Background fit / echo extraction window: piecewise-polynomial fit and residual."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QSpinBox,
)

from .analysis_window import AnalysisWindow
from .fitting import CRITERIA, MAX_SPLINE_DEGREE, MODELS, RESIDUALS, FitSettings, fit_background
from .loaders import Dataset
from .plot_tab import ROW_INDEX


class FitDialog(AnalysisWindow):
    window_title = "Background fit / echo extraction"
    table_title = "<b>Fit summary</b>"
    export_suffix = "background_fit"

    # ------------------------------------------------------------------ UI
    def build_groups(self):
        return [self._build_model_group(), self._build_output_group()]

    def on_ready(self) -> None:
        self._on_model()

    def _build_model_group(self) -> QGroupBox:
        box = QGroupBox("Background model")
        form = QFormLayout(box)
        self.model = QComboBox()
        self.model.addItems(MODELS)
        self.model.setToolTip(
            "Spline: one polynomial per piece, joined smoothly (no steps or kinks).\n"
            "Independent pieces: each piece fitted on its own, blended where they meet.\n"
            "Single polynomial: one polynomial over the whole range.")
        self.model.currentTextChanged.connect(self._on_model)
        form.addRow("Model", self.model)

        self.pieces = QSpinBox(minimum=1, maximum=500, value=8)
        self.pieces.setToolTip("Number of polynomials the range is split into.\n"
                               "More pieces follow the background more closely, but can start\n"
                               "to follow the echoes too.")
        self.pieces.valueChanged.connect(self._schedule)
        self.piece_length = QLabel()
        self.piece_length.setObjectName("hint")
        row = QHBoxLayout()
        row.addWidget(self.pieces)
        row.addWidget(self.piece_length, 1)
        form.addRow("Polynomials", row)

        self.degree_min = QSpinBox(minimum=0, maximum=15, value=1)
        self.degree_max = QSpinBox(minimum=0, maximum=15, value=5)
        for spin in (self.degree_min, self.degree_max):
            spin.valueChanged.connect(self._schedule)
        degrees = QHBoxLayout()
        degrees.addWidget(self.degree_min)
        degrees.addWidget(QLabel("to"))
        degrees.addWidget(self.degree_max)
        degrees.addStretch(1)
        form.addRow("Degree range", degrees)
        self.degree_note = QLabel(f"Splines use degrees up to {MAX_SPLINE_DEGREE}.")
        self.degree_note.setObjectName("hint")
        form.addRow(self.degree_note)

        self.criterion = QComboBox()
        self.criterion.addItems(CRITERIA)
        self.criterion.setToolTip(
            "Every degree in the range is fitted. Then either:\n"
            "• the best one is kept (BIC penalises extra terms more than AIC, so it stays\n"
            "  closer to the slow background), or\n"
            "• all of them are combined in a weighted average:\n"
            "  – Akaike / BIC weights: the statistically standard model average; with many\n"
            "    points they often give nearly all the weight to one degree,\n"
            "  – goodness of fit: weight 1/residual², a gentler blend,\n"
            "  – equal weights: a plain average.\n"
            "The table shows each degree's share.")
        self.criterion.currentTextChanged.connect(self._schedule)
        form.addRow("Degrees", self.criterion)

        self.blend = QSpinBox(minimum=0, maximum=100, singleStep=5, value=20, suffix=" % of a piece")
        self.blend.setToolTip("Neighbouring pieces overlap by this much and are blended smoothly.")
        self.blend.valueChanged.connect(self._schedule)
        self.blend_label = QLabel("Blend joins")
        form.addRow(self.blend_label, self.blend)

        self.exclude = QCheckBox("Leave marked regions out of the fit")
        self.exclude.setToolTip("Mark the echoes as regions first: the background is then fitted\n"
                                "without them, so it cannot bend to follow an echo.")
        self.exclude.toggled.connect(self._schedule)
        form.addRow(self.exclude)

        self.residual = QComboBox()
        self.residual.addItems(RESIDUALS)
        self.residual.currentTextChanged.connect(self._schedule)
        form.addRow("Residual", self.residual)
        return box

    def _build_output_group(self) -> QGroupBox:
        box = QGroupBox("Display && output")
        form = QFormLayout(box)
        self.show_edges = QCheckBox("Show where the pieces join")
        self.show_edges.setChecked(True)
        self.show_edges.toggled.connect(lambda _: self.draw())
        form.addRow(self.show_edges)
        self.add_buttons(form, [
            ("Export figure (PNG, SVG, PDF…)", self.export, True),
            ("Copy table", self.copy_table, False),
            ("Save fit && residual (CSV)…", self.save_curves, False),
            ("Add fit && residual to this plot", self.add_to_plot, False),
            ("Open residual as a new plot", self.open_residual, False),
            ("Analyse residual (FFT)", self.analyse_residual, True),
        ])
        return box

    def settings_widgets(self):
        return [("model", self.model), ("polynomials", self.pieces),
                ("degree_min", self.degree_min), ("degree_max", self.degree_max),
                ("criterion", self.criterion), ("blend", self.blend),
                ("exclude_marked", self.exclude), ("residual", self.residual),
                ("show_joins", self.show_edges)]

    def _on_model(self, *_) -> None:
        model = MODELS[self.model.currentText()]
        self.pieces.setEnabled(model != "single")
        self.blend.setEnabled(model == "pieces")
        self.blend_label.setEnabled(model == "pieces")
        self.degree_note.setVisible(model == "spline")
        self._schedule()

    # ------------------------------------------------------------- analysis
    def settings(self) -> FitSettings:
        return FitSettings(
            model=MODELS[self.model.currentText()], pieces=self.pieces.value(),
            degree_min=self.degree_min.value(), degree_max=self.degree_max.value(),
            criterion=CRITERIA[self.criterion.currentText()], blend=self.blend.value(),
            residual=RESIDUALS[self.residual.currentText()])

    def _excluded(self, x_range) -> list[tuple[float, float]]:
        if not self.exclude.isChecked():
            return []
        regions = [tuple(sorted((m.xmin, m.xmax))) for m in self.tab.region_marks]
        # The region chosen as the analysis range is the data, not an echo to skip.
        if x_range is not None:
            regions = [r for r in regions if not np.allclose(r, sorted(x_range))]
        return regions

    def run(self) -> None:
        settings = self.settings()
        x_range = self.x_range()
        self.results, self.errors = [], []
        for s in self.selected_series():
            x, y, rng, color = self.series_data(s, x_range)
            settings.exclude = [(lo - s.x_offset, hi - s.x_offset) for lo, hi in self._excluded(x_range)]
            try:
                result = fit_background(x, y, settings, self.tab._series_label(s), rng)
            except ValueError as exc:
                self.errors.append(f"{self.tab._series_label(s)}: {exc}")
                continue
            result.x = result.x + s.x_offset
            result.edges = result.edges + s.x_offset
            result.color = color
            self.results.append(result)
        if self.results and settings.model != "single":
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
        ax_fit, ax_res = fig.subplots(2, 1, sharex=True)
        t, grid = self.draw_theme(), self.grid_on()
        fig.set_facecolor(t.card)
        with matplotlib.rc_context(self._rc()):
            for ax in (ax_fit, ax_res):
                ax.set_facecolor(t.card)
                ax.grid(grid, alpha=0.3)
            for lo, hi in self._excluded(self.x_range()):
                for ax in (ax_fit, ax_res):
                    ax.axvspan(lo, hi, color=t.muted, alpha=0.15, lw=0)
            for r in self.results:
                ax_fit.plot(r.x, r.y, color=r.color, lw=1.0, alpha=0.75, label=f"{r.label}")
                ax_fit.plot(r.x, r.fit, color=t.text, lw=1.6, ls="--",
                            label=self._fit_label(r) if len(self.results) == 1 else None)
                ax_res.plot(r.x, r.residual, color=r.color, lw=1.0, label=r.label)
            if self.show_edges.isChecked() and self.results and len(self.results[0].edges) > 2:
                for edge in self.results[0].edges[1:-1]:
                    for ax in (ax_fit, ax_res):
                        ax.axvline(edge, color=t.muted, lw=0.8, ls=":")
            ax_res.axhline(0, color=t.muted, lw=0.8)
            ax_fit.set_title("Data and background fit", fontsize="medium", loc="left")
            direction = "data − fit" if self.residual.currentText() == "Data − fit" else "fit − data"
            ax_res.set_title(f"Residual ({direction}): the echoes", fontsize="medium", loc="left")
            ax_res.set_xlabel(self.tab.ax.get_xlabel())
            if self.results:
                ax_fit.legend(fontsize="small")
            if len(self.results) > 1:
                ax_res.legend(fontsize="small")
        self.canvas.draw_idle()
        self._update_info()

    def _fit_label(self, r) -> str:
        pieces = len(r.edges) - 1
        if r.weights:
            lo, hi = sorted((self.degree_min.value(), self.degree_max.value()))
            return f"fit ({pieces} × weighted average of degrees {lo}–{hi})"
        return f"fit ({pieces} × degree {r.degree_text})"

    def _update_info(self) -> None:
        if not self.results:
            self.info.setText("; ".join(self.errors) or "No line selected.")
            return
        excluded = self._excluded(self.x_range())
        parts = [f"{len(self.results[0].x)} points"]
        if excluded:
            parts.append(f"{len(excluded)} marked region(s) left out of the fit (shaded)")
        if self.errors:
            parts.append("skipped: " + "; ".join(self.errors))
        self.info.setText(" · ".join(parts))

    def headers(self) -> list[str]:
        return ["Line", "Model", "Polynomials", "Degree(s)", "RMS of residual", "R²", "Points fitted"]

    def rows(self) -> list[list]:
        model = self.model.currentText().split(" (")[0]
        return [[r.label, model, len(r.edges) - 1, r.degree_text, r.rms, r.r2, int(r.used.sum())]
                for r in self.results]

    def format_cell(self, column: int, value) -> str:
        if column in (2, 6):
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
                frame[f"{r.label} fit"] = put(r.fit)
            frame[f"{r.label} residual"] = put(r.residual)
        return frame

    def save_curves(self) -> None:
        if (path := self.ask_save("Save fit and residual", f"{self.tab.display_name}_fit.csv")):
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
        self.info.setText("Fit and residual added to the plot.")

    def _residual_dataset(self) -> Dataset:
        frame = self.curves_frame(include_data=False, include_fit=False)
        return Dataset(Path(f"{self.tab.display_name} residual.csv"), frame)

    def open_residual(self) -> None:
        if self._need_results():
            self.tab.openDatasetRequested.emit(self._residual_dataset(), {})

    def analyse_residual(self) -> None:
        if self._need_results():
            self.tab.openDatasetRequested.emit(self._residual_dataset(), {"fft": True})
