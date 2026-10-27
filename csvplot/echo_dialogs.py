"""Echo extraction windows: TWITE and matched-subspace echo matching."""

from __future__ import annotations

import matplotlib
import matplotlib.patches
import numpy as np
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
)

from .analysis import prepare
from .analysis_window import AnalysisWindow
from .echoes import EchoMatchParams, run_echo_match, run_twite
from .fit_dialog import frequency_display
from .help_widgets import add_full_row, add_row
from .plot_tab import RegionMark
from .twite import TwiteParams

INPUTS = {
    "Residual (background and oscillations removed)": "residual",
    "Background removed": "background_removed",
    "Line as it is": "raw",
}
INPUT_INFO = (
    "Which signal the method works on. Pre-processing uses the settings of the Background\n"
    "fit window (Analysis tab), so you tune it in one place:\n"
    "• Residual: background and oscillations removed, i.e. mostly echoes left.\n"
    "• Background removed: only the slow background removed (oscillations still in).\n"
    "• Line as it is: no pre-processing (use if the line is already background-removed).")


def _num_list(text: str) -> list[float]:
    return [float(v) for v in text.replace(";", ",").split(",") if v.strip()]


class EchoWindow(AnalysisWindow):
    """Common parts: input choice, pre-processing, marking echoes on the plot."""

    def _build_input_group(self, default: str) -> QGroupBox:
        box = QGroupBox("Input signal")
        form = QFormLayout(box)
        self.input_kind = QComboBox()
        self.input_kind.addItems(INPUTS)
        self.input_kind.setCurrentText(default)
        self.input_kind.currentTextChanged.connect(self._schedule)
        add_row(form, "Signal", self.input_kind, INPUT_INFO)
        settings = QPushButton("Pre-processing settings…")
        settings.setProperty("secondary", True)
        settings.clicked.connect(lambda: (self.tab.open_fit(), self._schedule()))
        form.addRow(settings)
        return box

    def prepared(self, x, y, rng, label, x_offset=0.0, x_range=None):
        """(x, y) on a uniform grid after the chosen pre-processing (x in plot coordinates)."""
        kind = INPUTS[self.input_kind.currentText()]
        if kind == "raw":
            xs, ys, _, _ = prepare(x, y, rng)
            return xs + x_offset, ys
        result = self.tab.fit_pipeline().process_arrays(x, y, rng, label, x_offset, x_range)
        values = result.residual if kind == "residual" else result.background_removed
        xs, ys, _, _ = prepare(result.x, values)
        return xs, ys

    def mark_regions(self, centres, half: float, color: str) -> None:
        for c in centres:
            self.tab.region_marks.append(RegionMark(c - half, c + half, color))
        self.tab._after_marks_changed()
        self.refresh_choices()
        self.info.setText(f"{len(centres)} echo(es) marked on the plot as regions.")


# ======================================================================= TWITE
TWITE_PRESETS = {
    "Notebook: residual (params_res)": dict(
        t_ignore=20.0, quiet_window=(420.0, 560.0), k_max=None, pct=0.60, gap=10.0,
        noise_mult=2.5, tol_harmonic=20.0, template_half=15.0, search_half=20.0,
        target_tol=10.0, corr_min=0.60, lowpass_mult=3.0),
    "Notebook: background-removed signal (params_sig)": dict(
        t_ignore=45.0, quiet_window=(300.0, 400.0), k_max=None, pct=0.10, gap=10.0,
        noise_mult=2.5, tol_harmonic=20.0, template_half=15.0, search_half=20.0,
        target_tol=10.0, corr_min=0.60, lowpass_mult=3.0),
    "twite.py defaults": TwiteParams().as_dict(),
}

TWITE_INFO = {
    "t_ignore": "Ignore everything before this x (early, non-echo oscillations). The first echo\n"
                "is searched after it.",
    "quiet": "An echo-free stretch used to measure the noise floor. Amplitude candidates must\n"
             "exceed 'noise multiple' × the noise there.",
    "k_max": "Highest echo order k searched for. 'auto' = as many as the record allows.",
    "pct": "Detector A (amplitude): candidate threshold, as a fraction of the trace's own\n"
           "peak |amplitude|. Lower finds weaker echoes but more false candidates.",
    "gap": "Detector A: threshold crossings closer together than this are merged into one\n"
           "excursion (one echo).",
    "noise_mult": "Detector A: a candidate must exceed this multiple of the noise floor\n"
                  "(measured in the quiet window).",
    "tol_harmonic": "Detector A: a candidate counts as echo k if it lies within this distance of\n"
                    "k × t1 (t1 = first echo time).",
    "template_half": "Detector B (shape): half-width of the echo-1 template that later echoes are\n"
                     "compared with.",
    "search_half": "Detector B: radius around k × t1 searched for a shape match.",
    "target_tol": "Detector B: a shape match is accepted only within this distance of k × t1.\n"
                  "Also the width of the marked boxes.",
    "corr_min": "Detector B: minimum |ZNCC| (zero-normalised cross-correlation, −1…1) with the\n"
                "echo-1 template. ZNCC compares shape only, so decayed echoes still match.",
    "lowpass_mult": "Echo-1 refinement: FFT low-pass cutoff, as a multiple of 1/t1. Locates the\n"
                    "real pulse peak instead of its leading edge.",
    "n_bin": "Block-average this many samples before running TWITE (the notebook used 5,\n"
             "i.e. 1 ps bins for 0.2 ps steps). 1 = no binning.",
    "preset": "Fill in the parameters used in the notebook (for the residual or for the\n"
              "background-removed signal) or the defaults of twite.py.",
}


class TwiteDialog(EchoWindow):
    window_title = "Echo extraction — TWITE"
    table_title = "<b>Identified echo train</b>"
    export_suffix = "twite"

    def build_groups(self):
        return [self._build_input_group("Residual (background and oscillations removed)"),
                self._build_params_group(), self._build_output_group()]

    def on_ready(self) -> None:
        self._apply_preset()

    def _spin(self, value=0.0, lo=0.0, hi=1e9, decimals=2, step=1.0):
        spin = QDoubleSpinBox(minimum=lo, maximum=hi, decimals=decimals, singleStep=step, value=value)
        spin.valueChanged.connect(self._schedule)
        return spin

    def _build_params_group(self) -> QGroupBox:
        box = QGroupBox("TWITE parameters")
        form = QFormLayout(box)
        self.preset = QComboBox()
        self.preset.addItems(TWITE_PRESETS)
        self.preset.currentTextChanged.connect(self._apply_preset)
        add_row(form, "Preset", self.preset, TWITE_INFO["preset"])

        form.addRow(QLabel("<b>Search region</b>"))
        self.t_ignore = self._spin(20)
        add_row(form, "Ignore before", self.t_ignore, TWITE_INFO["t_ignore"])
        self.quiet_lo, self.quiet_hi = self._spin(420), self._spin(560)
        quiet = QHBoxLayout()
        quiet.addWidget(self.quiet_lo)
        quiet.addWidget(QLabel("to"))
        quiet.addWidget(self.quiet_hi)
        add_row(form, "Quiet window", quiet, TWITE_INFO["quiet"])
        self.k_max = QSpinBox(minimum=0, maximum=200)
        self.k_max.setSpecialValueText("auto")
        self.k_max.valueChanged.connect(self._schedule)
        add_row(form, "Max. order k", self.k_max, TWITE_INFO["k_max"])

        form.addRow(QLabel("<b>Detector A: amplitude</b>"))
        self.pct = self._spin(0.6, 0, 1, 3, 0.01)
        self.gap = self._spin(10)
        self.noise_mult = self._spin(2.5, step=0.1)
        self.tol_harmonic = self._spin(20)
        add_row(form, "Threshold (× peak)", self.pct, TWITE_INFO["pct"])
        add_row(form, "Merge gap", self.gap, TWITE_INFO["gap"])
        add_row(form, "Noise multiple", self.noise_mult, TWITE_INFO["noise_mult"])
        add_row(form, "Harmonic tolerance", self.tol_harmonic, TWITE_INFO["tol_harmonic"])

        form.addRow(QLabel("<b>Detector B: shape</b>"))
        self.template_half = self._spin(15)
        self.search_half = self._spin(20)
        self.target_tol = self._spin(10)
        self.corr_min = self._spin(0.6, 0, 1, 3, 0.05)
        add_row(form, "Template half-width", self.template_half, TWITE_INFO["template_half"])
        add_row(form, "Search radius", self.search_half, TWITE_INFO["search_half"])
        add_row(form, "Accept within", self.target_tol, TWITE_INFO["target_tol"])
        add_row(form, "Min. |correlation|", self.corr_min, TWITE_INFO["corr_min"])

        form.addRow(QLabel("<b>Echo 1 and input</b>"))
        self.lowpass_mult = self._spin(3, step=0.5)
        add_row(form, "Low-pass (× 1/t1)", self.lowpass_mult, TWITE_INFO["lowpass_mult"])
        self.n_bin = QSpinBox(minimum=1, maximum=100, value=5)
        self.n_bin.valueChanged.connect(self._schedule)
        add_row(form, "Bin samples", self.n_bin, TWITE_INFO["n_bin"])
        return box

    def _build_output_group(self) -> QGroupBox:
        box = QGroupBox("Output")
        form = QFormLayout(box)
        self.add_buttons(form, [
            ("Export figure (PNG, SVG, PDF…)", self.export, True),
            ("Copy table", self.copy_table, False),
            ("Save table (CSV)…", self.save_table, False),
            ("Mark echoes on the plot", self.mark_on_plot, True),
        ])
        return box

    def _apply_preset(self, *_) -> None:
        values = TWITE_PRESETS[self.preset.currentText()]
        widgets = self._param_widgets()
        for name, widget in widgets.items():
            widget.blockSignals(True)
        for name, value in values.items():
            if name == "quiet_window":
                self.quiet_lo.setValue(value[0])
                self.quiet_hi.setValue(value[1])
            elif name == "k_max":
                self.k_max.setValue(value or 0)
            elif name in widgets:
                widgets[name].setValue(value)
        for widget in widgets.values():
            widget.blockSignals(False)
        self._schedule()

    def _param_widgets(self) -> dict:
        return {"t_ignore": self.t_ignore, "quiet_lo": self.quiet_lo, "quiet_hi": self.quiet_hi,
                "k_max": self.k_max, "pct": self.pct, "gap": self.gap,
                "noise_mult": self.noise_mult, "tol_harmonic": self.tol_harmonic,
                "template_half": self.template_half, "search_half": self.search_half,
                "target_tol": self.target_tol, "corr_min": self.corr_min,
                "lowpass_mult": self.lowpass_mult}

    def settings_widgets(self):
        return [("input", self.input_kind), ("preset", self.preset),
                *self._param_widgets().items(), ("bin", self.n_bin)]

    def params(self) -> TwiteParams:
        return TwiteParams(
            t_ignore=self.t_ignore.value(), quiet_window=(self.quiet_lo.value(), self.quiet_hi.value()),
            k_max=self.k_max.value() or None, pct=self.pct.value(), gap=self.gap.value(),
            noise_mult=self.noise_mult.value(), tol_harmonic=self.tol_harmonic.value(),
            template_half=self.template_half.value(), search_half=self.search_half.value(),
            target_tol=self.target_tol.value(), corr_min=self.corr_min.value(),
            lowpass_mult=self.lowpass_mult.value())

    def run(self) -> None:
        params = self.params()
        x_range = self.x_range()
        self.results, self.errors = [], []
        for s in self.selected_series():
            x, y, rng, color = self.series_data(s, x_range)
            label = self.tab._series_label(s)
            try:
                xs, ys = self.prepared(x, y, rng, label, s.x_offset, x_range)
                result = run_twite(xs, ys, params, self.n_bin.value())
            except ValueError as exc:
                self.errors.append(f"{label}: {exc}")
                continue
            result.update(label=label, color=color)
            self.results.append(result)
        self.draw()
        self.fill_table()

    def draw(self) -> None:
        fig = self.figure
        fig.clear()
        ax = fig.subplots()
        t, grid = self.draw_theme(), self.grid_on()
        fig.set_facecolor(t.card)
        p = self.params()
        with matplotlib.rc_context(self._rc()):
            ax.set_facecolor(t.card)
            ax.grid(grid, alpha=0.3)
            ax.axvspan(*p.quiet_window, color=t.muted, alpha=0.12, lw=0, label="quiet window")
            offset = 0.0
            for n, r in enumerate(self.results):
                ty, yy = r["t"], r["y"] - offset
                ax.axvspan(ty[0], p.t_ignore, color=t.muted, alpha=0.06, lw=0,
                           label="ignored" if n == 0 else None)
                ax.plot(ty, yy, color=r["color"], lw=0.9, label=r["label"])
                ax.axhline(-offset, color=t.muted, lw=0.6, ls="--")
                span = float(np.ptp(r["y"])) or 1.0
                box_h, half = 0.9 * span, p.target_tol / 2
                if r["t1"]:
                    k_max = int(ty[-1] // r["t1"])
                    for k in range(1, k_max + 1):
                        ax.axvline(k * r["t1"], color=t.border, lw=0.7, ls=":")
                for k, tc in sorted(r["echoes"].items()):
                    ax.add_patch(matplotlib.patches.Rectangle(
                        (tc - half, -offset - box_h / 2), 2 * half, box_h, facecolor=t.soft,
                        edgecolor=t.border, alpha=0.5, zorder=2))
                    ax.plot(tc, -offset, "o", color=t.accent, ms=7, mec=t.text, mew=0.5, zorder=5)
                    ax.annotate(str(k), (tc, -offset + box_h * 0.46), ha="center",
                                fontsize="small", color=t.text)
                offset += 1.3 * span
            ax.set_xlabel(self.tab.ax.get_xlabel())
            ax.set_title("TWITE: identified echo train (dotted: k × t1)", fontsize="medium", loc="left")
            if self.results:
                ax.legend(fontsize="small", loc="upper right")
        self.canvas.draw_idle()
        self._update_info()

    def _update_info(self) -> None:
        parts = []
        for r in self.results:
            if r["t1"] is None:
                parts.append(f"{r['label']}: no echoes found")
                continue
            n = len(r["echoes"])
            text = f"{r['label']}: t1 = {r['t1']:.4g}, {n} echo(es), noise σ = {r['noise_std']:.3g}"
            if n <= 1:
                text += " (only echo 1: check it — any trace yields a t1, a real train gives k ≥ 2)"
            parts.append(text)
        if self.errors:
            parts.append("skipped: " + "; ".join(self.errors))
        self.info.setText(" · ".join(parts) or "No line selected.")

    def headers(self) -> list[str]:
        return ["Line", "k", "Time", "k × t1", "Offset", "Found by", "|ZNCC|"]

    def rows(self) -> list[list]:
        rows = []
        for r in self.results:
            for k, tc in sorted(r["echoes"].items()):
                target = k * r["t1"]
                corr = r["corr"].get(k)
                rows.append([r["label"], str(k), tc, target, tc - target, r["methods"][k],
                             "" if corr is None else f"{abs(corr):.2f}"])
        return rows

    def mark_on_plot(self) -> None:
        centres = [tc for r in self.results for tc in r["echoes"].values()]
        self.mark_regions(centres, self.target_tol.value() / 2, self.tab.mark_color.color)


# ================================================================ echo match
EM_INFO = {
    "noise": "Where the noise model comes from:\n"
             "• Quiet window: noise with the spectrum of an echo-free stretch of this trace\n"
             "  (for a single averaged trace, e.g. one CSV column).\n"
             "• Repeats: every line shown on the plot is one repeat. They are averaged, and\n"
             "  random half-splits of them give pure-noise traces with the same noise as the\n"
             "  mean (as in the notebook). Needed for the timing bootstrap.",
    "quiet": "Quiet window: an echo-free stretch whose noise spectrum is used.",
    "n_sur": "Number of noise-only surrogate traces. More gives a more reliable threshold\n"
             "but is slower.",
    "floor": "Whitening: noise power floor (fraction of the maximum) so the filter does not\n"
             "blow up where the noise spectrum is tiny.",
    "seeds": "Times of a few clear echoes, used to learn the echo shape from the data\n"
             "(≥ 4 recommended). Empty: taken from TWITE (with the TWITE window's settings).",
    "half": "Half-width of the echo template (window length = 2 × this).",
    "k": "Number of echo shapes kept (SVD of the aligned seed snippets). 1 is usually enough;\n"
         "2 allows some change of shape between echoes.",
    "max_shift": "Seed snippets are aligned to the first one by shifting up to this much.",
    "poly": "Order of the local polynomial baseline removed in every window.",
    "freqs": "Oscillation (e.g. Brillouin) frequencies removed locally in every window.\n"
             "Empty: the strongest ones are estimated from the trace.",
    "n_auto": "How many frequencies to estimate when the list above is empty (0 = none).",
    "first_order": "Also remove first-order terms (τ·cos, τ·sin) for a slightly wrong frequency\n"
                   "or changing amplitude. Suppresses leakage but removes much echo energy\n"
                   "(off by default; better to get the frequencies right).",
    "alpha": "Family-wise false-alarm rate: the threshold is set so that pure noise produces\n"
             "a false detection anywhere in the range with this probability.",
    "max_iter": "Maximum number of detect-and-subtract iterations (maximum number of echoes).",
    "range": "Only look for echoes within this x range (empty = everywhere).",
    "min_sep": "After an echo is accepted, no new one within ± this distance (imperfect\n"
               "subtraction of strong echoes would otherwise create fake neighbours).",
    "n_boot": "Timing uncertainty: re-detect on this many bootstrap resamples of the repeats\n"
              "(repeats noise only; 0 = off). Slow for large values.",
    "tol": "Bootstrap: a re-detected echo matches the reference within this distance.",
}


class EchoMatchDialog(EchoWindow):
    window_title = "Echo extraction — matched-subspace echo matching"
    table_title = "<b>Detected echoes</b>"
    export_suffix = "echo_match"

    def build_groups(self):
        return [self._build_input_group("Background removed"), self._build_noise_group(),
                self._build_template_group(), self._build_detect_group(), self._build_output_group()]

    def on_ready(self) -> None:
        self._on_noise()

    def _spin(self, value, lo=0.0, hi=1e9, decimals=2, step=1.0):
        spin = QDoubleSpinBox(minimum=lo, maximum=hi, decimals=decimals, singleStep=step, value=value)
        spin.valueChanged.connect(self._schedule)
        return spin

    def _int(self, value, lo, hi):
        spin = QSpinBox(minimum=lo, maximum=hi, value=value)
        spin.valueChanged.connect(self._schedule)
        return spin

    def _edit(self, placeholder):
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        edit.editingFinished.connect(self._schedule)
        return edit

    def _build_noise_group(self) -> QGroupBox:
        box = QGroupBox("Noise model")
        form = QFormLayout(box)
        self.noise = QComboBox()
        self.noise.addItems(["Quiet window of the trace", "Repeats: all lines shown on the plot"])
        self.noise.currentTextChanged.connect(self._on_noise)
        add_row(form, "Noise from", self.noise, EM_INFO["noise"])
        self.quiet_lo, self.quiet_hi = self._spin(420), self._spin(560)
        quiet = QHBoxLayout()
        quiet.addWidget(self.quiet_lo)
        quiet.addWidget(QLabel("to"))
        quiet.addWidget(self.quiet_hi)
        add_row(form, "Quiet window", quiet, EM_INFO["quiet"])
        self.n_sur = self._int(200, 20, 5000)
        add_row(form, "Surrogates", self.n_sur, EM_INFO["n_sur"])
        self.floor = self._spin(1e-3, 0, 1, 5, 1e-4)
        add_row(form, "Whitening floor", self.floor, EM_INFO["floor"])
        return box

    def _build_template_group(self) -> QGroupBox:
        box = QGroupBox("Echo shape && nuisance")
        form = QFormLayout(box)
        self.seeds = self._edit("from TWITE, or e.g. 55.6, 111, 163")
        add_row(form, "Seed echo times", self.seeds, EM_INFO["seeds"])
        self.half = self._spin(7.5, 0.01, step=0.5)
        add_row(form, "Template half-width", self.half, EM_INFO["half"])
        self.k = self._int(1, 1, 5)
        add_row(form, "Echo shapes", self.k, EM_INFO["k"])
        self.max_shift = self._spin(3.0, step=0.5)
        add_row(form, "Alignment shift", self.max_shift, EM_INFO["max_shift"])
        self.poly = self._int(2, 0, 6)
        add_row(form, "Baseline order", self.poly, EM_INFO["poly"])
        self.freqs = self._edit("estimate")
        self.freq_unit = QLabel()
        freqs = QHBoxLayout()
        freqs.addWidget(self.freqs, 1)
        freqs.addWidget(self.freq_unit)
        add_row(form, "Oscillation freqs", freqs, EM_INFO["freqs"])
        self.n_auto = self._int(2, 0, 6)
        add_row(form, "Estimate how many", self.n_auto, EM_INFO["n_auto"])
        self.first_order = QCheckBox("First-order frequency terms")
        self.first_order.toggled.connect(self._schedule)
        add_full_row(form, self.first_order, EM_INFO["first_order"])
        return box

    def _build_detect_group(self) -> QGroupBox:
        box = QGroupBox("Detection")
        form = QFormLayout(box)
        self.alpha = self._spin(0.01, 0.0001, 0.5, 4, 0.005)
        add_row(form, "False-alarm rate α", self.alpha, EM_INFO["alpha"])
        self.max_iter = self._int(60, 1, 500)
        add_row(form, "Max. echoes", self.max_iter, EM_INFO["max_iter"])
        self.t_min, self.t_max = self._edit("start"), self._edit("end")
        rng = QHBoxLayout()
        rng.addWidget(self.t_min)
        rng.addWidget(QLabel("to"))
        rng.addWidget(self.t_max)
        add_row(form, "Search range", rng, EM_INFO["range"])
        self.min_sep = self._spin(4.0, step=0.5)
        add_row(form, "Min. separation", self.min_sep, EM_INFO["min_sep"])
        self.n_bin = self._int(0, 0, 100)
        self.n_bin.setSpecialValueText("auto (≈3000 points)")
        add_row(form, "Bin samples", self.n_bin,
                "Block-average this many samples before detection. Auto bins long traces down\n"
                "to about 3000 points (the size of the notebook's data), which keeps the\n"
                "detection fast. 1 = no binning.")
        self.n_boot = self._int(0, 0, 2000)
        add_row(form, "Bootstrap resamples", self.n_boot, EM_INFO["n_boot"])
        self.tol = self._spin(2.0, step=0.5)
        add_row(form, "Bootstrap tolerance", self.tol, EM_INFO["tol"])
        return box

    def _build_output_group(self) -> QGroupBox:
        box = QGroupBox("Output")
        form = QFormLayout(box)
        self.add_buttons(form, [
            ("Export figure (PNG, SVG, PDF…)", self.export, True),
            ("Copy table", self.copy_table, False),
            ("Save table (CSV)…", self.save_table, False),
            ("Mark echoes on the plot", self.mark_on_plot, True),
        ])
        return box

    def settings_widgets(self):
        return [("input", self.input_kind), ("noise", self.noise), ("quiet_lo", self.quiet_lo),
                ("quiet_hi", self.quiet_hi), ("surrogates", self.n_sur), ("floor", self.floor),
                ("seeds", self.seeds), ("half", self.half), ("shapes", self.k),
                ("max_shift", self.max_shift), ("poly", self.poly), ("freqs", self.freqs),
                ("n_auto", self.n_auto), ("first_order", self.first_order), ("alpha", self.alpha),
                ("max_iter", self.max_iter), ("t_min", self.t_min), ("t_max", self.t_max),
                ("min_sep", self.min_sep), ("bin", self.n_bin), ("bootstrap", self.n_boot),
                ("tolerance", self.tol)]

    def _on_noise(self, *_) -> None:
        repeats = self.noise.currentIndex() == 1
        self.quiet_lo.setEnabled(not repeats)
        self.quiet_hi.setEnabled(not repeats)
        self.n_boot.setEnabled(repeats)
        self.tol.setEnabled(repeats)
        self.freq_unit.setText(frequency_display(self.tab)[0])
        self._schedule()

    def params(self) -> EchoMatchParams:
        _, factor = frequency_display(self.tab)
        try:
            seeds, freqs = _num_list(self.seeds.text()), _num_list(self.freqs.text())
        except ValueError:
            self.errors.append("could not read the seed times or frequencies (use commas)")
            seeds, freqs = [], []
        return EchoMatchParams(
            noise="repeats" if self.noise.currentIndex() == 1 else "quiet",
            quiet=(self.quiet_lo.value(), self.quiet_hi.value()), n_sur=self.n_sur.value(),
            floor=self.floor.value(), seeds=seeds, half=self.half.value(), k=self.k.value(),
            max_shift=self.max_shift.value(), poly=self.poly.value(),
            freqs=[f / factor for f in freqs], n_auto_freqs=self.n_auto.value(),
            first_order=self.first_order.isChecked(), alpha=self.alpha.value(),
            max_iter=self.max_iter.value(), t_min=self.number(self.t_min),
            t_max=self.number(self.t_max), min_sep=self.min_sep.value(),
            n_boot=self.n_boot.value(), tol=self.tol.value())

    def _twite_params(self) -> TwiteParams:
        twite = getattr(self.tab, "twite_dialog", None)
        return twite.params() if twite is not None else TwiteParams(
            **TWITE_PRESETS["Notebook: background-removed signal (params_sig)"])

    def run(self) -> None:
        self.results, self.errors = [], []
        params = self.params()
        x_range = self.x_range()
        if params.noise == "repeats":
            jobs = [self._repeats_job(x_range)]
        else:
            jobs = [self._series_job(s, x_range) for s in self.selected_series()]
        for job in jobs:
            if job is None:
                continue
            label, color, t, y, reps, background = job
            n_bin = self.n_bin.value() or max(1, round(len(t) / 3000))
            if n_bin > 1:
                from .echoes import block_bin
                tb, y = block_bin(t, y, n_bin)
                if reps is not None:
                    reps = np.column_stack([block_bin(t, reps[:, j], n_bin)[1]
                                            for j in range(reps.shape[1])])
                    background = block_bin(t, background, n_bin)[1]
                t = tb
            try:
                result = run_echo_match(t, y, params, reps=reps, background=background,
                                        twite_params=self._twite_params())
            except (ValueError, np.linalg.LinAlgError, IndexError) as exc:
                self.errors.append(f"{label}: {exc}")
                continue
            result.update(label=label, color=color, t=t, y=y)
            self.results.append(result)
        self.draw()
        self.fill_table()

    def _series_job(self, s, x_range):
        x, y, rng, color = self.series_data(s, x_range)
        label = self.tab._series_label(s)
        try:
            t, yy = self.prepared(x, y, rng, label, s.x_offset, x_range)
        except ValueError as exc:
            self.errors.append(f"{label}: {exc}")
            return None
        return label, color, t, yy, None, None

    def _repeats_job(self, x_range):
        lines = [s for s in self.tab._plottable() if s.visible]
        datasets = {id(s.dataset) for s in lines}
        if len(lines) < 4 or len(datasets) != 1:
            self.errors.append("noise from repeats needs ≥ 4 lines shown, all from the same file")
            return None
        x, _, rng, color = self.series_data(lines[0], x_range)
        reps = np.column_stack([s.dataset.frame[s.column].to_numpy(dtype=float) for s in lines])
        ok = np.isfinite(x) & np.isfinite(reps).all(axis=1)
        x, reps = x[ok], reps[ok]
        if rng is not None:
            keep = (x >= min(rng)) & (x <= max(rng))
            x, reps = x[keep], reps[keep]
        order = np.argsort(x)
        x, reps = x[order], reps[order]
        mean = reps.mean(axis=1)
        label = f"mean of {len(lines)} repeats"
        t, y = self.prepared(x, mean, None, label, lines[0].x_offset, x_range)
        if len(t) != len(x) or not np.allclose(t, x + lines[0].x_offset):
            self.errors.append("repeats must be evenly spaced in x")
            return None
        background = mean - y
        return label, color, t, y, reps, background

    def draw(self) -> None:
        fig = self.figure
        fig.clear()
        axes = fig.subplots(3, 1, height_ratios=[3, 2, 1.4])
        ax_sig, ax_z, ax_tpl = axes
        ax_z.sharex(ax_sig)
        t, grid = self.draw_theme(), self.grid_on()
        fig.set_facecolor(t.card)
        with matplotlib.rc_context(self._rc()):
            for ax in axes:
                ax.set_facecolor(t.card)
                ax.grid(grid, alpha=0.3)
            for r in self.results[:1]:   # one trace at a time keeps the plot readable
                ax_sig.plot(r["t"], r["y"], color=r["color"], lw=0.9, label=r["label"])
                for s in r["seeds"]:
                    ax_sig.axvline(s, color=t.border, lw=0.8, ls=":")
                yr = float(np.ptp(r["y"])) or 1.0
                for e in r["events"]:
                    up = e["sign"] >= 0
                    ax_sig.plot(e["t"], (0.45 if up else -0.45) * yr, "^" if up else "v",
                                color=t.accent, ms=8, mfc="none" if e["near_strong"] else t.accent)
                ax_z.plot(r["t"], r["z"], color=r["color"], lw=0.9)
                ax_z.axhline(r["threshold"], color=t.accent, lw=1, ls="--",
                             label=f"threshold {r['threshold']:.2f}")
                for e in r["events"]:
                    ax_z.plot(e["t"], e["z"], "o", color=t.accent, ms=5,
                              mfc="none" if e["near_strong"] else t.accent)
                D = r["templates"]
                tau = (np.arange(D.shape[0]) - D.shape[0] // 2) * (r["t"][1] - r["t"][0])
                for j in range(D.shape[1]):
                    ax_tpl.plot(tau, D[:, j], lw=1.2, label=f"shape {j + 1}")
            ax_sig.set_title("Input signal: ▲▼ detected echoes (sign), hollow = next to a much "
                             "stronger one, dotted = seeds", fontsize="medium", loc="left")
            ax_z.set_title("Detection statistic z (whitened, before subtraction)",
                           fontsize="medium", loc="left")
            ax_z.set_xlabel(self.tab.ax.get_xlabel())
            ax_tpl.set_title("Learned echo shape(s) (whitened)", fontsize="medium", loc="left")
            ax_tpl.set_xlabel("offset from echo centre")
            if self.results:
                ax_sig.legend(fontsize="small", loc="upper right")
                ax_z.legend(fontsize="small", loc="upper right")
        self.canvas.draw_idle()
        self._update_info()

    def _update_info(self) -> None:
        parts = []
        unit, factor = frequency_display(self.tab)
        for r in self.results:
            freqs = ", ".join(f"{f * factor:.4g}" for f in r["freqs"]) or "none"
            parts.append(f"{r['label']}: {len(r['events'])} echo(es); oscillations removed at "
                         f"{freqs} {unit}")
            parts += r["warnings"]
        if len(self.results) > 1:
            parts.append(f"plot shows the first line; the table lists all {len(self.results)}")
        if self.errors:
            parts.append("skipped: " + "; ".join(self.errors))
        self.info.setText(" · ".join(parts) or "No line selected.")

    def headers(self) -> list[str]:
        return ["Line", "Time", "z", "Sign", "Removal", "Next to stronger", "Time σ (boot)",
                "Detection rate"]

    def rows(self) -> list[list]:
        rows = []
        for r in self.results:
            boot = r["boot"] or [(None, None)] * len(r["events"])
            for e, (sd, rate) in zip(r["events"], boot):
                rows.append([r["label"], float(e["t"]), float(e["z"]), "+" if e["sign"] >= 0 else "−",
                             e.get("removal", ""), "yes" if e["near_strong"] else "",
                             "" if sd is None or np.isnan(sd) else f"{sd:.3g}",
                             "" if rate is None else f"{100 * rate:.0f}%"])
        return rows

    def mark_on_plot(self) -> None:
        centres = [float(e["t"]) for r in self.results for e in r["events"] if not e["near_strong"]]
        self.mark_regions(centres, self.half.value(), self.tab.mark_color.color)
