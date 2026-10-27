"""High-quality figure export: format, resolution, size and background."""

from __future__ import annotations

from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from typing import Callable, ContextManager

import matplotlib
from matplotlib.figure import Figure
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QSpinBox,
    QVBoxLayout,
)

from . import theme

# name -> (extension, is_vector)
FORMATS = {
    "PNG image": ("png", False),
    "SVG vector": ("svg", True),
    "PDF document": ("pdf", True),
    "EPS vector": ("eps", True),
    "TIFF image": ("tiff", False),
    "JPEG image": ("jpg", False),
}
BACKGROUNDS = ["White", "Theme colour", "Transparent"]
CM_PER_INCH = 2.54

CUSTOM = "Custom (as on screen)"
# name -> (width cm, height cm, dpi, font size pt)
PRESETS = {
    "Journal, single column (8.5 cm)": (8.5, 6.5, 600, 8),
    "Journal, 1.5 column (12 cm)": (12.0, 8.0, 600, 9),
    "Journal, double column (17.8 cm)": (17.8, 9.5, 600, 9),
    "Slide 16:9": (25.4, 14.29, 200, 18),
    "Slide 4:3": (25.4, 19.05, 200, 18),
    "Poster panel (30 × 20 cm)": (30.0, 20.0, 300, 24),
}

Restyle = Callable[[theme.Theme, "float | None"], ContextManager]


PREVIEW_W, PREVIEW_H = 460, 360


class ExportDialog(QDialog):
    def __init__(self, figure: Figure, default_name: str, parent=None,
                 restyle: Restyle | None = None):
        super().__init__(parent)
        self.setWindowTitle("Export figure")
        self.figure = figure
        self.restyle = restyle
        self.default_name = default_name
        self.settings = QSettings("csvplot", "csvplot")
        self._applying_preset = False

        self.preset_combo = QComboBox()
        self.preset_combo.addItems([CUSTOM, *PRESETS])
        self.preset_combo.setToolTip("Optional: set size, resolution and font size in one click.")
        self.preset_combo.currentTextChanged.connect(self._on_figure_preset)

        self.font_size = QDoubleSpinBox(minimum=0, maximum=60, decimals=1, singleStep=1, suffix=" pt")
        self.font_size.setSpecialValueText("As on screen")
        self.font_size.setToolTip("Font size for the exported figure only.")

        self.format_combo = QComboBox()
        self.format_combo.addItems(FORMATS)
        self.dpi = QSpinBox(minimum=50, maximum=1200, singleStep=50, suffix=" dpi")
        self.dpi_presets = QComboBox()
        self.dpi_presets.addItems(["Screen (150)", "Print (300)", "Journal (600)", "Max (1200)", "Custom"])
        self.dpi_presets.currentTextChanged.connect(self._on_preset)
        self.dpi.valueChanged.connect(self._sync_preset)

        self.units = QComboBox()
        self.units.addItems(["cm", "in"])
        self.units.currentTextChanged.connect(self._on_units)
        self.width = QDoubleSpinBox(minimum=1, maximum=200, decimals=2, singleStep=0.5)
        self.height = QDoubleSpinBox(minimum=1, maximum=200, decimals=2, singleStep=0.5)
        self.width.valueChanged.connect(self._update_info)
        self.height.valueChanged.connect(self._update_info)

        self.background = QComboBox()
        self.background.addItems(BACKGROUNDS)
        self.info = QLabel()
        self.info.setStyleSheet(f"color: {theme.current.muted};")

        dpi_row = QHBoxLayout()
        dpi_row.addWidget(self.dpi_presets, 1)
        dpi_row.addWidget(self.dpi)
        size_row = QHBoxLayout()
        size_row.addWidget(self.width)
        size_row.addWidget(QLabel("×"))
        size_row.addWidget(self.height)
        size_row.addWidget(self.units)

        form = QFormLayout()
        form.addRow("Preset", self.preset_combo)
        form.addRow("Format", self.format_combo)
        form.addRow("Resolution", dpi_row)
        form.addRow("Size", size_row)
        form.addRow("Font size", self.font_size)
        form.addRow("Background", self.background)
        form.addRow(self.info)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Cancel).setProperty("secondary", True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        # Live preview, drawn exactly as the file will be (size, fonts, theme,
        # background), at a lower resolution.
        self.preview = QLabel("Rendering preview…")
        self.preview.setObjectName("export_preview")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setFixedSize(PREVIEW_W + 24, PREVIEW_H + 24)
        self.preview_caption = QLabel()
        self.preview_caption.setObjectName("hint")
        self.preview_caption.setAlignment(Qt.AlignCenter)
        self._preview_timer = QTimer(self, singleShot=True, interval=250)
        self._preview_timer.timeout.connect(self.update_preview)

        preview_col = QVBoxLayout()
        preview_col.addWidget(self.preview)
        preview_col.addWidget(self.preview_caption)
        settings_col = QVBoxLayout()
        settings_col.addLayout(form)
        settings_col.addStretch(1)
        settings_col.addWidget(buttons)
        layout = QHBoxLayout(self)
        layout.addLayout(settings_col)
        layout.addSpacing(12)
        layout.addLayout(preview_col)

        self.format_combo.currentTextChanged.connect(self._update_info)
        self._load_settings()
        # Editing any preset-controlled value by hand switches back to Custom.
        for spin in (self.width, self.height, self.dpi, self.font_size):
            spin.valueChanged.connect(self._preset_edited)
        for spin in (self.width, self.height, self.font_size):
            spin.valueChanged.connect(self._schedule_preview)
        for combo in (self.background, self.units):
            combo.currentTextChanged.connect(self._schedule_preview)
        self._schedule_preview()

    def _schedule_preview(self, *_) -> None:
        self._preview_timer.start()

    def update_preview(self) -> None:
        w_in, h_in = self.size_inches()
        # Render small but with the real size in inches, so text and lines
        # keep their true proportions.
        dpi = max(20.0, min(PREVIEW_W / w_in, PREVIEW_H / h_in))
        buffer = BytesIO()
        try:
            save_figure(self.figure, buffer, (w_in, h_in), dpi, self.background.currentText(),
                        self.restyle, self.export_font_size, fmt="png")
        except Exception as exc:  # noqa: BLE001 - show the problem instead of a preview
            self.preview.setText(f"Preview failed:\n{exc}")
            return
        pixmap = QPixmap()
        pixmap.loadFromData(buffer.getvalue(), "PNG")
        self.preview.setPixmap(pixmap)
        unit = self.units.currentText()
        self.preview_caption.setText(
            f"Preview: {self.width.value():.1f} × {self.height.value():.1f} {unit}, "
            f"{self.font_size.value() or 'screen'}{' pt' if self.font_size.value() else ''} font")

    def _on_figure_preset(self, name: str) -> None:
        if name not in PRESETS:
            return
        w_cm, h_cm, dpi, font = PRESETS[name]
        self._applying_preset = True
        self.units.blockSignals(True)
        self.units.setCurrentText("cm")
        self.units.blockSignals(False)
        self.width.setValue(w_cm)
        self.height.setValue(h_cm)
        self.dpi.setValue(dpi)
        self.font_size.setValue(font)
        self._applying_preset = False
        self._update_info()

    def _preset_edited(self, *_) -> None:
        if not self._applying_preset and self.preset_combo.currentText() != CUSTOM:
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentText(CUSTOM)
            self.preset_combo.blockSignals(False)

    # -------------------------------------------------------------- state
    def _load_settings(self) -> None:
        s = self.settings
        self.format_combo.setCurrentText(s.value("export/format", "PNG image"))
        self.units.blockSignals(True)
        self.units.setCurrentText(s.value("export/units", "cm"))
        self.units.blockSignals(False)
        self.background.setCurrentText(s.value("export/background", "White"))
        self.dpi.setValue(int(s.value("export/dpi", 300)))
        # Default size: what is on screen now.
        w_in, h_in = self.figure.get_size_inches()
        factor = self._factor()
        self.width.setValue(w_in * factor)
        self.height.setValue(h_in * factor)
        self._sync_preset()
        self._update_info()

    def _save_settings(self) -> None:
        s = self.settings
        s.setValue("export/format", self.format_combo.currentText())
        s.setValue("export/units", self.units.currentText())
        s.setValue("export/background", self.background.currentText())
        s.setValue("export/dpi", self.dpi.value())

    def _factor(self) -> float:
        return CM_PER_INCH if self.units.currentText() == "cm" else 1.0

    def _on_units(self, unit: str) -> None:
        scale = CM_PER_INCH if unit == "cm" else 1 / CM_PER_INCH
        self.width.setValue(self.width.value() * scale)
        self.height.setValue(self.height.value() * scale)

    def _on_preset(self, text: str) -> None:
        if "(" in text:
            self.dpi.setValue(int(text.split("(")[1].rstrip(")")))

    def _sync_preset(self) -> None:
        match = next((i for i in range(self.dpi_presets.count())
                      if f"({self.dpi.value()})" in self.dpi_presets.itemText(i)), None)
        self.dpi_presets.blockSignals(True)
        self.dpi_presets.setCurrentIndex(match if match is not None else self.dpi_presets.count() - 1)
        self.dpi_presets.blockSignals(False)
        self._update_info()

    def _update_info(self) -> None:
        ext, vector = FORMATS[self.format_combo.currentText()]
        w_in, h_in = self.size_inches()
        px = f"{round(w_in * self.dpi.value())} × {round(h_in * self.dpi.value())} px"
        if vector:
            self.info.setText(f"Vector format: sharp at any zoom ({px} for embedded images).")
        else:
            self.info.setText(f"Output: {px}")
        if ext in ("jpg",) and self.background.currentText() == "Transparent":
            self.background.setCurrentText("White")

    # ------------------------------------------------------------- output
    def size_inches(self) -> tuple[float, float]:
        f = self._factor()
        return self.width.value() / f, self.height.value() / f

    @property
    def export_font_size(self) -> float | None:
        return self.font_size.value() or None

    @property
    def extension(self) -> str:
        return FORMATS[self.format_combo.currentText()][0]

    def ask_path(self) -> str | None:
        name, ext = self.format_combo.currentText(), self.extension
        folder = self.settings.value("export/dir", str(Path.home()))
        start = str(Path(folder) / f"{self.default_name}.{ext}")
        path, _ = QFileDialog.getSaveFileName(self, "Export figure", start, f"{name} (*.{ext})")
        if not path:
            return None
        accepted = {ext, {"jpg": "jpeg", "tiff": "tif"}.get(ext, ext)}
        if Path(path).suffix.lower().lstrip(".") not in accepted:
            path += f".{ext}"
        self.settings.setValue("export/dir", str(Path(path).parent))
        return path


def save_figure(figure: Figure, path: "str | BytesIO", size_inches, dpi: float, background: str,
                restyle: Restyle | None = None, font_size: float | None = None,
                fmt: str | None = None) -> None:
    """Write ``figure`` to ``path`` at the given size and resolution.

    ``restyle(theme, font_size)`` is a context manager that temporarily redraws
    the plot. It is used for the export font size, and so that a dark theme
    exports with dark text on a white or transparent background. The on-screen
    figure is restored afterwards.
    """
    ext = fmt or Path(path).suffix.lower().lstrip(".")
    original = figure.get_size_inches().copy()
    kwargs = {"dpi": dpi, "format": {"jpg": "jpeg", "tif": "tiff"}.get(ext, ext)}
    if background == "Transparent":
        kwargs["transparent"] = True
    else:
        kwargs["facecolor"] = "white" if background == "White" else theme.current.card
    draw_theme = theme.export_theme(background)
    needs_restyle = font_size is not None or draw_theme is not theme.current
    context = restyle(draw_theme, font_size) if restyle and needs_restyle else nullcontext()
    # Some styling (e.g. gridlines) is resolved at save time, so keep the
    # export theme's rcParams active while writing.
    with context, matplotlib.rc_context(theme.mpl_rc(draw_theme)):
        _save(figure, path, size_inches, background, original, kwargs)


def _save(figure, path, size_inches, background, original, kwargs) -> None:
    axes_colors = [ax.get_facecolor() for ax in figure.axes]
    try:
        figure.set_size_inches(*size_inches, forward=False)
        # Lay out once at the new size first; otherwise the first save can use
        # stale tick-label sizes and clip the axes edge.
        figure.draw_without_rendering()
        if background == "White":
            for ax in figure.axes:
                ax.set_facecolor("white")
        figure.savefig(path, **kwargs)
    finally:
        for ax, color in zip(figure.axes, axes_colors):
            ax.set_facecolor(color)
        figure.set_size_inches(*original, forward=False)
        figure.canvas.draw_idle()


def export_figure(figure: Figure, default_name: str, parent=None,
                  restyle: Restyle | None = None) -> str | None:
    """Show the export dialog and save; returns the written path, if any."""
    dialog = ExportDialog(figure, default_name, parent, restyle)
    if dialog.exec() != QDialog.Accepted:
        return None
    dialog._save_settings()
    path = dialog.ask_path()
    if not path:
        return None
    try:
        save_figure(figure, path, dialog.size_inches(), dialog.dpi.value(),
                    dialog.background.currentText(), restyle, dialog.export_font_size)
    except Exception as exc:  # noqa: BLE001 - surface any writer error
        QMessageBox.warning(parent, "Export failed", f"Could not save {path}\n\n{exc}")
        return None
    return path
