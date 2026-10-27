"""High-quality figure export: format, resolution, size and background."""

from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
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


class ExportDialog(QDialog):
    def __init__(self, figure: Figure, default_name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export figure")
        self.figure = figure
        self.default_name = default_name
        self.settings = QSettings("csvplot", "csvplot")

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
        form.addRow("Format", self.format_combo)
        form.addRow("Resolution", dpi_row)
        form.addRow("Size", size_row)
        form.addRow("Background", self.background)
        form.addRow(self.info)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Cancel).setProperty("secondary", True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

        self.format_combo.currentTextChanged.connect(self._update_info)
        self._load_settings()

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


def save_figure(figure: Figure, path: str, size_inches, dpi: int, background: str) -> None:
    """Write ``figure`` to ``path`` at the given size and resolution.

    The on-screen figure size is restored afterwards.
    """
    ext = Path(path).suffix.lower().lstrip(".")
    original = figure.get_size_inches().copy()
    kwargs = {"dpi": dpi, "format": {"jpg": "jpeg", "tif": "tiff"}.get(ext, ext)}
    if background == "Transparent":
        kwargs["transparent"] = True
    else:
        kwargs["facecolor"] = "white" if background == "White" else theme.current.card
    axes_colors = [ax.get_facecolor() for ax in figure.axes]
    try:
        figure.set_size_inches(*size_inches, forward=False)
        if background == "White":
            for ax in figure.axes:
                ax.set_facecolor("white")
        figure.savefig(path, **kwargs)
    finally:
        for ax, color in zip(figure.axes, axes_colors):
            ax.set_facecolor(color)
        figure.set_size_inches(*original, forward=False)
        figure.canvas.draw_idle()


def export_figure(figure: Figure, default_name: str, parent=None) -> str | None:
    """Show the export dialog and save; returns the written path, if any."""
    dialog = ExportDialog(figure, default_name, parent)
    if dialog.exec() != QDialog.Accepted:
        return None
    dialog._save_settings()
    path = dialog.ask_path()
    if not path:
        return None
    try:
        save_figure(figure, path, dialog.size_inches(), dialog.dpi.value(),
                    dialog.background.currentText())
    except Exception as exc:  # noqa: BLE001 - surface any writer error
        QMessageBox.warning(parent, "Export failed", f"Could not save {path}\n\n{exc}")
        return None
    return path
