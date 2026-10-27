"""Small 'i' buttons that explain a parameter (hover for a tooltip, click to pin it)."""

from __future__ import annotations

from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QFormLayout, QHBoxLayout, QLayout, QToolButton, QToolTip, QWidget


class InfoButton(QToolButton):
    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self.setText("i")
        self.setObjectName("info")
        self.setToolTip(text)
        self.setAutoRaise(True)
        self.setFixedSize(20, 20)
        self._text = text
        self.clicked.connect(self._show)

    def _show(self) -> None:
        QToolTip.showText(self.mapToGlobal(QPoint(self.width(), self.height())), self._text, self)


def add_row(form: QFormLayout, label: str, field, info: str) -> None:
    """Add 'label: field (i)' to a form. ``field`` may be a widget or a layout."""
    row = QHBoxLayout()
    row.setSpacing(6)
    if isinstance(field, QLayout):
        row.addLayout(field, 1)
    else:
        row.addWidget(field, 1)
    row.addWidget(InfoButton(info))
    form.addRow(label, row)


def add_full_row(form: QFormLayout, field: QWidget, info: str) -> None:
    """A row without a label (e.g. a checkbox) with an info button."""
    row = QHBoxLayout()
    row.setSpacing(6)
    row.addWidget(field, 1)
    row.addWidget(InfoButton(info))
    form.addRow(row)
