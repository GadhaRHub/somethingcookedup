"""Read-only table view of a dataset's raw values."""

from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableView,
    QVBoxLayout,
)

from .loaders import Dataset


class DataFrameModel(QAbstractTableModel):
    """Exposes a DataFrame to Qt without copying it."""

    def __init__(self, frame: pd.DataFrame | None = None, parent=None):
        super().__init__(parent)
        self._frame = frame if frame is not None else pd.DataFrame()

    def set_frame(self, frame: pd.DataFrame) -> None:
        self.beginResetModel()
        self._frame = frame
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._frame)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else self._frame.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        value = self._frame.iat[index.row(), index.column()]
        if role == Qt.DisplayRole:
            if isinstance(value, (float, np.floating)):
                return "" if np.isnan(value) else f"{value:.8g}"
            return str(value)
        if role == Qt.TextAlignmentRole:
            numeric = isinstance(value, (int, float, np.number))
            return int((Qt.AlignRight if numeric else Qt.AlignLeft) | Qt.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._frame.columns[section])
        return str(section + 1)


class DataPreview(QFrame):
    """Card with a dataset picker and the raw table."""

    closeRequested = Signal()

    def __init__(self, datasets: list[Dataset], parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.datasets = datasets

        self.dataset_combo = QComboBox()
        self.dataset_combo.currentIndexChanged.connect(self._show_dataset)
        self.info = QLabel()
        close_btn = QPushButton("Close table")
        close_btn.setProperty("secondary", True)
        close_btn.clicked.connect(self.closeRequested)

        header = QHBoxLayout()
        header.addWidget(QLabel("<b>Dataset</b>"))
        header.addWidget(self.dataset_combo, 1)
        header.addWidget(self.info)
        header.addWidget(close_btn)

        self.model = DataFrameModel(parent=self)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setAlternatingRowColors(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setDefaultSectionSize(150)
        self.table.verticalHeader().setDefaultSectionSize(24)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.addLayout(header)
        layout.addWidget(self.table, 1)
        self.refresh()

    def refresh(self) -> None:
        current = max(self.dataset_combo.currentIndex(), 0)
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        self.dataset_combo.addItems([d.name for d in self.datasets])
        self.dataset_combo.blockSignals(False)
        self.dataset_combo.setEnabled(len(self.datasets) > 1)
        self.dataset_combo.setCurrentIndex(min(current, len(self.datasets) - 1))
        self._show_dataset(self.dataset_combo.currentIndex())

    def _show_dataset(self, index: int) -> None:
        if not 0 <= index < len(self.datasets):
            return
        frame = self.datasets[index].frame
        self.model.set_frame(frame)
        self.info.setText(f"{frame.shape[0]:,} rows × {frame.shape[1]} columns")
