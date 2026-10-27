import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from matplotlib.backend_bases import MouseEvent  # noqa: E402

from csvplot.loaders import load_file  # noqa: E402
from csvplot.main_window import MainWindow  # noqa: E402

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "dRR_Ti_10nm.csv"


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(app):
    w = MainWindow()
    w.show()
    w.open_files([SAMPLE])
    yield w
    w.close()


def click(tab, x, y, button=1):
    tab.canvas.draw()
    px, py = tab.ax.transData.transform((x, y))
    event = MouseEvent("button_press_event", tab.canvas, px, py, button=button)
    tab.canvas.callbacks.process("button_press_event", event)


def test_default_plot(window):
    tab = window.current_tab()
    assert tab.x_column == "t_ps"
    assert [line.get_label() for line in tab._data_lines] == ["dR_over_R", "dR_over_R_minus_bg"]


def test_ranges_and_colormap(window):
    tab = window.current_tab()
    tab.xmin_edit.setText("-1")
    tab.xmax_edit.setText("20")
    tab.apply_ranges()
    assert tab.ax.get_xlim() == (-1, 20)
    tab.use_cmap.setChecked(True)
    colors = {line.get_color() for line in tab._data_lines}
    assert len(colors) == 2
    assert tab.ax.get_xlim() == (-1, 20)  # restyling keeps the view


def test_point_mark_add_and_remove(window):
    tab = window.current_tab()
    tab.act_mark_point.setChecked(True)
    click(tab, 300, 0.015)
    assert len(tab.point_marks) == 1
    mark = tab.point_marks[0]
    click(tab, mark.x, mark.y, button=3)
    assert tab.point_marks == []


def test_overlay_and_detach(window):
    tab = window.current_tab()
    tab.add_dataset(load_file(SAMPLE))
    assert len(tab.datasets) == 2
    window.detach_current()
    assert window.tabs.count() == 0 and len(window.detached) == 1
    window.detached[0].close()
    assert window.tabs.count() == 1
