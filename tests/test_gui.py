import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
from PySide6 import QtCore  # noqa: E402

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


def test_switch_theme(window):
    from csvplot import theme

    tab = window.current_tab()
    window.set_theme("Forest")
    try:
        assert tab._data_lines[0].get_color() == theme.THEMES["Forest"].series_colors[0]
        assert tab.figure.get_facecolor()[:3] != (1, 1, 1)
    finally:
        window.set_theme("Berry")
    assert tab._data_lines[0].get_color() == theme.THEMES["Berry"].series_colors[0]


@pytest.mark.parametrize("ext", ["png", "svg", "pdf", "eps", "tiff", "jpg"])
def test_export_formats(window, tmp_path, ext):
    from csvplot.export import save_figure

    tab = window.current_tab()
    before = tuple(tab.figure.get_size_inches())
    path = tmp_path / f"plot.{ext}"
    save_figure(tab.figure, str(path), (6, 4), 300, "White")
    assert path.stat().st_size > 1000
    assert tuple(tab.figure.get_size_inches()) == before
    if ext == "png":
        from matplotlib.image import imread

        assert imread(path).shape[:2] == (1200, 1800)


def test_individual_line_style_and_hiding(window):
    tab = window.current_tab()
    first, second = tab._plottable()
    tab._width_spins()[tab.series.index(first)].setValue(4)
    assert tab._data_lines[0].get_linewidth() == 4
    assert tab._data_lines[1].get_linewidth() == 1.5

    tab.series_table.selectRow(tab.series.index(second))
    tab.series_linestyle.setCurrentText("Dashed")
    tab.series_marker.setCurrentText("Square")
    tab.series_name.setText("background-subtracted")
    tab.series_name.editingFinished.emit()
    line = tab._data_lines[1]
    assert (line.get_linestyle(), line.get_marker(), line.get_label()) == ("--", "s", "background-subtracted")

    tab.set_series_visible(first, False)
    assert len(tab._data_lines) == 1
    assert tab.series_table.item(tab.series.index(first), 0).checkState() == QtCore.Qt.Unchecked


def test_legend_click_toggles_line(window):
    from types import SimpleNamespace

    tab = window.current_tab()
    target = tab._plottable()[0]
    handle = next(a for a, s in tab._legend_map.items() if s is target)
    event = SimpleNamespace(artist=handle, mouseevent=SimpleNamespace(button=1))
    tab._on_pick(event)
    assert not target.visible
    handle = next(a for a, s in tab._legend_map.items() if s is target)
    tab._on_pick(SimpleNamespace(artist=handle, mouseevent=SimpleNamespace(button=1)))
    assert target.visible


def test_session_and_tab_names(window, monkeypatch):
    window.set_session_name("Ti film pump-probe")
    assert window.windowTitle() == "Ti film pump-probe — CSV Plotter"
    window.set_session_name("")
    assert window.windowTitle() == "CSV Plotter"

    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *a, **k: ("10 nm sample", True))
    window.rename_tab(0)
    assert window.tabs.tabText(0) == "10 nm sample"
    assert window.current_tab().display_name == "10 nm sample"
    assert not window.windowIcon().isNull()


def test_data_preview_optional(window):
    tab = window.current_tab()
    assert tab.data_preview is None and tab.views.count() == 1
    window.toggle_data()
    assert tab.data_preview is not None and tab.views.count() == 2
    model = tab.data_preview.model
    assert model.rowCount() == 12021 and model.headerData(0, QtCore.Qt.Horizontal) == "t_ps"
    tab.close_data_preview()
    assert tab.data_preview is None and tab.views.count() == 1 and not tab.act_data.isChecked()


def test_accent_axes_and_night_theme(window):
    from matplotlib.colors import to_hex

    from csvplot import theme

    tab = window.current_tab()
    tab.accent_axes.setChecked(True)
    assert to_hex(tab.ax.spines["left"].get_edgecolor()) == theme.current.accent.lower()
    window.set_theme("Night")
    try:
        assert to_hex(tab.figure.get_facecolor()) == theme.THEMES["Night"].card.lower()
        # White export of a dark theme draws with the light variant, then restores.
        with tab.export_style(theme.export_theme("White"), 8):
            assert to_hex(tab.ax.xaxis.label.get_color()) == theme.THEMES["Berry"].accent.lower()
            assert tab.ax.xaxis.label.get_fontsize() == 8
        assert to_hex(tab.figure.get_facecolor()) == theme.THEMES["Night"].card.lower()
    finally:
        window.set_theme("Berry")


def test_export_with_preset_font(window, tmp_path):
    from csvplot.export import PRESETS, save_figure

    tab = window.current_tab()
    w_cm, h_cm, dpi, font = PRESETS["Journal, single column (8.5 cm)"]
    path = tmp_path / "fig.png"
    save_figure(tab.figure, str(path), (w_cm / 2.54, h_cm / 2.54), dpi, "White",
                tab.export_style, font)
    assert path.exists()
    assert tab.ax.xaxis.label.get_fontsize() == tab.font_size.value()


def test_major_and_minor_ticks(window):
    from matplotlib.ticker import AutoMinorLocator, MultipleLocator, NullLocator

    tab = window.current_tab()
    tab.major_step["x"].setText("100")
    tab.major_step["x"].editingFinished.emit()
    tab.minor_ticks["x"].setCurrentText("4")
    tab.minor_ticks["y"].setCurrentText("Off")
    tab.mirror_ticks.setChecked(True)
    tab.tick_direction.setCurrentText("Inside")
    assert isinstance(tab.ax.xaxis.get_major_locator(), MultipleLocator)
    assert list(tab.ax.get_xticks()[:3]) == [-100, 0, 100]
    assert isinstance(tab.ax.xaxis.get_minor_locator(), AutoMinorLocator)
    assert isinstance(tab.ax.yaxis.get_minor_locator(), NullLocator)
    assert tab.ax.spines["top"].get_visible()

    # A step that would create thousands of ticks falls back to automatic.
    tab.major_step["x"].setText("0.001")
    tab.major_step["x"].editingFinished.emit()
    assert not isinstance(tab.ax.xaxis.get_major_locator(), MultipleLocator)


def test_tick_width_and_length(window):
    tab = window.current_tab()
    tab.tick_width["major"].setValue(1.5)
    tab.tick_length["minor"].setValue(4)
    tab.minor_ticks["x"].setCurrentText("2")
    tab.canvas.draw()
    major = tab.ax.xaxis.get_major_ticks()[0].tick1line
    minor = tab.ax.xaxis.get_minor_ticks()[0].tick1line
    assert major.get_markeredgewidth() == 1.5
    assert minor.get_markersize() == 4


def test_export_preview(window):
    from csvplot.export import ExportDialog

    tab = window.current_tab()
    before = tuple(tab.figure.get_size_inches())
    dialog = ExportDialog(tab.figure, "x", window, tab.export_style)
    dialog.preset_combo.setCurrentText("Slide 16:9")
    dialog.update_preview()
    pixmap = dialog.preview.pixmap()
    assert not pixmap.isNull()
    assert abs(pixmap.width() / pixmap.height() - 16 / 9) < 0.05
    assert tuple(tab.figure.get_size_inches()) == before
    dialog.reject()


def test_plain_and_grid_background(window, tmp_path):
    from csvplot.export import save_figure

    tab = window.current_tab()

    def gridlines_visible():
        return any(line.get_visible() for line in tab.ax.get_xgridlines())

    assert tab.plot_background.currentText() == "Grid" and gridlines_visible()
    tab.plot_background.setCurrentText("Plain")
    assert not gridlines_visible()

    # The export can override the on-screen choice without changing it.
    seen = []
    original = tab.export_style

    def spy(*args):
        ctx = original(*args)

        class Wrap:
            def __enter__(self):
                ctx.__enter__()
                seen.append(gridlines_visible())

            def __exit__(self, *exc):
                return ctx.__exit__(*exc)

        return Wrap()

    save_figure(tab.figure, str(tmp_path / "g.png"), (4, 3), 100, "White", spy, grid=True)
    assert seen == [True]
    assert tab.plot_background.currentText() == "Plain" and not gridlines_visible()


def test_icon_follows_theme(window):
    from csvplot.main_window import icon_path

    assert icon_path().name == "icon-berry.svg"
    window.set_theme("Forest")
    try:
        assert icon_path().name == "icon-forest.svg"
        assert "icon-forest.svg" in window.empty_label.text()
    finally:
        window.set_theme("Berry")
