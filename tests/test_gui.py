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


def _mouse(tab, kind, x, y, button=1, dblclick=False):
    tab.canvas.draw()
    px, py = tab.ax.transData.transform((x, y))
    event = MouseEvent(kind, tab.canvas, px, py, button=button, dblclick=dblclick)
    tab.canvas.callbacks.process(kind, event)


def test_region_highlight_move_and_resize(window):
    from csvplot.plot_tab import RegionMark

    tab = window.current_tab()
    region = RegionMark(100, 200, "#CC088A")
    tab.region_marks.append(region)
    tab._after_marks_changed()

    # Selecting it in the list highlights it.
    tab.marks_list.item(0).setSelected(True)
    patch = region.artists[0]
    assert patch.get_alpha() > 0.3 and patch.get_linewidth() == 2

    # Drag the middle: moves; drag the right edge: resizes.
    _mouse(tab, "button_press_event", 150, 0.01)
    _mouse(tab, "motion_notify_event", 200, 0.01)
    _mouse(tab, "button_release_event", 200, 0.01)
    assert (round(region.xmin), round(region.xmax)) == (150, 250)
    _mouse(tab, "button_press_event", 250, 0.01)
    _mouse(tab, "motion_notify_event", 300, 0.01)
    _mouse(tab, "button_release_event", 300, 0.01)
    assert (round(region.xmin), round(region.xmax)) == (150, 300)
    assert tab.marks_list.item(0).isSelected()
    assert "150" in tab.marks_list.item(0).text()


def test_x_axis_name(window):
    tab = window.current_tab()
    tab.xname_edit.setText("Delay (ps)")
    tab.xname_edit.editingFinished.emit()
    assert tab.ax.get_xlabel() == "Delay (ps)"
    assert tab.xlabel_edit.text() == "Delay (ps)"
    tab.xlabel_edit.setText("")
    tab.redraw()
    assert tab.ax.get_xlabel() == "t_ps" and tab.xname_edit.text() == ""


def test_series_offsets(window):
    tab = window.current_tab()
    first, second = tab._plottable()
    tab.series_table.selectRow(tab.series.index(second))
    tab.series_y_offset.setText("0.01")
    tab.series_x_offset.setText("2")
    tab.series_x_offset.editingFinished.emit()
    raw_x = second.dataset.frame["t_ps"].to_numpy()
    raw_y = second.dataset.frame[second.column].to_numpy()
    line = tab._data_lines[1]
    assert abs(line.get_xdata()[0] - (raw_x[0] + 2)) < 1e-12
    assert abs(line.get_ydata()[0] - (raw_y[0] + 0.01)) < 1e-12
    assert tab._data_lines[0].get_ydata()[0] == first.dataset.frame[first.column].iloc[0]


def test_data_table_beside_plot(window):
    tab = window.current_tab()
    tab.show_data_preview()
    assert tab.views.count() == 2 and tab.data_preview.isVisibleTo(tab)
    assert tab.views.orientation() == QtCore.Qt.Vertical
    tab.data_preview.layout_button.click()
    assert tab.views.orientation() == QtCore.Qt.Horizontal
    assert tab.controls.isVisibleTo(tab)
    tab.close_data_preview()
    assert tab.views.count() == 1


def test_legend_drag_is_kept_and_options(window):
    tab = window.current_tab()
    tab._legend._loc = (0.1, 0.2)          # what dragging the legend does
    tab.redraw()
    assert tab._legend._loc == (0.1, 0.2) and tab.legend_reset.isEnabled()
    tab.legend_line_length.setValue(4)
    tab.legend_box.setChecked(False)
    tab.legend_cols.setValue(2)
    legend = tab.ax.get_legend()
    assert legend._loc == (0.1, 0.2)       # still where it was dragged
    assert legend.handlelength == 4 and not legend.get_frame_on() and legend._ncols == 2
    tab.legend_reset.click()
    assert tab.ax.get_legend()._loc != (0.1, 0.2) and not tab.legend_reset.isEnabled()


def test_math_labels_and_invalid_math(window):
    tab = window.current_tab()
    tab.ylabel_edit.setText(r"$\Delta R/R$ (10$^{-3}$)")
    tab.redraw()
    assert tab.ax.get_ylabel() == r"$\Delta R/R$ (10$^{-3}$)"
    messages = []
    tab.statusMessage.connect(messages.append)
    tab.title_edit.setText(r"$\frac{a}$")   # broken math must not break drawing
    tab.redraw()
    tab.canvas.draw()
    assert tab.ax.get_title() == r"\$\frac{a}\$" and messages


def test_insert_math_helpers(app):
    from csvplot.symbols import insert_math

    edit = QtWidgets.QLineEdit("cm-1")
    edit.setSelection(2, 2)                 # select "-1"
    insert_math(edit, "^{{{}}}|2")
    assert edit.text() == "cm$^{-1}$"
    edit.setText(r"$\Delta R$")
    edit.setCursorPosition(7)               # inside the math, after "\Delta"
    insert_math(edit, r"\mu")
    assert edit.text() == r"$\Delta\mu R$"


def test_text_and_equation_marks(window, monkeypatch):
    tab = window.current_tab()
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText",
                        lambda *a, **k: (r"$E = mc^2$", True))
    tab.act_add_text.setChecked(True)
    assert not tab.act_mark_point.isChecked()
    _mouse(tab, "button_press_event", 300, 0.02)
    assert [m.content for m in tab.text_marks] == [r"$E = mc^2$"]
    mark = tab.text_marks[0]
    assert mark.artists[0].get_text() == r"$E = mc^2$"
    assert "Text" in tab.marks_list.item(tab.marks_list.count() - 1).text()
    # Right-click removes it.
    tab.canvas.draw()
    bbox = mark.artists[0].get_window_extent()
    event = MouseEvent("button_press_event", tab.canvas, bbox.x0 + 2, bbox.y0 + 2, button=3)
    tab.canvas.callbacks.process("button_press_event", event)
    assert tab.text_marks == []


def test_renaming_a_line_keeps_the_zoom(window):
    tab = window.current_tab()
    tab.xmin_edit.setText("-1")
    tab.xmax_edit.setText("20")
    tab.apply_ranges()
    tab.series_table.selectRow(tab.series.index(tab._plottable()[0]))
    tab.series_name.setText("renamed")
    tab.series_name.editingFinished.emit()
    assert tab.ax.get_xlim() == (-1, 20)


def test_full_latex_mode(window, tmp_path):
    from csvplot.symbols import latex_available

    if not latex_available()[0]:
        pytest.skip("no LaTeX installation")
    tab = window.current_tab()
    tab.ylabel_edit.setText(r"$\Delta R/R$ in \% of $R_0$")
    tab.full_latex.setChecked(True)
    tab.canvas.draw()
    assert tab.ax.yaxis.label.get_usetex()
    assert tab.ax.get_title() == r"dRR\_Ti\_10nm.csv"      # underscores escaped
    from csvplot.export import save_figure

    save_figure(tab.figure, str(tmp_path / "tex.pdf"), (4, 3), 300, "White")
    assert (tmp_path / "tex.pdf").stat().st_size > 1000

    # A LaTeX error falls back to built-in math instead of breaking the app.
    tab.title_edit.setText(r"$\undefinedcommand$")
    tab.redraw()
    tab.canvas.draw()
    assert not tab.full_latex.isChecked()
