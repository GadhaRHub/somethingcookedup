"""Colour themes shared by the Qt widgets and the matplotlib plots.

A theme is a set of colour roles. Add a new palette by adding a
:class:`Theme` to ``THEMES``; it then appears under View → Theme and as a
colormap of the same name.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib
from matplotlib.colors import LinearSegmentedColormap


@dataclass(frozen=True)
class Theme:
    name: str
    bar: str            # menu bar / main toolbar
    bar_text: str
    accent: str         # primary buttons, selected tab
    accent_hover: str
    check: str          # ticked checkboxes
    soft: str           # pills, unselected tabs, selections
    soft_light: str     # hovers, table header
    border: str
    deep: str           # checkbox outlines, scrollbar hover
    bg: str             # window background
    card: str           # panels and plot background
    text: str
    muted: str
    series_colors: tuple[str, ...]
    cmap_colors: tuple[str, ...]


THEMES = {
    "Berry": Theme(
        name="Berry",
        bar="#4F0336", bar_text="#F2EEE5",
        accent="#CC088A", accent_hover="#B0077A", check="#CC088A",
        soft="#D9C0FA", soft_light="#EDE2FD", border="#D9C0FA", deep="#8F63D9",
        bg="#F2EEE5", card="#FFFCF7", text="#4F0336", muted="#9C7A8F",
        series_colors=("#CC088A", "#4F0336", "#8F63D9", "#E58A2E", "#1F8A8A",
                       "#E86AB8", "#3D5A99", "#7A9E3B", "#B0413E", "#6B6B6B"),
        cmap_colors=("#4F0336", "#CC088A", "#B78BF0"),
    ),
    "Forest": Theme(
        name="Forest",
        bar="#186049", bar_text="#ECE8AF",
        accent="#D98327", accent_hover="#C06F1C", check="#4A943E",
        soft="#ECE8AF", soft_light="#F5F3D6", border="#CAD8A6", deep="#4A943E",
        bg="#F3F0DC", card="#FFFEF6", text="#186049", muted="#7F9A88",
        series_colors=("#D98327", "#186049", "#4A943E", "#8C5A9E", "#2E6FA3",
                       "#B0413E", "#C9A227", "#1F8A8A", "#E86AB8", "#6B6B6B"),
        cmap_colors=("#186049", "#4A943E", "#D98327"),
    ),
}

DEFAULT_THEME = "Berry"
current: Theme = THEMES[DEFAULT_THEME]


def set_current(name: str) -> Theme:
    global current
    current = THEMES.get(name, THEMES[DEFAULT_THEME])
    return current


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r}, {g}, {b}, {alpha})"


_ICONS = {
    "check": '<path d="M3.5 8.5l3 3 6-7" fill="none" stroke="{white}" stroke-width="2.2" '
             'stroke-linecap="round" stroke-linejoin="round"/>',
    "down": '<path d="M3.5 6l4.5 4.5L12.5 6" fill="none" stroke="{text}" stroke-width="2" '
            'stroke-linecap="round" stroke-linejoin="round"/>',
    "up": '<path d="M3.5 10.5L8 6l4.5 4.5" fill="none" stroke="{text}" stroke-width="2" '
          'stroke-linecap="round" stroke-linejoin="round"/>',
    "close": '<path d="M4.5 4.5l7 7M11.5 4.5l-7 7" fill="none" stroke="{text}" '
             'stroke-width="1.8" stroke-linecap="round"/>',
    "close-light": '<path d="M4.5 4.5l7 7M11.5 4.5l-7 7" fill="none" stroke="{white}" '
                   'stroke-width="1.8" stroke-linecap="round"/>',
}


def _icon_paths(t: Theme) -> dict[str, str]:
    """Write the stylesheet's SVG icons in the theme's colours; return their paths."""
    folder = Path(tempfile.gettempdir()) / "csvplot-icons" / t.name
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, body in _ICONS.items():
        path = folder / f"{name}.svg"
        svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
               + body.format(text=t.text, white="#FFFFFF") + "</svg>")
        if not path.exists() or path.read_text() != svg:
            path.write_text(svg)
        paths[name] = path.as_posix()
    return paths


def register_colormaps() -> None:
    for t in THEMES.values():
        if t.name not in matplotlib.colormaps:
            matplotlib.colormaps.register(LinearSegmentedColormap.from_list(t.name, list(t.cmap_colors)))


def apply_matplotlib_style(t: Theme | None = None) -> None:
    t = t or current
    register_colormaps()
    matplotlib.rcParams.update({
        "figure.facecolor": t.card,
        "axes.facecolor": t.card,
        "axes.edgecolor": t.text,
        "axes.labelcolor": t.text,
        "axes.titlecolor": t.text,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": t.text,
        "xtick.color": t.text,
        "ytick.color": t.text,
        "grid.color": t.border,
        "legend.facecolor": "white",
        "legend.edgecolor": t.border,
        "legend.framealpha": 0.92,
        "savefig.facecolor": "white",
    })


def stylesheet(t: Theme | None = None) -> str:
    t = t or current
    icon = _icon_paths(t)
    return f"""
    QMainWindow, QDialog, QMessageBox {{ background: {t.bg}; }}
    QWidget {{ color: {t.text}; }}

    QMenuBar {{ background: {t.bar}; color: {t.bar_text}; padding: 4px 6px; }}
    QMenuBar::item {{ background: transparent; padding: 5px 12px; border-radius: 8px; }}
    QMenuBar::item:selected {{ background: {t.accent}; color: white; }}
    QMenu {{ background: {t.card}; border: 1px solid {t.border}; border-radius: 10px; padding: 6px; }}
    QMenu::item {{ padding: 6px 22px; border-radius: 6px; color: {t.text}; }}
    QMenu::item:selected {{ background: {t.soft}; }}
    QMenu::item:disabled {{ color: {t.muted}; }}
    QMenu::separator {{ height: 1px; background: {t.border}; margin: 4px 8px; }}

    QToolBar#main_toolbar {{ background: {t.bar}; border: none; padding: 4px 10px 10px 10px; spacing: 8px; }}
    QToolBar#main_toolbar QToolButton {{
        margin-left: 4px;
        color: {t.bar_text}; background: {_rgba(t.soft, 0.16)}; border: none;
        border-radius: 12px; padding: 7px 16px; font-weight: 600;
    }}
    QToolBar#main_toolbar QToolButton:hover {{ background: {t.accent}; color: white; }}
    QToolBar#main_toolbar QToolButton:pressed {{ background: {t.accent_hover}; }}
    QToolBar#main_toolbar QToolButton:disabled {{ color: {_rgba(t.bar_text, 0.4)}; background: {_rgba(t.soft, 0.06)}; }}

    QTabWidget::pane {{ border: none; }}
    QTabBar {{ qproperty-drawBase: 0; }}
    QTabBar::tab {{
        background: {t.soft}; color: {t.text}; padding: 8px 8px 8px 16px;
        margin: 10px 2px 0 0; border-radius: 12px; font-weight: 600;
    }}
    QTabBar::tab:first {{ margin-left: 10px; }}
    QTabBar::tab:selected {{ background: {t.accent}; color: white; }}
    QTabBar::tab:hover:!selected {{ background: {t.soft_light}; }}
    QTabBar::close-button {{
        image: url({icon["close"]}); subcontrol-origin: padding; subcontrol-position: center right;
        width: 16px; height: 16px; margin-left: 6px; border-radius: 8px;
    }}
    QTabBar::close-button:selected {{ image: url({icon["close-light"]}); }}
    QTabBar::close-button:hover {{ background: {_rgba(t.text, 0.15)}; }}

    QFrame#card {{ background: {t.card}; border: 1px solid {t.border}; border-radius: 18px; }}
    QLabel#empty_state {{
        background: {t.card}; border: 2px dashed {t.border}; border-radius: 24px;
        padding: 40px 60px; color: {t.text};
    }}

    QGroupBox {{
        background: {t.card}; border: 1px solid {t.border}; border-radius: 16px;
        margin-top: 16px; padding: 18px 10px 10px 10px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin; subcontrol-position: top left; left: 14px;
        padding: 3px 12px; background: {t.soft}; color: {t.text};
        border-radius: 9px; font-weight: bold;
    }}

    QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox {{
        background: white; border: 1px solid {t.border}; border-radius: 9px;
        padding: 4px 8px; min-height: 20px;
        selection-background-color: {t.accent}; selection-color: white;
    }}
    QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus {{ border: 1px solid {t.accent}; }}
    QLineEdit:disabled, QComboBox:disabled, QDoubleSpinBox:disabled, QSpinBox:disabled {{ background: {t.bg}; color: {t.muted}; }}
    QComboBox::drop-down {{ border: none; width: 24px; }}
    QComboBox::down-arrow {{ image: url({icon["down"]}); width: 11px; height: 11px; }}
    QComboBox QAbstractItemView {{
        background: white; border: 1px solid {t.border}; padding: 4px; outline: 0;
        selection-background-color: {t.soft}; selection-color: {t.text};
    }}
    QDoubleSpinBox::up-button, QDoubleSpinBox::down-button, QSpinBox::up-button, QSpinBox::down-button {{
        subcontrol-origin: border; width: 20px; border: none; background: transparent;
    }}
    QDoubleSpinBox::up-button, QSpinBox::up-button {{ subcontrol-position: top right; }}
    QDoubleSpinBox::down-button, QSpinBox::down-button {{ subcontrol-position: bottom right; }}
    QDoubleSpinBox::up-arrow, QSpinBox::up-arrow {{ image: url({icon["up"]}); width: 9px; height: 9px; }}
    QDoubleSpinBox::down-arrow, QSpinBox::down-arrow {{ image: url({icon["down"]}); width: 9px; height: 9px; }}

    QPushButton {{
        background: {t.accent}; color: white; border: none; border-radius: 11px;
        padding: 7px 16px; font-weight: 600;
    }}
    QPushButton:hover {{ background: {t.accent_hover}; }}
    QPushButton:pressed {{ background: {t.text}; }}
    QPushButton[secondary="true"] {{ background: white; color: {t.text}; border: 1px solid {t.border}; }}
    QPushButton[secondary="true"]:hover {{ background: {t.soft_light}; }}
    QPushButton[secondary="true"]:pressed {{ background: {t.soft}; }}
    QMessageBox QPushButton {{ min-width: 70px; }}

    QCheckBox {{ spacing: 8px; }}
    QCheckBox::indicator, QTableView::indicator {{
        width: 16px; height: 16px; border: 1px solid {t.deep};
        border-radius: 5px; background: white;
    }}
    QCheckBox::indicator:checked, QTableView::indicator:checked {{
        background: {t.check}; border-color: {t.check}; image: url({icon["check"]});
    }}
    QCheckBox::indicator:hover, QTableView::indicator:hover {{ border-color: {t.accent}; }}
    QCheckBox::indicator:disabled {{ background: {t.bg}; border-color: {t.border}; }}
    QCheckBox:disabled {{ color: {t.muted}; }}

    QTableWidget, QListWidget {{
        background: white; border: 1px solid {t.border}; border-radius: 12px;
        padding: 2px; outline: 0;
    }}
    QListWidget::item {{ padding: 4px 6px; border-radius: 6px; }}
    QListWidget::item:selected {{ background: {t.soft}; color: {t.text}; }}
    QTableWidget::item:selected {{ background: {t.soft_light}; color: {t.text}; }}
    QHeaderView {{ background: transparent; }}
    QHeaderView::section {{
        background: {t.soft_light}; color: {t.text}; border: none;
        padding: 5px 8px; font-weight: bold;
    }}
    QHeaderView::section:first {{ border-top-left-radius: 10px; }}
    QHeaderView::section:last {{ border-top-right-radius: 10px; }}

    QScrollArea, QWidget#controls_panel {{ background: transparent; border: none; }}
    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 4px 2px; }}
    QScrollBar::handle:vertical {{ background: {t.border}; border-radius: 3px; min-height: 30px; }}
    QScrollBar::handle:vertical:hover {{ background: {t.deep}; }}
    QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px 4px; }}
    QScrollBar::handle:horizontal {{ background: {t.border}; border-radius: 3px; min-width: 30px; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

    QSplitter::handle {{ background: transparent; width: 10px; }}

    QFrame#card QToolBar {{ background: {t.card}; border: none; spacing: 4px; padding: 2px; }}
    QFrame#card QToolButton {{
        background: transparent; border: none; border-radius: 9px; padding: 5px 9px;
        color: {t.text}; font-weight: 600;
    }}
    QFrame#card QToolButton:hover {{ background: {t.soft_light}; }}
    QFrame#card QToolButton:checked {{ background: {t.soft}; }}
    QFrame#card QToolBar QLabel {{ color: {t.muted}; }}
    QLabel#coords {{
        background: {t.bg}; border-radius: 10px; padding: 6px 12px;
        font-family: monospace; color: {t.text};
    }}

    QStatusBar {{ background: {t.bg}; color: {t.text}; }}
    QToolTip {{ background: {t.bar}; color: {t.bar_text}; border: none; padding: 5px 8px; }}
    """


def apply_theme(app, name: str | None = None) -> Theme:
    t = set_current(name) if name else current
    app.setStyle("Fusion")
    app.setStyleSheet(stylesheet(t))
    apply_matplotlib_style(t)
    return t
