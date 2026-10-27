"""Save and restore a whole session: every plot with its data, settings and marks.

A session is a JSON file (``*.csvplot``). Each dataset stores the path of its
file and, so the session still opens if the file moves or for data made inside
the app (such as a spectrum), a copy of the data itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .loaders import Dataset, load_file

SESSION_SUFFIX = ".csvplot"
SESSION_FILTER = f"CSV Plotter session (*{SESSION_SUFFIX})"
FORMAT_VERSION = 1


def _frame_to_json(frame: pd.DataFrame) -> dict:
    columns = {}
    for col in frame.columns:
        values = frame[col]
        if pd.api.types.is_numeric_dtype(values):
            arr = values.to_numpy(dtype=float)
            columns[col] = [None if np.isnan(v) else float(v) for v in arr]
        else:
            columns[col] = [None if pd.isna(v) else str(v) for v in values]
    return {"columns": list(frame.columns), "data": columns}


def _frame_from_json(blob: dict) -> pd.DataFrame:
    frame = pd.DataFrame({c: blob["data"][c] for c in blob["columns"]})
    for col in frame.columns:
        converted = pd.to_numeric(frame[col], errors="coerce")
        if converted.notna().sum() == frame[col].notna().sum():
            frame[col] = converted.astype(float)
    return frame


def session_dict(main) -> dict:
    tabs = [(main.tabs.widget(i), False) for i in range(main.tabs.count())]
    tabs += [(window.tab, True) for window in main.detached]
    plots = []
    for tab, detached in tabs:
        plots.append({
            "datasets": [{"path": str(ds.path), "from_file": ds.path.is_file(),
                          "embedded": _frame_to_json(ds.frame)} for ds in tab.datasets],
            "detached": detached,
            "state": tab.get_state(),
        })
    return {
        "format": "csvplot-session",
        "version": FORMAT_VERSION,
        "session_name": main.session_name,
        "theme": main.current_theme_name(),
        "current_tab": main.tabs.currentIndex(),
        "plots": plots,
    }


def save_session(main, path: str | Path) -> None:
    path = Path(path)
    text = json.dumps(session_dict(main), indent=1, allow_nan=False)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)  # never leave a half-written session behind


def _load_dataset(entry: dict, base: Path, notes: list[str]) -> Dataset:
    path = Path(entry["path"])
    if entry.get("from_file", True):
        candidates = [path] if path.is_absolute() else [base / path, path]
        found = next((c for c in candidates if c.is_file()), None)
        if found is not None:
            try:
                return load_file(found)   # re-read: picks up changes to the file
            except Exception as exc:  # noqa: BLE001 - fall back to the saved copy
                notes.append(f"{found.name}: could not be read ({exc}); used the saved copy")
        elif entry.get("embedded"):
            notes.append(f"{path.name}: file not found; used the copy saved in the session")
    if entry.get("embedded"):
        return Dataset(path, _frame_from_json(entry["embedded"]))
    raise FileNotFoundError(f"{path} was not found and the session has no copy of it")


def load_session(main, path: str | Path) -> list[str]:
    """Open the plots of a session in ``main``. Returns notes for the user."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != "csvplot-session":
        raise ValueError(f"{path.name} is not a CSV Plotter session")
    notes: list[str] = []
    if data.get("theme"):
        main.set_theme(data["theme"])
    main.set_session_name(data.get("session_name", ""))
    to_detach = []
    for plot in data.get("plots", []):
        try:
            datasets = [_load_dataset(entry, path.parent, notes) for entry in plot["datasets"]]
        except (FileNotFoundError, ValueError) as exc:
            notes.append(str(exc))
            continue
        tab = main._new_tab(datasets[0])
        for ds in datasets[1:]:
            tab.add_dataset(ds)
        try:
            tab.apply_state(plot.get("state", {}))
        except Exception as exc:  # noqa: BLE001 - keep the plot, report the problem
            notes.append(f"{tab.display_name}: some settings could not be restored ({exc})")
        main.tabs.setTabText(main.tabs.indexOf(tab), tab.display_name)
        main.tabs.setTabToolTip(main.tabs.indexOf(tab), "\n".join(str(d.path) for d in tab.datasets))
        if plot.get("detached"):
            to_detach.append(tab)
    current = data.get("current_tab", 0)
    if 0 <= current < main.tabs.count():
        main.tabs.setCurrentIndex(current)
    for tab in to_detach:
        main.tabs.setCurrentWidget(tab)
        main.detach_current()
    return notes
