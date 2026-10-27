"""File loaders.

Each loader turns a file into a :class:`Dataset` (a pandas DataFrame plus some
metadata). New formats are added by decorating a function with
:func:`register_loader`; the Open dialog and drag-and-drop pick them up
automatically.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

LoaderFunc = Callable[[Path], pd.DataFrame]

_LOADERS: dict[str, tuple[str, LoaderFunc]] = {}


@dataclass
class Dataset:
    path: Path
    frame: pd.DataFrame

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def numeric_columns(self) -> list[str]:
        return [c for c in self.frame.columns if pd.api.types.is_numeric_dtype(self.frame[c])]


def register_loader(description: str, *extensions: str):
    """Register ``func`` as the loader for the given file extensions."""

    def decorator(func: LoaderFunc) -> LoaderFunc:
        for ext in extensions:
            _LOADERS[ext.lower()] = (description, func)
        return func

    return decorator


def supported_extensions() -> list[str]:
    return sorted(_LOADERS)


def file_dialog_filter() -> str:
    """Filter string for QFileDialog, e.g. ``"Data files (*.csv *.tsv);;..."``."""
    by_desc: dict[str, list[str]] = {}
    for ext, (desc, _) in _LOADERS.items():
        by_desc.setdefault(desc, []).append(f"*{ext}")
    all_patterns = " ".join(f"*{ext}" for ext in supported_extensions())
    parts = [f"All supported files ({all_patterns})"]
    parts += [f"{desc} ({' '.join(sorted(p))})" for desc, p in sorted(by_desc.items())]
    parts.append("All files (*)")
    return ";;".join(parts)


def load_file(path: str | Path) -> Dataset:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {path}")
    _, func = _LOADERS.get(path.suffix.lower(), _LOADERS[".csv"])
    frame = func(path)
    if frame.empty or not frame.columns.size:
        raise ValueError(f"{path.name} contains no data")
    frame = _clean_frame(frame)
    if not any(pd.api.types.is_numeric_dtype(frame[c]) for c in frame.columns):
        raise ValueError(f"{path.name} has no numeric columns to plot")
    return Dataset(path=path, frame=frame)


def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    # Unique, stripped, string column names.
    names, seen = [], {}
    for col in frame.columns:
        name = str(col).strip() or "column"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        names.append(name)
    frame.columns = names

    # Convert text columns that are mostly numbers (e.g. a few "NaN"/"--" cells).
    for col in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[col]):
            converted = pd.to_numeric(frame[col], errors="coerce")
            if converted.notna().sum() >= 0.5 * frame[col].notna().sum() > 0:
                frame[col] = converted
    return frame.dropna(how="all").reset_index(drop=True)


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


@register_loader("Delimited text", ".csv", ".tsv", ".txt", ".dat")
def load_delimited(path: Path) -> pd.DataFrame:
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as fh:
        head = [fh.readline() for _ in range(200)]

    # Leading "#" comment lines are skipped.
    skip = 0
    for line in head:
        if line.strip() and not line.lstrip().startswith("#"):
            break
        skip += 1
    lines = [ln for ln in head[skip:] if ln.strip()]
    if not lines:
        return pd.DataFrame()

    sample = "".join(lines)
    try:
        sep = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        sep = r"\s+" if len(lines[0].split()) > 1 else ","

    first = lines[0].strip()
    fields = first.split() if sep == r"\s+" else next(csv.reader([first], delimiter=sep))
    has_header = not all(_is_number(f.strip()) for f in fields if f.strip())

    frame = pd.read_csv(
        path,
        sep=sep,
        skiprows=skip,
        header=0 if has_header else None,
        skipinitialspace=True,
        encoding="utf-8-sig",
        engine="python" if sep == r"\s+" else "c",
    )
    if not has_header:
        frame.columns = [f"col {i + 1}" for i in range(frame.shape[1])]
    return frame
