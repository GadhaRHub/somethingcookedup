# CSV Plotter

A desktop app for viewing data files as interactive plots. It reads CSV, TSV and
other delimited text files, and you can add more formats later (see below).

## Run

```bash
pip install -r requirements.txt
python -m csvplot                         # then File → Open… (Ctrl+O)
python -m csvplot samples/dRR_Ti_10nm.csv # or pass files on the command line
```

You can also drag files onto the window.

## Features

- **Multiple files**: each file opens in its own tab. Tabs can be closed, reordered,
  or **detached** into separate windows (View → Detach, Ctrl+D). Closing a detached
  window puts the plot back in a tab.
  **File → Add to current plot** (Ctrl+Shift+O) overlays another file on the open plot.
- **Automatic plotting from column names**: the first numeric column is used as X and
  the other columns are plotted against it. You can pick any column for X, or use
  the row number, and tick which columns to show as Y series.
- **Interactive canvas**
  - Mouse wheel zooms around the cursor (hold Shift to zoom only X, Ctrl to zoom only Y)
  - Pan tool: drag to move the plot. Zoom tool: drag a box to zoom into it.
  - Home / Back / Forward step through your view history
  - The cursor's x/y position and the nearest data point are shown under the plot
  - The legend and mark labels can be dragged
- **Marks**: *Mark point* (click to mark a point; it snaps to the nearest data point
  if snapping is on) and *Mark region* (drag to shade an x-range). Right-click a mark,
  or use the Marks list, to remove it. Mark colours are configurable.
- **Styling**: choose a colour per series, or colour all series from a **colormap**
  (viridis, plasma, turbo, coolwarm, tab10, …, can be reversed). Also: line, marker
  or line+marker style, line width, marker size, title, axis labels, font size,
  legend on/off and position, and grid.
- **Axis ranges**: type the min/max values and Apply, or use Auto-fit. The boxes
  update as you pan and zoom. Log scale for X and/or Y.
- **Export**: File → Save figure (PNG, SVG, PDF, …).

## Adding a file format

Loaders live in `csvplot/loaders.py`. Register a function that returns a
`pandas.DataFrame`:

```python
@register_loader("Excel workbook", ".xlsx")
def load_excel(path):
    return pd.read_excel(path)
```

The Open dialog and drag-and-drop pick it up automatically.

## Tests

```bash
pip install pytest
QT_QPA_PLATFORM=offscreen python -m pytest
```
