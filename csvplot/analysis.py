"""Spectral analysis: windowed FFT, Welch averaging, spectrogram and period extraction.

All functions are independent of the GUI so they can be tested and reused.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from scipy import signal

# name -> scipy window spec (a parameter, if any, comes from FFTSettings.window_param)
WINDOWS = {
    "Hann": "hann",
    "Hamming": "hamming",
    "Blackman": "blackman",
    "Blackman-Harris": "blackmanharris",
    "Flat top (accurate amplitudes)": "flattop",
    "Bartlett (triangular)": "bartlett",
    "Tukey (tapered cosine)": "tukey",
    "Kaiser": "kaiser",
    "Rectangular (no window)": "boxcar",
}
WINDOW_PARAMS = {"tukey": ("α", 0.5, 0.0, 1.0, 0.05), "kaiser": ("β", 8.6, 0.0, 30.0, 0.5)}

METHODS = {
    "Single FFT (whole range)": "single",
    "Welch (averaged sliding windows)": "welch",
    "Spectrogram (sliding windows along x)": "spectrogram",
}
DETRENDS = {"None": False, "Remove mean": "constant", "Remove linear trend": "linear"}

# x unit -> (frequency unit name, factor to convert cycles-per-x-unit into it)
_FREQ_UNITS = {
    "fs": ("PHz", 1.0), "ps": ("THz", 1.0), "ns": ("GHz", 1.0), "µs": ("MHz", 1.0),
    "us": ("MHz", 1.0), "ms": ("kHz", 1.0), "s": ("Hz", 1.0),
    "min": ("1/min", 1.0), "h": ("1/h", 1.0),
}


@dataclass
class FFTSettings:
    method: str = "single"            # single | welch | spectrogram
    window: str = "hann"
    window_param: float | None = None
    segment: float | None = None      # sliding-window length in x units (welch/spectrogram)
    overlap: float = 50.0             # percent
    detrend: str | bool = "linear"
    zero_pad: int = 4                 # FFT length = segment points × zero_pad
    n_peaks: int = 5
    min_height: float = 5.0           # percent of the largest peak
    f_min: float | None = None
    f_max: float | None = None


@dataclass
class Peak:
    frequency: float
    amplitude: float
    relative: float                   # amplitude / largest amplitude in the search range

    @property
    def period(self) -> float:
        return 1.0 / self.frequency if self.frequency else float("inf")


@dataclass
class SpectrumResult:
    label: str
    x: np.ndarray                     # data actually analysed (uniform spacing)
    y: np.ndarray
    dx: float
    resampled: bool
    frequency: np.ndarray
    amplitude: np.ndarray
    peaks: list[Peak] = field(default_factory=list)
    segment_points: int = 0
    n_segments: int = 1
    # spectrogram only
    seg_centers: np.ndarray | None = None
    spectrogram: np.ndarray | None = None

    @property
    def resolution(self) -> float:
        """Frequency resolution set by the analysed length (not by zero padding)."""
        return 1.0 / (self.segment_points * self.dx) if self.segment_points else float("nan")


def frequency_unit(x_label: str) -> tuple[str, str]:
    """('THz', 'ps') style units from an axis name like 'Delay (ps)'; falls back
    to cycles per x unit."""
    match = re.search(r"[\(\[]\s*([^\)\]]+?)\s*[\)\]]\s*$", x_label or "")
    unit = (match.group(1) if match else (x_label or "").split("_")[-1]).strip()
    unit = unit.replace("$", "").replace("\\mu ", "µ").replace("\\mu", "µ")
    if unit in _FREQ_UNITS:
        return _FREQ_UNITS[unit][0], unit
    return (f"1/{unit}" if unit else "cycles per x unit"), unit


def prepare(x: np.ndarray, y: np.ndarray, x_range: tuple[float, float] | None = None
            ) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Sort, drop NaNs, crop to ``x_range`` and resample onto a uniform grid if
    the spacing is uneven. Returns (x, y, dx, resampled)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x_range is not None:
        lo, hi = sorted(x_range)
        keep = (x >= lo) & (x <= hi)
        x, y = x[keep], y[keep]
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    x, unique = np.unique(x, return_index=True)
    y = y[unique]
    if len(x) < 8:
        raise ValueError("Need at least 8 data points in the chosen range.")
    steps = np.diff(x)
    dx = float(np.median(steps))
    if dx <= 0:
        raise ValueError("The x values must increase.")
    resampled = bool(np.max(np.abs(steps - dx)) > 1e-3 * dx)
    if resampled:
        n = int(round((x[-1] - x[0]) / dx)) + 1
        grid = x[0] + dx * np.arange(n)
        y = np.interp(grid, x, y)
        x = grid
    return x, y, dx, resampled


def _window(settings: FFTSettings):
    if settings.window in WINDOW_PARAMS:
        param = settings.window_param
        if param is None:
            param = WINDOW_PARAMS[settings.window][1]
        return (settings.window, param)
    return settings.window


def _segment_points(settings: FFTSettings, n: int, dx: float) -> int:
    if settings.method == "single" or not settings.segment:
        return n
    return int(np.clip(round(settings.segment / dx), 8, n))


def spectrum(x, y, settings: FFTSettings, label: str = "",
             x_range: tuple[float, float] | None = None) -> SpectrumResult:
    """Amplitude spectrum (same units as y) using the chosen window and method."""
    x, y, dx, resampled = prepare(x, y, x_range)
    n = len(y)
    fs = 1.0 / dx
    seg = _segment_points(settings, n, dx)
    nfft = max(seg, int(2 ** np.ceil(np.log2(seg * max(1, settings.zero_pad)))))
    win = _window(settings)
    noverlap = int(seg * np.clip(settings.overlap, 0, 95) / 100)

    result = SpectrumResult(label, x, y, dx, resampled, np.array([]), np.array([]),
                            segment_points=seg)
    if settings.method == "single":
        data = signal.detrend(y, type=settings.detrend) if settings.detrend else y - 0
        w = signal.get_window(win, n)
        spec = np.fft.rfft(data * w, nfft)
        result.frequency = np.fft.rfftfreq(nfft, dx)
        # Amplitude-correct: a sine of amplitude A gives a peak of height A.
        result.amplitude = 2 * np.abs(spec) / np.sum(w)
    elif settings.method == "welch":
        f, pxx = signal.welch(y, fs=fs, window=win, nperseg=seg, noverlap=noverlap, nfft=nfft,
                              detrend=settings.detrend, scaling="spectrum")
        result.frequency, result.amplitude = f, np.sqrt(2 * pxx)
        result.n_segments = 1 + max(0, (n - seg) // max(1, seg - noverlap))
    else:
        f, t, sxx = signal.spectrogram(y, fs=fs, window=win, nperseg=seg, noverlap=noverlap,
                                       nfft=nfft, detrend=settings.detrend, scaling="spectrum",
                                       mode="psd")
        amp = np.sqrt(2 * sxx)
        result.frequency, result.amplitude = f, amp.mean(axis=1)
        result.seg_centers, result.spectrogram = x[0] + t, amp
        result.n_segments = amp.shape[1]
    if len(result.amplitude):
        result.amplitude[0] = result.amplitude[0] / 2  # DC term is not doubled
    result.peaks = find_peaks(result.frequency, result.amplitude, settings)
    return result


def find_peaks(freq: np.ndarray, amp: np.ndarray, settings: FFTSettings) -> list[Peak]:
    """Strongest spectral peaks (DC excluded), refined between bins by fitting a
    parabola to the log-amplitude of the three highest points."""
    mask = freq > 0
    if settings.f_min is not None:
        mask &= freq >= settings.f_min
    if settings.f_max is not None:
        mask &= freq <= settings.f_max
    idx = np.flatnonzero(mask)
    if len(idx) < 3:
        return []
    sub = amp[idx]
    top = float(sub.max())
    if top <= 0:
        return []
    found, props = signal.find_peaks(sub, height=top * settings.min_height / 100,
                                     prominence=top * settings.min_height / 200)
    order = np.argsort(props["peak_heights"])[::-1][: settings.n_peaks]
    df = freq[1] - freq[0]
    peaks = []
    for k in found[order]:
        i = idx[k]
        f, a = freq[i], amp[i]
        if 0 < i < len(amp) - 1 and min(amp[i - 1:i + 2]) > 0:
            l, c, r = np.log(amp[i - 1:i + 2])
            denom = l - 2 * c + r
            if denom < 0:
                delta = 0.5 * (l - r) / denom
                f = freq[i] + delta * df
                a = float(np.exp(c - 0.25 * (l - r) * delta))
        peaks.append(Peak(float(f), float(a), 0.0))
    # Interpolated peaks can be a little higher than the highest bin.
    top = max([top, *(p.amplitude for p in peaks)])
    for p in peaks:
        p.relative = p.amplitude / top
    return peaks
