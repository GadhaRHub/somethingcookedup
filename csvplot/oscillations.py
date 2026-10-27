"""Removal of oscillations (e.g. Brillouin oscillations) by windowed sinusoid fits.

In each window along x, the strongest spectral peaks are found (windowed FFT,
local offset and slope removed first). Sinusoids at those frequencies, plus a
local offset and slope, are fitted by least squares. Overlapping windows are
cross-faded with trapezoid weights into one continuous curve. Subtracting the
oscillation part from a background-removed trace leaves the echoes.

This follows the "identified oscillations" procedure of the Extracting_echos
notebook, except that the noise level comes from the spectrum itself (the
median amplitude in the search band), because a CSV file usually holds a
single averaged trace rather than the individual repeats.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks


@dataclass
class OscillationSettings:
    window: float = 50.0          # window length (x units)
    overlap: float = 20.0         # overlap between neighbouring windows (x units)
    f_min: float | None = None    # lowest frequency searched (cycles per x unit); None = 1/window
    f_max: float | None = None    # highest frequency searched; None = Nyquist
    snr_min: float = 5.0          # peak must exceed this × the median spectral level
    rel_min: float = 0.3          # ...and this fraction of the strongest peak in the window
    max_freqs: int = 4            # at most this many frequencies per window
    bands: list[tuple[float, float]] = field(default_factory=list)  # only remove these bands
    n_fft: int = 4096


@dataclass
class OscillationResult:
    curve: np.ndarray             # fitted oscillations (without the local offset/slope)
    windows: list[tuple[float, list[float]]]   # (window start, frequencies found)


def _spectrum(v, w, n_fft):
    return np.abs(np.fft.rfft(v * w, n=n_fft)) / w.sum() * 2


def _detrend(tw, v):
    return v - np.polyval(np.polyfit(tw, v, 1), tw)


def window_starts(t0: float, t1: float, window: float, overlap: float, dt: float) -> np.ndarray:
    step = max(window - overlap, dt)
    starts = np.arange(t0, t1 - window + dt, step)
    if starts.size == 0:
        return np.array([t0])
    if starts[-1] + window < t1 - dt:   # one last window flush with the end
        starts = np.append(starts, t1 - window)
    return starts


def find_frequencies(tw, yw, s: OscillationSettings) -> list[float]:
    dt = tw[1] - tw[0]
    w = np.hanning(tw.size)
    a = _spectrum(_detrend(tw, yw), w, s.n_fft)
    f = np.fft.rfftfreq(s.n_fft, d=dt)
    f_lo = s.f_min if s.f_min is not None else 1.0 / (tw[-1] - tw[0])   # ≥ 1 cycle in the window
    f_hi = s.f_max if s.f_max is not None else f[-1]
    band = (f >= f_lo) & (f <= f_hi)
    if not band.any():
        return []
    noise = np.median(a[band])
    peaks, _ = find_peaks(a)
    peaks = peaks[band[peaks]]
    if not peaks.size:
        return []
    peaks = peaks[(a[peaks] > s.snr_min * noise) & (a[peaks] >= s.rel_min * a[peaks].max())]
    peaks = peaks[np.argsort(a[peaks])[::-1]][: s.max_freqs]
    freqs = [float(f[k]) for k in peaks]
    if s.bands:
        freqs = [fq for fq in freqs if any(lo <= fq <= hi for lo, hi in s.bands)]
    return freqs


def remove_oscillations(t, y, s: OscillationSettings) -> OscillationResult:
    """Fitted oscillation curve for (t, y); ``y - curve`` has them removed.
    ``t`` must be increasing and roughly uniform."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    dt = float(np.median(np.diff(t)))
    window = min(s.window, t[-1] - t[0])
    overlap = float(np.clip(s.overlap, 0, 0.9 * window))
    starts = window_starts(t[0], t[-1], window, overlap, dt)

    total_w = np.zeros_like(t)
    osc = np.zeros_like(t)
    found = []
    for t0 in starts:
        sel = (t >= t0) & (t <= t0 + window)
        if sel.sum() < 8:
            continue
        tw, yw = t[sel], y[sel]
        freqs = find_frequencies(tw, yw, s)
        found.append((float(t0), freqs))
        # Trapezoid cross-fade: ramps over the overlap at each end.
        if overlap > 0:
            up = np.clip((tw - t0) / overlap, 0, 1)
            down = np.clip((t0 + window - tw) / overlap, 0, 1)
            wt = np.minimum(up, down) + 1e-6
        else:
            wt = np.ones_like(tw)
        total_w[sel] += wt
        if not freqs:
            continue
        cols = [np.ones_like(tw), tw - tw.mean()]
        for fq in freqs:
            om = 2 * np.pi * fq
            cols += [np.cos(om * tw), np.sin(om * tw)]
        x = np.column_stack(cols)
        coef, *_ = np.linalg.lstsq(x, yw, rcond=None)
        osc[sel] += wt * (x[:, 2:] @ coef[2:])
    with np.errstate(invalid="ignore", divide="ignore"):
        curve = np.where(total_w > 0, osc / np.where(total_w > 0, total_w, 1), 0.0)
    return OscillationResult(curve, found)
