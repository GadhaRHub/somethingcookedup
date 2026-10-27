# Ported unchanged (except for removing the matplotlib plotting helper) from
# twite.py supplied with the Extracting_echos notebooks. The app draws the
# results itself.
"""
TWITE - Two-Way Identification of Trailing Echoes
=================================================

Locates the first echo and its trailing harmonic train (echo 1, 2, 3, ...) in a
single background-removed pump-probe trace.

Two independent detectors are run and merged:

  A. amplitude - threshold crossings above `pct` of the trace peak, clustered into
                 excursions, required to exceed `noise_mult` x the late-time noise
                 floor, and to fall within `tol_harmonic` of k*t1.
  B. shape     - zero-normalised cross-correlation (ZNCC) against the echo-1
                 waveform, searched only within +/- `search_half` of k*t1 and
                 accepted within +/- `target_tol` of it, above `corr_min`.

Detector A is amplitude-sensitive but shape-blind; detector B is shape-sensitive
but amplitude-blind (ZNCC is scale-invariant), so later, decayed echoes that A
misses are typically still recovered by B.

Echo 1 itself is refined by an FFT low-pass reconstruction, which locates the
true pulse peak rather than its leading edge.

Only the harmonic train (k*t1) is reported; unrelated features are ignored.

Usage
-----
    from twite import twite, TwiteParams

    p = TwiteParams(t_ignore=50.0, noise_mult=2.5)      # override what you need
    result = twite(t, y, p)

    result["echoes"]    # {1: 60.4, 2: 119.2, 3: 179.3, 4: 235.2}
    result["methods"]   # {1: 'amplitude', 2: 'both', ...}

    twite(t, y, p, plot=True, label="8 mm")             # same, plus a marked-up plot

Defaults were tuned on one 15-spot line scan; treat them as a starting point.
On a new sample, revisit `t_ignore`, `quiet_window` and `noise_mult` first.

------------------------------------------------------------------------------
SIGNAL INPUT - read outside this module
------------------------------------------------------------------------------
twite() takes plain arrays (t, y) and does no file reading itself, on purpose:
loading and pre-processing the signal is the caller's job, done in the notebook
cell, so you stay in control of exactly what gets passed in (which channel,
which spot, background-removed or raw, etc.) and can swap datasets freely
without touching this file. A typical notebook cell looks like:

    import scipy.io as sio
    m = sio.loadmat("matlab.mat")
    t = m["delays"].ravel() / 1e3                       # fs -> ps
    Y = m["measured_array_ExtLockin_exp"]                # (time, spot)
    t_sig = t[-Y.shape[0]:]                              # align to Y's length
    y_sig = Y[:, spot_index]                              # pick one spot

    from twite import twite, TwiteParams
    result = twite(t_sig, y_sig, TwiteParams())

------------------------------------------------------------------------------
TUNABLE PARAMETERS - quick reference (see TwiteParams below for full detail)
------------------------------------------------------------------------------
  Search region
    t_ignore       float        ignore delays before this (early oscillations)
    quiet_window   (lo, hi)     echo-free window used to measure the noise floor
    k_max          int | None   highest echo order to search for (None = auto)

  Detector A - amplitude (threshold-crossing)
    pct            float        candidate threshold, fraction of the trace peak
    gap            float        merge threshold-crossings closer than this (ps)
    noise_mult     float        candidate must exceed noise_mult x noise floor
    tol_harmonic   float        matching tolerance around k*t1 (ps)

  Detector B - shape (ZNCC template match)
    template_half  float        half-width of the echo-1 template (ps)
    search_half    float        radius searched around k*t1 (ps)
    target_tol     float        accepted offset from k*t1 for a shape match (ps)
    corr_min       float        minimum |correlation| to accept a shape match

  Echo-1 refinement
    lowpass_mult   float        FFT low-pass cutoff, as a multiple of 1/t1_guess
------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any

import numpy as np

__all__ = ["TwiteParams", "twite", "zncc"]


# ----------------------------------------------------------------------------
# parameters
# ----------------------------------------------------------------------------

@dataclass
class TwiteParams:
    """Tunable parameters for :func:`twite`. All times are in the same unit as `t`."""

    # --- search region -------------------------------------------------------
    t_ignore: float = 50.0
    """Ignore everything before this delay (early non-echo oscillations)."""

    quiet_window: Tuple[float, float] = (150.0, 250.0)
    """(lo, hi) echo-free region used to estimate the noise floor."""

    k_max: Optional[int] = None
    """Highest echo order to look for. None = as many as the record allows."""

    # --- detector A: amplitude ----------------------------------------------
    pct: float = 0.09
    """Candidate threshold, as a fraction of the trace's own peak |amplitude|."""

    gap: float = 5.0
    """Threshold-crossing runs closer together than this are merged into one echo."""

    noise_mult: float = 2.5
    """An amplitude candidate must exceed this multiple of the noise floor."""

    tol_harmonic: float = 20.0
    """Matching tolerance around k*t1 for the amplitude detector."""

    # --- detector B: shape ---------------------------------------------------
    template_half: float = 15.0
    """Half-width of the echo-1 template (full template window = 2x this)."""

    search_half: float = 50.0
    """Radius around k*t1 searched by the shape detector."""

    target_tol: float = 10.0
    """Accepted offset from k*t1 for a shape match."""

    corr_min: float = 0.60
    """Minimum |ZNCC| for a shape match to be accepted."""

    # --- echo-1 refinement ---------------------------------------------------
    lowpass_mult: float = 3.0
    """FFT low-pass cutoff, as a multiple of 1/t1_guess.

    Low-pass (not band-pass): an echo is a localised pulse and needs a broad band
    of frequencies to reconstruct sharply.
    """

    def as_dict(self) -> Dict[str, Any]:
        """Parameters as a plain dict, e.g. to record alongside a result."""
        return asdict(self)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def zncc(sig: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Zero-normalised cross-correlation at every integer shift.

    Returns an array of length ``len(sig) - len(template) + 1``; entry *s* is the
    correlation coefficient (-1..+1) between ``template`` and ``sig[s:s+len(template)]``.
    Amplitude-invariant: it compares shape only, so a decayed echo still scores
    highly. A negative score means an inverted (mirror-image) match.
    """
    tpl = np.asarray(template, float)
    tpl = tpl - tpl.mean()
    tpl_norm = np.linalg.norm(tpl)
    n_shifts = len(sig) - len(tpl) + 1
    out = np.full(max(n_shifts, 0), np.nan)
    if n_shifts < 1 or tpl_norm < 1e-12:
        return out
    for s in range(n_shifts):
        seg = sig[s:s + len(tpl)]
        seg_c = seg - seg.mean()
        seg_norm = np.linalg.norm(seg_c)
        if seg_norm > 1e-12:
            out[s] = float(np.dot(seg_c, tpl) / (seg_norm * tpl_norm))
    return out


def _empty_result(noise_std: float = np.nan) -> Dict[str, Any]:
    return {"t1": None, "echoes": {}, "corr": {}, "methods": {},
            "noise_std": float(noise_std), "candidates": []}


def _candidate_excursions(t, y, p: TwiteParams):
    """Threshold crossings above `pct` of peak, clustered into excursions.

    Returns (times, peak_amplitudes): the midpoint between the first and last
    baseline crossing of each excursion, and the largest |signal| inside it.
    """
    dev = np.abs(y)
    peak = float(np.max(dev))
    if peak <= 0:
        return [], []
    above = (t >= p.t_ignore) & (dev > p.pct * peak)
    if not above.any():
        return [], []

    idx = np.where(above)[0]
    runs = [r for r in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1) if len(r) >= 2]
    if not runs:
        return [], []

    starts = np.array([t[r[0]] for r in runs])
    groups = np.split(np.arange(len(runs)), np.where(np.diff(starts) > p.gap)[0] + 1)

    times, amps = [], []
    for g in groups:
        i0, i1 = runs[g[0]][0], runs[g[-1]][-1]
        times.append(float(0.5 * (t[i0] + t[i1])))
        amps.append(float(dev[i0:i1 + 1].max()))
    return times, amps


def _refine_first_echo(t, y, guess: float, p: TwiteParams) -> Optional[float]:
    """Locate echo 1's true peak via an FFT low-pass reconstruction.

    The raw threshold-crossing midpoint can land on the leading edge of a bipolar
    echo rather than its dominant lobe; low-passing removes high-frequency noise
    while preserving the pulse, so the peak search lands on the real feature.
    """
    dt = t[1] - t[0]
    freqs = np.fft.rfftfreq(t.size, d=dt)
    spec = np.fft.rfft(y - y.mean())
    recon = np.fft.irfft(np.where(freqs < p.lowpass_mult / guess, spec, 0), n=t.size)
    win = (t >= p.t_ignore) & (t <= guess + p.template_half)
    if not win.any():
        return None
    return float(t[win][np.argmax(np.abs(recon[win]))])


# ----------------------------------------------------------------------------
# main entry point
# ----------------------------------------------------------------------------

def twite(t, y, params: Optional[TwiteParams] = None, *,
          plot: bool = False, ax=None, label: Optional[str] = None) -> Dict[str, Any]:
    """Identify the trailing echo train in a single trace.

    Parameters
    ----------
    t, y : array_like
        Delay axis and background-removed signal; 1-D, equal length, `t` uniformly
        spaced and increasing.
    params : TwiteParams, optional
        Tunable parameters. Defaults are used if omitted.
    plot : bool
        Draw the trace with identified echoes marked.
    ax : matplotlib Axes, optional
        Existing axes to draw into; a new figure is made if omitted.
    label : str, optional
        Trace label for the plot title/legend.

    Returns
    -------
    dict
        ``t1``         refined first-echo time, or None if nothing was found
        ``echoes``     {k: time} for every identified echo, including k=1
        ``corr``       {k: zncc} for orders confirmed by the shape detector
        ``methods``    {k: 'amplitude' | 'shape' | 'both'}
        ``noise_std``  late-time noise level used for gating
        ``candidates`` all clustered excursion times considered (diagnostic)

    Notes
    -----
    A trace containing only noise can still yield a ``t1``, because `pct` is
    relative to the trace's own peak: something always crosses it. Require
    ``len(result["echoes"]) > 1`` before trusting a detection, since a genuine
    train gives at least k=2.
    """
    p = params if params is not None else TwiteParams()

    t = np.asarray(t, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if t.size != y.size:
        raise ValueError(f"t and y must have equal length (got {t.size} and {y.size})")
    if t.size < 3:
        raise ValueError("need at least 3 samples")
    if not np.all(np.diff(t) > 0):
        raise ValueError("t must be strictly increasing")

    dt = t[1] - t[0]
    t_end = float(t[-1])

    lo, hi = p.quiet_window
    quiet = (t >= lo) & (t <= hi)
    noise_std = float(y[quiet].std()) if quiet.sum() > 1 else float(y.std())

    # --- candidates ---------------------------------------------------------
    cand_t, cand_amp = _candidate_excursions(t, y, p)
    if not cand_t:
        return _empty_result(noise_std)

    # --- echo 1 -------------------------------------------------------------
    t1 = _refine_first_echo(t, y, cand_t[0], p)
    if t1 is None or t1 <= 0:
        return _empty_result(noise_std)

    k_max = p.k_max if p.k_max is not None else int(np.floor((t_end + p.search_half) / t1))

    # --- detector A: amplitude ---------------------------------------------
    amp_hits: Dict[int, float] = {1: t1}
    used = {t1}
    for k in range(2, k_max + 1):
        target = k * t1
        if target - p.tol_harmonic > t_end:
            break
        cands = [(tc, a) for tc, a in zip(cand_t, cand_amp)
                 if tc not in used
                 and abs(tc - target) <= p.tol_harmonic
                 and a > p.noise_mult * noise_std]
        if cands:
            best = min(cands, key=lambda ca: abs(ca[0] - target))[0]
            amp_hits[k] = best
            used.add(best)

    # --- detector B: shape --------------------------------------------------
    shape_hits: Dict[int, float] = {}
    shape_corr: Dict[int, float] = {}
    n_tpl = int(round(2 * p.template_half / dt)) + 1
    i0 = int(np.searchsorted(t, t1 - p.template_half))
    if i0 >= 0 and i0 + n_tpl <= t.size:
        template = y[i0:i0 + n_tpl]
        for k in range(2, k_max + 1):
            target = k * t1
            if target - p.search_half > t_end:
                break
            w = np.where((t >= target - p.search_half) & (t <= target + p.search_half))[0]
            if w.size < n_tpl:
                continue
            scores = zncc(y[w[0]:w[-1] + 1], template)
            if np.all(np.isnan(scores)):
                continue
            centres = t[w[0]:w[0] + scores.size] + p.template_half
            j = int(np.nanargmax(np.abs(scores)))
            if abs(centres[j] - target) <= p.target_tol and abs(scores[j]) >= p.corr_min:
                shape_hits[k] = float(centres[j])
                shape_corr[k] = float(scores[j])

    # --- merge --------------------------------------------------------------
    echoes: Dict[int, float] = {}
    methods: Dict[int, str] = {}
    for k in sorted(set(amp_hits) | set(shape_hits)):
        in_a, in_s = k in amp_hits, k in shape_hits
        echoes[k] = amp_hits[k] if in_a else shape_hits[k]   # amplitude time preferred
        methods[k] = "both" if (in_a and in_s) else ("amplitude" if in_a else "shape")

    result = {"t1": t1, "echoes": echoes, "corr": shape_corr, "methods": methods,
              "noise_std": noise_std, "candidates": cand_t}

    if plot:
        raise NotImplementedError("plotting is done by the app (csvplot.echo_dialogs)")

    return result
