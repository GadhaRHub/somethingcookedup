"""Glue for the two echo-extraction methods (TWITE and matched-subspace echo
matching) so the app can run them on a single trace or on repeats."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import welch

from . import echo_match as em
from .oscillations import OscillationSettings, find_frequencies
from .twite import TwiteParams, twite


def block_bin(t, y, n: int):
    """Block average of n samples (as in the notebook). n <= 1 returns the input."""
    if n <= 1:
        return np.asarray(t, float), np.asarray(y, float)
    n_use = (len(t) // n) * n
    return (np.asarray(t[:n_use], float).reshape(-1, n).mean(1),
            np.asarray(y[:n_use], float).reshape(-1, n).mean(1))


def run_twite(t, y, params: TwiteParams, n_bin: int = 1) -> dict:
    tb, yb = block_bin(t, y, n_bin)
    result = twite(tb, yb, params)
    result["t"], result["y"] = tb, yb
    return result


# ------------------------------------------------------------ echo matching
@dataclass
class EchoMatchParams:
    noise: str = "quiet"                       # "quiet" (single trace) or "repeats"
    quiet: tuple[float, float] = (420.0, 560.0)
    n_sur: int = 200
    floor: float = 1e-3
    seeds: list[float] = field(default_factory=list)   # empty: take them from TWITE
    half: float = 7.5
    k: int = 1
    max_shift: float = 3.0
    poly: int = 2
    freqs: list[float] = field(default_factory=list)   # cycles per x unit; empty: estimate
    n_auto_freqs: int = 2
    first_order: bool = False
    alpha: float = 0.01
    max_iter: int = 60
    t_min: float | None = None
    t_max: float | None = None
    min_sep: float = 4.0
    n_boot: int = 0
    tol: float = 2.0
    seed: int = 0


def quiet_surrogates(t, y, quiet, n_sur: int, seed: int = 0) -> np.ndarray:
    """Noise-only traces for a single averaged trace: Gaussian noise with the
    spectrum (and level) of an echo-free 'quiet' stretch of the trace."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    lo, hi = sorted(quiet)
    sel = (t >= lo) & (t <= hi)
    if sel.sum() < 16:
        raise ValueError(f"The quiet window {lo:g}–{hi:g} has fewer than 16 points.")
    seg = y[sel]
    seg = seg - np.polyval(np.polyfit(t[sel], seg, 1), t[sel])
    dt = t[1] - t[0]
    f_q, psd = welch(seg, fs=1 / dt, nperseg=min(256, seg.size))
    n = t.size
    amp = np.sqrt(np.interp(np.fft.rfftfreq(n, dt), f_q, psd))
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((n_sur, n))
    out = np.fft.irfft(np.fft.rfft(white, axis=1) * amp, n=n, axis=1)
    return out * (seg.std() / out.std(axis=1, keepdims=True))


def auto_frequencies(t, y, n: int) -> list[float]:
    """The strongest oscillation frequencies of the whole trace."""
    s = OscillationSettings(snr_min=5, rel_min=0.1, max_freqs=n, n_fft=2 ** 15)
    return find_frequencies(np.asarray(t, float), np.asarray(y, float), s)


def run_echo_match(t, y, p: EchoMatchParams, reps=None, background=None,
                   twite_params: TwiteParams | None = None) -> dict:
    """Matched-subspace echo detection (echo_match.py) on the uniform trace (t, y).

    reps: (n_t, n_rep) individual repeats (raw) for p.noise == "repeats";
    background: what was subtracted from their mean to get y (for bootstrap).
    """
    em.WARNINGS.clear()
    t, y = np.asarray(t, float), np.asarray(y, float)
    dt = t[1] - t[0]
    notes = []

    if p.noise == "repeats":
        if reps is None or reps.shape[1] < 4:
            raise ValueError("Noise from repeats needs at least 4 lines shown on the plot.")
        sur = em.half_split_surrogates(reps, n_sur=p.n_sur, seed=p.seed)
    else:
        sur = quiet_surrogates(t, y, p.quiet, p.n_sur, p.seed)
    whiten = em.make_whitener(sur, floor=p.floor)
    xw, surw = whiten(y), whiten(sur)

    freqs = list(p.freqs)
    if not freqs and p.n_auto_freqs > 0:
        freqs = auto_frequencies(t, y, p.n_auto_freqs)
        notes.append("oscillation frequencies estimated from the trace")
    L = 2 * int(round(p.half / dt)) + 1
    P = em.nuisance_projector(L, dt, freqs, poly=p.poly, first_order=p.first_order)

    seeds = list(p.seeds)
    if not seeds:
        tw = twite(t, y, twite_params or TwiteParams())
        seeds = sorted(tw["echoes"].values())
        notes.append(f"seed times from TWITE: {', '.join(f'{s:.4g}' for s in seeds) or 'none'}")
    if not seeds:
        raise ValueError("No seed echo times: enter some, or adjust TWITE so it finds echoes.")
    D, S = em.learn_templates(xw, t, seeds, p.half, k=min(p.k, len(seeds)),
                              max_shift_ps=p.max_shift, P=P)

    events, thr, cleaned = em.detect(t, xw, surw, D, P, alpha=p.alpha, max_iter=p.max_iter,
                                     t_min=p.t_min, t_max=p.t_max, min_sep_ps=p.min_sep)

    # z(t) before any subtraction, for display only (a subset of the surrogates is
    # enough for that; the detection threshold above uses all of them).
    Qd, _ = np.linalg.qr(P @ D)
    null = np.array([em._stat(s, Qd)[0] for s in surw[:50]])
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # edges have no statistic
        rms = np.sqrt(np.nanmean(null ** 2, axis=0))
        z0 = em._stat(xw, Qd)[0] / rms

    boot = None
    if p.n_boot > 0 and events:
        if p.noise != "repeats" or reps is None:
            notes.append("bootstrap needs repeats; skipped")
        else:
            bg = 0.0 if background is None else np.asarray(background)[:, None]
            boot = em.bootstrap_times(t, reps - bg, whiten, D, P, events, n_boot=p.n_boot,
                                      tol_ps=p.tol, surw=surw, alpha=p.alpha, max_iter=p.max_iter,
                                      t_min=p.t_min, t_max=p.t_max, min_sep_ps=p.min_sep)
    return dict(events=events, threshold=float(thr), z=z0, whitened=xw, cleaned=cleaned,
                templates=D, singular=S, freqs=freqs, seeds=seeds, boot=boot,
                warnings=list(em.WARNINGS) + notes)
