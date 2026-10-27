"""Exponential background models.

``double_exponential`` is a port of ``minus_exp_fun_single3`` from
diffR_functions_mine.py (the background removal used in the
Extracting_echos notebooks): f(x) = a·exp(b·x) + c·exp(d·x), fitted on a
normalised x axis from several starting points (a Prony / linear-prediction
estimate plus fixed guesses), keeping the lowest-cost fit. With ``force_decay``
the rates b, d are kept ≤ 0. If every fit fails, the mean is used instead,
as in the original.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


def _model(x, a, b, c, d):
    return a * np.exp(np.clip(b * x, -700, 700)) + c * np.exp(np.clip(d * x, -700, 700))


def double_exponential(time, signal, force_decay: bool = True, fit_mask=None):
    """Fit a·exp(b·t) + c·exp(d·t) to (time, signal).

    Returns (background evaluated on ``time``, params (a, b, c, d) in the
    original time units, or None if the fit failed and the mean was used).
    ``fit_mask`` optionally selects the points used for fitting (e.g. to leave
    marked echoes out); the background is still evaluated everywhere.
    """
    time = np.asarray(time, float).ravel()
    signal = np.asarray(signal, float).ravel()
    mask = np.isfinite(time) & np.isfinite(signal)
    if fit_mask is not None:
        mask &= np.asarray(fit_mask, bool)
    mean_value = float(np.mean(signal[mask])) if mask.any() else 0.0
    fallback = (np.full_like(time, mean_value), None)
    if mask.sum() < 4:
        return fallback

    t, y = time[mask], signal[mask]
    t_min, t_max = float(t.min()), float(t.max())
    span = t_max - t_min
    if not (span > 0 and np.isfinite(span)):
        return fallback
    scale = span
    x = (t - t_min) / scale

    def prony_seed():
        dt = float(np.median(np.diff(t)))
        a_mat = np.column_stack((y[1:-1], y[:-2]))
        c1, c2 = np.linalg.lstsq(a_mat, y[2:], rcond=None)[0]
        roots = np.roots([1.0, -c1, -c2])
        with np.errstate(divide="ignore", invalid="ignore"):
            rates = np.real(np.log(roots.astype(complex)) / dt)
        if not np.all(np.isfinite(rates)):
            raise ValueError("non-finite Prony rates")
        b_orig, d_orig = sorted(rates, reverse=True)
        if force_decay:
            if b_orig >= 0:
                b_orig = -abs(b_orig) if abs(b_orig) > 1e-12 else -1.0 / span
            if d_orig >= 0:
                d_orig = -abs(d_orig) if abs(d_orig) > 1e-12 else -10.0 / span
        b_n, d_n = b_orig * scale, d_orig * scale
        if np.isclose(b_n, d_n, rtol=1e-6, atol=1e-12):
            d_n = 2.0 * b_n if b_n != 0 else -1.0
        m = np.column_stack((np.exp(np.clip(b_n * x, -700, 700)), np.exp(np.clip(d_n * x, -700, 700))))
        a_n, c_n = np.linalg.lstsq(m, y, rcond=None)[0]
        return [a_n, b_n, c_n, d_n]

    tail = max(1, y.size // 10)
    baseline = float(np.mean(y[-tail:]))
    amplitude = float(y[0] - baseline)
    if not np.isfinite(amplitude) or amplitude == 0:
        amplitude = float(np.ptp(y)) or 1.0

    guesses = []
    try:
        guesses.append(prony_seed())
    except Exception:  # noqa: BLE001 - the fixed guesses below still apply
        pass
    guesses += [
        [amplitude, -1.0, baseline, -0.1],
        [amplitude, -5.0, baseline, -0.5],
        [amplitude, -0.3, baseline, -3.0],
        [amplitude, -10.0, baseline, -1.0],
        [amplitude, -1.0, 0.0, -0.1],
        [float(np.mean(y)), -1.0, float(np.mean(y)), -0.1],
    ]
    if force_decay:
        lower, upper = [-np.inf, -1000.0, -np.inf, -1000.0], [np.inf, 0.0, np.inf, 0.0]
    else:
        lower, upper = [-np.inf] * 4, [np.inf] * 4

    def prepare(p0):
        p0 = np.asarray(p0, float)
        if p0.size != 4 or not np.all(np.isfinite(p0)):
            return None
        if force_decay:
            if p0[1] >= 0:
                p0[1] = -abs(p0[1]) if abs(p0[1]) > 1e-12 else -1.0
            if p0[3] >= 0:
                p0[3] = -abs(p0[3]) if abs(p0[3]) > 1e-12 else -0.1
            if np.isclose(p0[1], p0[3], rtol=1e-6, atol=1e-12):
                p0[3] = 2.0 * p0[1] if p0[1] != 0 else -1.0
        for i in range(4):
            if np.isfinite(lower[i]) and p0[i] <= lower[i]:
                p0[i] = lower[i] + 1e-9 * max(1.0, abs(lower[i]))
            if np.isfinite(upper[i]) and p0[i] >= upper[i]:
                p0[i] = upper[i] - 1e-9 * max(1.0, abs(upper[i]))
        return p0

    kwargs = {"bounds": (lower, upper)} if force_decay else {}
    # Speed-up (not in the original): compare the starting points on a thinned
    # copy of long traces, then refine the winner once on all points.
    step = max(1, x.size // 2000)
    xs, ys = x[::step], y[::step]
    best, best_cost = None, np.inf
    for p0 in guesses:
        p0 = prepare(p0)
        if p0 is None:
            continue
        try:
            popt, _ = curve_fit(_model, xs, ys, p0=p0, maxfev=20000, **kwargs)
        except Exception:  # noqa: BLE001 - try the next starting point
            continue
        cost = float(np.sum((ys - _model(xs, *popt)) ** 2))
        if np.isfinite(cost) and cost < best_cost:
            best, best_cost = popt, cost
    if best is None:
        return fallback
    if step > 1:
        try:
            refined, _ = curve_fit(_model, x, y, p0=prepare(best), maxfev=20000, **kwargs)
            if np.sum((y - _model(x, *refined)) ** 2) <= np.sum((y - _model(x, *best)) ** 2):
                best = refined
        except Exception:  # noqa: BLE001 - keep the thinned-data fit
            pass

    fit = _model((time - t_min) / scale, *best)
    a_n, b_n, c_n, d_n = best
    b_o, d_o = float(b_n / scale), float(d_n / scale)
    params = (float(a_n * np.exp(np.clip(-b_o * t_min, -700, 700))), b_o,
              float(c_n * np.exp(np.clip(-d_o * t_min, -700, 700))), d_o)
    if not (np.all(np.isfinite(fit[np.isfinite(time)])) and np.all(np.isfinite(params))):
        return fallback
    return fit, params
