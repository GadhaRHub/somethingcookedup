"""
Ported from echo_match.py (Extracting_echos). Changes: frequencies are in
cycles per x unit (the original used GHz with t in ps), warnings are
collected in WARNINGS instead of printed.

echo_match.py — data-driven echo detection by pattern matching (no forward model).

Pipeline
    1. half_split_surrogates : pure-noise traces from the repeats (same noise as the mean)
    2. make_whitener         : whitening filter from the surrogate noise spectrum (red noise -> white)
    3. learn_templates       : echo shape subspace from aligned snippets of the data itself (SVD)
    4. detect                : matched-subspace detector with local Brillouin + baseline projected out,
                               iterative detect-and-subtract, family-wise threshold from the surrogates
    5. bootstrap_times       : timing uncertainty by resampling repeats

All matching happens in the whitened domain. Inputs: t (ps, uniform), y (mean trace, e.g. y_sig),
reps (n_t x n_rep, same time grid as y).
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# Messages that the original printed; the app shows them in the window.
WARNINGS: list[str] = []


# ---------------------------------------------------------------- noise
def half_split_surrogates(reps, n_sur=200, seed=0):
    """(mean of half A - mean of half B)/2 over random splits.
    Signal cancels exactly; noise variance ~ that of the full mean (sigma^2/n_rep)."""
    rng = np.random.default_rng(seed)
    n = reps.shape[1]
    m = n // 2
    out = np.empty((n_sur, reps.shape[0]))
    for s in range(n_sur):
        p = rng.permutation(n)
        out[s] = 0.5 * (reps[:, p[:m]].mean(1) - reps[:, p[m:2 * m]].mean(1))
    return out


def make_whitener(sur, floor=1e-3):
    """Whitening filter W(f) = 1/sqrt(noise PSD). Mirror padding avoids wrap-around edge spikes.
    An LTI filter maps sinusoids to sinusoids and low-order polynomials to low-order polynomials,
    so the local nuisance model in `detect` stays valid after whitening."""
    N = sur.shape[1]
    ext = lambda x: np.concatenate([x, x[..., ::-1]], axis=-1)
    psd = (np.abs(np.fft.rfft(ext(sur), axis=-1)) ** 2).mean(0)
    W = 1.0 / np.sqrt(psd + floor * psd.max())

    def whiten(x):
        return np.fft.irfft(np.fft.rfft(ext(x), axis=-1) * W, n=2 * N, axis=-1)[..., :N]
    return whiten


# ---------------------------------------------------------------- nuisance + templates
def nuisance_projector(L, dt, freqs_ghz, poly=2, first_order=False):
    """P = I - Q Q^T, removing a local polynomial baseline and sinusoids at the Brillouin frequencies.
    Shifting a sinusoid only rotates its cos/sin pair, so ONE projector serves every window position.
    first_order=True also removes tau*cos and tau*sin (first-order frequency error / amplitude change).
    It suppresses Brillouin leakage but also removes a lot of echo energy (detection z dropped ~3x in
    tests), so it is off by default. Better: get the frequencies right with estimate_freq."""
    tau = (np.arange(L) - L // 2) * dt
    taun = tau / np.abs(tau).max()
    cols = [taun ** p for p in range(poly + 1)]
    for f in freqs_ghz:
        w = 2 * np.pi * f * tau
        cols += [np.cos(w), np.sin(w)]
        if first_order:
            cols += [taun * np.cos(w), taun * np.sin(w)]
    Q, _ = np.linalg.qr(np.column_stack(cols))
    return np.eye(L) - Q @ Q.T


def estimate_freq(t, y, t_lo, t_hi, f_lo, f_hi, pad=16, remove=()):
    """Brillouin frequency (GHz) from one long stretch where it dominates: zero-padded, Hann-windowed
    FFT peak in [f_lo, f_hi], refined by a parabola through the peak bin.
    remove: frequencies (GHz) already estimated that are also present in this stretch; they are
    least-squares fitted and removed first, so their spectral leakage cannot pull the peak."""
    sel = (t >= t_lo) & (t <= t_hi)
    tw, yw_ = t[sel], y[sel]
    u = (tw - tw.mean()) / np.ptp(tw)
    cols = [np.ones_like(u), u, u ** 2]
    for f0 in remove:
        cols += [np.cos(2 * np.pi * f0 * tw), np.sin(2 * np.pi * f0 * tw)]
    B = np.column_stack(cols)
    yw_ = (yw_ - B @ np.linalg.lstsq(B, yw_, rcond=None)[0]) * np.hanning(tw.size)
    n = pad * tw.size
    A = np.abs(np.fft.rfft(yw_, n=n))
    f = np.fft.rfftfreq(n, d=tw[1] - tw[0])
    band = np.flatnonzero((f >= f_lo) & (f <= f_hi))
    k = band[np.argmax(A[band])]
    if k in (band[0], band[-1]):
        WARNINGS.append(f"estimate_freq: peak at the edge of [{f_lo:g}, {f_hi:g}] -> no clear "
                        "oscillation peak in this stretch; check the spectrum or the range.")
    a, b, c = np.log(A[k - 1:k + 2])
    d = np.clip(0.5 * (a - c) / (a - 2 * b + c), -0.5, 0.5) if (a - 2 * b + c) < 0 else 0.0
    return f[k] + d * (f[1] - f[0])


def _snip(x, i, h):
    return x[i - h: i + h + 1]


def learn_templates(xw, t, seed_times, half_ps, k=1, max_shift_ps=3.0, P=None):
    """Echo-shape subspace from the data: cut whitened snippets at seed_times, align each to the
    first (shift + sign), SVD -> top-k shapes. Returns D (L x k), singular values.

    Pass P (the nuisance projector) to ALIGN on the projected snippets. Otherwise the alignment locks
    onto whatever dominates the window -- often the Brillouin oscillation, not the echo -- and the
    'template' becomes a piece of sinusoid. The template itself is still built from the unprojected
    snippets: with several seeds the Brillouin under each echo has a different phase and averages out,
    while the full echo shape (including its content at the Brillouin frequencies) is kept, which is
    what lets `detect` subtract echoes cleanly. Use >= 4 seeds where possible."""
    dt = t[1] - t[0]
    h, s = int(round(half_ps / dt)), int(round(max_shift_ps / dt))
    proj = (lambda v: P @ v) if P is not None else (lambda v: v)
    idx = [np.argmin(np.abs(t - tc)) for tc in seed_times]
    idx = [i for i in idx if h + s <= i < t.size - h - s]
    ref = proj(_snip(xw, idx[0], h))
    rows = []
    for i in idx:
        xc = [proj(_snip(xw, i + d, h)) @ ref for d in range(-s, s + 1)]
        j = int(np.argmax(np.abs(xc)))
        seg = np.sign(xc[j]) * _snip(xw, i + j - s, h)
        rows.append(seg / np.linalg.norm(seg))
    U, S, _ = np.linalg.svd(np.array(rows).T, full_matrices=False)
    return U[:, :k], S


# ---------------------------------------------------------------- detector
def _stat(x, Qd):
    """Matched-subspace statistic at every centre: ||Qd^T x_window|| and the coefficients."""
    L = Qd.shape[0]
    h = L // 2
    c = sliding_window_view(x, L) @ Qd
    stat = np.full(x.size, np.nan)
    coef = np.full((x.size, Qd.shape[1]), np.nan)
    stat[h:h + c.shape[0]] = np.sqrt((c ** 2).sum(1))
    coef[h:h + c.shape[0]] = c
    return stat, coef


def detect(t, xw, surw, D, P, alpha=0.01, max_iter=60, t_min=None, t_max=None, min_sep_ps=4.0):
    """Iterative detect-and-subtract on the whitened mean xw.
    z(t) = stat(t) / rms of the same statistic on the whitened surrogates at that t
           (handles noise that changes with delay).
    Threshold = (1-alpha) quantile of max_t z over surrogates -> family-wise false-alarm rate alpha
           over all positions and all template shapes (the look-elsewhere effect is built in).
    min_sep_ps: after an event is accepted, no new event within +-min_sep_ps. Subtraction is never
           perfect for strong echoes; without this the leftover splits into a cascade of fake events."""
    Qd, R = np.linalg.qr(P @ D)            # orthonormal basis of the projected template subspace
    retained = np.linalg.norm(P @ D, axis=0) / np.linalg.norm(D, axis=0)
    if np.any(retained < 0.5):
        WARNINGS.append(f"templates keep only {np.round(retained, 2)} of their norm after removing "
                        "baseline + oscillations. The template is mostly oscillation/trend, not echo: "
                        "check seeds, frequencies and half-width.")
    L = Qd.shape[0]
    h = L // 2
    keep = np.ones(t.size, bool)
    if t_min is not None: keep &= t >= t_min
    if t_max is not None: keep &= t <= t_max

    null = np.array([_stat(s, Qd)[0] for s in surw])
    with np.errstate(invalid='ignore'), __import__('warnings').catch_warnings():
        __import__('warnings').simplefilter('ignore', RuntimeWarning)
        rms = np.sqrt(np.nanmean(null ** 2, axis=0))
    znull = null / rms
    thr = np.quantile(np.nanmax(np.where(keep, znull, np.nan), axis=1), 1 - alpha)

    x = xw.copy()
    events = []
    sep = int(round(min_sep_ps / (t[1] - t[0])))
    for _ in range(max_iter):
        stat, coef = _stat(x, Qd)
        z = np.where(keep, stat / rms, np.nan)
        if np.all(np.isnan(z)):
            break
        i = int(np.nanargmax(z))
        if z[i] < thr:
            break
        # sub-sample peak position (parabola through 3 points)
        if 0 < i < t.size - 1 and np.isfinite(z[i - 1:i + 2]).all():
            a, b, c3 = z[i - 1], z[i], z[i + 1]
            den = a - 2 * b + c3
            d = 0.5 * (a - c3) / den if den != 0 else 0.0
        else:
            d = 0.0
        events.append(dict(t=t[i] + d * (t[1] - t[0]), z=z[i], sign=int(np.sign(coef[i, 0])),
                           coef=coef[i].copy()))
        # Two ways to remove the echo:
        #  full : joint LS of [D, nuisance] gives a = R^-1 c; subtract D a. Removes the echo completely,
        #         but if D is nearly inside the nuisance span (small R) a blows up and the subtraction
        #         INJECTS energy -> runaway cascade of ever-larger fake events.
        #  proj : subtract Qd c (orthogonal projection). Can never add energy, but leaves the echo's
        #         Brillouin-frequency content behind (small leakage, handled by min_sep / near_strong).
        # Rule: use whichever leaves less energy in a +-L neighbourhood. A subtraction may never add energy.
        a = np.linalg.solve(R, coef[i])
        lo, hi = max(i - L, 0), min(i + L + 1, x.size)
        full = x.copy(); full[i - h:i + h + 1] -= D @ a
        proj = x.copy(); proj[i - h:i + h + 1] -= Qd @ coef[i]
        if np.sum(full[lo:hi] ** 2) <= np.sum(proj[lo:hi] ** 2):
            x, events[-1]['coef'], events[-1]['removal'] = full, a, 'full'
        else:
            x, events[-1]['removal'] = proj, 'proj'
        keep[max(i - sep, 0):i + sep + 1] = False
    events.sort(key=lambda e: e['t'])
    # Flag weak events sitting next to much stronger ones: possible leftovers of imperfect subtraction.
    win_ps = h * (t[1] - t[0])
    for e in events:
        nb = [o['z'] for o in events if o is not e and abs(o['t'] - e['t']) <= win_ps]
        e['near_strong'] = bool(nb) and bool(e['z'] < 0.2 * max(nb))
    return events, thr, x


# ---------------------------------------------------------------- uncertainty
def bootstrap_times(t, reps, whiten, D, P, thr_events, n_boot=100, tol_ps=2.0, seed=1, **kw):
    """Resample repeats with replacement, re-detect, match to the reference events within tol_ps.
    reps must be background-subtracted the same way as the mean (reps - background[:, None]).
    Returns per-event (std of time, detection rate)."""
    rng = np.random.default_rng(seed)
    n = reps.shape[1]
    ref = np.array([e['t'] for e in thr_events])
    hits = [[] for _ in ref]
    sur = kw.pop('surw')
    for _ in range(n_boot):
        yb = reps[:, rng.integers(0, n, n)].mean(1)
        ev, _, _ = detect(t, whiten(yb), sur, D, P, **kw)
        tb = np.array([e['t'] for e in ev])
        for j, r in enumerate(ref):
            if tb.size and np.min(np.abs(tb - r)) <= tol_ps:
                hits[j].append(tb[np.argmin(np.abs(tb - r))])
    return [(np.std(h) if len(h) > 1 else np.nan, len(h) / n_boot) for h in hits]
