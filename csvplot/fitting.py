"""Background fitting with piecewise polynomials, for echo extraction.

The slow background of a signal is modelled by polynomials over N pieces of the
x range. For each piece (or the whole spline) every degree in the chosen range
is tried and the best is picked by an information criterion (BIC or AIC), which
penalises extra terms so the fit follows the background rather than fast
features. Subtracting the fit from the data leaves the residual: the echoes.
Marked regions (e.g. known echoes) can be left out of the fit so the
background cannot bend to absorb them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import make_lsq_spline

MODELS = {
    "Spline (smoothly joined polynomials)": "spline",
    "Independent pieces (blended at the joins)": "pieces",
    "Single polynomial": "single",
}
CRITERIA = {
    "Best degree by BIC (prefers simpler)": "bic",
    "Best degree by AIC": "aic",
    "Fixed: always the maximum degree": "fixed",
}
RESIDUALS = {"Data − fit": "data-fit", "Fit − data": "fit-data"}
MAX_SPLINE_DEGREE = 5


@dataclass
class FitSettings:
    model: str = "spline"
    pieces: int = 8
    degree_min: int = 1
    degree_max: int = 5
    criterion: str = "bic"
    blend: float = 20.0                 # % of a piece's width used to blend neighbours
    exclude: list[tuple[float, float]] = field(default_factory=list)
    residual: str = "data-fit"


@dataclass
class FitResult:
    label: str
    x: np.ndarray
    y: np.ndarray
    fit: np.ndarray
    residual: np.ndarray
    used: np.ndarray                    # points that took part in the fit
    degrees: list[int]                  # chosen degree per piece (one entry for a spline/single)
    edges: np.ndarray                   # piece boundaries
    rms: float = float("nan")           # RMS of (data - fit) over the fitted points
    r2: float = float("nan")

    @property
    def degree_text(self) -> str:
        if len(set(self.degrees)) == 1:
            return str(self.degrees[0])
        return ", ".join(map(str, self.degrees))


def _score(rss: float, n: int, k: int, criterion: str) -> float:
    if criterion == "fixed":
        return -k                       # largest degree wins
    rss = max(rss, 1e-300)
    penalty = k * np.log(n) if criterion == "bic" else 2 * k
    return n * np.log(rss / n) + penalty


def _best_polynomial(x, y, degrees, criterion):
    best = None
    for deg in degrees:
        if len(x) < deg + 2:            # not enough points to judge this degree
            continue
        poly = Polynomial.fit(x, y, deg)
        rss = float(np.sum((poly(x) - y) ** 2))
        score = _score(rss, len(x), deg + 1, criterion)
        if best is None or score < best[0]:
            best = (score, deg, poly)
    return best


def _degrees(settings: FitSettings, cap: int | None = None) -> list[int]:
    lo, hi = sorted((settings.degree_min, settings.degree_max))
    if cap is not None:
        lo, hi = min(lo, cap), min(hi, cap)
    if settings.criterion == "fixed":
        return [hi]
    return list(range(lo, hi + 1))


def _taper(x, lo, hi, ramp):
    """Weight 1 inside [lo, hi], cosine ramps of width ``ramp`` outside it."""
    w = np.ones_like(x)
    if ramp > 0:
        left = (x < lo) & (x > lo - ramp)
        right = (x > hi) & (x < hi + ramp)
        w[left] = 0.5 * (1 + np.cos(np.pi * (lo - x[left]) / ramp))
        w[right] = 0.5 * (1 + np.cos(np.pi * (x[right] - hi) / ramp))
    w[(x <= lo - ramp) | (x >= hi + ramp)] = 0.0
    return w


def fit_background(x, y, settings: FitSettings, label: str = "",
                   x_range: tuple[float, float] | None = None) -> FitResult:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x_range is not None:
        lo, hi = sorted(x_range)
        keep = (x >= lo) & (x <= hi)
        x, y = x[keep], y[keep]
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    if len(x) < 4:
        raise ValueError("Need at least 4 data points in the chosen range.")

    used = np.ones(len(x), bool)
    for lo, hi in settings.exclude:
        used &= ~((x >= min(lo, hi)) & (x <= max(lo, hi)))
    if used.sum() < 4:
        raise ValueError("Too few points left after leaving out the marked regions.")
    xu, yu = x[used], y[used]
    x0, x1 = x[0], x[-1]
    pieces = 1 if settings.model == "single" else max(1, int(settings.pieces))
    edges = np.linspace(x0, x1, pieces + 1)

    if settings.model == "spline":
        fit, degrees = _fit_spline(x, xu, yu, edges, settings)
    elif settings.model == "pieces":
        fit, degrees = _fit_pieces(x, xu, yu, edges, settings)
    else:
        best = _best_polynomial(xu, yu, _degrees(settings), settings.criterion)
        if best is None:
            raise ValueError("Too few points for the chosen degrees.")
        fit, degrees = best[2](x), [best[1]]

    data_minus_fit = y - fit
    residual = data_minus_fit if settings.residual == "data-fit" else -data_minus_fit
    r_used = data_minus_fit[used]
    rms = float(np.sqrt(np.mean(r_used ** 2)))
    ss_tot = float(np.sum((yu - yu.mean()) ** 2))
    r2 = 1 - float(np.sum(r_used ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    return FitResult(label, x, y, fit, residual, used, degrees, edges, rms, r2)


def _fit_pieces(x, xu, yu, edges, settings):
    """Separate polynomial per piece, blended smoothly where pieces overlap."""
    width = edges[1] - edges[0]
    ramp = width * np.clip(settings.blend, 0, 100) / 100
    total = np.zeros_like(x)
    weights = np.zeros_like(x)
    degrees = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        # Fit on the piece plus its blending zones, so neighbours agree there.
        sel = (xu >= lo - ramp) & (xu <= hi + ramp)
        best = _best_polynomial(xu[sel], yu[sel], _degrees(settings), settings.criterion)
        if best is None:
            degrees.append(-1)          # no usable points (e.g. all excluded)
            continue
        degrees.append(best[1])
        w = _taper(x, lo, hi, ramp)
        total += w * best[2](x)
        weights += w
    with np.errstate(invalid="ignore", divide="ignore"):
        fit = total / weights
    if np.isnan(fit).any():
        # Gaps (pieces without points) are bridged linearly.
        good = ~np.isnan(fit)
        if not good.any():
            raise ValueError("No piece had enough points to fit.")
        fit = np.interp(x, x[good], fit[good])
    return fit, degrees


def _fit_spline(x, xu, yu, edges, settings):
    """Least-squares spline: one polynomial per piece, joined smoothly
    (continuous value and derivatives up to degree − 1)."""
    best = None
    for k in _degrees(settings, cap=MAX_SPLINE_DEGREE):
        k = max(k, 0)
        interior = edges[1:-1]
        # Drop knots with no data between them (e.g. inside a left-out region).
        counts = np.histogram(xu, bins=edges)[0]
        keep_knot = (counts[:-1] > 0) & (counts[1:] > 0)
        interior = interior[keep_knot]
        t = np.r_[[x[0]] * (k + 1), interior, [x[-1]] * (k + 1)]
        try:
            spline = make_lsq_spline(xu, yu, t, k)
        except (ValueError, np.linalg.LinAlgError):
            continue                    # e.g. too few points between knots
        n_coef = len(t) - k - 1
        rss = float(np.sum((spline(xu) - yu) ** 2))
        score = _score(rss, len(xu), n_coef, settings.criterion)
        if best is None or score < best[0]:
            best = (score, k, spline)
    if best is None:
        raise ValueError("The spline could not be fitted; try fewer pieces or lower degrees.")
    return best[2](x), [best[1]]
