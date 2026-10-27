"""Background fitting with piecewise polynomials, for echo extraction.

The slow background of a signal is modelled by polynomials over N pieces of the
x range. For each piece (or the whole spline) every degree in the chosen range
is fitted. Then either the best degree is kept, picked by an information
criterion (BIC or AIC, which penalise extra terms so the fit follows the
background rather than fast features), or all degrees are combined in a
weighted average. Subtracting the fit from the data leaves the residual: the
echoes. Marked regions (e.g. known echoes) can be left out of the fit so the
background cannot bend to absorb them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import make_lsq_spline

from .background import double_exponential
from .oscillations import OscillationSettings, remove_oscillations

MODELS = {
    "Double exponential a·e^(bx) + c·e^(dx)": "exp2",
    "Spline (smoothly joined polynomials)": "spline",
    "Independent pieces (blended at the joins)": "pieces",
    "Single polynomial": "single",
}
CRITERIA = {
    "Best degree by BIC (prefers simpler)": "bic",
    "Best degree by AIC": "aic",
    "Fixed: always the maximum degree": "fixed",
    "Weighted average: Akaike weights (AIC)": "avg_aic",
    "Weighted average: BIC weights": "avg_bic",
    "Weighted average: by goodness of fit (1/residual²)": "avg_fit",
    "Weighted average: equal weights": "avg_equal",
}
RESIDUALS = {"Data − fit": "data-fit", "Fit − data": "fit-data"}
MAX_SPLINE_DEGREE = 5


def is_average(criterion: str) -> bool:
    return criterion.startswith("avg_")


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
    force_decay: bool = True            # double exponential: keep both rates ≤ 0
    # Oscillations (e.g. Brillouin) removed after the background; None = keep them.
    oscillations: OscillationSettings | None = None


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
    # Weighted average only: {degree: weight} per piece (one entry for a spline/single).
    weights: list[dict[int, float]] = field(default_factory=list)
    oscillations: np.ndarray | None = None      # fitted oscillations (if removed)
    osc_windows: list = field(default_factory=list)   # (window start, frequencies found)
    exp_params: tuple | None = None             # double exponential (a, b, c, d)

    @property
    def background_removed(self) -> np.ndarray:
        return self.y - self.fit

    @property
    def degree_text(self) -> str:
        if self.weights:
            # Average weight of each degree over the pieces.
            total: dict[int, float] = {}
            for piece in self.weights:
                for deg, w in piece.items():
                    total[deg] = total.get(deg, 0.0) + w / len(self.weights)
            ranked = sorted(total.items(), key=lambda kv: -kv[1])
            return "avg: " + ", ".join(f"{d} ({100 * w:.0f}%)" for d, w in ranked if w >= 0.005)
        if not self.degrees:
            return "–"
        if len(set(self.degrees)) == 1:
            return str(self.degrees[0])
        return ", ".join(map(str, self.degrees))


@dataclass
class _Candidate:
    degree: int
    curve: object        # callable: x -> fitted values
    rss: float
    n_params: int


@dataclass
class _Choice:
    """The outcome for one piece: a curve plus how it was made."""

    curve: object
    degree: int                          # best (or most-weighted) degree
    weights: dict[int, float] | None     # set for weighted averages


def _information(c: _Candidate, n: int, kind: str) -> float:
    rss = max(c.rss, 1e-300)
    penalty = c.n_params * np.log(n) if kind == "bic" else 2 * c.n_params
    return n * np.log(rss / n) + penalty


def _choose(candidates: list[_Candidate], n: int, criterion: str) -> _Choice | None:
    if not candidates:
        return None
    if criterion == "fixed":
        best = max(candidates, key=lambda c: c.degree)
        return _Choice(best.curve, best.degree, None)
    if not is_average(criterion):
        best = min(candidates, key=lambda c: _information(c, n, criterion))
        return _Choice(best.curve, best.degree, None)

    if criterion in ("avg_aic", "avg_bic"):
        scores = np.array([_information(c, n, criterion[4:]) for c in candidates])
        # Akaike / BIC weights: exp(-Δ/2), normalised.
        raw = np.exp(-0.5 * (scores - scores.min()))
    elif criterion == "avg_fit":
        raw = np.array([1.0 / max(c.rss, 1e-300) for c in candidates])
    else:
        raw = np.ones(len(candidates))
    weights = raw / raw.sum()
    curves = [(w, c.curve) for w, c in zip(weights, candidates)]

    def averaged(x, curves=curves):
        return sum(w * curve(x) for w, curve in curves)

    by_degree = {c.degree: float(w) for w, c in zip(weights, candidates)}
    top = max(by_degree, key=by_degree.get)
    return _Choice(averaged, top, by_degree)


def _polynomial_candidates(x, y, degrees) -> list[_Candidate]:
    out = []
    for deg in degrees:
        if len(x) < deg + 2:            # not enough points to judge this degree
            continue
        poly = Polynomial.fit(x, y, deg)
        out.append(_Candidate(deg, poly, float(np.sum((poly(x) - y) ** 2)), deg + 1))
    return out


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
    pieces = 1 if settings.model == "single" else max(1, int(settings.pieces))
    edges = np.linspace(x[0], x[-1], pieces + 1)

    exp_params = None
    if settings.model == "exp2":
        fit, exp_params = double_exponential(x, y, settings.force_decay, fit_mask=used)
        choices, edges = [], np.array([x[0], x[-1]])
    elif settings.model == "spline":
        fit, choices = _fit_spline(x, xu, yu, edges, settings)
    elif settings.model == "pieces":
        fit, choices = _fit_pieces(x, xu, yu, edges, settings)
    else:
        choice = _choose(_polynomial_candidates(xu, yu, _degrees(settings)), len(xu),
                         settings.criterion)
        if choice is None:
            raise ValueError("Too few points for the chosen degrees.")
        fit, choices = choice.curve(x), [choice]

    background_removed = y - fit
    osc, osc_windows = None, []
    if settings.oscillations is not None:
        found = remove_oscillations(x, background_removed, settings.oscillations)
        osc, osc_windows = found.curve, found.windows
    data_minus_fit = background_removed - (osc if osc is not None else 0.0)
    residual = data_minus_fit if settings.residual == "data-fit" else -data_minus_fit
    r_used = background_removed[used]
    rms = float(np.sqrt(np.mean(r_used ** 2)))
    ss_tot = float(np.sum((yu - yu.mean()) ** 2))
    r2 = 1 - float(np.sum(r_used ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    degrees = [c.degree if c is not None else -1 for c in choices]
    weights = ([c.weights for c in choices if c is not None]
               if is_average(settings.criterion) else [])
    return FitResult(label, x, y, fit, residual, used, degrees, edges, rms, r2, weights,
                     osc, osc_windows, exp_params)


def _fit_pieces(x, xu, yu, edges, settings):
    """Separate polynomial per piece, blended smoothly where pieces overlap."""
    width = edges[1] - edges[0]
    ramp = width * np.clip(settings.blend, 0, 100) / 100
    total = np.zeros_like(x)
    weights = np.zeros_like(x)
    choices = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        # Fit on the piece plus its blending zones, so neighbours agree there.
        sel = (xu >= lo - ramp) & (xu <= hi + ramp)
        candidates = _polynomial_candidates(xu[sel], yu[sel], _degrees(settings))
        choice = _choose(candidates, int(sel.sum()), settings.criterion)
        choices.append(choice)
        if choice is None:
            continue                    # no usable points (e.g. all left out)
        w = _taper(x, lo, hi, ramp)
        total += w * choice.curve(x)
        weights += w
    with np.errstate(invalid="ignore", divide="ignore"):
        fit = total / weights
    if np.isnan(fit).any():
        # Gaps (pieces without points) are bridged linearly.
        good = ~np.isnan(fit)
        if not good.any():
            raise ValueError("No piece had enough points to fit.")
        fit = np.interp(x, x[good], fit[good])
    return fit, choices


def _fit_spline(x, xu, yu, edges, settings):
    """Least-squares spline: one polynomial per piece, joined smoothly
    (continuous value and derivatives up to degree − 1)."""
    interior = edges[1:-1]
    # Drop knots with no data between them (e.g. inside a left-out region).
    counts = np.histogram(xu, bins=edges)[0]
    interior = interior[(counts[:-1] > 0) & (counts[1:] > 0)]
    candidates = []
    for k in _degrees(settings, cap=MAX_SPLINE_DEGREE):
        k = max(k, 0)
        t = np.r_[[x[0]] * (k + 1), interior, [x[-1]] * (k + 1)]
        try:
            spline = make_lsq_spline(xu, yu, t, k)
        except (ValueError, np.linalg.LinAlgError):
            continue                    # e.g. too few points between knots
        rss = float(np.sum((spline(xu) - yu) ** 2))
        candidates.append(_Candidate(k, spline, rss, len(t) - k - 1))
    choice = _choose(candidates, len(xu), settings.criterion)
    if choice is None:
        raise ValueError("The spline could not be fitted; try fewer pieces or lower degrees.")
    return choice.curve(x), [choice]
