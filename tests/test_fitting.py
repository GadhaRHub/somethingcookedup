import numpy as np
import pytest

from csvplot.fitting import FitSettings, fit_background

X = np.linspace(0, 600, 6001)
BACKGROUND = 0.025 * np.exp(-X / 400) + 1e-6 * X
ECHO = 1e-3 * np.exp(-((X - 470) / 6) ** 2) * np.sin(2 * np.pi * X / 6.7)
Y = BACKGROUND + ECHO


@pytest.mark.parametrize("model", ["spline", "pieces", "single"])
def test_extracts_the_echo(model):
    result = fit_background(X, Y, FitSettings(model=model, pieces=8, degree_min=1, degree_max=5))
    error = np.sqrt(np.mean((result.residual - ECHO) ** 2))
    assert error < 5e-5                        # < 5 % of the echo amplitude
    assert result.r2 > 0.999


@pytest.mark.parametrize("model", ["spline", "pieces", "single"])
def test_leaving_out_the_echo_region(model):
    result = fit_background(X, Y, FitSettings(model=model, pieces=8, exclude=[(440, 500)]))
    assert np.sqrt(np.mean((result.residual - ECHO) ** 2)) < 1e-6
    assert not result.used[(X > 450) & (X < 490)].any()


def test_degree_choice_and_direction():
    # A pure quadratic: BIC should pick degree 2, not the maximum.
    y = 3 + 2 * X - 0.01 * X ** 2
    assert fit_background(X, y, FitSettings(model="single", degree_max=8)).degrees == [2]
    assert fit_background(X, y, FitSettings(model="single", degree_max=8,
                                            criterion="fixed")).degrees == [8]
    flipped = fit_background(X, Y, FitSettings(residual="fit-data"))
    normal = fit_background(X, Y, FitSettings())
    assert np.allclose(flipped.residual, -normal.residual)


def test_pieces_report_a_degree_each_and_range():
    result = fit_background(X, Y, FitSettings(model="pieces", pieces=6), x_range=(100, 400))
    assert len(result.degrees) == 6 and len(result.edges) == 7
    assert result.x[0] >= 100 and result.x[-1] <= 400


def test_errors():
    with pytest.raises(ValueError):
        fit_background(X, Y, FitSettings(), x_range=(0, 0.1))
    with pytest.raises(ValueError):
        fit_background(X, Y, FitSettings(exclude=[(-1, 700)]))
