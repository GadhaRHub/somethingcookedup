import numpy as np
import pytest

from csvplot.analysis import FFTSettings, frequency_unit, prepare, spectrum

X = np.arange(0, 200, 0.05)
Y = 2.0 * np.sin(2 * np.pi * 0.25 * X) + 0.5 * np.sin(2 * np.pi * 1.3 * X) + 0.1 * X


@pytest.mark.parametrize("method,segment", [("single", None), ("welch", 50), ("spectrogram", 50)])
@pytest.mark.parametrize("window", ["hann", "hamming", "blackmanharris", "boxcar"])
def test_finds_frequencies_and_amplitudes(method, segment, window):
    result = spectrum(X, Y, FFTSettings(method=method, window=window, segment=segment))
    (p1, p2) = result.peaks[:2]
    assert p1.frequency == pytest.approx(0.25, abs=0.005)
    assert p1.period == pytest.approx(4.0, rel=0.02)
    assert p1.amplitude == pytest.approx(2.0, rel=0.03)
    assert p2.frequency == pytest.approx(1.3, abs=0.005)
    assert p2.amplitude == pytest.approx(0.5, rel=0.05)
    assert p1.relative == pytest.approx(1.0)


def test_parametrised_windows_and_overlap():
    for window, param in (("tukey", 0.3), ("kaiser", 12)):
        result = spectrum(X, Y, FFTSettings(method="welch", window=window, window_param=param,
                                            segment=40, overlap=75))
        assert result.peaks[0].frequency == pytest.approx(0.25, abs=0.01)
        assert result.n_segments > 10


def test_range_and_frequency_limits():
    result = spectrum(X, Y, FFTSettings(f_min=1.0), x_range=(50, 150))
    assert result.x[0] >= 50 and result.x[-1] <= 150
    assert result.peaks[0].frequency == pytest.approx(1.3, abs=0.01)


def test_uneven_spacing_is_resampled():
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 100, 3000))
    x2, y2, dx, resampled = prepare(x, np.sin(2 * np.pi * 0.2 * x))
    assert resampled and np.allclose(np.diff(x2), dx)
    result = spectrum(x, np.sin(2 * np.pi * 0.2 * x), FFTSettings())
    assert result.peaks[0].frequency == pytest.approx(0.2, abs=0.005)


def test_units():
    assert frequency_unit("Delay (ps)") == ("THz", "ps")
    assert frequency_unit("t_ps") == ("THz", "ps")
    assert frequency_unit("time [s]") == ("Hz", "s")
    assert frequency_unit("Delay $\\tau$ (fs)") == ("PHz", "fs")


def test_too_few_points():
    with pytest.raises(ValueError):
        spectrum(X, Y, FFTSettings(), x_range=(0, 0.2))
