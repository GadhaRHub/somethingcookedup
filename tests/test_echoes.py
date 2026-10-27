import numpy as np
import pytest

from csvplot.echoes import EchoMatchParams, block_bin, quiet_surrogates, run_echo_match, run_twite
from csvplot.twite import TwiteParams

T = np.arange(10, 600, 0.2)
TRUE = [58.0, 116.0, 174.0, 232.0, 290.0]


def pulse(c):
    return np.exp(-((T - c) / 2.5) ** 2) * np.sin(2 * np.pi * (T - c) / 6)


def make(n_rep=20, seed=1):
    rng = np.random.default_rng(seed)
    sig = 0.02 * sum(0.85 ** k * pulse(c) for k, c in enumerate(TRUE))
    osc = 0.01 * np.sin(2 * np.pi * T / 28.7 + 0.3) * np.exp(-T / 500)
    reps = np.stack([sig + osc + np.convolve(rng.normal(0, 0.01, T.size), np.ones(3) / 3, "same")
                     for _ in range(n_rep)], 1)
    return reps


def test_echo_match_quiet_window_finds_the_train():
    reps = make()
    result = run_echo_match(T, reps.mean(1), EchoMatchParams(
        noise="quiet", quiet=(400, 590), seeds=TRUE[:4], freqs=[1 / 28.7], n_sur=100))
    found = sorted(e["t"] for e in result["events"])
    assert len(found) == 5
    assert np.allclose(found, TRUE, atol=0.5)
    assert result["threshold"] > 0 and result["templates"].shape[1] == 1


def test_echo_match_with_repeats_and_bootstrap():
    reps = make()
    result = run_echo_match(T, reps.mean(1), EchoMatchParams(
        noise="repeats", seeds=TRUE[:4], freqs=[1 / 28.7], n_sur=60, n_boot=10), reps=reps)
    strong = sorted(e["t"] for e in result["events"] if not e["near_strong"])
    for c in TRUE:
        assert min(abs(np.array(strong) - c)) < 0.6
    assert result["boot"] is not None and len(result["boot"]) == len(result["events"])


def test_echo_match_estimates_frequencies_and_seeds():
    reps = make()
    result = run_echo_match(T, reps.mean(1), EchoMatchParams(quiet=(400, 590), n_sur=60),
                            twite_params=TwiteParams(t_ignore=30, quiet_window=(400, 590)))
    assert any(abs(f - 1 / 28.7) < 0.003 for f in result["freqs"])
    assert result["seeds"]
    assert any("TWITE" in w for w in result["warnings"])


def test_quiet_surrogates_have_the_noise_level():
    rng = np.random.default_rng(3)
    y = rng.normal(0, 0.5, T.size)
    sur = quiet_surrogates(T, y, (100, 500), 20)
    assert sur.shape == (20, T.size)
    assert np.std(sur) == pytest.approx(0.5, rel=0.1)
    with pytest.raises(ValueError):
        quiet_surrogates(T, y, (100, 101), 5)


def test_twite_port_and_binning():
    y = 0.02 * sum(0.85 ** k * pulse(c) for k, c in enumerate(TRUE))
    y = y + np.random.default_rng(2).normal(0, 1e-4, T.size)
    result = run_twite(T, y, TwiteParams(t_ignore=30, quiet_window=(400, 590), pct=0.3), n_bin=5)
    assert len(result["t"]) == T.size // 5
    assert result["t1"] is not None and len(result["echoes"]) >= 2
    assert set(result["methods"].values()) <= {"amplitude", "shape", "both"}
    tb, yb = block_bin(T, y, 1)
    assert np.array_equal(yb, y)
