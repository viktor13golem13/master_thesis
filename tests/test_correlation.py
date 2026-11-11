import pytest
import numpy as np

from thesis.correlation import (
    correlation_for_year, single_index_shrinkage, window_weights, yearly_correlation,
)
import pandas as pd


def test_single_window_matches_numpy():
    X = np.random.default_rng(0).normal(size=(125, 5))
    assert np.allclose(yearly_correlation(X, window_length=125), np.corrcoef(X.T))


def test_uniform_weighting_averages_every_window():
    X = np.random.default_rng(1).normal(size=(260, 5))
    expected = np.mean([np.corrcoef(X[end - 125:end].T) for end in range(125, 261)], axis=0)
    assert np.allclose(yearly_correlation(X, window_length=125, weighting="uniform"), expected)


def test_window_weightings():
    paper = window_weights(125, 125, "paper")
    recent = window_weights(125, 125, "recent")
    assert paper[0] > paper[-1]                     # paper: older windows count more
    assert np.allclose(recent, paper[::-1])
    assert np.isclose(paper[0] / paper[-1], np.exp(124 / 125))
    assert np.isclose(paper.sum(), 1)


def test_series_that_never_moves_has_zero_correlation():
    X = np.random.default_rng(2).normal(size=(125, 3))
    X[:, 2] = 0.0
    assert np.allclose(yearly_correlation(X, window_length=125)[2], [0, 0, 1])


def ledoit_wolf_reference(x):
    """Shrinkage intensity by the formulas of Ledoit & Wolf (2003), written as loops."""
    t, n = x.shape
    y = x - x.mean(axis=0)
    m = y.mean(axis=1)
    s = y.T @ y / t
    s_m = y.T @ m / t
    s_00 = m @ m / t
    f = np.outer(s_m, s_m) / s_00
    np.fill_diagonal(f, np.diag(s))
    pi = sum(np.mean((y[:, i] * y[:, j] - s[i, j]) ** 2) for i in range(n) for j in range(n))
    rho = sum(np.mean((y[:, i] * y[:, i] - s[i, i]) ** 2) for i in range(n))
    for i in range(n):
        for j in range(n):
            if i != j:
                r = ((s_m[j] * s_00 * y[:, i] + s_m[i] * s_00 * y[:, j] - s_m[i] * s_m[j] * m)
                     * m * y[:, i] * y[:, j] / s_00 ** 2 - f[i, j] * s[i, j])
                rho += np.mean(r)
    gamma = ((f - s) ** 2).sum()
    return float(np.clip((pi - rho) / gamma / t, 0, 1))


def test_shrinkage_matches_ledoit_wolf_formulas():
    rng = np.random.default_rng(3)
    market = rng.normal(size=(60, 1))
    x = market @ rng.random((1, 8)) + 0.8 * rng.normal(size=(60, 8))
    shrinkage, target = single_index_shrinkage(x)
    assert 0 < shrinkage < 1
    assert np.isclose(shrinkage, ledoit_wolf_reference(x))
    assert np.allclose(np.diag(target), 1)


def test_single_window_covariance_matches_numpy():
    X = np.random.default_rng(4).normal(size=(125, 5))
    assert np.allclose(yearly_correlation(X, window_length=125, covariance=True), np.cov(X.T))


def test_uniform_weighting_averages_every_window_covariance():
    X = np.random.default_rng(5).normal(size=(260, 5))
    expected = np.mean([np.cov(X[end - 125:end].T) for end in range(125, 261)], axis=0)
    assert np.allclose(yearly_correlation(X, window_length=125, covariance=True), expected)


@pytest.mark.parametrize("shrink", [False, True])
def test_scaled_covariance_stays_between_minus_one_and_one(shrink):
    rng = np.random.default_rng(6)
    days = pd.bdate_range("2019-01-01", "2019-12-31")
    X = rng.normal(size=(len(days), 6)) @ rng.normal(size=(6, 6)) * np.array([1, 2, 3, 4, 5, 6]) / 100
    C = correlation_for_year(pd.DataFrame(X, index=days), 2019, shrink=shrink, covariance=True)
    assert np.abs(C).max() <= 1 + 1e-12
    assert np.isclose(np.diag(C).max(), 1.0)
    assert np.argmax(np.diag(C)) == np.argmax(X.std(axis=0))
