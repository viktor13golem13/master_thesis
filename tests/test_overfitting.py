import numpy as np
from scipy.stats import norm

from thesis.grid_search import SeriesStore
from thesis.metrics import sharpe_ratio_lo
from thesis.overfitting import (backtest_overfitting, deflated_sharpe_ratio, expected_best_of,
                                expected_maximum_sharpe, probabilistic_sharpe_ratio, sharpe_lo_columns)

RNG = np.random.default_rng(0)


def test_expected_maximum_matches_simulated_maxima():
    n, sd = 1000, 0.5
    simulated = RNG.normal(scale=sd, size=(2000, n)).max(axis=1).mean()
    assert abs(expected_maximum_sharpe(n, sd ** 2) - simulated) < 0.02


def test_probabilistic_sharpe_ratio_by_hand():
    # (0.1 - 0.05) √99 / √(1 - (-0.5)(0.1) + (4 - 1)/4 · 0.01) = 0.4818...
    expected = norm.cdf(0.05 * np.sqrt(99) / np.sqrt(1 + 0.05 + 0.0075))
    assert np.isclose(probabilistic_sharpe_ratio(0.1, 0.05, 100, -0.5, 4.0), expected)


def test_one_trial_is_not_deflated():
    excess = RNG.normal(0.001, 0.01, size=500)
    result = deflated_sharpe_ratio(excess, 0.5, np.array([0.5, 0.1]), n_trials=1)
    assert result["expected best SR_Lo without skill"] == 0.0


def test_more_trials_lower_the_dsr():
    excess = RNG.normal(0.001, 0.01, size=500)
    trials = RNG.normal(0, 0.3, size=100)
    dsr = [deflated_sharpe_ratio(excess, 1.0, trials, n)["DSR"] for n in (10, 100, 1000)]
    assert dsr[0] > dsr[1] > dsr[2]


def test_sharpe_lo_columns_matches_the_metric():
    returns = RNG.normal(0.0002, 0.003, size=(400, 5))
    rf = np.full(400, 0.0001)
    expected = [sharpe_ratio_lo(returns[:, j], rf) for j in range(5)]
    assert np.allclose(sharpe_lo_columns(returns, rf), expected)


def test_pure_noise_overfits_about_half_the_time_or_more():
    returns = RNG.normal(0, 0.003, size=(1000, 200))
    pbo = backtest_overfitting(returns, np.zeros(1000), n_blocks=10)
    assert len(pbo["omega"]) == 252
    assert pbo["PBO"] > 0.3


def test_a_truly_better_strategy_does_not_overfit():
    # Long series: on short noise series Lo's volatility can collapse by chance
    # (20 sample autocorrelations summing to almost -1/2), giving a huge SR_Lo.
    returns = RNG.normal(0, 0.003, size=(4000, 50))
    returns[:, 7] += 0.002
    pbo = backtest_overfitting(returns, np.zeros(4000), n_blocks=10)
    assert pbo["PBO"] == 0.0


def test_expected_best_of_one_draw_is_the_mean_and_of_many_the_maximum():
    values = np.array([3.0, 1.0, 2.0, np.nan])
    assert np.isclose(expected_best_of(values, 1), 2.0)
    assert np.isclose(expected_best_of(values, 10_000), 3.0)


def test_series_store_keeps_each_distinct_series_once():
    a, b = np.arange(4.0), np.ones(4)
    store = SeriesStore()
    rows = store.add_rows([{"daily": a}, {"daily": b}, {"daily": a.copy()}])
    assert [row["series"] for row in rows] == [0, 1, 0]
    assert "daily" not in rows[0]
    assert np.array_equal(store.matrix(), np.column_stack([a, b]).astype(np.float32))
