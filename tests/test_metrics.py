import numpy as np
import pytest

from thesis import metrics
from thesis.metrics import METRICS, all_metrics, metric_name

RP = np.array([0.01, -0.02, 0.03, -0.01, 0.02])     # daily portfolio returns
RF = np.full(5, 0.001)                               # daily risk-free returns


def test_cumulative_return():
    assert np.isclose(metrics.cumulative_return(RP, RF), 1.01 * 0.98 * 1.03 * 0.99 * 1.02 - 1)


def test_expected_return():
    assert np.isclose(metrics.expected_return(RP, RF), 0.006 * metrics.TRADING_DAYS_PER_YEAR)


def test_volatility():
    assert np.isclose(metrics.volatility(RP, RF), np.std(RP, ddof=1) * np.sqrt(metrics.TRADING_DAYS_PER_YEAR))


def test_sharpe_ratio():
    assert np.isclose(metrics.sharpe_ratio(RP, RF), (0.006 - 0.001) / np.std(RP, ddof=1) * np.sqrt(metrics.TRADING_DAYS_PER_YEAR))


def test_sharpe_ratio_of_constant_returns_is_undefined():
    assert np.isnan(metrics.sharpe_ratio(np.full(5, 0.01), RF))


def test_lo_volatility_matches_lo_formula():
    r = np.random.default_rng(5).normal(size=1566) / 100
    x = r - r.mean()
    rho = [np.sum(x[k:] * x[:len(x) - k]) / np.sum(x * x) for k in range(21)]
    factor = 1 + 2 * sum((1 - k / 261) * rho[k] for k in range(1, 21))
    assert np.isclose(metrics.volatility_lo(r, r), metrics.volatility(r, r) * np.sqrt(factor))
    rf = np.full(len(r), 0.0001)
    assert np.isclose(metrics.sharpe_ratio_lo(r, rf), metrics.sharpe_ratio(r, rf) / np.sqrt(factor))


def test_block_volatilities_compound_each_block():
    r = np.random.default_rng(1).normal(size=3 * 261 + 10) / 100      # 10 leftover days are dropped
    months = [np.prod(1 + r[i:i + 22]) - 1 for i in range(0, 36 * 22, 22)]
    years = [np.prod(1 + r[i:i + 261]) - 1 for i in range(0, 3 * 261, 261)]
    assert np.isclose(metrics.volatility_monthly(r, r), np.std(months, ddof=1) * np.sqrt(261 / 22))
    assert np.isclose(metrics.volatility_yearly(r, r), np.std(years, ddof=1))


def test_volatility_versions_agree_on_independent_returns():
    r = np.random.default_rng(2).normal(size=50 * 261) / 100
    daily = metrics.volatility(r, r)
    for version in (metrics.volatility_lo, metrics.volatility_monthly, metrics.volatility_yearly):
        assert abs(version(r, r) / daily - 1) < 0.25


def test_volatility_versions_grow_with_positive_autocorrelation():
    noise = np.random.default_rng(3).normal(size=30 * 261 + 1) / 100
    r = noise[1:] + 0.5 * noise[:-1]                                   # lag-1 autocorrelation 0.4
    daily = metrics.volatility(r, r)
    assert metrics.volatility_lo(r, r) > 1.2 * daily
    assert metrics.volatility_monthly(r, r) > 1.2 * daily


def test_sharpe_versions_share_the_excess_return():
    r = np.random.default_rng(4).normal(size=6 * 261) / 100
    rf = np.full(len(r), 0.0001)
    excess = metrics.expected_return(r - rf, rf)
    assert np.isclose(metrics.sharpe_ratio_monthly(r, rf), excess / metrics.volatility_monthly(r, rf))
    assert np.isclose(metrics.sharpe_ratio_yearly(r, rf), excess / metrics.volatility_yearly(r, rf))
    assert np.isclose(metrics.sharpe_ratio_lo(r, rf), excess / metrics.volatility_lo(r, rf))


def test_block_volatility_needs_two_blocks():
    assert np.isnan(metrics.volatility_yearly(RP, RF))
    assert np.isnan(metrics.sharpe_ratio_yearly(RP, RF))


def test_value_at_risk_and_cvar():
    returns = np.arange(-50, 50) / 1000              # 100 days from -5% to +4.9%
    assert np.isclose(metrics.value_at_risk(returns, returns), np.quantile(returns, 0.05))
    assert np.isclose(metrics.conditional_value_at_risk(returns, returns), np.mean(returns[:5]))


def test_max_drawdown():
    # value path 1.10, 0.88, 0.968: peak 1.10, trough 0.88 -> 20% drawdown
    assert np.isclose(metrics.max_drawdown(np.array([0.10, -0.20, 0.10]), RF[:3]), 0.20)


def test_downside_ratios():
    shortfall = np.maximum(RF - RP, 0)
    excess = RP - RF
    assert np.isclose(metrics.omega_ratio(RP, RF), excess.mean() / shortfall.mean() + 1)
    assert np.isclose(metrics.sortino_ratio(RP, RF), excess.mean() / np.sqrt((shortfall ** 2).mean()))
    assert np.isclose(metrics.upside_potential_ratio(RP, RF),
                      np.maximum(excess, 0).mean() / np.sqrt((shortfall ** 2).mean()))


@pytest.mark.parametrize("name, function", METRICS.items())
def test_every_metric_has_the_same_form(name, function):
    assert isinstance(function(RP, RF), float)
    assert metric_name(function) == name


def test_all_metrics_returns_every_metric():
    assert set(all_metrics(RP, RF)) == set(METRICS)
