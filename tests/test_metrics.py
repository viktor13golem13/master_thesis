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
