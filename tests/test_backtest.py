import numpy as np

from thesis.backtest import portfolio_returns, select_bonds


def test_select_bonds():
    ranking = np.array([4, 2, 0, 1, 3])              # most central first
    assert list(select_bonds(ranking, "central", 2)) == [4, 2]
    assert list(select_bonds(ranking, "peripheral", 2)) == [3, 1]


def test_portfolio_returns_average_the_holdings_of_each_year():
    returns_by_year = {
        2018: np.array([[0.01, 0.02, 0.03], [0.00, 0.04, -0.02]]),   # 2 days, 3 bonds
        2019: np.array([[0.05, -0.01, 0.00]]),                       # 1 day
    }
    holdings_by_year = {2018: np.array([2, 0]), 2019: np.array([1, 0])}
    daily = portfolio_returns(returns_by_year, holdings_by_year)

    expected = np.array([
        (0.03 + 0.01) / 2,      # 2018, day 1: bonds 2 and 0
        (-0.02 + 0.00) / 2,     # 2018, day 2
        (-0.01 + 0.05) / 2,     # 2019: bonds 1 and 0
    ])
    assert np.allclose(daily, expected)
