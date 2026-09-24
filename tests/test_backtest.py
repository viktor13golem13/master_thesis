import numpy as np

from thesis.backtest import portfolio_returns, select_bonds


def test_select_bonds():
    ranking = np.array([4, 2, 0, 1, 3])              # most central first
    assert list(select_bonds(ranking, "central", 2)) == [4, 2]
    assert list(select_bonds(ranking, "peripheral", 2)) == [3, 1]


def test_portfolio_returns_hold_the_picks_of_the_previous_year():
    returns_by_year = {
        2018: np.array([[0.01, 0.02, 0.03], [0.00, 0.04, -0.02]]),   # 2 days, 3 bonds
        2019: np.array([[0.05, -0.01, 0.00]]),                       # 1 day
    }
    picks_by_year = {2017: np.array([2, 0, 1]), 2018: np.array([1, 0, 2])}
    daily = portfolio_returns(returns_by_year, picks_by_year, sizes=(1, 2))

    expected = np.array([
        [0.03, (0.03 + 0.01) / 2],       # 2018, day 1: bond 2 | bonds 2 and 0
        [-0.02, (-0.02 + 0.00) / 2],     # 2018, day 2
        [-0.01, (-0.01 + 0.05) / 2],     # 2019: bond 1 | bonds 1 and 0
    ])
    assert np.allclose(daily, expected)
