import numpy as np
import pandas as pd

from thesis import config
from thesis.random_baseline import run_random_baseline, share_beaten


def small_market(monkeypatch):
    monkeypatch.setattr(config, "PORTFOLIO_SIZES", (2, 3))
    monkeypatch.setattr(config, "RANDOM_PORTFOLIOS", 5)
    dates = pd.bdate_range("2018-01-01", "2019-12-31")
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(rng.normal(0, 0.003, size=(len(dates), 6)), index=dates)
    rf = pd.Series(0.0001, index=dates)
    return returns, rf


def test_random_baseline_is_reproducible(monkeypatch):
    returns, rf = small_market(monkeypatch)
    first = run_random_baseline(returns, rf, [2018], verbose=False)
    second = run_random_baseline(returns, rf, [2018], verbose=False)
    assert len(first) == 2 * 5
    pd.testing.assert_frame_equal(first, second)


def test_random_draws_are_m_different_bonds(monkeypatch):
    returns, rf = small_market(monkeypatch)
    rng = np.random.default_rng(config.RANDOM_SEED)
    draw = rng.choice(returns.shape[1], size=3, replace=False)
    assert len(set(draw)) == 3 and draw.min() >= 0 and draw.max() < returns.shape[1]


def test_share_beaten():
    random_values = pd.Series([0.1, 0.2, 0.3, 0.4])
    assert share_beaten(0.35, random_values) == 0.75
    assert share_beaten(0.05, random_values) == 0.0
