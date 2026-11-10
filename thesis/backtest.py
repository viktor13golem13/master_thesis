"""
Step 5 — How would a strategy have performed? (paper, Sec. 2.2-2.3 and 4)

Walk forward one year at a time: the centrality scores computed from the
data of formation year y select m bonds, which are held with equal weights
(1/m each, rebalanced daily, r_p = xᵀr) throughout year y + 1. When bonds
tie for the last places, those places are shared by all tied bonds
(select_weights). The holding years are chained into one series of daily
portfolio returns, which the metrics in thesis.metrics then score.
"""
import numpy as np
import pandas as pd

from thesis.centrality import tie_values
from thesis.metrics import all_metrics


def split_by_year(series: pd.DataFrame | pd.Series, years: list[int]) -> dict[int, np.ndarray]:
    """The rows of each requested calendar year, as numpy arrays."""
    return {y: series.loc[series.index.year == y].to_numpy(dtype=float) for y in years}


def select_bonds(ranking: np.ndarray, selection: str, m: int) -> np.ndarray:
    """The m most central bonds ("central") or the m most peripheral ones ("peripheral")."""
    if selection == "central":
        return ranking[:m]
    if selection == "peripheral":
        return ranking[::-1][:m]
    raise ValueError(f"Unknown selection {selection!r}")


def select_weights(scores: np.ndarray, selection: str, m: int) -> np.ndarray:
    """
    Portfolio weights of the m most central ("central") or most peripheral
    ("peripheral") bonds: 1/m each.

    When several bonds tie for the last places (see centrality.tie_values),
    e.g. many isolated nodes, those places are split evenly over all tied
    bonds: with k bonds clearly in and t tied, each tied bond gets
    (m - k) / (m t). This is the average portfolio over every random way of
    breaking the tie, so the result does not depend on luck; the portfolio
    then holds more than m bonds.
    """
    values = tie_values(scores)
    if selection == "peripheral":
        values = -values
    elif selection != "central":
        raise ValueError(f"Unknown selection {selection!r}")
    boundary = np.sort(values)[::-1][m - 1]          # value of the m-th best bond
    clearly_in = values > boundary
    tied = values == boundary

    weights = np.zeros(len(scores))
    weights[clearly_in] = 1.0 / m
    weights[tied] = (m - clearly_in.sum()) / (m * tied.sum())
    return weights


def portfolio_returns(returns_by_year: dict[int, np.ndarray], holdings_by_year: dict[int, np.ndarray]) -> np.ndarray:
    """
    Daily returns of an equal-weight portfolio, rebalanced daily (r_p = xᵀr).

    holdings_by_year[y] are the bonds held during year y; each day the
    portfolio's return is the average of their returns. The years are joined
    in chronological order.
    """
    return np.concatenate([
        returns_by_year[year][:, bonds].mean(axis=1) for year, bonds in sorted(holdings_by_year.items())
    ])


def evaluate_portfolio(
    returns_by_year: dict[int, np.ndarray],
    rf_by_year: dict[int, np.ndarray],
    holdings_by_year: dict[int, np.ndarray],
) -> dict[str, float]:
    """All metrics for one portfolio, measured over the years in holdings_by_year."""
    daily = portfolio_returns(returns_by_year, holdings_by_year)
    rf = np.concatenate([rf_by_year[year] for year in sorted(holdings_by_year)])
    return all_metrics(daily, rf)


def weighted_portfolio_returns(returns_by_year: dict[int, np.ndarray], weights_by_year: dict[int, np.ndarray]) -> np.ndarray:
    """
    Daily returns of a portfolio with fixed weights, rebalanced daily (r_p = xᵀr).
    weights_by_year[y] has one weight per bond, summing to 1, held during year y.
    """
    return np.concatenate([returns_by_year[year] @ weights for year, weights in sorted(weights_by_year.items())])


def evaluate_weighted_portfolio(
    returns_by_year: dict[int, np.ndarray],
    rf_by_year: dict[int, np.ndarray],
    weights_by_year: dict[int, np.ndarray],
) -> dict[str, float]:
    """All metrics for one portfolio with fixed weights, like evaluate_portfolio."""
    daily = weighted_portfolio_returns(returns_by_year, weights_by_year)
    rf = np.concatenate([rf_by_year[year] for year in sorted(weights_by_year)])
    return all_metrics(daily, rf)


def equal_weight_benchmark(
    returns_by_year: dict[int, np.ndarray],
    rf_by_year: dict[int, np.ndarray],
    holding_years: list[int],
) -> dict[str, float]:
    """All metrics for the equal-weight portfolio of every bond, rebalanced daily."""
    daily = np.concatenate([returns_by_year[y].mean(axis=1) for y in holding_years])
    rf = np.concatenate([rf_by_year[y] for y in holding_years])
    return all_metrics(daily, rf)
