"""
Step 5 — How would a strategy have performed? (paper, Sec. 2.2-2.3 and 4)

Walk forward one year at a time: the centrality ranking computed from the
data of formation year y selects m bonds, which are held with equal weights
(1/m each, rebalanced daily, r_p = xᵀr) throughout year y + 1. The holding
years are chained into one series of daily portfolio returns, which the
metrics in thesis.metrics then score.
"""
import numpy as np
import pandas as pd

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


def equal_weight_benchmark(
    returns_by_year: dict[int, np.ndarray],
    rf_by_year: dict[int, np.ndarray],
    holding_years: list[int],
) -> dict[str, float]:
    """All metrics for the equal-weight portfolio of every bond, rebalanced daily."""
    daily = np.concatenate([returns_by_year[y].mean(axis=1) for y in holding_years])
    rf = np.concatenate([rf_by_year[y] for y in holding_years])
    return all_metrics(daily, rf)
