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


def select_bonds(ranking: np.ndarray, selection: str, max_size: int) -> np.ndarray:
    """
    The first max_size bonds a strategy would pick, in order of preference:
    the most central first ("central") or the most peripheral first ("peripheral").
    """
    if selection == "central":
        return ranking[:max_size]
    if selection == "peripheral":
        return ranking[::-1][:max_size]
    raise ValueError(f"Unknown selection {selection!r}")


def portfolio_returns(
    returns_by_year: dict[int, np.ndarray],
    picks_by_year: dict[int, np.ndarray],
    sizes: tuple[int, ...],
) -> np.ndarray:
    """
    Daily returns of equal-weight portfolios holding the first m picks,
    for every m in sizes at once.

    picks_by_year[y] are the bonds chosen with data of formation year y
    (in order of preference); they are held during year y + 1.

    The average of the first m columns for every m is a running sum divided
    by m, so all portfolio sizes cost one cumulative sum.

    Returns
    -------
    np.ndarray of shape (days, len(sizes)), days in chronological order.
    """
    columns = np.asarray(sizes) - 1
    counts = np.asarray(sizes, dtype=float)
    years = []
    for formation_year in sorted(picks_by_year):
        held = returns_by_year[formation_year + 1][:, picks_by_year[formation_year]]
        years.append(np.cumsum(held, axis=1)[:, columns] / counts)
    return np.concatenate(years)


def holding_period_risk_free(rf_by_year: dict[int, np.ndarray], formation_years: list[int]) -> np.ndarray:
    """Daily risk-free returns over the holding years, matching portfolio_returns."""
    return np.concatenate([rf_by_year[y + 1] for y in sorted(formation_years)])


def evaluate_portfolios(
    returns_by_year: dict[int, np.ndarray],
    rf_by_year: dict[int, np.ndarray],
    picks_by_year: dict[int, np.ndarray],
    sizes: tuple[int, ...],
) -> list[dict[str, float]]:
    """All metrics for each portfolio size, in the order of sizes."""
    daily = portfolio_returns(returns_by_year, picks_by_year, sizes)
    rf = holding_period_risk_free(rf_by_year, list(picks_by_year))
    return [all_metrics(daily[:, k], rf) for k in range(len(sizes))]


def equal_weight_benchmark(
    returns_by_year: dict[int, np.ndarray],
    rf_by_year: dict[int, np.ndarray],
    holding_years: list[int],
) -> dict[str, float]:
    """All metrics for the equal-weight portfolio of every bond, rebalanced daily."""
    daily = np.concatenate([returns_by_year[y].mean(axis=1) for y in holding_years])
    rf = np.concatenate([rf_by_year[y] for y in holding_years])
    return all_metrics(daily, rf)
