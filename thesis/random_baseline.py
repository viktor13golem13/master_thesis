"""
Baseline: portfolios of m bonds chosen at random (literature study, Sec. 1.3).

For every formation year, m of the bonds are drawn uniformly at random and
held with equal weights during the following year, exactly like a graph
strategy. Many such portfolios give, for each m, the distribution of results
that luck alone produces under the same rules; a strategy is judged by the
share of random portfolios it beats.
"""
import numpy as np
import pandas as pd

from thesis import config
from thesis.backtest import evaluate_portfolio, split_by_year


def run_random_baseline(returns: pd.DataFrame, rf_returns: pd.Series,
                        formation_years: list[int], verbose: bool = True) -> pd.DataFrame:
    """One row per random portfolio: its size m, its draw number and all metrics."""
    holding_years = [year + 1 for year in formation_years]
    returns_by_year = split_by_year(returns, holding_years)
    rf_by_year = split_by_year(rf_returns, holding_years)
    n_bonds = returns.shape[1]
    rng = np.random.default_rng(config.RANDOM_SEED)

    rows = []
    for m in config.PORTFOLIO_SIZES:
        for draw in range(config.RANDOM_PORTFOLIOS):
            holdings = {year + 1: rng.choice(n_bonds, size=m, replace=False) for year in formation_years}
            rows.append({"m": m, "draw": draw, **evaluate_portfolio(returns_by_year, rf_by_year, holdings)})
        if verbose:
            print(f"  m={m}: {config.RANDOM_PORTFOLIOS} random portfolios", flush=True)
    return pd.DataFrame(rows)


def share_beaten(value: float, random_values: pd.Series) -> float:
    """Share of random portfolios with a lower value of the metric (0.95 = better than 95% of them)."""
    return float((random_values < value).mean())
