import numpy as np
import pandas as pd

from metrics import portfolio_return, risk_free_rate

RF_COL = "FTSE 3-Month Treasury Bill Index - Total Return"


def make_portfolio(ranking: list, top_n: int, use_top: bool) -> list[tuple[str, float]]:
    """Select top_n or bottom_n bonds from a centrality ranking, equal-weighted."""
    bonds = ranking[:top_n] if use_top else ranking[-top_n:]
    fraction = 1.0 / top_n
    return [(bond, fraction) for bond, _ in bonds]


def avg_sharpe_from_rankings(
    df: pd.DataFrame,
    year_rankings: dict,
    top_n: int,
    use_top: bool,
) -> float:
    """
    Evaluate a strategy by averaging per-year Sharpe ratios.

    Each year is treated independently: portfolio return and volatility are
    computed within that year and combined into a Sharpe ratio, then the
    results are averaged across years. Equivalent to restarting with 1 euro
    each year regardless of prior performance.
    """
    from metrics import sharpe_ratio
    sharpes = []
    for eval_year, (last_ranking, last_matrix, col_names) in year_rankings.items():
        portfolio = make_portfolio(last_ranking, top_n, use_top)
        year_df = df[df["Date"].dt.year == eval_year]
        try:
            sr = sharpe_ratio(year_df, portfolio, col_names, last_matrix, RF_COL)
            if np.isfinite(sr):
                sharpes.append(sr)
        except Exception:
            pass
    return np.mean(sharpes) if sharpes else float("nan")


def geometric_sharpe_from_rankings(
    df: pd.DataFrame,
    year_rankings: dict,
    top_n: int,
    use_top: bool,
) -> float:
    """
    Evaluate a strategy using a compounded (geometric) Sharpe ratio.

    Returns are compounded across years — profits and losses carry forward
    into the next year's base. The Sharpe is computed from the annualised
    geometric return and the standard deviation of annual returns.

    Example: +10% year 1, -10% year 2 → 1.10 × 0.90 = 0.99 → -1% total,
    not 0% as arithmetic averaging would suggest.
    """
    annual_returns = []
    annual_rf_rates = []

    for eval_year, (last_ranking, last_matrix, col_names) in year_rankings.items():
        portfolio = make_portfolio(last_ranking, top_n, use_top)
        year_df = df[df["Date"].dt.year == eval_year]
        try:
            ret = portfolio_return(year_df, portfolio)
            rf = risk_free_rate(year_df, RF_COL)
            annual_returns.append(ret)
            annual_rf_rates.append(rf)
        except Exception:
            pass

    if len(annual_returns) < 2:
        return float("nan")

    r = np.array(annual_returns)
    rf = np.array(annual_rf_rates)

    n_years = len(r)
    annualized_return = (np.prod(1 + r)) ** (1 / n_years) - 1
    avg_rf = np.mean(rf)
    vol = np.std(r, ddof=1)

    if vol == 0:
        return float("nan")

    return (annualized_return - avg_rf) / vol
