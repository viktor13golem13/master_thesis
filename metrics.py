import numpy as np
import pandas as pd


def portfolio_return(
    dataset: pd.DataFrame,
    portfolio: list[tuple[str, float]],
) -> float:
    """
    Calculate the weighted return for a portfolio over the given period.

    Parameters
    ----------
    dataset : pd.DataFrame
        Bond index data with a 'Date' column and one column per bond.
        Rows should be in chronological order.
    portfolio : list of (bond_name, fraction) tuples
        Fractions should sum to 1.

    Returns
    -------
    float
        Weighted portfolio return (e.g. 0.04 means 4%).
    """
    total = 0.0
    for bond, fraction in portfolio:
        series = dataset[bond].dropna()
        bond_return = (series.iloc[-1] - series.iloc[0]) / series.iloc[0]
        total += fraction * bond_return
    return total


def risk_free_rate(dataset: pd.DataFrame, rf_col: str) -> float:
    """
    Extract the annualised risk-free rate from a bond index column.

    Parameters
    ----------
    dataset : pd.DataFrame
        Bond index data with a 'Date' column.
    rf_col : str
        Column name of the risk-free instrument (e.g. a treasury bill index).

    Returns
    -------
    float
        Annualised risk-free return over the period covered by the dataset.
    """
    series = dataset[rf_col].dropna()
    n_days = len(series)
    total_return = (series.iloc[-1] - series.iloc[0]) / series.iloc[0]
    return (1 + total_return) ** (252 / n_days) - 1


def portfolio_volatility(
    dataset: pd.DataFrame,
    portfolio: list[tuple[str, float]],
    col_names: list[str],
    corr_matrix: np.ndarray,
) -> float:
    """
    Compute annualised portfolio volatility using sigma_p = sqrt(w^T Sigma w).

    The covariance matrix is built from the correlation matrix and individual
    bond return standard deviations: Sigma_ij = rho_ij * sigma_i * sigma_j.

    Parameters
    ----------
    dataset : pd.DataFrame
        Bond index data used to compute individual standard deviations.
    portfolio : list of (bond_name, fraction) tuples
    col_names : list of str
        Column names corresponding to the axes of corr_matrix.
    corr_matrix : np.ndarray of shape (n, n)

    Returns
    -------
    float
        Annualised portfolio volatility.
    """
    value_cols = [c for c in dataset.columns if c != "Date"]
    daily_returns = dataset[value_cols].pct_change().dropna()

    stds = np.array([daily_returns[col].std() * np.sqrt(252) for col in col_names])

    D = np.diag(stds)
    cov = D @ corr_matrix @ D

    col_index = {name: i for i, name in enumerate(col_names)}
    w = np.zeros(len(col_names))
    for bond, fraction in portfolio:
        w[col_index[bond]] = fraction

    return np.sqrt(w @ cov @ w)


def sharpe_ratio(
    dataset: pd.DataFrame,
    portfolio: list[tuple[str, float]],
    col_names: list[str],
    corr_matrix: np.ndarray,
    rf_col: str,
) -> float:
    """
    Compute the Sharpe Ratio: SR = (ER - R_f) / sigma_p.

    Parameters
    ----------
    dataset : pd.DataFrame
        Bond index data for the evaluation period.
    portfolio : list of (bond_name, fraction) tuples
    col_names : list of str
        Column names corresponding to the axes of corr_matrix.
    corr_matrix : np.ndarray of shape (n, n)
    rf_col : str
        Column name of the risk-free instrument in dataset.

    Returns
    -------
    float
        Sharpe Ratio.
    """
    er = portfolio_return(dataset, portfolio)
    rf = risk_free_rate(dataset, rf_col)
    vol = portfolio_volatility(dataset, portfolio, col_names, corr_matrix)
    return (er - rf) / vol
