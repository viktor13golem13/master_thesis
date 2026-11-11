"""
Step 2 — One correlation matrix per year (paper, Sec. 3.1).

For each year, Pearson correlations are computed over rolling windows of 125
trading days; every window that fits inside the year (about 137) is then
averaged into a single matrix C. The paper averages only the last 125. Optionally C is shrunk towards a simpler
single-index model (Ledoit & Wolf 2003) to reduce estimation noise.

The covariance graph uses the same windows, weights and shrinkage, but
averages the windows' covariances instead of their correlations, and divides
the result by the largest variance (correlation_for_year with covariance=True).
"""
import numpy as np
import pandas as pd

from thesis import config


# ------------------------------------------------------- rolling correlation

def window_weights(n_windows: int, window_length: int, weighting: str) -> np.ndarray:
    """
    How much each rolling window counts in the yearly average, oldest first.

    "uniform"  every window counts the same.
    "paper"    the formula as printed in the paper: the window ending t days
               before year end gets weight exp((t - τ) / τ), so older windows
               count more (the oldest about e ≈ 2.7 times the newest).
    "recent"   the mirror image of "paper": newer windows count more, in the
               spirit of Pozzi et al. (2013).
    """
    if weighting == "uniform":
        return np.full(n_windows, 1.0 / n_windows)
    t = np.arange(n_windows, 0, -1)          # oldest window has the largest t
    weights = np.exp((t - window_length) / window_length)
    if weighting == "recent":
        weights = weights[::-1]
    elif weighting != "paper":
        raise ValueError(f"Unknown weighting {weighting!r}; expected one of {config.WINDOW_WEIGHTINGS}")
    return weights / weights.sum()


def window_correlation(returns: np.ndarray) -> np.ndarray:
    """
    Pearson correlation matrix of one window of daily returns (days x series).
    A series that does not move in the window gets correlation 0 with all others.
    """
    centred = returns - returns.mean(axis=0)
    norms = np.sqrt((centred ** 2).sum(axis=0))
    norms[norms == 0] = np.inf
    standardised = centred / norms
    return standardised.T @ standardised


def window_covariance(returns: np.ndarray) -> np.ndarray:
    """Sample covariance matrix of one window of daily returns (days x series)."""
    centred = returns - returns.mean(axis=0)
    return centred.T @ centred / (len(returns) - 1)


def yearly_correlation(
    returns: np.ndarray,
    window_length: int = config.WINDOW_LENGTH,
    weighting: str = "uniform",
    covariance: bool = False,
) -> np.ndarray:
    """
    Weighted average of the rolling-window correlation matrices of one year,
    or of the covariance matrices if covariance is True.

    Parameters
    ----------
    returns : np.ndarray of shape (days in the year, series)
    window_length : int
        τ, the window length in trading days.
    weighting : str
        See window_weights.
    covariance : bool
        Average covariance matrices instead of correlation matrices.

    Returns
    -------
    np.ndarray of shape (series, series); correlations have 1 on the diagonal.
    """
    n_days, n_series = returns.shape
    n_windows = n_days - window_length + 1          # every window that fits in the year
    if n_windows < 1:
        raise ValueError(f"The year has {n_days} days, fewer than the window length {window_length}")
    weights = window_weights(n_windows, window_length, weighting)
    window_ends = range(n_days - n_windows + 1, n_days + 1)

    C = np.zeros((n_series, n_series))
    for weight, end in zip(weights, window_ends):
        window = returns[end - window_length:end]
        C += weight * (window_covariance(window) if covariance else window_correlation(window))
    if not covariance:
        np.fill_diagonal(C, 1.0)
    return C


# ------------------------------------------------------------------ shrinkage

def single_index_shrinkage(returns: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Ledoit & Wolf (2003): the optimal weight δ on the single-index model.

    The single-index model explains every covariance through one common
    "market" factor, here the equal-weight average of all series. The paper's
    estimator is δ = clip(κ / T, 0, 1) with κ = (π - ρ) / γ, where
        π  measures the estimation noise in the sample covariance S,
        ρ  corrects π for noise shared with the model F,
        γ  is the distance between S and F.
    Direct port of Ledoit and Wolf's reference code (covMarket).

    Returns
    -------
    shrinkage : float
        δ in [0, 1].
    target_correlation : np.ndarray
        The single-index model F written as a correlation matrix.
    """
    T = returns.shape[0]
    x = returns - returns.mean(axis=0)
    market = x.mean(axis=1)

    S = x.T @ x / T                                   # sample covariance
    cov_market = x.T @ market / T                     # covariance of each series with the market
    var_market = market @ market / T
    F = np.outer(cov_market, cov_market) / var_market
    np.fill_diagonal(F, np.diag(S))                   # the model keeps the sample variances

    gamma = np.linalg.norm(S - F, "fro") ** 2

    x2 = x ** 2
    pi = (x2.T @ x2).sum() / T - (S ** 2).sum()

    rho_diag = (x2 ** 2).sum() / T - (np.diag(S) ** 2).sum()
    z = x * market[:, None]
    v1 = x2.T @ z / T - cov_market[:, None] * S
    rho_off1 = ((v1 * cov_market[None, :]).sum() - (np.diag(v1) * cov_market).sum()) / var_market
    v3 = z.T @ z / T - var_market * S
    rho_off3 = ((v3 * np.outer(cov_market, cov_market)).sum()
                - (np.diag(v3) * cov_market ** 2).sum()) / var_market ** 2
    rho = rho_diag + 2 * rho_off1 - rho_off3

    shrinkage = float(np.clip((pi - rho) / gamma / T, 0.0, 1.0))

    sd = np.sqrt(np.diag(S))
    sd[sd == 0] = np.inf
    target_correlation = F / np.outer(sd, sd)
    np.fill_diagonal(target_correlation, 1.0)
    return shrinkage, target_correlation


def shrink_correlation(C: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """
    Pull C towards the single-index model: δ·F + (1 - δ)·C (paper, Sec. 3.1).

    The paper shrinks the covariance and converts back to correlations; since
    both matrices are rescaled by the same standard deviations, that is the
    same as mixing the correlation matrices directly.
    """
    shrinkage, target = single_index_shrinkage(returns)
    return shrinkage * target + (1.0 - shrinkage) * C


def shrink_covariance(S: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """
    Pull the covariance S towards the single-index model: δ·F + (1 - δ)·S,
    the paper's form of the shrinkage. F is the single-index correlation
    turned back into a covariance with the year's sample variances.
    """
    shrinkage, target = single_index_shrinkage(returns)
    sd = returns.std(axis=0)                        # ddof 0, as in single_index_shrinkage
    return shrinkage * target * np.outer(sd, sd) + (1.0 - shrinkage) * S


# ---------------------------------------------------------------- per year

def correlation_for_year(
    returns: pd.DataFrame,
    year: int,
    weighting: str = "uniform",
    shrink: bool = False,
    covariance: bool = False,
) -> np.ndarray:
    """
    The (optionally shrunk) yearly correlation matrix of one calendar year.

    With covariance=True, the covariance matrix divided by its largest
    variance instead. Every entry then stays in [-1, 1]: the matrix is
    positive semidefinite, so |C_ij| <= √(C_ii C_jj) <= max C_kk.
    Pairs of calm bonds get weak edges.
    """
    X = returns.loc[returns.index.year == year].to_numpy(dtype=float)
    C = yearly_correlation(X, weighting=weighting, covariance=covariance)
    if shrink:
        C = shrink_covariance(C, X) if covariance else shrink_correlation(C, X)
    if covariance:
        C = C / np.diag(C).max()
    return C


def volatilities_for_year(returns: pd.DataFrame, year: int) -> np.ndarray:
    """Standard deviation of each series' daily returns in one calendar year."""
    return returns.loc[returns.index.year == year].to_numpy(dtype=float).std(axis=0, ddof=1)
