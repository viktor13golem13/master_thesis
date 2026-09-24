import numpy as np
import pandas as pd

WINDOW_WEIGHTINGS = ("uniform", "paper", "recent")


def daily_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Daily simple returns of price levels indexed by Date (first row dropped)."""
    return prices.pct_change(fill_method=None).iloc[1:]


def window_weights(n_windows: int, tau: int, weighting: str) -> np.ndarray:
    """
    Weights for the last n_windows rolling windows of a year, oldest first.

    weighting
        "uniform" — every window gets the same weight.
        "paper"   — as printed in Arslan, Noferini & Vrontos (2024, Sec. 3.1): the
                    window ending t trading days before year end (t = 1, ..., tau)
                    gets w(t) = w0 * exp((t - tau) / tau), so older windows weigh
                    more (the oldest about e times the newest).
        "recent"  — the mirror image of "paper": newer windows weigh more, in the
                    spirit of the exponential smoothing of Pozzi et al. (2013).
    """
    if weighting == "uniform":
        return np.full(n_windows, 1.0 / n_windows)
    t = np.arange(n_windows, 0, -1)  # oldest window has the largest t
    w = np.exp((t - tau) / tau)
    if weighting == "recent":
        w = w[::-1]
    elif weighting != "paper":
        raise ValueError(f"Unknown window weighting {weighting!r}; expected one of {WINDOW_WEIGHTINGS}.")
    return w / w.sum()


def yearly_correlation(returns: np.ndarray, tau: int = 125, weighting: str = "uniform") -> np.ndarray:
    """
    Weighted average of rolling Pearson correlation matrices within one year.

    Uses the last tau windows of length tau that fit inside the year (fewer if
    the year is shorter than 2 * tau - 1 days). Series with zero variance in a
    window contribute correlation 0 for that window.

    Parameters
    ----------
    returns : np.ndarray of shape (n_days, n_series)
        Daily returns for a single calendar year.
    tau : int
        Window length in trading days (default 125, as in the paper).
    weighting : str
        How the windows are weighted; see window_weights.

    Returns
    -------
    np.ndarray of shape (n_series, n_series)
        Averaged correlation matrix with 1 on the diagonal.
    """
    n_days, p = returns.shape
    n_windows = min(tau, n_days - tau + 1)
    if n_windows < 1:
        raise ValueError(f"Year has {n_days} days, fewer than the window length {tau}.")
    weights = window_weights(n_windows, tau, weighting)

    C = np.zeros((p, p))
    first_end = n_days - n_windows + 1
    for w, end in zip(weights, range(first_end, n_days + 1)):
        X = returns[end - tau:end]
        X = X - X.mean(axis=0)
        std = np.sqrt((X ** 2).sum(axis=0))
        std[std == 0] = np.inf
        Z = X / std
        C += w * (Z.T @ Z)
    np.fill_diagonal(C, 1.0)
    return C


def single_index_shrinkage(returns: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Ledoit & Wolf (2003) shrinkage towards the single-index model.

    The market is the equal-weight average of all series. This is a direct
    port of Ledoit and Wolf's reference implementation (covMarket).

    Parameters
    ----------
    returns : np.ndarray of shape (n_days, n_series)

    Returns
    -------
    shrinkage : float in [0, 1]
        Optimal weight on the single-index target.
    target_corr : np.ndarray of shape (n_series, n_series)
        The single-index target expressed as a correlation matrix.
    """
    t, n = returns.shape
    x = returns - returns.mean(axis=0)
    xmkt = x.mean(axis=1)

    sample = x.T @ x / t
    covmkt = x.T @ xmkt / t
    varmkt = xmkt @ xmkt / t
    prior = np.outer(covmkt, covmkt) / varmkt
    np.fill_diagonal(prior, np.diag(sample))

    c = np.linalg.norm(sample - prior, "fro") ** 2
    y = x ** 2
    p = (y.T @ y).sum() / t - (sample ** 2).sum()
    rdiag = (y ** 2).sum() / t - (np.diag(sample) ** 2).sum()
    z = x * xmkt[:, None]
    v1 = y.T @ z / t - covmkt[:, None] * sample
    roff1 = ((v1 * covmkt[None, :]).sum() - (np.diag(v1) * covmkt).sum()) / varmkt
    v3 = z.T @ z / t - varmkt * sample
    roff3 = ((v3 * np.outer(covmkt, covmkt)).sum() - (np.diag(v3) * covmkt ** 2).sum()) / varmkt ** 2
    r = rdiag + 2 * roff1 - roff3
    shrinkage = float(np.clip((p - r) / c / t, 0.0, 1.0))

    sd = np.sqrt(np.diag(sample))
    sd[sd == 0] = np.inf
    target_corr = prior / np.outer(sd, sd)
    np.fill_diagonal(target_corr, 1.0)
    return shrinkage, target_corr


def shrink_correlation(C: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """
    Shrink a correlation matrix towards the single-index model (paper, Sec. 3.1).

    The shrinkage weight and target are estimated from the same year's daily
    returns. Because both matrices are rescaled by the same standard
    deviations, shrinking the covariance and converting back is the same as
    taking this convex combination of correlation matrices.
    """
    shrinkage, target = single_index_shrinkage(returns)
    return shrinkage * target + (1.0 - shrinkage) * C


def compute_yearly_correlations(
    returns: pd.DataFrame,
    years: list[int],
    tau: int = 125,
    weighting: str = "uniform",
    shrink: bool = False,
) -> dict[int, np.ndarray]:
    """Map each requested calendar year to its averaged (optionally shrunk) correlation matrix."""
    result = {}
    for year in years:
        X = returns.loc[returns.index.year == year].to_numpy(dtype=float)
        C = yearly_correlation(X, tau, weighting)
        result[year] = shrink_correlation(C, X) if shrink else C
    return result
