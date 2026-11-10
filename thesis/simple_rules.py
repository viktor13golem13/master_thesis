"""
Baseline: rank bonds without building a graph.

Peripheral bonds might simply be bonds with low correlations or low
volatility. These rules rank bonds by such plain statistics of the formation
year; if they do as well as the graph strategies, the graph adds little.

Every rule has the same form:

    rule(C, volatilities) -> scores, one per bond

where C is the yearly correlation matrix (step 2) and volatilities the
daily standard deviations of that year. As with centralities, "central"
picks the highest scores and "peripheral" the lowest.
"""
import numpy as np

from thesis.correlation import covariance_from_correlation


def average_correlation(C: np.ndarray, volatilities: np.ndarray) -> np.ndarray:
    """Average correlation with every other bond."""
    return (C.sum(axis=1) - np.diag(C)) / (C.shape[0] - 1)


def average_covariance(C: np.ndarray, volatilities: np.ndarray) -> np.ndarray:
    """
    Average covariance with every bond, itself included. This is proportional
    to the bond's contribution to the variance of the equal-weight portfolio
    of all bonds: Σ_j Σ_ij = Cov(r_i, Σ_j r_j).
    """
    return covariance_from_correlation(C, volatilities).mean(axis=1)


def volatility(C: np.ndarray, volatilities: np.ndarray) -> np.ndarray:
    """The bond's own volatility; does not use C."""
    return volatilities


# Rules that depend on C get one version per window weighting and shrinkage
# option; volatility has only one.
CORRELATION_RULES = [average_correlation, average_covariance]
