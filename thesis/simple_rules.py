"""
Baseline: rank bonds without building a graph.

Peripheral bonds might simply be bonds with low correlations or low
volatility. These rules rank bonds by such plain statistics of the formation
year; if they do as well as the graph strategies, the graph adds little.

Every rule has the same form:

    rule(C, volatilities) -> scores, one per bond

where C is the yearly correlation or scaled covariance matrix (step 2, as
listed in MATRIX_RULES) and volatilities the daily standard deviations of
that year. As with centralities, "central"
picks the highest scores and "peripheral" the lowest.
"""
import numpy as np


def average_correlation(C: np.ndarray, volatilities: np.ndarray) -> np.ndarray:
    """Average correlation with every other bond."""
    return (C.sum(axis=1) - np.diag(C)) / (C.shape[0] - 1)


def average_covariance(C: np.ndarray, volatilities: np.ndarray) -> np.ndarray:
    """
    Average covariance with every bond, itself included (C is the scaled
    covariance matrix here). This is proportional
    to the bond's contribution to the variance of the equal-weight portfolio
    of all bonds: Σ_j Σ_ij = Cov(r_i, Σ_j r_j).
    """
    return C.mean(axis=1)


def volatility(C: np.ndarray, volatilities: np.ndarray) -> np.ndarray:
    """The bond's own volatility; does not use C."""
    return volatilities


# Rules that use a matrix, and which one; they get one version per window
# weighting and shrinkage option. volatility has only one.
MATRIX_RULES = {average_correlation: "correlation", average_covariance: "covariance"}
