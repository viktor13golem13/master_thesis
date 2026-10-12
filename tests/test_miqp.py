import itertools

import numpy as np
import pytest

pytest.importorskip("gurobipy")

from thesis.correlation import single_index_shrinkage
from thesis.miqp import best_bonds


def shrunk_covariance(returns):
    """δF + (1 - δ)S written out in full, for checking the compact formulation."""
    T = len(returns)
    x = returns - returns.mean(axis=0)
    S = x.T @ x / T
    market = x.mean(axis=1)
    c = x.T @ market / T
    F = np.outer(c, c) / (market @ market / T)
    np.fill_diagonal(F, np.diag(S))
    delta, _ = single_index_shrinkage(returns)
    return delta * F + (1 - delta) * S


@pytest.mark.parametrize("objective", ["min_variance", "min_variance_target_return"])
def test_best_bonds_matches_brute_force(objective):
    rng = np.random.default_rng(0)
    market = rng.normal(size=(60, 1))
    returns = 0.004 * market @ rng.random((1, 12)) + 0.005 * rng.normal(size=(60, 12)) + 0.001 * rng.normal(size=12)
    cov = shrunk_covariance(returns)
    means = returns.mean(axis=0)

    feasible = [set(c) for c in itertools.combinations(range(12), 3)
                if objective == "min_variance" or means[list(c)].mean() >= returns.mean()]
    expected = min(feasible, key=lambda c: cov[np.ix_(list(c), list(c))].sum())

    chosen, gap = best_bonds(returns, 3, objective, time_limit=60)
    assert set(chosen) == expected
    assert gap < 1e-4


@pytest.mark.parametrize("objective", ["min_variance", "min_variance_target_return"])
def test_best_weights_matches_scipy(objective):
    from scipy.optimize import minimize
    from thesis.miqp import best_weights

    rng = np.random.default_rng(1)
    market = rng.normal(size=(60, 1))
    returns = 0.004 * market @ rng.random((1, 12)) + 0.005 * rng.normal(size=(60, 12)) + 0.001 * rng.normal(size=12)
    cov = shrunk_covariance(returns)
    means = returns.mean(axis=0)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if objective == "min_variance_target_return":
        constraints.append({"type": "ineq", "fun": lambda w: means @ w - returns.mean()})
    reference = minimize(lambda w: w @ cov @ w, np.full(12, 1 / 12), bounds=[(0, 1 / 3)] * 12,
                         constraints=constraints, method="SLSQP", options={"ftol": 1e-15, "maxiter": 1000}).x

    weights = best_weights(returns, 3, objective, time_limit=60)
    assert np.isclose(weights.sum(), 1) and weights.max() <= 1 / 3 + 1e-9
    assert weights @ cov @ weights <= (reference @ cov @ reference) * (1 + 1e-6)   # at least as good as scipy
    assert np.allclose(weights, reference, atol=1e-3)


def test_best_bonds_is_exact_for_realistically_small_returns():
    # Daily bond returns of ~0.01% give variances near 1e-8, far below the
    # solver's default tolerances; best_bonds must rescale to stay exact.
    rng = np.random.default_rng(2)
    market = rng.normal(size=(250, 1))
    returns = 1e-4 * (market @ rng.random((1, 12)) + rng.normal(size=(250, 12)) * rng.uniform(0.05, 1, 12))
    cov = shrunk_covariance(returns)
    expected = min((set(c) for c in itertools.combinations(range(12), 3)),
                   key=lambda c: cov[np.ix_(list(c), list(c))].sum())
    chosen, _ = best_bonds(returns, 3, "min_variance", time_limit=60)
    assert set(chosen) == expected
