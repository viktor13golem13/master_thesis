import numpy as np

from thesis.simple_rules import average_correlation, average_covariance, volatility

RETURNS = np.random.default_rng(0).normal(size=(260, 6)) * np.array([1, 2, 3, 1, 5, 2]) / 100
C = np.corrcoef(RETURNS.T)
SD = RETURNS.std(axis=0, ddof=1)


def test_average_correlation_leaves_out_the_bond_itself():
    expected = [np.mean([C[i, j] for j in range(6) if j != i]) for i in range(6)]
    assert np.allclose(average_correlation(C, SD), expected)


def test_average_covariance_ranks_by_contribution_to_equal_weight_variance():
    # Cov(r_i, Σ_j r_j): bond i's share of the variance of the equal-weight portfolio
    contribution = [np.cov(RETURNS[:, i], RETURNS.sum(axis=1))[0, 1] for i in range(6)]
    assert np.array_equal(np.argsort(average_covariance(np.cov(RETURNS.T), SD)), np.argsort(contribution))


def test_volatility_rule_is_the_bond_volatility():
    assert np.array_equal(volatility(None, SD), SD)
