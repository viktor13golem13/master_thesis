import numpy as np
import pytest
from scipy.linalg import expm

from centrality import (
    nbtw_matrix, nbtw_radius, rank_by_centrality, spectral_centralities,
)
from correlation import single_index_shrinkage, window_weights, yearly_correlation
from graph import GRAPH_TYPES, GraphType, adjacency_matrix
from metrics import performance_metrics

# Toy correlation matrix (3.2) of Arslan, Noferini & Vrontos (2024).
C_TOY = np.array([
    [1, -0.1378, 0.2025, 0.4683, -0.2583],
    [-0.1378, 1, 0.4373, 0.1050, -0.1738],
    [0.2025, 0.4373, 1, 0.4245, 0.4108],
    [0.4683, 0.1050, 0.4245, 1, -0.0465],
    [-0.2583, -0.1738, 0.4108, -0.0465, 1],
])


def random_weighted_graph(n, seed, loops=False, density=0.6):
    rng = np.random.default_rng(seed)
    A = rng.random((n, n)) * (rng.random((n, n)) < density)
    A = np.triu(A, 1)
    A = A + A.T
    if loops:
        A[np.diag_indices(n)] = rng.random(n)
    return A


# ---------------------------------------------------------------- correlation

def test_single_window_matches_numpy():
    X = np.random.default_rng(0).normal(size=(125, 5))
    assert np.allclose(yearly_correlation(X, tau=125), np.corrcoef(X.T))


def test_uniform_average_of_rolling_windows():
    X = np.random.default_rng(1).normal(size=(260, 5))
    expected = np.mean([np.corrcoef(X[e - 125:e].T) for e in range(136, 261)], axis=0)
    assert np.allclose(yearly_correlation(X, tau=125, weighting="uniform"), expected)


def test_window_weightings():
    paper = window_weights(125, 125, "paper")
    recent = window_weights(125, 125, "recent")
    assert paper[0] > paper[-1]                     # older windows weigh more
    assert np.allclose(recent, paper[::-1])
    assert np.isclose(paper[0] / paper[-1], np.exp(124 / 125))


def test_flat_series_has_zero_correlation():
    X = np.random.default_rng(2).normal(size=(125, 3))
    X[:, 2] = 0.0
    assert np.allclose(yearly_correlation(X, tau=125)[2], [0, 0, 1])


def ledoit_wolf_reference(x):
    """Shrinkage intensity by the formulas of Ledoit & Wolf (2003), written as loops."""
    t, n = x.shape
    y = x - x.mean(axis=0)
    m = y.mean(axis=1)
    s = y.T @ y / t
    s_m = y.T @ m / t
    s_00 = m @ m / t
    f = np.outer(s_m, s_m) / s_00
    np.fill_diagonal(f, np.diag(s))
    pi = sum(np.mean((y[:, i] * y[:, j] - s[i, j]) ** 2) for i in range(n) for j in range(n))
    rho = sum(np.mean((y[:, i] * y[:, i] - s[i, i]) ** 2) for i in range(n))
    for i in range(n):
        for j in range(n):
            if i != j:
                r = ((s_m[j] * s_00 * y[:, i] + s_m[i] * s_00 * y[:, j] - s_m[i] * s_m[j] * m)
                     * m * y[:, i] * y[:, j] / s_00 ** 2 - f[i, j] * s[i, j])
                rho += np.mean(r)
    gamma = ((f - s) ** 2).sum()
    return float(np.clip((pi - rho) / gamma / t, 0, 1))


def test_shrinkage_matches_reference_formulas():
    rng = np.random.default_rng(3)
    market = rng.normal(size=(60, 1))
    x = market @ rng.random((1, 8)) + 0.8 * rng.normal(size=(60, 8))
    shrinkage, target = single_index_shrinkage(x)
    assert 0 < shrinkage < 1
    assert np.isclose(shrinkage, ledoit_wolf_reference(x))
    assert np.allclose(np.diag(target), 1)


# ---------------------------------------------------------------------- graph

@pytest.mark.parametrize("option, graph_type, expected", [
    (1, GraphType("positive", True, False), [[1, 0, 0, 1, 0], [0, 1, 1, 0, 0], [0, 1, 1, 1, 1], [1, 0, 1, 1, 0], [0, 0, 1, 0, 1]]),
    (2, GraphType("absolute", True, False), [[1, 0, 0, 1, 1], [0, 1, 1, 0, 0], [0, 1, 1, 1, 1], [1, 0, 1, 1, 0], [1, 0, 1, 0, 1]]),
    (3, GraphType("positive", False, False), [[0, 0, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 1], [1, 0, 1, 0, 0], [0, 0, 1, 0, 0]]),
    (4, GraphType("absolute", False, False), [[0, 0, 0, 1, 1], [0, 0, 1, 0, 0], [0, 1, 0, 1, 1], [1, 0, 1, 0, 0], [1, 0, 1, 0, 0]]),
])
def test_unweighted_options_match_paper_figure_2(option, graph_type, expected):
    assert np.array_equal(adjacency_matrix(C_TOY, 0.25, graph_type), expected)


@pytest.mark.parametrize("graph_type", [g for g in GRAPH_TYPES if g.weighted])
def test_weighted_options_keep_values_above_threshold(graph_type):
    A = adjacency_matrix(C_TOY, 0.25, graph_type)
    unweighted = adjacency_matrix(C_TOY, 0.25, GraphType(graph_type.sign, graph_type.loops, False))
    T = {"positive": C_TOY, "negative": -C_TOY, "absolute": np.abs(C_TOY)}[graph_type.sign]
    assert np.allclose(A, unweighted * T)


def test_there_are_ten_distinct_graph_types():
    assert len(GRAPH_TYPES) == len(set(GRAPH_TYPES)) == 10


# ----------------------------------------------------------------- centrality

def test_spectral_centralities_match_direct_formulas():
    A = random_weighted_graph(12, seed=4, loops=True)
    rho = np.max(np.abs(np.linalg.eigvalsh(A)))
    scores = spectral_centralities(A)
    one = np.ones(12)
    R = np.linalg.inv(np.eye(12) - 0.5 / rho * A)
    assert np.allclose(scores[("katz", 0.5)], R @ one)
    assert np.allclose(scores[("katz_subgraph", 0.5)], np.diag(R))
    assert np.allclose(scores[("degree", None)], A @ one)
    E = expm(0.7 * A)
    lam_max = np.linalg.eigvalsh(A).max()
    scale = np.exp(-0.7 * lam_max)  # the shift rescales, rankings are unchanged
    assert np.allclose(scores[("exponential", 0.7)], scale * E @ one)
    assert np.allclose(scores[("exponential_subgraph", 0.7)], scale * np.diag(E))


def test_exponential_does_not_overflow_on_dense_graphs():
    A = np.ones((745, 745)) - np.eye(745)       # rho = 744 > log(max float)
    scores = spectral_centralities(A)[("exponential", 1.0)]
    assert np.all(np.isfinite(scores))


def test_katz_min_matches_direct_formula_for_small_rho():
    A = random_weighted_graph(12, seed=9) * 0.3
    rho = np.max(np.abs(np.linalg.eigvalsh(A)))
    alpha = (1 - np.exp(-rho)) / rho
    direct = np.linalg.solve(np.eye(12) - alpha * A, np.ones(12))
    scores = spectral_centralities(A)[("katz_min", None)]
    assert np.allclose(scores / scores.sum(), direct / direct.sum())


def test_katz_min_tends_to_eigenvector_centrality_for_large_rho():
    A = random_weighted_graph(60, seed=10, density=0.9) * 3     # rho well above 37
    lam, Q = np.linalg.eigh(A)
    eigenvector = np.abs(Q[:, -1])
    scores = spectral_centralities(A)[("katz_min", None)]
    assert np.all(np.isfinite(scores))
    assert np.array_equal(np.argsort(scores), np.argsort(eigenvector))


def test_nbtw_matches_unweighted_closed_form():
    A = (random_weighted_graph(10, seed=5) > 0).astype(float)
    D = np.diag(A.sum(axis=1))
    alpha = 0.2
    closed = (np.eye(10) - alpha * A + alpha ** 2 * (D - np.eye(10))) / (1 - alpha ** 2)
    assert np.allclose(nbtw_matrix(A, alpha), closed)


def nbtw_series(A, alpha, K):
    """Σ_{k<=K} α^k P_k by dynamic programming over (previous, current) node."""
    n = A.shape[0]
    total = np.eye(n) + alpha * A
    # W[s, p, c]: weighted NBTWs of the current length from s, previous node p, current node c
    W = np.einsum("sc,sp->spc", A, np.eye(n))
    for k in range(2, K + 1):
        step = np.einsum("spc,cd->scd", W, A)
        backtrack = np.einsum("spc,cp->scp", W, A)       # the forbidden move c -> p
        W = step - backtrack
        total += alpha ** k * W.sum(axis=1)
    return total


@pytest.mark.parametrize("loops", [False, True])
def test_nbtw_inverse_counts_nonbacktracking_walks(loops):
    A = random_weighted_graph(7, seed=6, loops=loops)
    alpha = 0.3 * nbtw_radius(A)
    assert np.allclose(np.linalg.inv(nbtw_matrix(A, alpha)), nbtw_series(A, alpha, 60), atol=1e-10)


@pytest.mark.parametrize("loops", [False, True])
def test_nbtw_radius_is_inverse_spectral_radius_of_edge_matrix(loops):
    A = random_weighted_graph(8, seed=7, loops=loops)
    arcs = [(u, v) for u in range(8) for v in range(8) if A[u, v] > 0]
    index = {arc: k for k, arc in enumerate(arcs)}
    B = np.zeros((len(arcs), len(arcs)))
    for (u, v) in arcs:
        for w in range(8):
            if A[v, w] > 0 and w != u:
                B[index[(u, v)], index[(v, w)]] = A[v, w]
    expected = 1 / np.max(np.abs(np.linalg.eigvals(B)))
    assert np.isclose(nbtw_radius(A), min(expected, 1 / A.max()), rtol=1e-3)


def test_ties_are_broken_randomly_not_by_column_order():
    scores = np.array([3.0, 1.0, 1.0 + 1e-15, 1.0, 2.0])
    orders = {tuple(rank_by_centrality(scores, np.random.default_rng(s))[2:]) for s in range(20)}
    assert all(set(o[:3]) == {1, 2, 3} for o in orders)
    assert len(orders) > 1
    assert list(rank_by_centrality(scores, np.random.default_rng(0))[:2]) == [0, 4]


# -------------------------------------------------------------------- metrics

def test_performance_metrics_basic_identities():
    rng = np.random.default_rng(8)
    rp = rng.normal(0.0004, 0.003, size=500)
    rf = np.full(500, 0.0001)
    m = performance_metrics(rp, rf)
    assert np.isclose(m["CR"], np.prod(1 + rp) - 1)
    assert np.isclose(m["SR"], (rp - rf).mean() / rp.std(ddof=1) * np.sqrt(252))
    assert 0 <= m["MaxDD"] < 1
    assert m["CVaR"] <= m["VaR"]
