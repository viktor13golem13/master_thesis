import math

import numpy as np
import pytest
from scipy.linalg import expm

from thesis.centrality import (
    degree, exponential, exponential_subgraph, katz, katz_min, katz_subgraph,
    nbtw_exponential, nbtw_exponential_subgraph, nbtw_matrix, nbtw_radius, rank_by_centrality,
)
from thesis.graph import Graph

from tests.helpers import random_weighted_graph


def test_walk_based_centralities_match_direct_formulas():
    A = random_weighted_graph(12, seed=4, loops=True)
    graph = Graph(A, weighted=True)
    rho = np.max(np.abs(np.linalg.eigvalsh(A)))
    one = np.ones(12)

    resolvent = np.linalg.inv(np.eye(12) - 0.5 / rho * A)
    assert np.allclose(katz(graph, 0.5), resolvent @ one)
    assert np.allclose(katz_subgraph(graph, 0.5), np.diag(resolvent))
    assert np.allclose(degree(graph), A @ one)

    # exponentials are divided by e^{0.7 λ_max}, which does not change rankings
    E = expm(0.7 * A) * np.exp(-0.7 * np.linalg.eigvalsh(A).max())
    assert np.allclose(exponential(graph, 0.7), E @ one)
    assert np.allclose(exponential_subgraph(graph, 0.7), np.diag(E))


def test_exponential_does_not_overflow_on_dense_graphs():
    A = np.ones((745, 745)) - np.eye(745)       # ρ = 744: e^744 overflows a float
    assert np.all(np.isfinite(exponential(Graph(A, weighted=True), 1.0)))


def test_katz_min_matches_direct_formula_for_small_rho():
    A = random_weighted_graph(12, seed=9) * 0.3
    rho = np.max(np.abs(np.linalg.eigvalsh(A)))
    alpha = (1 - np.exp(-rho)) / rho
    direct = np.linalg.solve(np.eye(12) - alpha * A, np.ones(12))
    scores = katz_min(Graph(A, weighted=True))
    assert np.allclose(scores / scores.sum(), direct / direct.sum())


def test_katz_min_tends_to_eigenvector_centrality_for_large_rho():
    A = random_weighted_graph(60, seed=10, density=0.9) * 3     # ρ well above 37
    eigenvector = np.abs(np.linalg.eigh(A)[1][:, -1])
    scores = katz_min(Graph(A, weighted=True))
    assert np.all(np.isfinite(scores))
    assert np.array_equal(np.argsort(scores), np.argsort(eigenvector))


def test_graph_without_edges_gives_equal_scores():
    graph = Graph(np.zeros((5, 5)), weighted=True)
    assert np.allclose(katz(graph, 0.5), 1)


# ------------------------------------------------------------------- NBTW

def test_nbtw_matrix_matches_unweighted_closed_form():
    A = (random_weighted_graph(10, seed=5) > 0).astype(float)
    D = np.diag(A.sum(axis=1))
    alpha = 0.2
    closed_form = (np.eye(10) - alpha * A + alpha ** 2 * (D - np.eye(10))) / (1 - alpha ** 2)
    assert np.allclose(nbtw_matrix(A, alpha), closed_form)


def count_nonbacktracking_walks(A, alpha, max_length):
    """Σ_{k <= max_length} α^k P_k, counting walks step by step, remembering the previous node."""
    n = A.shape[0]
    total = np.eye(n) + alpha * A
    # W[s, p, c]: weighted walks of the current length from s, previous node p, now at c
    W = np.einsum("sc,sp->spc", A, np.eye(n))
    for k in range(2, max_length + 1):
        step = np.einsum("spc,cd->scd", W, A)
        step_back = np.einsum("spc,cp->scp", W, A)        # the forbidden move c -> p
        W = step - step_back
        total += alpha ** k * W.sum(axis=1)
    return total


@pytest.mark.parametrize("loops", [False, True])
def test_nbtw_inverse_counts_nonbacktracking_walks(loops):
    A = random_weighted_graph(7, seed=6, loops=loops)
    alpha = 0.3 * nbtw_radius(A)
    counted = count_nonbacktracking_walks(A, alpha, 60)
    assert np.allclose(np.linalg.inv(nbtw_matrix(A, alpha)), counted, atol=1e-10)


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


# ---------------------------------------------------------------- ranking

def test_ties_are_broken_randomly_not_by_column_order():
    scores = np.array([3.0, 1.0, 1.0 + 1e-15, 1.0, 2.0])
    tails = {tuple(rank_by_centrality(scores, np.random.default_rng(s))[2:]) for s in range(20)}
    assert all(set(tail) == {1, 2, 3} for tail in tails)
    assert len(tails) > 1
    assert list(rank_by_centrality(scores, np.random.default_rng(0))[:2]) == [0, 4]


# ------------------------------------------------------ exponential NBTW

def nonbacktracking_walks_by_length(A, max_length):
    """[p_0(A), p_1(A), ...]: weighted NBTW counts of each length, counted step by step."""
    n = A.shape[0]
    counts = [np.eye(n), A.copy()]
    W = np.einsum("sc,sp->spc", A, np.eye(n))
    for _ in range(2, max_length + 1):
        W = np.einsum("spc,cd->scd", W, A) - np.einsum("spc,cp->scp", W, A)
        counts.append(W.sum(axis=1))
    return counts


@pytest.mark.parametrize("loops", [False, True])
def test_nbtw_exponential_matches_brute_force_walk_count(loops):
    A = (random_weighted_graph(8, seed=11, loops=loops, density=0.5) > 0).astype(float)
    A[7, :] = A[:, 7] = 0
    A[7, 0] = A[0, 7] = 1                                    # node 7 is a leaf
    graph = Graph(A, weighted=False)
    c = max(np.max(np.abs(np.linalg.eigvalsh(A))), 1.0)
    p = nonbacktracking_walks_by_length(A, 60)
    for beta in (0.3, 1.0):
        F = sum(beta ** k * p[k] / math.factorial(k) for k in range(61)) * np.exp(-beta * c)
        assert np.allclose(nbtw_exponential(graph, beta), F.sum(axis=1))
        assert np.allclose(nbtw_exponential_subgraph(graph, beta), np.diag(F))


def test_nbtw_exponential_is_skipped_on_weighted_graphs():
    graph = Graph(random_weighted_graph(6, seed=12), weighted=True)
    assert nbtw_exponential(graph, 0.5) is None
    assert nbtw_exponential_subgraph(graph, 0.5) is None


def test_nbtw_exponential_does_not_overflow_on_dense_graphs():
    A = np.ones((300, 300)) - np.eye(300)                     # ρ = 299
    scores = nbtw_exponential(Graph(A, weighted=False), 1.0)
    assert np.all(np.isfinite(scores)) and np.all(scores > 0)
