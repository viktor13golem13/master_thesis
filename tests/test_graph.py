import numpy as np
import pytest

from thesis.config import GRAPH_TYPES
from thesis.correlation import covariance_from_correlation
from thesis.graph import GraphType, adjacency_matrix, matched_threshold

from tests.helpers import C_TOY


# Figure 2 of the paper: the unweighted graphs of the toy matrix at θ = 0.25.
@pytest.mark.parametrize("paper_option, graph_type, expected", [
    (1, GraphType("positive", True, False), [[1, 0, 0, 1, 0], [0, 1, 1, 0, 0], [0, 1, 1, 1, 1], [1, 0, 1, 1, 0], [0, 0, 1, 0, 1]]),
    (2, GraphType("absolute", True, False), [[1, 0, 0, 1, 1], [0, 1, 1, 0, 0], [0, 1, 1, 1, 1], [1, 0, 1, 1, 0], [1, 0, 1, 0, 1]]),
    (3, GraphType("positive", False, False), [[0, 0, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 1], [1, 0, 1, 0, 0], [0, 0, 1, 0, 0]]),
    (4, GraphType("absolute", False, False), [[0, 0, 0, 1, 1], [0, 0, 1, 0, 0], [0, 1, 0, 1, 1], [1, 0, 1, 0, 0], [1, 0, 1, 0, 0]]),
])
def test_unweighted_graphs_match_paper_figure_2(paper_option, graph_type, expected):
    assert np.array_equal(adjacency_matrix(C_TOY, 0.25, graph_type), expected)


@pytest.mark.parametrize("graph_type", [g for g in GRAPH_TYPES if g.weighted])
def test_weighted_graphs_keep_the_correlation_on_each_edge(graph_type):
    A = adjacency_matrix(C_TOY, 0.25, graph_type)
    edges = adjacency_matrix(C_TOY, 0.25, GraphType(graph_type.sign, graph_type.loops, False))
    T = {"positive": C_TOY, "absolute": np.abs(C_TOY)}[graph_type.sign]
    assert np.allclose(A, edges * T)


def test_there_are_eight_distinct_graph_types():
    assert len(GRAPH_TYPES) == len(set(GRAPH_TYPES)) == 8


@pytest.mark.parametrize("sign", ["positive", "absolute"])
@pytest.mark.parametrize("threshold", [0.0, 0.3, 0.8])
def test_matched_threshold_gives_the_same_number_of_edges(sign, threshold):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(200, 20)) @ rng.normal(size=(20, 20)) * rng.random(20)
    C = np.corrcoef(X.T)
    C = (C + C.T) / 2                     # np.corrcoef is symmetric only up to rounding; the pipeline's C is exact
    M = covariance_from_correlation(C, X.std(axis=0, ddof=1))
    graph_type = GraphType(sign, loops=False, weighted=True)
    n_edges = (adjacency_matrix(C, threshold, graph_type) > 0).sum()
    A = adjacency_matrix(M, matched_threshold(M, C, threshold, graph_type), graph_type)
    assert (A > 0).sum() == n_edges
    assert (A >= 0).all()
