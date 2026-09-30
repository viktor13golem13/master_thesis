import numpy as np
import pytest

from thesis.config import GRAPH_TYPES
from thesis.graph import GraphType, adjacency_matrix

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
