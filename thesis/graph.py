"""
Step 3 — From a correlation matrix to a graph (paper, Sec. 3.2).

Each bond index is a node. Two nodes are joined by an edge when their
correlation passes the threshold θ; in a weighted graph the edge keeps the
correlation as its weight, in an unweighted graph every edge has weight 1.
"""
from dataclasses import dataclass
from functools import cached_property

import numpy as np


@dataclass(frozen=True)
class GraphType:
    """
    Which correlations become edges, and how.

    sign
        "positive"  edges where C_ij > θ          (paper transforms A1, A2)
        "negative"  edges where -C_ij > θ         (paper transform A3, (-C)_+)
        "absolute"  edges where |C_ij| > θ        (paper transform A4)
    loops
        Keep each node's edge to itself (paper options 1, 2, 5, 6)
        or remove it (options 3, 4, 7, 8).
    weighted
        Edges carry the correlation value (options 5-8) or 1 (options 1-4).

    Because θ >= 0, the paper's 4 transforms x 8 options give only 10
    different graphs: raw C and its positive part (A1, A2) keep the same
    edges, and the negative part has no loops since its diagonal is 0.
    config.GRAPH_TYPES lists all 10.
    """
    sign: str
    loops: bool
    weighted: bool


def adjacency_matrix(C: np.ndarray, threshold: float, graph_type: GraphType) -> np.ndarray:
    """
    Weighted:   A = [T > θ] ∘ T      (keep values above θ, zero the rest)
    Unweighted: A = [T > θ]          (1 above θ, 0 otherwise)

    where T is C, -C or |C| according to graph_type.sign, with the diagonal
    set to 0 unless graph_type.loops.
    """
    if graph_type.sign == "positive":
        T = C.copy()
    elif graph_type.sign == "negative":
        T = -C
    elif graph_type.sign == "absolute":
        T = np.abs(C)
    else:
        raise ValueError(f"Unknown sign {graph_type.sign!r}")
    if not graph_type.loops:
        np.fill_diagonal(T, 0.0)
    edges = T > threshold
    return np.where(edges, T, 0.0) if graph_type.weighted else edges.astype(float)


class Graph:
    """
    An adjacency matrix plus quantities that several centralities share.

    Each shared quantity (such as the eigendecomposition) is computed the
    first time it is needed and then reused, so computing all 59 centrality
    variants on a graph costs little more than computing one.
    """

    def __init__(self, adjacency: np.ndarray, weighted: bool):
        self.A = adjacency
        self.weighted = weighted
        self.n_nodes = adjacency.shape[0]
        self._cache = {}

    @cached_property
    def eigendecomposition(self) -> tuple[np.ndarray, np.ndarray]:
        """Eigenvalues λ (ascending) and orthonormal eigenvectors Q, with A = Q diag(λ) Qᵀ."""
        return np.linalg.eigh(self.A)

    @cached_property
    def spectral_radius(self) -> float:
        """ρ(A), the largest absolute eigenvalue."""
        return float(np.max(np.abs(self.eigendecomposition[0])))

    @cached_property
    def has_edges(self) -> bool:
        return self.spectral_radius > 1e-12

    @cached_property
    def isolated_fraction(self) -> float:
        """Share of nodes with no edge to any other node."""
        off_diagonal = self.A - np.diag(np.diag(self.A))
        return float(np.mean(off_diagonal.sum(axis=1) == 0))

    def row_sums_of(self, f) -> np.ndarray:
        """f(A)·1 for a function f of the eigenvalues: Q f(λ) Qᵀ1."""
        lam, Q = self.eigendecomposition
        return Q @ (f(lam) * (Q.T @ np.ones(self.n_nodes)))

    def diagonal_of(self, f) -> np.ndarray:
        """diag f(A) for a function f of the eigenvalues: (Q ∘ Q) f(λ)."""
        lam, Q = self.eigendecomposition
        return (Q ** 2) @ f(lam)

    def cached(self, key, compute):
        """Return compute(), computing it only the first time key is requested."""
        if key not in self._cache:
            self._cache[key] = compute()
        return self._cache[key]
