"""
Step 3 — From a correlation matrix to a graph (paper, Sec. 3.2).

Each bond index is a node. Two nodes are joined by an edge when their
correlation passes the threshold θ; in a weighted graph the edge keeps the
correlation as its weight, in an unweighted graph every edge has weight 1.

The covariance graph is built the same way from the scaled covariance
matrix, but with the threshold from matched_threshold, so that it has as
many edges as the correlation graph at θ.
"""
from dataclasses import dataclass
from functools import cached_property

import numpy as np

# "damped" graphs: a negative correlation x counts as c·|x|, between "positive"
# (c = 0, negatives ignored) and "absolute" (c = 1). It becomes an edge only if
# |x| > θ / c, so it must be stronger than a positive correlation to count.
NEGATIVE_DAMPING = 0.75


@dataclass(frozen=True)
class GraphType:
    """
    Which correlations become edges, and how.

    sign
        "positive"  edges where C_ij > θ          (paper transforms A1, A2)
        "absolute"  edges where |C_ij| > θ        (paper transforms A3, A4)
        "damped"    edges where max(C_ij, -c·C_ij) > θ, c = NEGATIVE_DAMPING
                    (not in the paper)
    loops
        Keep each node's edge to itself (paper options 1, 2, 5, 6)
        or remove it (options 3, 4, 7, 8).
    weighted
        Edges carry the correlation value (options 5-8) or 1 (options 1-4).

    Because θ >= 0, the paper's 4 transforms x 8 options give only 8
    different graphs: raw C and its positive part (A1, A2) keep the same
    edges. The paper describes A3 as "the positive part of the negative
    elements", but its A3 results (Tables 6-7) are identical to its |C|
    results (Tables 2-3), so A3 is treated as |C| here. The literal reading,
    edges where -C_ij > θ, leaves bond graphs almost empty above θ = 0.2.
    config.GRAPH_TYPES lists all 8.
    """
    sign: str
    loops: bool
    weighted: bool


def adjacency_matrix(C: np.ndarray, threshold: float, graph_type: GraphType) -> np.ndarray:
    """
    Weighted:   A = [T > θ] ∘ T      (keep values above θ, zero the rest)
    Unweighted: A = [T > θ]          (1 above θ, 0 otherwise)

    where T is C, |C| or max(C, -cC) according to graph_type.sign, with the diagonal set
    to 0 unless graph_type.loops.
    """
    T = _signed(C, graph_type)
    if not graph_type.loops:
        np.fill_diagonal(T, 0.0)
    edges = T > threshold
    return np.where(edges, T, 0.0) if graph_type.weighted else edges.astype(float)


def matched_threshold(M: np.ndarray, C: np.ndarray, threshold: float, graph_type: GraphType) -> float:
    """
    The threshold that gives the graph of M as many edges as the graph of C
    has at `threshold` (loops not counted): the strongest entries of M are kept.

    Used for the covariance graph. Covariances are much smaller than
    correlations, so at the same θ the covariance graph would be almost empty;
    matching the number of edges compares the two graphs at equal density.
    Never below 0, so the graph never gets negative weights.
    """
    pairs = np.triu_indices_from(C, k=1)
    n_edges = int((_signed(C, graph_type)[pairs] > threshold).sum())
    values = np.sort(_signed(M, graph_type)[pairs])[::-1]
    cutoff = values[n_edges] if n_edges < len(values) else -np.inf
    return max(float(cutoff), 0.0)


def _signed(C: np.ndarray, graph_type: GraphType) -> np.ndarray:
    """C itself ("positive"), |C| ("absolute") or max(C, -cC) ("damped"), as a new array."""
    if graph_type.sign == "positive":
        return C.copy()
    if graph_type.sign == "absolute":
        return np.abs(C)
    if graph_type.sign == "damped":
        return np.maximum(C, -NEGATIVE_DAMPING * C)
    raise ValueError(f"Unknown sign {graph_type.sign!r}")


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
