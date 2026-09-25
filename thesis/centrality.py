"""
Step 4 — How central is each bond in the graph? (paper, Sec. 2.1)

Every centrality has the same form:

    centrality(graph, parameter) -> scores, one per node (higher = more central)

or None if the measure is not available for that kind of graph (the NBTW
exponential measures exist only for unweighted graphs).

The parameter is a fraction of the measure's largest allowed parameter
(α / α_max), as the paper reports it, or None for parameter-free measures.
config.CENTRALITIES lists which (function, parameter) pairs are tried.

Most measures count walks through the graph — sequences of edges — with
longer walks counting less:
    Katz          all walks from a node, a walk of length k weighted α^k
    exponential   the same with weight β^k / k!
    subgraph      only closed walks (returning to the node itself)
    NBTW          only nonbacktracking walks (never going straight back i -> j -> i)
"""
import numpy as np
import igraph as ig
from scipy.linalg import cho_factor, cho_solve, expm

from thesis.graph import Graph


# ----------------------------------------------------------- simple measures

def degree(graph: Graph, parameter=None) -> np.ndarray:
    """Sum of a node's edge weights (number of edges if unweighted) (Sec. 2.1.1)."""
    return graph.A.sum(axis=1)


def betweenness(graph: Graph, parameter=None) -> np.ndarray:
    """
    How often a node lies on the shortest path between two other nodes
    (Sec. 2.1.10). Loops are ignored. In a weighted graph an edge's length is
    1 / weight, so strongly correlated bonds are close; in an unweighted
    graph every edge has length 1.
    """
    i, j = np.nonzero(np.triu(graph.A, k=1))
    network = ig.Graph(n=graph.n_nodes, edges=np.column_stack([i, j]).tolist())
    lengths = (1.0 / graph.A[i, j]).tolist() if graph.weighted else None
    return np.asarray(network.betweenness(weights=lengths))


# ------------------------------------------------------------------- Katz

def katz(graph: Graph, fraction: float) -> np.ndarray:
    """
    Katz centrality (I - αA)⁻¹·1 with α = fraction / ρ(A) (Sec. 2.1.2).
    Small α approaches degree, α near 1/ρ approaches eigenvector centrality.
    """
    if not graph.has_edges:
        return np.ones(graph.n_nodes)
    alpha = fraction / graph.spectral_radius
    return graph.row_sums_of(lambda lam: 1.0 / (1.0 - alpha * lam))


def katz_subgraph(graph: Graph, fraction: float) -> np.ndarray:
    """Katz subgraph centrality diag (I - αA)⁻¹, closed walks only (Sec. 2.1.3)."""
    if not graph.has_edges:
        return np.ones(graph.n_nodes)
    alpha = fraction / graph.spectral_radius
    return graph.diagonal_of(lambda lam: 1.0 / (1.0 - alpha * lam))


def katz_min(graph: Graph, parameter=None) -> np.ndarray:
    """
    Katz centrality with α = (1 - e^{-ρ}) / ρ, the value whose ranking is
    closest to exponential centrality (Sec. 3.4, Aprahamian et al. 2016).

    Then 1 - αλ = (ρ - λ)/ρ + (λ/ρ)·e^{-ρ}. For ρ above about 37, e^{-ρ}
    rounds to 0 and 1 - αρ would become exactly 0, so the denominator is
    multiplied by e^{ρ} (capped to stay finite). That rescales every score
    by the same factor and leaves the ranking unchanged.
    """
    if not graph.has_edges:
        return np.ones(graph.n_nodes)
    rho = graph.spectral_radius
    return graph.row_sums_of(
        lambda lam: 1.0 / ((rho - lam) / rho * np.exp(min(rho, 700.0)) + lam / rho)
    )


# ------------------------------------------------------------ exponential

def exponential(graph: Graph, beta: float) -> np.ndarray:
    """
    Exponential centrality e^{βA}·1 (Sec. 2.1.4).

    Computed as e^{β(A - λ_max I)}·1, which divides every score by the same
    number e^{βλ_max}: the ranking is unchanged, but nothing overflows on
    dense graphs, where λ_max can be in the hundreds.
    """
    lam_max = graph.eigendecomposition[0].max()
    return graph.row_sums_of(lambda lam: np.exp(beta * (lam - lam_max)))


def exponential_subgraph(graph: Graph, beta: float) -> np.ndarray:
    """Exponential subgraph centrality diag e^{βA}, closed walks only (Sec. 2.1.5)."""
    lam_max = graph.eigendecomposition[0].max()
    return graph.diagonal_of(lambda lam: np.exp(beta * (lam - lam_max)))


# ------------------------------------------------------------------- NBTW

def nbtw_matrix(A: np.ndarray, alpha: float) -> np.ndarray:
    """
    Ψ(α) of Theorem 2.1 for an undirected, weighted graph, possibly with loops:

        Ψ_ii = 1 + Σ_j α²A_ij² / (1 - α²A_ij²) - αA_ii / (1 - α²A_ii²)
        Ψ_ij = -αA_ij / (1 - α²A_ij²)                         (i ≠ j)

    Its inverse counts nonbacktracking walks: Ψ(α)⁻¹ = Σ_k α^k P_k, where
    (P_k)_ij is the weighted number of such walks of length k from i to j.
    """
    aA = alpha * A
    denom = 1.0 - aA ** 2
    Psi = -aA / denom
    Psi[np.diag_indices_from(Psi)] += 1.0 + (aA ** 2 / denom).sum(axis=1)
    return Psi


def _is_positive_definite(M: np.ndarray) -> bool:
    try:
        np.linalg.cholesky(M)
        return True
    except np.linalg.LinAlgError:
        return False


def nbtw_radius(A: np.ndarray, rel_tol: float = 1e-4) -> float:
    """
    μ, the largest α for which the NBTW walk count Σ α^k P_k converges.

    The counts P_k are nonnegative, so (Pringsheim's theorem) the series
    first breaks down on the positive real axis: at the first α > 0 where
    Ψ(α), which equals I at α = 0, stops being positive definite — or where
    an entry of Ψ blows up, at α = 1 / max A_ij. Found by bisection; the
    paper notes μ > 1/ρ(A), which gives the starting lower bound.
    """
    a_max = float(A.max())
    if a_max <= 0:
        return np.inf
    upper = 1.0 / a_max

    rho = float(np.max(np.abs(np.linalg.eigvalsh(A))))
    lo = min(1.0 / rho, upper) * (1 - 1e-9)
    if not _is_positive_definite(nbtw_matrix(A, lo)):
        lo = 0.0
    hi = upper * (1 - 1e-12)
    if _is_positive_definite(nbtw_matrix(A, hi)):
        return upper
    while hi - lo > rel_tol * hi:
        mid = 0.5 * (lo + hi)
        if _is_positive_definite(nbtw_matrix(A, mid)):
            lo = mid
        else:
            hi = mid
    return lo


def _nbtw_factor(graph: Graph, fraction: float):
    """Cholesky factor of Ψ(fraction · μ), shared by nbtw and nbtw_subgraph."""
    mu = graph.cached("nbtw_radius", lambda: nbtw_radius(graph.A))
    return graph.cached(("nbtw_factor", fraction),
                        lambda: cho_factor(nbtw_matrix(graph.A, fraction * mu)))


def nbtw(graph: Graph, fraction: float) -> np.ndarray:
    """NBTW centrality Ψ(α)⁻¹·1 with α = fraction · μ (Sec. 2.1.6)."""
    if not graph.has_edges:
        return np.ones(graph.n_nodes)
    return cho_solve(_nbtw_factor(graph, fraction), np.ones(graph.n_nodes))


def nbtw_subgraph(graph: Graph, fraction: float) -> np.ndarray:
    """NBTW subgraph centrality diag Ψ(α)⁻¹, closed walks only (Sec. 2.1.7)."""
    if not graph.has_edges:
        return np.ones(graph.n_nodes)
    return np.diag(cho_solve(_nbtw_factor(graph, fraction), np.eye(graph.n_nodes)))


# ------------------------------------------------------ exponential NBTW

# NBTW exponential centralities are computed for β = step, 2·step, ...,
# 1.0 in one go (see _nbtw_exponential_by_beta), so every β must be a
# multiple of this step.
NBTW_EXPONENTIAL_STEP = 0.1


def _nbtw_exponential_by_beta(A: np.ndarray, rho: float) -> dict[int, np.ndarray]:
    """
    F(β) = Σ_k β^k p_k(A) / k! for β = k·step, k = 1, ..., 10, each divided
    by e^{βc} (c = max(ρ(A), 1)) so nothing overflows; rankings are unchanged.

    Unweighted graphs only. Arrigo, Grindrod, Higham & Noferini (2018,
    Thm 2.1): with D = diag(A²) and the 2n x 2n matrix Y = [0 I; I-D A],

        F(β) = I + β [I 0] ψ₁(βY) W,     W = [A; A² - D],   ψ₁(z) = (e^z - 1)/z.

    ψ₁ is obtained from one matrix exponential (Saad 1992):

        exp([β(Y - cI)  W; 0  -βcI]) = [e^{β(Y-cI)}  e^{-βc} ψ₁(βY) W; 0  e^{-βc} I].

    That exponential is computed once for β = step and raised to the powers
    k = 2, ..., 10 by repeated multiplication, using the block structure.
    The formula also holds with loops (D then counts the loop); the tests
    check it against a brute-force count of nonbacktracking walks.
    """
    n = A.shape[0]
    h = NBTW_EXPONENTIAL_STEP
    c = max(rho, 1.0)
    I = np.eye(n)
    D = np.diag(np.diag(A @ A))
    Y = np.block([[np.zeros((n, n)), I], [I - D, A]])
    W = np.vstack([A, A @ A - D])

    augmented = np.zeros((3 * n, 3 * n))
    augmented[:2 * n, :2 * n] = h * (Y - c * np.eye(2 * n))
    augmented[:2 * n, 2 * n:] = W
    augmented[2 * n:, 2 * n:] = -h * c * I
    step = expm(augmented)
    E1, T1, e1 = step[:2 * n, :2 * n], step[:2 * n, 2 * n:], np.exp(-h * c)

    # exp(k·augmented) = [E_k T_k; 0 e1^k I]; its top-right block is k times
    # the one needed for β = k·h, because it is linear in W.
    result = {}
    E, T = E1, T1
    for k in range(1, int(round(1.0 / h)) + 1):
        if k > 1:
            E, T = E @ E1, E @ T1 + T * e1
        result[k] = np.exp(-k * h * c) * I + h * T[:n]      # = e^{-βc} F(β)
    return result


def _nbtw_exponential_matrix(graph: Graph, beta: float) -> np.ndarray | None:
    """e^{-βc} F(β) from _nbtw_exponential_by_beta, or None for weighted graphs."""
    if graph.weighted:
        return None
    k = int(round(beta / NBTW_EXPONENTIAL_STEP))
    if not np.isclose(k * NBTW_EXPONENTIAL_STEP, beta):
        raise ValueError(f"beta={beta} is not a multiple of {NBTW_EXPONENTIAL_STEP}")
    by_beta = graph.cached("nbtw_exponential",
                           lambda: _nbtw_exponential_by_beta(graph.A, graph.spectral_radius))
    return by_beta[k]


def nbtw_exponential(graph: Graph, beta: float) -> np.ndarray | None:
    """
    NBTW exponential centrality Σ_k β^k p_k(A) / k! · 1 (Sec. 2.1.8).

    Unweighted graphs only: returns None for weighted graphs. For those no
    formula on the n x n node level exists (Arrigo, Higham, Noferini &
    Wood 2024, Sec. 4); their edge-level formula (Thm 4.6) needs an m x m
    matrix, m up to ~550,000 directed edges here, which is not feasible.
    """
    F = _nbtw_exponential_matrix(graph, beta)
    return None if F is None else F.sum(axis=1)


def nbtw_exponential_subgraph(graph: Graph, beta: float) -> np.ndarray | None:
    """NBTW exponential subgraph centrality diag Σ_k β^k p_k(A) / k! (Sec. 2.1.9); unweighted graphs only."""
    F = _nbtw_exponential_matrix(graph, beta)
    return None if F is None else np.diag(F).copy()


# ---------------------------------------------------------------- ranking

def rank_by_centrality(scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Node indices from most to least central.

    Scores equal to 10 significant digits are ties (e.g. isolated nodes,
    which differ only by rounding noise). Ties are ordered randomly, so the
    column order of the data file cannot decide which bonds are picked.
    """
    scale = np.max(np.abs(scores))
    rounded = np.round(scores / scale, 10) if scale > 0 else np.zeros_like(scores)
    return np.lexsort((rng.random(len(scores)), -rounded))
