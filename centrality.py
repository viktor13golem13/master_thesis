"""
Centrality measures from Arslan, Noferini & Vrontos (2024, Sec. 2.1).

Every function takes a symmetric, nonnegative adjacency matrix A and returns
a dict {(measure, parameter): scores}. The parameter is the fraction of the
measure's maximal parameter (alpha / alpha_max), or None for parameter-free
measures, matching how the paper reports alpha.
"""
import numpy as np
import igraph as ig
from scipy.linalg import cho_factor, cho_solve

KATZ_FRACTIONS = tuple(k / 10 for k in range(1, 10))         # alpha = f / rho(A)
NBTW_FRACTIONS = tuple(k / 10 for k in range(1, 10))         # alpha = f * mu
EXPONENTIAL_BETAS = tuple(k / 10 for k in range(1, 11))      # e^{beta A}, alpha_max = 1

NO_PARAM = None


def spectral_centralities(A: np.ndarray) -> dict[tuple[str, float | None], np.ndarray]:
    """
    Degree plus every walk-based centrality that is a matrix function of A.

    With A = Q diag(λ) Qᵀ, f(A)·1 = Q f(λ) Qᵀ1 and diag f(A) = (Q∘Q) f(λ), so
    one eigendecomposition serves all parameters:

        degree                A·1
        katz (α)              (I - αA)⁻¹·1,        α = f / ρ(A)
        katz_min              Katz with α = (1 - e^{-ρ}) / ρ  (Aprahamian et al. 2016)
        katz_subgraph (α)     diag (I - αA)⁻¹
        exponential (β)       e^{βA}·1
        exponential_subgraph  diag e^{βA}

    Exponentials are computed as e^{β(A - λ_max I)}, which rescales every
    score by the same factor e^{-βλ_max}: rankings are unchanged, but e^{βλ}
    can no longer overflow for dense graphs (λ_max can be in the hundreds).
    """
    n = A.shape[0]
    result = {("degree", NO_PARAM): A.sum(axis=1)}

    lam, Q = np.linalg.eigh(A)
    rho = float(np.max(np.abs(lam)))
    Qt1 = Q.T @ np.ones(n)
    Q2 = Q ** 2

    if rho < 1e-12:  # no edges: every node is equally (un)important
        flat = np.ones(n)
        for f in KATZ_FRACTIONS:
            result[("katz", f)] = flat
            result[("katz_subgraph", f)] = flat
        result[("katz_min", NO_PARAM)] = flat
    else:
        for f in KATZ_FRACTIONS:
            g = 1.0 / (1.0 - (f / rho) * lam)
            result[("katz", f)] = Q @ (g * Qt1)
            result[("katz_subgraph", f)] = Q2 @ g
        # Katz-min uses α = (1 - e^{-ρ}) / ρ, so 1 - αλ = (ρ - λ)/ρ + (λ/ρ) e^{-ρ}.
        # For ρ above ~37, e^{-ρ} underflows and 1 - αρ becomes exactly 0, so
        # the denominator is multiplied by e^{ρ} instead (capped to stay finite;
        # rankings are unchanged). As ρ grows this tends to eigenvector centrality.
        scaled_denom = (rho - lam) / rho * np.exp(min(rho, 700.0)) + lam / rho
        result[("katz_min", NO_PARAM)] = Q @ (Qt1 / scaled_denom)

    shifted = lam - lam.max()
    for beta in EXPONENTIAL_BETAS:
        g = np.exp(beta * shifted)
        result[("exponential", beta)] = Q @ (g * Qt1)
        result[("exponential_subgraph", beta)] = Q2 @ g
    return result


def nbtw_matrix(A: np.ndarray, alpha: float) -> np.ndarray:
    """
    Ψ(α) of Theorem 2.1 for an undirected, weighted graph, possibly with loops.

        Ψ_ii = 1 + Σ_j α²A_ij² / (1 - α²A_ij²) - αA_ii / (1 - α²A_ii²)
        Ψ_ij = -αA_ij / (1 - α²A_ij²)                      (i ≠ j)

    Ψ(α)⁻¹ = Σ_k α^k P_k, where (P_k)_ij is the weighted count of
    nonbacktracking walks of length k from i to j.
    """
    aA = alpha * A
    denom = 1.0 - aA ** 2
    Psi = -aA / denom
    Psi[np.diag_indices_from(Psi)] += 1.0 + (aA ** 2 / denom).sum(axis=1)
    return Psi


def nbtw_radius(A: np.ndarray, rel_tol: float = 1e-4) -> float:
    """
    Radius of convergence μ of the NBTW generating function Σ α^k P_k.

    The coefficients P_k are nonnegative, so by Pringsheim's theorem the
    first singularity lies on the positive real axis: the first α > 0 where
    the symmetric matrix Ψ(α) (= I at α = 0) stops being positive definite,
    or where an entry of Ψ has a pole (α = 1 / max A_ij). Found by bisection
    on a Cholesky test; the paper notes μ > 1/ρ(A), which gives the lower
    bracket.
    """
    a_max = float(A.max())
    if a_max <= 0:
        return np.inf
    upper = 1.0 / a_max

    def positive_definite(alpha):
        try:
            np.linalg.cholesky(nbtw_matrix(A, alpha))
            return True
        except np.linalg.LinAlgError:
            return False

    rho = float(np.max(np.abs(np.linalg.eigvalsh(A))))
    lo = min(1.0 / rho, upper) * (1 - 1e-9)
    if not positive_definite(lo):
        lo = 0.0
    hi = upper * (1 - 1e-12)
    if positive_definite(hi):
        return upper
    while hi - lo > rel_tol * hi:
        mid = 0.5 * (lo + hi)
        if positive_definite(mid):
            lo = mid
        else:
            hi = mid
    return lo


def nbtw_centralities(A: np.ndarray) -> dict[tuple[str, float | None], np.ndarray]:
    """
    NBTW centrality Ψ(α)⁻¹·1 and NBTW subgraph centrality diag Ψ(α)⁻¹
    (paper Sec. 2.1.6-2.1.7) for α = f·μ, f in NBTW_FRACTIONS.
    """
    n = A.shape[0]
    mu = nbtw_radius(A)
    result = {}
    if not np.isfinite(mu):  # no edges
        for f in NBTW_FRACTIONS:
            result[("nbtw", f)] = np.ones(n)
            result[("nbtw_subgraph", f)] = np.ones(n)
        return result
    for f in NBTW_FRACTIONS:
        factor = cho_factor(nbtw_matrix(A, f * mu))
        result[("nbtw", f)] = cho_solve(factor, np.ones(n))
        result[("nbtw_subgraph", f)] = np.diag(cho_solve(factor, np.eye(n)))
    return result


def betweenness_centrality(A: np.ndarray, weighted: bool) -> dict[tuple[str, float | None], np.ndarray]:
    """
    Betweenness centrality (paper Sec. 2.1.10), ignoring loops.

    For weighted graphs an edge's length is 1 / weight, so strongly
    correlated pairs are close; unweighted graphs use hop counts.
    """
    i, j = np.nonzero(np.triu(A, k=1))
    graph = ig.Graph(n=A.shape[0], edges=np.column_stack([i, j]).tolist())
    lengths = (1.0 / A[i, j]).tolist() if weighted else None
    return {("betweenness", NO_PARAM): np.asarray(graph.betweenness(weights=lengths))}


def all_centralities(A: np.ndarray, weighted: bool) -> dict[tuple[str, float | None], np.ndarray]:
    """Every centrality variant used in the grid search, keyed by (measure, parameter)."""
    return {
        **spectral_centralities(A),
        **nbtw_centralities(A),
        **betweenness_centrality(A, weighted),
    }


def rank_by_centrality(scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Node indices sorted from most to least central.

    Scores equal to 10 significant digits count as ties (e.g. isolated nodes,
    which differ only by rounding noise) and are ordered randomly, so column
    order in the data cannot decide which bonds are picked.
    """
    scale = np.max(np.abs(scores))
    rounded = np.round(scores / scale, 10) if scale > 0 else np.zeros_like(scores)
    return np.lexsort((rng.random(len(scores)), -rounded))
