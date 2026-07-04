import warnings
import numpy as np
from scipy.sparse.linalg import eigsh, expm_multiply


def exponential_centrality(matrix: np.ndarray) -> np.ndarray:
    """
    Compute exponential (subgraph) centrality for each node.

    For a symmetric adjacency matrix A, the centrality is the row sum of the
    matrix exponential e^A, i.e. expm(A) @ ones. Computed via Krylov
    subspace method to avoid forming the full n×n matrix exponential.

    Parameters
    ----------
    matrix : np.ndarray of shape (n, n)
        Symmetric adjacency matrix with edge weights in [0, 1] and 0 on the
        diagonal.

    Returns
    -------
    np.ndarray of shape (n,)
        Exponential centrality score for each node.
    """
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    ones = np.ones(matrix.shape[0])
    return expm_multiply(matrix, ones)


def katz_centrality(matrix: np.ndarray, alpha: float | None = None) -> np.ndarray:
    """
    Compute Katz centrality for each node.

    Centrality is the row sum of (I - αA)⁻¹. The series converges only when
    α < 1 / spectral_radius(A); a warning is raised if the provided α violates
    this.

    Parameters
    ----------
    matrix : np.ndarray of shape (n, n)
        Symmetric adjacency matrix with edge weights in [0, 1] and 0 on the
        diagonal.
    alpha : float or None
        Attenuation factor. If None, defaults to 0.5 / spectral_radius(A).

    Returns
    -------
    np.ndarray of shape (n,)
        Katz centrality score for each node.
    """
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        eigenvalues, _ = eigsh(matrix, k=1, which="LM")
        spectral_radius = np.abs(eigenvalues).max()
    except Exception:
        spectral_radius = 0.0
    if spectral_radius == 0:
        return np.ones(matrix.shape[0])
    alpha_max = 1.0 / spectral_radius

    if alpha is None:
        alpha = 0.5 * alpha_max
    elif alpha > alpha_max:
        warnings.warn(
            f"alpha={alpha:.6f} exceeds 1/spectral_radius={alpha_max:.6f}. "
            "The matrix (I - αA) may be singular or ill-conditioned.",
            UserWarning,
            stacklevel=2,
        )

    n = matrix.shape[0]
    return np.linalg.solve(np.eye(n) - alpha * matrix, np.ones(n))


def compute_centralities(
    centrality_fn,
    dates: np.ndarray,
    col_names: list[str],
    matrices: np.ndarray,
) -> dict[str, list[tuple[str, float]]]:
    """
    Apply a centrality function to each date's matrix and return sorted rankings.

    Parameters
    ----------
    centrality_fn : callable
        A centrality function that takes a (n, n) matrix and returns an
        (n,) array of scores (e.g. exponential_centrality, katz_centrality).
    dates : np.ndarray of shape (n_dates,)
    col_names : list of str, length n_cols
    matrices : np.ndarray of shape (n_dates, n_cols, n_cols)

    Returns
    -------
    dict mapping each date (as str) to a list of (col_name, score) tuples
    sorted by centrality descending.
    """
    result = {}
    for date, matrix in zip(dates, matrices):
        scores = centrality_fn(matrix)
        ranking = sorted(zip(col_names, scores.tolist()), key=lambda x: x[1], reverse=True)
        result[str(date)] = ranking
    return result


if __name__ == "__main__":
    from graph import load_graph
    dates, col_names, matrices = load_graph('graph.npz')
    rankings = compute_centralities(katz_centrality, dates, col_names, matrices)
    m = 10
    for key, value in rankings.items():
        print(key)
        print(value[:m])
        print(value[-m:])
        print()