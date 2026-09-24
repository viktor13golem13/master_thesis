import numpy as np

from metrics import performance_metrics


def portfolio_returns_by_size(
    returns_by_year: dict[int, np.ndarray],
    picks_by_year: dict[int, np.ndarray],
    sizes: tuple[int, ...],
) -> np.ndarray:
    """
    Daily returns of walk-forward equal-weight portfolios of several sizes.

    picks_by_year[y] lists series indices in selection order (most central
    first, or most peripheral first) using data up to the end of formation
    year y. The first m of them are held with weight 1/m throughout year
    y + 1, i.e. rebalanced daily as in r_p,t = x^T r_t.

    Returns
    -------
    np.ndarray of shape (n_days, len(sizes)); column k is the portfolio of
    the sizes[k] first picks, days in chronological order.
    """
    columns = np.asarray(sizes) - 1
    counts = np.asarray(sizes, dtype=float)
    pieces = []
    for formation_year in sorted(picks_by_year):
        held = returns_by_year[formation_year + 1][:, picks_by_year[formation_year]]
        pieces.append(np.cumsum(held, axis=1)[:, columns] / counts)
    return np.concatenate(pieces)


def evaluate_sizes(
    returns_by_year: dict[int, np.ndarray],
    rf_by_year: dict[int, np.ndarray],
    picks_by_year: dict[int, np.ndarray],
    sizes: tuple[int, ...],
) -> list[dict[str, float]]:
    """Performance metrics of each portfolio size, in the order of sizes."""
    rp = portfolio_returns_by_size(returns_by_year, picks_by_year, sizes)
    rf = np.concatenate([rf_by_year[y + 1] for y in sorted(picks_by_year)])
    return [performance_metrics(rp[:, k], rf) for k in range(len(sizes))]


def equal_weight_all(
    returns_by_year: dict[int, np.ndarray],
    rf_by_year: dict[int, np.ndarray],
    holding_years: list[int],
) -> dict[str, float]:
    """Benchmark: equal weight in every series, rebalanced daily."""
    rp = np.concatenate([returns_by_year[y].mean(axis=1) for y in holding_years])
    rf = np.concatenate([rf_by_year[y] for y in holding_years])
    return performance_metrics(rp, rf)
