"""
Parallel walk-forward grid search.

A graph configuration (window weighting, shrinkage, graph type, threshold) is
the unit of work: for every formation year it builds the graph, computes all
centralities, and then evaluates every (centrality, selection, portfolio
size) strategy on the following years.
"""
import itertools
import multiprocessing as mp
import zlib
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from centrality import all_centralities, rank_by_centrality
from correlation import WINDOW_WEIGHTINGS, compute_yearly_correlations
from evaluation import evaluate_sizes
from graph import GRAPH_TYPES, GraphType, adjacency_matrix

THRESHOLDS = tuple(k / 10 for k in range(10))
PORTFOLIO_SIZES = (5, 10, 15, 20, 25, 30, 40, 50, 75, 100)
SHRINKAGE = (False, True)


@dataclass(frozen=True)
class GraphConfig:
    weighting: str
    shrink: bool
    graph_type: GraphType
    threshold: float

    def seed(self) -> int:
        """Stable seed for tie-breaking, so results are reproducible."""
        return zlib.crc32(repr(self).encode())


def all_graph_configs() -> list[GraphConfig]:
    return [
        GraphConfig(weighting, shrink, graph_type, threshold)
        for weighting, shrink, graph_type, threshold in itertools.product(
            WINDOW_WEIGHTINGS, SHRINKAGE, GRAPH_TYPES, THRESHOLDS,
        )
    ]


# Data shared with worker processes. Set before the pool is created; with the
# fork start method the workers inherit it without copying or pickling.
_SHARED: dict = {}


def _correlation_task(args):
    weighting, shrink, year = args
    C = compute_yearly_correlations(_SHARED["returns"], [year], weighting=weighting, shrink=shrink)[year]
    return (weighting, shrink, year), C


def _graph_config_task(config: GraphConfig) -> list[dict]:
    correlations = _SHARED["correlations"]
    formation_years = _SHARED["formation_years"]
    returns_by_year = _SHARED["returns_by_year"]
    rf_by_year = _SHARED["rf_by_year"]
    n_picks = max(PORTFOLIO_SIZES)

    picks = {}      # (measure, param, selection) -> {formation_year: indices}
    isolated = []
    for year in formation_years:
        A = adjacency_matrix(correlations[(config.weighting, config.shrink, year)], config.threshold, config.graph_type)
        isolated.append(float(np.mean((A - np.diag(np.diag(A))).sum(axis=1) == 0)))
        rng = np.random.default_rng([config.seed(), year])
        for (measure, param), scores in all_centralities(A, config.graph_type.weighted).items():
            order = rank_by_centrality(scores, rng)
            picks.setdefault((measure, param, "central"), {})[year] = order[:n_picks]
            picks.setdefault((measure, param, "peripheral"), {})[year] = order[::-1][:n_picks]

    base = {
        "weighting": config.weighting,
        "shrink": config.shrink,
        **{k: v for k, v in asdict(config.graph_type).items()},
        "threshold": config.threshold,
        "isolated_frac": float(np.mean(isolated)),
    }
    rows = []
    for (measure, param, selection), by_year in picks.items():
        metrics = evaluate_sizes(returns_by_year, rf_by_year, by_year, PORTFOLIO_SIZES)
        for size, m in zip(PORTFOLIO_SIZES, metrics):
            rows.append({**base, "centrality": measure, "alpha": param,
                         "selection": selection, "m": size, **m})
    return rows


def run_grid_search(
    returns: pd.DataFrame,
    rf_returns: pd.Series,
    formation_years: list[int],
    configs: list[GraphConfig] | None = None,
    processes: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Evaluate every strategy of the grid, walking forward year by year.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily simple returns of the investable series, covering every
        formation year and the year after each.
    rf_returns : pd.Series
        Daily simple risk-free returns on the same days.
    formation_years : list of int
        Years whose correlations select the portfolio held in the next year.
    configs : list of GraphConfig, optional
        Graph configurations to run (default: the full grid).
    processes : int, optional
        Worker processes (default: all cores).

    Returns
    -------
    pd.DataFrame, one row per strategy, sorted by Sharpe ratio (descending).
    """
    configs = configs if configs is not None else all_graph_configs()
    years = returns.index.year
    _SHARED.clear()
    _SHARED.update(
        returns=returns,
        formation_years=formation_years,
        returns_by_year={y + 1: returns.loc[years == y + 1].to_numpy(dtype=float) for y in formation_years},
        rf_by_year={y + 1: rf_returns.loc[rf_returns.index.year == y + 1].to_numpy(dtype=float) for y in formation_years},
    )

    context = mp.get_context("fork")
    correlation_keys = sorted({(c.weighting, c.shrink) for c in configs})
    with context.Pool(processes) as pool:
        tasks = [(w, s, y) for (w, s) in correlation_keys for y in formation_years]
        if verbose:
            print(f"Computing {len(tasks)} yearly correlation matrices...", flush=True)
        _SHARED["correlations"] = dict(pool.map(_correlation_task, tasks))

    rows = []
    with context.Pool(processes) as pool:  # new pool: workers must inherit the correlations
        if verbose:
            print(f"Evaluating {len(configs)} graph configurations...", flush=True)
        for k, result in enumerate(pool.imap_unordered(_graph_config_task, configs), 1):
            rows.extend(result)
            if verbose and (k % 25 == 0 or k == len(configs)):
                print(f"  {k}/{len(configs)} graphs done ({len(rows)} strategies)", flush=True)

    return pd.DataFrame(rows).sort_values("SR", ascending=False, ignore_index=True)
